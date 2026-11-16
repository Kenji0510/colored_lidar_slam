//! Region-coherent RGB reprojection, independent of the grayscale 9x9 ESIKF update.
//! 1 m regions are management cells. Adjacent cells are assigned one source
//! image together; an existing continuous assignment is never split by a local
//! image choice. RGB is sampled at the current mean without changing geometry.
//! Normals are averaged over 30-frame observation windows to rank whole images.
//! They never gate per-point RGB; missing/oblique normals cannot create holes.
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, VecDeque},
    path::{Path, PathBuf},
    sync::Arc,
};

use image::RgbImage;
use nalgebra::{Matrix4, Point3, Vector3};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::{
    camera::{CameraIntrinsics, project_camera_point},
    voxel_map::{LOCALMap, SurfaceStatus, VoxelCell, VoxelKey},
};

#[derive(Debug, Clone)]
pub struct MapRecolorConfig {
    /// Completed LiDAR frames, NOT camera frames. Default: 30, 60, 90, ...
    pub update_every_frames: u64,
    /// Fixed world-grid region side length [m]. Not the 9x9 alignment patch.
    pub region_size_m: f64,
    pub min_observed_frames: u64,
    /// New source images must cover this fraction of mature points in the region.
    pub min_coverage: f64,
    /// Quality-only hysteresis. A better mean-normal view/current-window image
    /// can replace a source without satisfying this quality improvement.
    pub switch_improvement_ratio: f64,
    /// Quality floor when a mean normal is unavailable. With a mean normal,
    /// angle ranks images and this floor cannot eliminate the best camera view.
    pub min_group_quality_ratio: f64,
    /// Preference only: if every image exceeds this angle, still use the most
    /// frontal available image. No point or image is rejected by this angle.
    pub preferred_view_angle_deg: f64,
    pub max_recent_images: usize,
    pub max_cached_rgb_images: usize,
    pub max_score_samples_per_region: usize,
}

impl Default for MapRecolorConfig {
    fn default() -> Self {
        Self {
            update_every_frames: 30,
            region_size_m: 1.0,
            min_observed_frames: 2,
            min_coverage: 0.90,
            switch_improvement_ratio: 0.15,
            min_group_quality_ratio: 0.75,
            preferred_view_angle_deg: 45.0,
            max_recent_images: 30,
            max_cached_rgb_images: 8,
            max_score_samples_per_region: 64,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MapRecolorLog {
    pub ran: bool,
    pub final_flush: bool,
    pub completed_frames: u64,
    pub recent_images: usize,
    pub evaluated_regions: usize,
    pub recolored_regions: usize,
    pub switched_regions: usize,
    pub deferred_regions: usize,
    pub colored_points: usize,
    pub cleared_points: usize,
    pub failed_images: usize,
    pub recolored_groups: usize,
    pub merged_groups: usize,
    pub largest_group_regions: usize,
    pub window_first_frame: u64,
    pub window_last_frame: u64,
    pub window_images: usize,
    pub window_observations: u64,
    pub window_normal_samples: u64,
    /// Sign-aligned sum of available per-frame voxel normals / sample count.
    pub window_mean_normal: Option<[f64; 3]>,
    pub window_normal_coherence: Option<f64>,
    pub window_centroid: Option<[f64; 3]>,
    pub preferred_view_groups: usize,
    pub best_oblique_view_groups: usize,
    pub no_normal_view_groups: usize,
    pub retained_image_groups: usize,
    pub group_views: Vec<GroupViewLog>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupViewLog {
    pub image_path: PathBuf,
    pub regions: usize,
    pub view_cosine: Option<f64>,
    pub preferred_angle_passed: Option<bool>,
    pub from_current_window: bool,
}

#[derive(Clone, Default, PartialEq)]
struct WindowView {
    observations: u64,
    normal_samples: u64,
    mean_normal: Option<Vector3<f64>>,
    normal: Option<Vector3<f64>>,
    centroid: Option<Point3<f64>>,
}

#[derive(Default)]
struct WindowObservation {
    frames: u64,
    normal_samples: u64,
    normal_sum: Vector3<f64>,
    position_sum: Vector3<f64>,
}

fn valid_map_normal(cell: &VoxelCell) -> Option<Vector3<f64>> {
    if cell.surface_status != SurfaceStatus::Planar
        || !cell.mean.coords.iter().all(|v| v.is_finite())
    {
        return None;
    }
    let normal = cell.surface_plane?.normal.cast::<f64>();
    let length = normal.norm();
    (length.is_finite() && length > 1e-8).then(|| normal / length)
}

fn align_normal(mut normal: Vector3<f64>, reference: &Vector3<f64>) -> Vector3<f64> {
    let scale = normal.norm() * reference.norm();
    let dot = normal.dot(reference);
    if scale > 1e-12 && dot < -1e-12 * scale {
        return -normal;
    }
    if scale <= 1e-12 || dot.abs() <= 1e-12 * scale {
        let axis = (0..3)
            .max_by(|a, b| normal[*a].abs().total_cmp(&normal[*b].abs()))
            .unwrap();
        if normal[axis] < 0.0 {
            normal = -normal;
        }
    }
    normal
}

type RegionKey = (i32, i32, i32);

#[derive(Debug, Clone, Serialize)]
pub struct MapRegionSource {
    pub region: [i32; 3],
    pub image_timestamp_sec: f64,
    pub image_path: PathBuf,
}

struct RegionData {
    key: RegionKey,
    voxels: Vec<VoxelKey>,
    samples: Vec<Point3<f32>>,
}

/// One indivisible, connected set of regions already using the same image.
/// A previously uncolored region is initially a one-region unit.
struct AssignmentUnit {
    regions: Vec<usize>,
    source: Option<Arc<SourceImage>>,
    neighbors: Vec<usize>,
    dirty: bool,
    points: usize,
    baseline_score: f64,
}

struct ImageOffers {
    source: Arc<SourceImage>,
    /// Point-count-weighted scores; None means the entire unit is ineligible.
    scores: Vec<Option<f64>>,
    verified: Vec<bool>,
    view_cosine: Option<f64>,
    from_current_window: bool,
}

struct GroupProposal {
    offer: usize,
    source_id: u64,
    units: Vec<usize>,
    regions: usize,
    points: usize,
    score: f64,
    view_cosine: Option<f64>,
    preferred_angle_passed: bool,
    from_current_window: bool,
}

impl PartialEq for GroupProposal {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for GroupProposal {}
impl PartialOrd for GroupProposal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for GroupProposal {
    fn cmp(&self, other: &Self) -> Ordering {
        // Keep large coherent regions. Among images covering the same extent,
        // prefer THIS window, then the best mean-normal view, then image quality.
        self.regions
            .cmp(&other.regions)
            .then_with(|| self.from_current_window.cmp(&other.from_current_window))
            .then_with(|| {
                self.preferred_angle_passed
                    .cmp(&other.preferred_angle_passed)
            })
            .then_with(|| {
                self.view_cosine
                    .unwrap_or(-1.0)
                    .total_cmp(&other.view_cosine.unwrap_or(-1.0))
            })
            .then_with(|| self.score.total_cmp(&other.score))
            .then_with(|| self.points.cmp(&other.points))
            .then_with(|| other.source_id.cmp(&self.source_id))
            .then_with(|| other.units.cmp(&self.units))
    }
}

fn adjacent_regions(key: RegionKey) -> impl Iterator<Item = RegionKey> {
    // Face adjacency in the fixed world grid; disconnected surfaces stay apart.
    [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]
    .into_iter()
    .filter_map(move |(x, y, z)| {
        Some((
            key.0.checked_add(x)?,
            key.1.checked_add(y)?,
            key.2.checked_add(z)?,
        ))
    })
}

/// Compact local image-quality grid. No full-resolution RGB history is retained.
/// Laplacian variance ranks sharpness; clipping penalizes under/overexposure.
struct QualityGrid {
    columns: usize,
    tiles: Vec<(f64, f64)>,
}

impl QualityGrid {
    const TILE: u32 = 32;

    fn new(image: &RgbImage) -> Self {
        let columns = image.width().div_ceil(Self::TILE) as usize;
        let rows = image.height().div_ceil(Self::TILE) as usize;
        let mut sums = vec![(0.0, 0.0, 0.0, 0usize); columns * rows];
        let gray = |x, y| {
            let p = image.get_pixel(x, y).0;
            0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64
        };
        // Sample on a 4-pixel grid, but use original-resolution neighbours.
        for y in (1..image.height().saturating_sub(1)).step_by(4) {
            for x in (1..image.width().saturating_sub(1)).step_by(4) {
                let center = gray(x, y);
                let lap = gray(x - 1, y) + gray(x + 1, y) + gray(x, y - 1) + gray(x, y + 1)
                    - 4.0 * center;
                let sum =
                    &mut sums[(y / Self::TILE) as usize * columns + (x / Self::TILE) as usize];
                sum.0 += lap;
                sum.1 += lap * lap;
                sum.2 += if center <= 5.0 || center >= 250.0 {
                    1.0
                } else {
                    0.0
                };
                sum.3 += 1;
            }
        }
        let tiles = sums
            .into_iter()
            .map(|(s, s2, clipped, n)| {
                if n == 0 {
                    return (0.0, 0.0);
                }
                let variance = (s2 / n as f64 - (s / n as f64).powi(2)).max(0.0);
                (variance, clipped / n as f64)
            })
            .collect();
        Self { columns, tiles }
    }

    fn weight(&self, u: u32, v: u32) -> f64 {
        let (variance, clipped) =
            self.tiles[(v / Self::TILE) as usize * self.columns + (u / Self::TILE) as usize];
        // Textureless walls remain usable. Sharpness is a preference, not a gate.
        (0.2 + 0.8 * (variance / (variance + 100.0)).sqrt()) * (1.0 - 0.8 * clipped)
    }
}

struct SourceImage {
    id: u64,
    path: PathBuf,
    camera_from_world: Matrix4<f64>,
    camera_center_world: Point3<f64>,
    intrinsics: CameraIntrinsics,
    quality: QualityGrid,
}

impl SourceImage {
    fn view_cosine(&self, window: &WindowView) -> Option<f64> {
        let normal = window.normal.as_ref()?;
        let to_camera = self.camera_center_world - window.centroid.as_ref()?;
        let distance = to_camera.norm();
        if !distance.is_finite() || distance < 1e-8 {
            return None;
        }
        let cosine = normal.dot(&(to_camera / distance)).abs();
        cosine.is_finite().then_some(cosine.clamp(0.0, 1.0))
    }
    fn project(&self, world: &Point3<f32>) -> Option<(u32, u32, f64)> {
        let camera_point = self.camera_from_world.transform_point(&world.cast::<f64>());
        let (u, v) = project_camera_point(&camera_point, &self.intrinsics)?;
        Some((u, v, camera_point.coords.norm()))
    }

    fn score(&self, points: &[Point3<f32>]) -> (f64, f64) {
        let mut count = 0;
        let mut quality_resolution = 0.0;
        for point in points {
            if let Some((u, v, distance)) = self.project(point) {
                count += 1;
                // Geometric coverage and image quality only. The window mean
                // normal ranks whole-image proposals separately, never pixels.
                quality_resolution += self.quality.weight(u, v)
                    * (self.intrinsics.fx * self.intrinsics.fy).sqrt()
                    / distance.max(0.01);
            }
        }
        let coverage = count as f64 / points.len().max(1) as f64;
        let score = quality_resolution / count.max(1) as f64 * coverage.powi(3);
        (coverage, score)
    }
}

/// Source-image metadata outlives the recent window while any region uses it.
/// Decoded RGB is a bounded LRU; old selected images can be reloaded from disk.
pub struct MapRecolorer {
    config: MapRecolorConfig,
    completed_frames: u64,
    recent: VecDeque<Arc<SourceImage>>,
    selected: FxHashMap<RegionKey, Arc<SourceImage>>,
    dirty: FxHashSet<RegionKey>,
    rgb_cache: VecDeque<(u64, Arc<RgbImage>)>,
    window_first_frame: u64,
    window_image_ids: FxHashSet<u64>,
    /// One sample per accepted map voxel per observed LiDAR frame, not per call.
    window_observations: FxHashMap<VoxelKey, WindowObservation>,
    pending_normal_keys: Vec<VoxelKey>,
    last_noted_frame_id: Option<u64>,
    active_view: WindowView,
    preferred_view_cosine: f64,
}

impl MapRecolorer {
    pub fn new(config: MapRecolorConfig) -> Self {
        assert!(config.update_every_frames > 0);
        assert!(config.region_size_m.is_finite() && config.region_size_m > 0.0);
        assert!(
            config.min_coverage.is_finite()
                && config.min_coverage > 0.0
                && config.min_coverage <= 1.0
        );
        assert!(
            config.switch_improvement_ratio.is_finite() && config.switch_improvement_ratio >= 0.0
        );
        assert!(
            config.min_group_quality_ratio.is_finite()
                && config.min_group_quality_ratio > 0.0
                && config.min_group_quality_ratio <= 1.0
        );
        assert!(
            config.preferred_view_angle_deg.is_finite()
                && (0.0..=90.0).contains(&config.preferred_view_angle_deg)
        );
        assert!(
            config.max_recent_images > 0
                && config.max_cached_rgb_images > 0
                && config.max_score_samples_per_region > 0
        );
        Self {
            preferred_view_cosine: config.preferred_view_angle_deg.to_radians().cos(),
            config,
            completed_frames: 0,
            recent: VecDeque::new(),
            selected: FxHashMap::default(),
            dirty: FxHashSet::default(),
            rgb_cache: VecDeque::new(),
            window_first_frame: 1,
            window_image_ids: FxHashSet::default(),
            window_observations: FxHashMap::default(),
            pending_normal_keys: Vec::new(),
            last_noted_frame_id: None,
            active_view: WindowView::default(),
        }
    }

    /// Call only after an accepted LiDAR registration and the visual pose update.
    /// The pose MUST be the camera pose at the image timestamp, not scan start.
    /// Timestamp bits identify a recorded image across repeated nearest matches.
    pub fn observe_image(
        &mut self,
        timestamp_sec: f64,
        path: &Path,
        rgb: &RgbImage,
        intrinsics: &CameraIntrinsics,
        camera_from_world: &Matrix4<f64>,
    ) -> bool {
        self.begin_window_if_needed();
        if !timestamp_sec.is_finite()
            || rgb.width() < 3
            || rgb.height() < 3
            || rgb.dimensions() != (intrinsics.width, intrinsics.height)
            || ![intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]
                .iter()
                .chain(intrinsics.distortion.iter())
                .all(|v| v.is_finite())
            || intrinsics.fx <= 0.0
            || intrinsics.fy <= 0.0
            || !camera_from_world.iter().all(|v| v.is_finite())
        {
            return false;
        }
        let id = timestamp_sec.to_bits();
        if self.recent.iter().any(|image| image.id == id) {
            self.window_image_ids.insert(id);
            return false;
        }
        let Some(world_from_camera) = camera_from_world.try_inverse() else {
            return false;
        };
        let camera_center_world = world_from_camera.transform_point(&Point3::origin());
        if !camera_center_world.coords.iter().all(|v| v.is_finite()) {
            return false;
        }
        self.window_image_ids.insert(id);
        self.recent.push_back(Arc::new(SourceImage {
            id,
            path: path.to_path_buf(),
            camera_from_world: *camera_from_world,
            camera_center_world,
            intrinsics: intrinsics.clone(),
            quality: QualityGrid::new(rgb),
        }));
        while self.recent.len() > self.config.max_recent_images {
            self.recent.pop_front();
        }
        true
    }

    /// Use stable voxel-grid centres for region ownership: a moving mean must not
    /// move the same map cell between different source-image owners.
    fn region_key(&self, key: &VoxelKey, voxel_size: f32) -> RegionKey {
        let index =
            |i| (((i as f64 + 0.5) * voxel_size as f64) / self.config.region_size_m).floor() as i32;
        (index(key.ix), index(key.iy), index(key.iz))
    }

    /// Call immediately after each accepted global-map update, including frames
    /// without an RGB image. Changed means/new cells are recolored at the next tick.
    pub fn note_map_update(&mut self, map: &LOCALMap) {
        self.begin_window_if_needed();
        if let Some(frame) = map.frame_index.back() {
            let new_frame = self.last_noted_frame_id != Some(frame.frame_id);
            for key in &frame.dirty_keys {
                self.dirty
                    .insert(self.region_key(key, map.config.index_voxel_size));
                if new_frame {
                    let observation = self.window_observations.entry(*key).or_default();
                    observation.frames = observation.frames.saturating_add(1);
                    self.pending_normal_keys.push(*key);
                }
            }
            self.last_noted_frame_id = Some(frame.frame_id);
        }
    }

    fn begin_window_if_needed(&mut self) {
        // Reset at the NEXT input frame, not immediately after the 30th tick.
        // A final flush at frame 30 can then reuse that same window's data after
        // final plane classification. Normal windows still remain disjoint.
        if self.completed_frames >= self.window_first_frame
            && self.completed_frames % self.config.update_every_frames == 0
        {
            self.window_first_frame = self.completed_frames + 1;
            self.window_observations.clear();
            self.pending_normal_keys.clear();
            self.window_image_ids.clear();
            self.active_view = WindowView::default();
        }
    }

    fn summarize_window(&self, map: &LOCALMap) -> WindowView {
        let mut observations: Vec<_> = self.window_observations.iter().collect();
        observations.sort_unstable_by_key(|(key, _)| (key.ix, key.iy, key.iz));
        let mut view = WindowView::default();
        let mut sum = Vector3::<f64>::zeros();
        let mut position_sum = Vector3::<f64>::zeros();
        let mut anchor: Option<Vector3<f64>> = None;
        for (key, observation) in observations {
            view.observations = view.observations.saturating_add(observation.frames);
            let mut normal_sum = observation.normal_sum;
            let mut positions = observation.position_sum;
            let mut samples = observation.normal_samples;
            // Preserve each frame's known normal; only backfill observations
            // whose normals were not yet available because of delayed fitting.
            let missing = observation.frames.saturating_sub(samples);
            if missing > 0 {
                if let Some(cell) = map.voxel_map.get(key) {
                    if let Some(normal) = valid_map_normal(cell) {
                        normal_sum += align_normal(normal, &normal_sum) * missing as f64;
                        positions += cell.mean.coords.cast::<f64>() * missing as f64;
                        samples = samples.saturating_add(missing);
                    }
                }
            }
            if samples == 0 {
                continue;
            }
            if let Some(anchor) = &anchor {
                normal_sum = align_normal(normal_sum, anchor);
            } else {
                normal_sum = align_normal(normal_sum, &Vector3::zeros());
                if normal_sum.norm() > 1e-8 {
                    anchor = Some(normal_sum.normalize());
                }
            }
            sum += normal_sum;
            position_sum += positions;
            view.normal_samples = view.normal_samples.saturating_add(samples);
        }
        if view.normal_samples > 0 {
            let mean = sum / view.normal_samples as f64;
            let length = mean.norm();
            if length.is_finite() && length > 1e-8 {
                view.mean_normal = Some(mean);
                view.normal = Some(mean / length);
                view.centroid = Some(Point3::from(position_sum / view.normal_samples as f64));
            }
        }
        view
    }

    fn sample_frame_normals(&mut self, map: &LOCALMap) {
        // Called after main's delayed classifier, once per completed frame.
        // No raw point history or full RGB frames need to be retained.
        for key in self.pending_normal_keys.drain(..) {
            let Some(cell) = map.voxel_map.get(&key) else {
                continue;
            };
            let Some(normal) = valid_map_normal(cell) else {
                continue;
            };
            let Some(observation) = self.window_observations.get_mut(&key) else {
                continue;
            };
            observation.normal_sum += align_normal(normal, &observation.normal_sum);
            observation.position_sum += cell.mean.coords.cast::<f64>();
            observation.normal_samples = observation.normal_samples.saturating_add(1);
        }
    }

    /// Count completed LiDAR frames, even those lacking a camera image or having
    /// rejected ICP. No RGB map updates occur between these scheduled calls.
    pub fn finish_frame<F>(&mut self, map: &mut LOCALMap, loader: F) -> MapRecolorLog
    where
        F: FnMut(&Path) -> anyhow::Result<RgbImage>,
    {
        self.begin_window_if_needed();
        self.sample_frame_normals(map);
        self.completed_frames += 1;
        if self.completed_frames % self.config.update_every_frames == 0 {
            self.recolor(map, loader, false)
        } else {
            MapRecolorLog {
                completed_frames: self.completed_frames,
                ..Default::default()
            }
        }
    }

    /// Save the last incomplete window too (e.g. frames 61..74 or a short run).
    pub fn flush<F>(&mut self, map: &mut LOCALMap, loader: F) -> MapRecolorLog
    where
        F: FnMut(&Path) -> anyhow::Result<RgbImage>,
    {
        self.recolor(map, loader, true)
    }

    /// Diagnostics: neighboring management regions should usually share paths.
    pub fn assignments(&self) -> Vec<MapRegionSource> {
        let mut assignments: Vec<_> = self
            .selected
            .iter()
            .map(|(key, source)| MapRegionSource {
                region: [key.0, key.1, key.2],
                image_timestamp_sec: f64::from_bits(source.id),
                image_path: source.path.clone(),
            })
            .collect();
        assignments.sort_unstable_by_key(|a| a.region);
        assignments
    }

    fn load_rgb<F>(&mut self, source: &SourceImage, loader: &mut F) -> anyhow::Result<Arc<RgbImage>>
    where
        F: FnMut(&Path) -> anyhow::Result<RgbImage>,
    {
        if let Some(index) = self.rgb_cache.iter().position(|(id, _)| *id == source.id) {
            let entry = self.rgb_cache.remove(index).unwrap();
            let image = Arc::clone(&entry.1);
            self.rgb_cache.push_back(entry);
            return Ok(image);
        }
        let image = loader(&source.path)?;
        anyhow::ensure!(
            image.dimensions() == (source.intrinsics.width, source.intrinsics.height),
            "source image dimensions changed: {}",
            source.path.display()
        );
        let image = Arc::new(image);
        self.rgb_cache.push_back((source.id, Arc::clone(&image)));
        while self.rgb_cache.len() > self.config.max_cached_rgb_images {
            self.rgb_cache.pop_front();
        }
        Ok(image)
    }

    fn collect_regions(&self, map: &LOCALMap) -> Vec<RegionData> {
        // Include clean neighbors: group selection must cross dirty-region
        // boundaries and must see the ENTIRE existing image assignment.
        let mut cells: FxHashMap<RegionKey, Vec<VoxelKey>> = FxHashMap::default();
        for (key, cell) in &map.voxel_map {
            if cell.observed_frames >= self.config.min_observed_frames
                && cell.mean.coords.iter().all(|v| v.is_finite())
            {
                cells
                    .entry(self.region_key(key, map.config.index_voxel_size))
                    .or_default()
                    .push(*key);
            }
        }
        let mut regions: Vec<_> = cells
            .into_iter()
            .map(|(key, mut voxels)| {
                voxels.sort_unstable_by_key(|k| (k.ix, k.iy, k.iz));
                let count = voxels.len().min(self.config.max_score_samples_per_region);
                let samples = (0..count)
                    .map(|i| map.voxel_map[&voxels[i * voxels.len() / count]].mean)
                    .collect();
                RegionData {
                    key,
                    voxels,
                    samples,
                }
            })
            .collect();
        regions.sort_unstable_by_key(|region| region.key);
        regions
    }

    fn assignment_units(&self, regions: &[RegionData]) -> Vec<AssignmentUnit> {
        let indices: FxHashMap<_, _> = regions
            .iter()
            .enumerate()
            .map(|(i, r)| (r.key, i))
            .collect();
        let mut owner = vec![usize::MAX; regions.len()];
        let mut units = Vec::new();
        for start in 0..regions.len() {
            if owner[start] != usize::MAX {
                continue;
            }
            let source = self.selected.get(&regions[start].key).cloned();
            let mut members = vec![start];
            owner[start] = units.len();
            let mut cursor = 0;
            while cursor < members.len() {
                let index = members[cursor];
                cursor += 1;
                // Unassigned cells start separately, so a new image need not
                // cover an entire unbounded, previously uncolored building.
                let Some(source) = &source else {
                    continue;
                };
                for key in adjacent_regions(regions[index].key) {
                    let Some(&next) = indices.get(&key) else {
                        continue;
                    };
                    if owner[next] == usize::MAX
                        && self.selected.get(&key).is_some_and(|s| s.id == source.id)
                    {
                        owner[next] = units.len();
                        members.push(next);
                    }
                }
            }
            members.sort_unstable();
            let points = members.iter().map(|i| regions[*i].voxels.len()).sum();
            let baseline_score = source.as_ref().map_or(0.0, |s| {
                members
                    .iter()
                    .map(|i| s.score(&regions[*i].samples).1 * regions[*i].voxels.len() as f64)
                    .sum()
            });
            let dirty = members
                .iter()
                .any(|i| self.dirty.contains(&regions[*i].key));
            units.push(AssignmentUnit {
                regions: members,
                source,
                dirty,
                points,
                baseline_score,
                neighbors: Vec::new(),
            });
        }
        for (index, region) in regions.iter().enumerate() {
            let unit = owner[index];
            for key in adjacent_regions(region.key) {
                if let Some(&other) = indices.get(&key) {
                    if owner[other] != unit {
                        units[unit].neighbors.push(owner[other]);
                    }
                }
            }
        }
        for unit in &mut units {
            unit.neighbors.sort_unstable();
            unit.neighbors.dedup();
        }
        units
    }

    fn image_offers(&self, units: &[AssignmentUnit], regions: &[RegionData]) -> Vec<ImageOffers> {
        let mut images: FxHashMap<u64, Arc<SourceImage>> =
            self.recent.iter().map(|s| (s.id, Arc::clone(s))).collect();
        // Current-window candidates rank ahead of old cached images for the
        // same extent. Keep cached images as a fallback when this window has
        // no compatible image, instead of creating a new uncolored hole.
        // A retained old image can expand into adjacent new regions even after
        // leaving the recent window. Avoid considering all historical images.
        for unit in units.iter().filter(|u| u.dirty) {
            for candidate in std::iter::once(unit).chain(unit.neighbors.iter().map(|i| &units[*i]))
            {
                if let Some(source) = &candidate.source {
                    images
                        .entry(source.id)
                        .or_insert_with(|| Arc::clone(source));
                }
            }
        }
        let mut images: Vec<_> = images.into_values().collect();
        images.sort_unstable_by_key(|s| s.id);
        images
            .into_iter()
            .map(|source| {
                let scores = units
                    .iter()
                    .map(|unit| {
                        let mut total = 0.0;
                        for index in &unit.regions {
                            let region = &regions[*index];
                            let (coverage, score) = source.score(&region.samples);
                            if coverage < self.config.min_coverage {
                                return None;
                            }
                            total += score * region.voxels.len() as f64;
                        }
                        // Coherence may trade a little local quality for fewer seams,
                        // but must not replace a good group with a very poor image.
                        (self.active_view.normal.is_some()
                            || total >= unit.baseline_score * self.config.min_group_quality_ratio)
                            .then_some(total)
                    })
                    .collect();
                ImageOffers {
                    view_cosine: source.view_cosine(&self.active_view),
                    from_current_window: self.window_image_ids.contains(&source.id),
                    source,
                    scores,
                    verified: vec![false; units.len()],
                }
            })
            .collect()
    }

    fn enqueue_components(
        &self,
        offer_index: usize,
        offers: &[ImageOffers],
        units: &[AssignmentUnit],
        subset: &[usize],
        claimed: &[bool],
        queue: &mut BinaryHeap<GroupProposal>,
    ) {
        let offer = &offers[offer_index];
        let mut remaining: FxHashSet<_> = subset
            .iter()
            .copied()
            .filter(|i| {
                if claimed[*i] {
                    return false;
                }
                let Some(score) = offer.scores[*i] else {
                    return false;
                };
                if units[*i].source.is_some() {
                    return true;
                }
                // With a window normal, angle ranks whole images. Do not let
                // a local quality floor eliminate the most frontal candidate.
                if self.active_view.normal.is_some() {
                    return true;
                }
                // Dynamic floor: an image rejected by full coverage or I/O
                // must not prevent a valid, slightly lower-quality fallback.
                let best = offers
                    .iter()
                    .filter_map(|o| o.scores[*i])
                    .fold(0.0, f64::max);
                score >= best * self.config.min_group_quality_ratio
            })
            .collect();
        // Sorted seeds make tie handling independent of hash-map iteration.
        for &start in subset {
            if !remaining.remove(&start) {
                continue;
            }
            let mut members = vec![start];
            let mut cursor = 0;
            while cursor < members.len() {
                let index = members[cursor];
                cursor += 1;
                for &neighbor in &units[index].neighbors {
                    if remaining.remove(&neighbor) {
                        members.push(neighbor);
                    }
                }
            }
            if !members.iter().any(|i| units[*i].dirty) {
                continue;
            }
            members.sort_unstable();
            let sum: f64 = members.iter().map(|i| offer.scores[*i].unwrap()).sum();
            if members.len() == 1 {
                let unit = &units[members[0]];
                let same = unit
                    .source
                    .as_ref()
                    .is_some_and(|s| s.id == offer.source.id);
                let better_view = unit.source.as_ref().is_some_and(|old| {
                    self.active_view.normal.is_some()
                        && offer.from_current_window
                        && (!self.window_image_ids.contains(&old.id)
                            || offer.view_cosine.unwrap_or(-1.0)
                                > old.view_cosine(&self.active_view).unwrap_or(-1.0) + 1e-12)
                });
                if unit.source.is_some()
                    && !same
                    && !better_view
                    && sum <= unit.baseline_score * (1.0 + self.config.switch_improvement_ratio)
                {
                    continue;
                }
            }
            let regions = members.iter().map(|i| units[*i].regions.len()).sum();
            let points: usize = members.iter().map(|i| units[*i].points).sum();
            queue.push(GroupProposal {
                offer: offer_index,
                source_id: offer.source.id,
                units: members,
                regions,
                points,
                score: sum / points.max(1) as f64,
                view_cosine: offer.view_cosine,
                preferred_angle_passed: offer
                    .view_cosine
                    .is_some_and(|c| c >= self.preferred_view_cosine),
                from_current_window: offer.from_current_window,
            });
        }
    }

    fn proposals(
        &self,
        offers: &[ImageOffers],
        units: &[AssignmentUnit],
        claimed: &[bool],
    ) -> BinaryHeap<GroupProposal> {
        let indices: Vec<_> = (0..units.len()).collect();
        let mut queue = BinaryHeap::new();
        for i in 0..offers.len() {
            self.enqueue_components(i, offers, units, &indices, claimed, &mut queue);
        }
        queue
    }

    fn covers_unit(
        &self,
        source: &SourceImage,
        unit: &AssignmentUnit,
        regions: &[RegionData],
        map: &LOCALMap,
    ) -> bool {
        unit.regions.iter().all(|index| {
            let voxels = &regions[*index].voxels;
            let covered = voxels
                .iter()
                .filter(|key| source.project(&map.voxel_map[key].mean).is_some())
                .count();
            covered as f64 / voxels.len() as f64 >= self.config.min_coverage
        })
    }

    fn apply_group(
        &mut self,
        source: &Arc<SourceImage>,
        image: &RgbImage,
        members: &[usize],
        units: &[AssignmentUnit],
        regions: &[RegionData],
        map: &mut LOCALMap,
        stats: &mut MapRecolorLog,
    ) {
        let mut region_count = 0;
        // All coverage checks and RGB loading succeed BEFORE the first write.
        // This never assigns only part of a previous continuous image group.
        for index in members {
            for region_index in &units[*index].regions {
                let region = &regions[*region_index];
                region_count += 1;
                for key in &region.voxels {
                    let cell = map.voxel_map.get_mut(key).unwrap();
                    if let Some((u, v, distance)) = source.project(&cell.mean) {
                        cell.rgb = Some(image.get_pixel(u, v).0);
                        cell.color_observation_distance_m = Some(distance);
                        stats.colored_points += 1;
                    } else {
                        if cell.rgb.is_some() {
                            stats.cleared_points += 1;
                        }
                        cell.rgb = None;
                        cell.color_observation_distance_m = None;
                    }
                }
                if !self
                    .selected
                    .get(&region.key)
                    .is_some_and(|s| s.id == source.id)
                {
                    stats.switched_regions += 1;
                }
                self.selected.insert(region.key, Arc::clone(source));
                self.dirty.remove(&region.key);
                stats.recolored_regions += 1;
            }
        }
        stats.recolored_groups += 1;
        stats.merged_groups += members.len().saturating_sub(1);
        stats.largest_group_regions = stats.largest_group_regions.max(region_count);
        let cosine = source.view_cosine(&self.active_view);
        let preferred = cosine.map(|c| c >= self.preferred_view_cosine);
        match preferred {
            Some(true) => stats.preferred_view_groups += 1,
            Some(false) => stats.best_oblique_view_groups += 1,
            None => stats.no_normal_view_groups += 1,
        }
        let from_current_window = self.window_image_ids.contains(&source.id);
        if !from_current_window {
            stats.retained_image_groups += 1;
        }
        stats.group_views.push(GroupViewLog {
            image_path: source.path.clone(),
            regions: region_count,
            view_cosine: cosine,
            preferred_angle_passed: preferred,
            from_current_window,
        });
    }

    fn recolor<F>(&mut self, map: &mut LOCALMap, mut loader: F, final_flush: bool) -> MapRecolorLog
    where
        F: FnMut(&Path) -> anyhow::Result<RgbImage>,
    {
        let window_view = self.summarize_window(map);
        if window_view != self.active_view {
            // Final/delayed classification may supply normals after a scheduled
            // update. Re-evaluate that same window, without inventing frame 31.
            for key in self.window_observations.keys() {
                self.dirty
                    .insert(self.region_key(key, map.config.index_voxel_size));
            }
        }
        self.active_view = window_view;
        let mut stats = MapRecolorLog {
            ran: true,
            final_flush,
            completed_frames: self.completed_frames,
            recent_images: self.recent.len(),
            window_first_frame: self.window_first_frame,
            window_last_frame: self.completed_frames,
            window_images: self.window_image_ids.len(),
            window_observations: self.active_view.observations,
            window_normal_samples: self.active_view.normal_samples,
            window_mean_normal: self.active_view.mean_normal.map(|v| v.into()),
            window_normal_coherence: self.active_view.mean_normal.map(|v| v.norm()),
            window_centroid: self.active_view.centroid.map(|p| p.coords.into()),
            ..Default::default()
        };
        if self.dirty.is_empty() {
            return stats;
        }
        let regions = self.collect_regions(map);
        let units = self.assignment_units(&regions);
        if !units.iter().any(|unit| unit.dirty) {
            stats.deferred_regions = self.dirty.len();
            return stats;
        }
        let mut offers = self.image_offers(&units, &regions);
        let mut claimed = vec![false; units.len()];
        let mut queue = self.proposals(&offers, &units, &claimed);
        let mut failed_ids = FxHashSet::default();
        let mut evaluated = FxHashSet::default();
        while let Some(proposal) = queue.pop() {
            let index = proposal.offer;
            if failed_ids.contains(&proposal.source_id) {
                continue;
            }
            // Another, larger assignment may have claimed an overlapping unit.
            // Reconsider only WHOLE remaining units, split into connected sets.
            if proposal.units.iter().any(|i| claimed[*i]) {
                self.enqueue_components(
                    index,
                    &offers,
                    &units,
                    &proposal.units,
                    &claimed,
                    &mut queue,
                );
                continue;
            }
            let mut valid = true;
            for &unit in &proposal.units {
                for region in &units[unit].regions {
                    evaluated.insert(*region);
                }
                if !offers[index].verified[unit] {
                    if !self.covers_unit(&offers[index].source, &units[unit], &regions, map) {
                        offers[index].scores[unit] = None;
                        valid = false;
                    }
                    offers[index].verified[unit] = true;
                } else if offers[index].scores[unit].is_none() {
                    valid = false;
                }
            }
            if !valid {
                queue = self.proposals(&offers, &units, &claimed);
                continue;
            }
            let source = Arc::clone(&offers[index].source);
            let image = match self.load_rgb(&source, &mut loader) {
                Ok(image) => image,
                Err(error) => {
                    log::warn!(
                        "Regional recolor: cannot load {}: {error}",
                        source.path.display()
                    );
                    stats.failed_images += 1;
                    failed_ids.insert(source.id);
                    offers[index].scores.fill(None);
                    queue = self.proposals(&offers, &units, &claimed);
                    continue;
                }
            };
            self.apply_group(
                &source,
                &image,
                &proposal.units,
                &units,
                &regions,
                map,
                &mut stats,
            );
            for unit in proposal.units {
                claimed[unit] = true;
            }
        }
        // If geometry grew outside the old FOV and no replacement covers the
        // WHOLE group, keep its source and reproject the entire old group. Clear
        // uncovered cells instead of filling them from unrelated images.
        for (index, unit) in units.iter().enumerate() {
            if claimed[index] || !unit.dirty {
                continue;
            }
            for region in &unit.regions {
                evaluated.insert(*region);
            }
            let Some(source) = &unit.source else {
                continue;
            };
            if failed_ids.contains(&source.id) {
                continue;
            }
            let image = match self.load_rgb(source, &mut loader) {
                Ok(image) => image,
                Err(error) => {
                    log::warn!(
                        "Regional recolor: cannot reload {}: {error}",
                        source.path.display()
                    );
                    stats.failed_images += 1;
                    failed_ids.insert(source.id);
                    continue;
                }
            };
            self.apply_group(source, &image, &[index], &units, &regions, map, &mut stats);
        }
        stats.evaluated_regions = evaluated.len();
        stats.deferred_regions = self.dirty.len();
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel_map::{FrameEntry, LocalMapConfig, VoxelCell, voxel_key};

    fn camera() -> CameraIntrinsics {
        CameraIntrinsics {
            width: 96,
            height: 96,
            fx: 60.0,
            fy: 60.0,
            cx: 48.0,
            cy: 48.0,
            distortion: [0.0; 5],
        }
    }

    fn image(rgb: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(96, 96, image::Rgb(rgb))
    }

    fn map(points: &[Point3<f32>], voxel_size: f32) -> LOCALMap {
        let mut map = LOCALMap::new(LocalMapConfig {
            index_voxel_size: voxel_size,
            max_points_per_voxel: 100,
            min_points_per_voxel: 1,
            min_observed_frames_per_voxel: 2,
            max_frames: 1000,
            max_distance: 100.0,
        });
        for point in points {
            let key = voxel_key(point, voxel_size);
            let mut cell = VoxelCell::new();
            cell.mean = *point;
            cell.point = (*point, 0);
            cell.voxel_key = key;
            cell.observed_frames = 2;
            cell.sample_count = 2;
            map.voxel_map.insert(key, cell);
        }
        dirty(&mut map);
        map
    }

    fn dirty(map: &mut LOCALMap) {
        map.frame_index.push_back(FrameEntry {
            frame_id: map.frame_index.len() as u64,
            origin: Point3::origin(),
            dirty_keys: map.voxel_map.keys().copied().collect(),
        });
    }

    fn observe(recolor: &mut MapRecolorer, id: f64, rgb: &RgbImage, transform: Matrix4<f64>) {
        assert!(recolor.observe_image(id, Path::new(&id.to_string()), rgb, &camera(), &transform));
    }

    fn tick(recolor: &mut MapRecolorer, map: &mut LOCALMap, rgb: &RgbImage) -> MapRecolorLog {
        recolor.finish_frame(map, |_| Ok(rgb.clone()))
    }

    fn instant() -> MapRecolorer {
        MapRecolorer::new(MapRecolorConfig {
            update_every_frames: 1,
            ..Default::default()
        })
    }

    fn set_normal(cell: &mut VoxelCell, normal: Vector3<f32>) {
        cell.surface_status = SurfaceStatus::Planar;
        cell.surface_plane = Some(crate::voxel_map::SurfacePlane {
            normal,
            d: -normal.dot(&cell.mean.coords),
            rmse_m: 0.0,
            neighbor_count: 8,
            inlier_count: 8,
        });
    }

    fn camera_pose(center: Point3<f64>, point: Point3<f64>) -> Matrix4<f64> {
        let forward = (point - center).normalize();
        let right = Vector3::y().cross(&forward).normalize();
        let down = forward.cross(&right).normalize();
        let rotation = nalgebra::Matrix3::from_rows(&[
            right.transpose(),
            down.transpose(),
            forward.transpose(),
        ]);
        let mut pose = Matrix4::identity();
        pose.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
        pose.fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&(-rotation * center.coords));
        pose
    }

    #[test]
    fn window_mean_is_weighted_by_observed_frames_and_aligns_normal_signs() {
        let mut r = MapRecolorer::new(Default::default());
        let mut m = map(
            &[Point3::new(0.2, 0.2, 4.0), Point3::new(0.5, 0.2, 4.0)],
            0.1,
        );
        let mut keys: Vec<_> = m.voxel_map.keys().copied().collect();
        keys.sort_by_key(|k| k.ix);
        set_normal(m.voxel_map.get_mut(&keys[0]).unwrap(), -Vector3::z());
        set_normal(m.voxel_map.get_mut(&keys[1]).unwrap(), -Vector3::x());
        r.note_map_update(&m);
        r.note_map_update(&m); // same frame must count once
        r.finish_frame(&mut m, |_| panic!("not at frame 30"));
        m.frame_index.push_back(FrameEntry {
            frame_id: 1,
            origin: Point3::origin(),
            dirty_keys: [keys[0]].into_iter().collect(),
        });
        r.note_map_update(&m);
        r.finish_frame(&mut m, |_| panic!("not at frame 30"));
        let view = r.summarize_window(&m);
        assert_eq!(view.observations, 3);
        assert_eq!(view.normal_samples, 3);
        assert!(
            (view.mean_normal.unwrap() - Vector3::new(1.0 / 3.0, 0.0, 2.0 / 3.0)).norm() < 1e-12
        );
        assert!((view.centroid.unwrap().x - 0.3).abs() < 1e-7);
        for cell in m.voxel_map.values_mut() {
            cell.surface_plane.as_mut().unwrap().normal *= -1.0;
        }
        assert!(
            (r.summarize_window(&m).mean_normal.unwrap() - view.mean_normal.unwrap()).norm()
                < 1e-12
        );
    }

    #[test]
    fn old_untouched_map_normals_do_not_enter_the_current_window_mean() {
        let mut r = MapRecolorer::new(Default::default());
        let mut m = map(
            &[Point3::new(0.2, 0.2, 4.0), Point3::new(8.2, 0.2, 4.0)],
            0.1,
        );
        let mut keys: Vec<_> = m.voxel_map.keys().copied().collect();
        keys.sort_by_key(|k| k.ix);
        set_normal(m.voxel_map.get_mut(&keys[0]).unwrap(), Vector3::z());
        set_normal(m.voxel_map.get_mut(&keys[1]).unwrap(), Vector3::x());
        m.frame_index.back_mut().unwrap().dirty_keys = [keys[0]].into_iter().collect();
        r.note_map_update(&m);
        let view = r.summarize_window(&m);
        assert_eq!(view.observations, 1);
        assert_eq!(view.normal_samples, 1);
        assert_eq!(view.mean_normal.unwrap(), Vector3::z());
    }

    #[test]
    fn all_angles_failing_still_selects_the_best_view_and_colors_every_point() {
        let mut r = instant();
        let point = Point3::new(0.2, 0.2, 4.0);
        let p = point.cast::<f64>();
        let mut m = map(&[point], 0.1);
        set_normal(m.voxel_map.values_mut().next().unwrap(), Vector3::z());
        let bad = image([180, 80, 20]);
        let good = image([20, 80, 180]);
        let a = 80.0_f64.to_radians();
        let b = 60.0_f64.to_radians();
        observe(
            &mut r,
            1.0,
            &bad,
            camera_pose(p + Vector3::new(2.0 * a.sin(), 0.0, -2.0 * a.cos()), p),
        );
        observe(
            &mut r,
            2.0,
            &good,
            camera_pose(p + Vector3::new(5.0 * b.sin(), 0.0, -5.0 * b.cos()), p),
        );
        r.note_map_update(&m);
        let stats = r.finish_frame(&mut m, |path| {
            assert_eq!(path, Path::new("2"));
            Ok(good.clone())
        });
        assert_eq!(stats.best_oblique_view_groups, 1);
        assert_eq!(stats.colored_points, 1);
        assert_eq!(stats.cleared_points, 0);
        assert!((stats.group_views[0].view_cosine.unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(
            m.voxel_map.values().next().unwrap().rgb,
            Some([20, 80, 180])
        );
    }

    #[test]
    fn most_frontal_image_is_not_eliminated_by_distance_or_a_quality_floor() {
        let mut r = instant();
        let point = Point3::new(0.2, 0.2, 4.0);
        let p = point.cast::<f64>();
        let mut m = map(&[point], 0.1);
        set_normal(m.voxel_map.values_mut().next().unwrap(), Vector3::z());
        let rgb = image([100, 120, 60]);
        let theta = 60.0_f64.to_radians();
        observe(
            &mut r,
            1.0,
            &rgb,
            camera_pose(p + Vector3::new(theta.sin(), 0.0, -theta.cos()), p),
        );
        observe(
            &mut r,
            2.0,
            &rgb,
            camera_pose(p + Vector3::new(0.0, 0.0, -8.0), p),
        );
        r.note_map_update(&m);
        let stats = r.finish_frame(&mut m, |path| {
            assert_eq!(path, Path::new("2"));
            Ok(rgb.clone())
        });
        assert_eq!(stats.preferred_view_groups, 1);
        assert_eq!(stats.colored_points, 1);
    }

    #[test]
    fn missing_or_invalid_normals_use_quality_fallback_without_creating_holes() {
        for kind in 0..4 {
            let mut r = instant();
            let mut m = map(&[Point3::new(0.2, 0.2, 4.0)], 0.1);
            let cell = m.voxel_map.values_mut().next().unwrap();
            match kind {
                0 => {}
                1 => set_normal(cell, Vector3::zeros()),
                2 => set_normal(cell, Vector3::new(f32::NAN, 0.0, 0.0)),
                _ => {
                    set_normal(cell, Vector3::z());
                    cell.surface_status = SurfaceStatus::NonPlanar;
                }
            }
            let rgb = image([100, 120, 60]);
            observe(&mut r, 1.0, &rgb, Matrix4::identity());
            r.note_map_update(&m);
            let stats = tick(&mut r, &mut m, &rgb);
            assert_eq!(stats.window_normal_samples, 0);
            assert_eq!(stats.no_normal_view_groups, 1);
            assert_eq!(stats.colored_points, 1);
            assert!(m.voxel_map.values().all(|c| c.rgb.is_some()));
        }
    }

    #[test]
    fn individual_oblique_or_unknown_normal_points_are_not_masked() {
        let mut r = instant();
        let points: Vec<_> = (0..10)
            .map(|i| Point3::new(0.05 + 0.1 * i as f32, 0.05, 4.0))
            .collect();
        let mut m = map(&points, 0.1);
        let mut keys: Vec<_> = m.voxel_map.keys().copied().collect();
        keys.sort_by_key(|k| k.ix);
        for cell in m.voxel_map.values_mut() {
            set_normal(cell, Vector3::z());
        }
        set_normal(m.voxel_map.get_mut(&keys[0]).unwrap(), Vector3::x());
        m.voxel_map.get_mut(&keys[1]).unwrap().surface_status = SurfaceStatus::Unknown;
        m.voxel_map.get_mut(&keys[1]).unwrap().surface_plane = None;
        let rgb = image([100, 120, 60]);
        observe(&mut r, 1.0, &rgb, Matrix4::identity());
        r.note_map_update(&m);
        let stats = tick(&mut r, &mut m, &rgb);
        assert_eq!(stats.window_observations, 10);
        assert_eq!(stats.window_normal_samples, 9);
        assert_eq!(stats.colored_points, 10);
        assert_eq!(stats.cleared_points, 0);
        assert!(m.voxel_map.values().all(|c| c.rgb == Some([100, 120, 60])));
    }

    #[test]
    fn thirty_frame_windows_do_not_mix_normals_or_unobserved_old_images() {
        let mut r = MapRecolorer::new(Default::default());
        let point = Point3::new(0.2, 0.2, 4.0);
        let p = point.cast::<f64>();
        let mut m = map(&[point], 0.1);
        let orange = image([180, 80, 20]);
        let blue = image([20, 80, 180]);
        for frame in 1..=60 {
            let normal = if frame <= 30 {
                Vector3::z()
            } else {
                Vector3::x()
            };
            set_normal(m.voxel_map.values_mut().next().unwrap(), normal);
            dirty(&mut m);
            r.note_map_update(&m);
            if frame == 1 {
                observe(
                    &mut r,
                    1.0,
                    &orange,
                    camera_pose(p + Vector3::new(0.0, 0.0, -4.0), p),
                );
            }
            if frame == 31 {
                observe(
                    &mut r,
                    31.0,
                    &blue,
                    camera_pose(p + Vector3::new(-4.0, 0.0, 0.0), p),
                );
            }
            let stats = r.finish_frame(&mut m, |path| {
                Ok(if path == Path::new("1") {
                    orange.clone()
                } else {
                    blue.clone()
                })
            });
            if frame == 30 || frame == 60 {
                assert_eq!(stats.window_observations, 30);
                assert_eq!(stats.window_normal_samples, 30);
                assert_eq!(stats.window_first_frame, if frame == 30 { 1 } else { 31 });
                assert_eq!(stats.window_last_frame, frame);
                assert_eq!(stats.window_images, 1);
                assert!(stats.group_views.iter().all(|v| v.from_current_window));
                let expected: [f64; 3] = if frame == 30 {
                    [0.0, 0.0, 1.0]
                } else {
                    [1.0, 0.0, 0.0]
                };
                assert_eq!(stats.window_mean_normal.unwrap(), expected);
            }
            if frame >= 30 {
                assert_eq!(
                    m.voxel_map.values().next().unwrap().rgb,
                    Some(if frame < 60 {
                        [180, 80, 20]
                    } else {
                        [20, 80, 180]
                    })
                );
            }
        }
    }

    #[test]
    fn repeated_nearest_image_is_a_candidate_in_the_new_window() {
        let mut r = instant();
        let mut m = map(&[Point3::new(0.2, 0.2, 4.0)], 0.1);
        set_normal(m.voxel_map.values_mut().next().unwrap(), Vector3::z());
        let rgb = image([100, 120, 60]);
        observe(&mut r, 1.0, &rgb, Matrix4::identity());
        r.note_map_update(&m);
        tick(&mut r, &mut m, &rgb);
        assert!(!r.observe_image(1.0, Path::new("1"), &rgb, &camera(), &Matrix4::identity()));
        dirty(&mut m);
        r.note_map_update(&m);
        let stats = tick(&mut r, &mut m, &rgb);
        assert_eq!(stats.window_images, 1);
        assert_eq!(stats.retained_image_groups, 0);
        assert_eq!(stats.colored_points, 1);
    }

    #[test]
    fn final_classification_can_rerank_the_same_completed_window_without_new_frames() {
        let mut r = MapRecolorer::new(MapRecolorConfig {
            update_every_frames: 2,
            ..Default::default()
        });
        let point = Point3::new(0.2, 0.2, 4.0);
        let p = point.cast::<f64>();
        let mut m = map(&[point], 0.1);
        let rgb = image([100, 120, 60]);
        let theta = 70.0_f64.to_radians();
        observe(
            &mut r,
            1.0,
            &rgb,
            camera_pose(p + Vector3::new(0.0, 0.0, -8.0), p),
        );
        observe(
            &mut r,
            2.0,
            &rgb,
            camera_pose(p + Vector3::new(theta.sin(), 0.0, -theta.cos()), p),
        );
        for _ in 0..2 {
            dirty(&mut m);
            r.note_map_update(&m);
            tick(&mut r, &mut m, &rgb);
        }
        assert!(r.selected.values().all(|s| s.id == 2.0_f64.to_bits()));
        assert!(r.dirty.is_empty());
        set_normal(m.voxel_map.values_mut().next().unwrap(), Vector3::z());
        let stats = r.flush(&mut m, |path| {
            assert_eq!(path, Path::new("1"));
            Ok(rgb.clone())
        });
        assert_eq!(stats.window_first_frame, 1);
        assert_eq!(stats.window_last_frame, 2);
        assert_eq!(stats.window_normal_samples, 2);
        assert_eq!(stats.switched_regions, 1);
        assert!(r.selected.values().all(|s| s.id == 1.0_f64.to_bits()));
        assert_eq!(
            r.flush(&mut m, |_| panic!("idempotent flush"))
                .colored_points,
            0
        );
    }

    fn wall_points() -> Vec<Point3<f32>> {
        vec![
            Point3::new(0.2, 0.2, 4.0),
            Point3::new(1.2, 0.2, 4.0),
            Point3::new(2.2, 0.2, 4.0),
        ]
    }

    fn narrow_left_pose() -> Matrix4<f64> {
        let mut pose = Matrix4::identity();
        pose[(2, 3)] = -1.0;
        pose
    }

    fn narrow_right_pose() -> Matrix4<f64> {
        let mut pose = narrow_left_pose();
        pose[(0, 3)] = -2.5;
        pose
    }

    fn wide_pose() -> Matrix4<f64> {
        let mut pose = Matrix4::identity();
        pose[(0, 3)] = -1.0;
        pose[(2, 3)] = -0.5;
        pose
    }

    fn legacy_wall() -> (MapRecolorer, LOCALMap) {
        let mut r = instant();
        let mut m = map(&wall_points(), 0.1);
        observe(&mut r, 1.0, &image([180, 80, 20]), narrow_left_pose());
        observe(&mut r, 2.0, &image([30, 80, 180]), narrow_right_pose());
        let left = Arc::clone(&r.recent[0]);
        let right = Arc::clone(&r.recent[1]);
        for (key, cell) in &mut m.voxel_map {
            let region = r.region_key(key, m.config.index_voxel_size);
            let source = if region.0 < 2 { &left } else { &right };
            r.selected.insert(region, Arc::clone(source));
            cell.rgb = Some(if region.0 < 2 {
                [180, 80, 20]
            } else {
                [30, 80, 180]
            });
        }
        // Only one region changed. Other groups must still participate in merging.
        r.dirty.insert((2, 0, 4));
        (r, m)
    }

    #[test]
    fn one_wide_image_wins_over_independently_better_local_images() {
        let mut r = instant();
        let mut m = map(&wall_points(), 0.1);
        observe(&mut r, 1.0, &image([180, 80, 20]), narrow_left_pose());
        observe(&mut r, 2.0, &image([30, 80, 180]), narrow_right_pose());
        let wide = image([100, 120, 60]);
        observe(&mut r, 3.0, &wide, wide_pose());
        assert!(
            r.recent[0].score(&[wall_points()[0]]).1 > r.recent[2].score(&[wall_points()[0]]).1
        );
        assert!(
            r.recent[1].score(&[wall_points()[2]]).1 > r.recent[2].score(&[wall_points()[2]]).1
        );
        r.note_map_update(&m);
        let stats = r.finish_frame(&mut m, |path| {
            assert_eq!(path, Path::new("3"));
            Ok(wide.clone())
        });
        assert_eq!(stats.recolored_groups, 1);
        assert_eq!(stats.largest_group_regions, 3);
        assert_eq!(stats.recolored_regions, 3);
        assert!(
            r.selected
                .values()
                .all(|source| source.id == 3.0_f64.to_bits())
        );
        assert!(
            m.voxel_map
                .values()
                .all(|cell| cell.rgb == Some([100, 120, 60]))
        );
    }

    #[test]
    fn better_local_image_cannot_split_an_existing_wide_assignment() {
        let mut r = instant();
        let mut m = map(&wall_points(), 0.1);
        let wide = image([100, 120, 60]);
        observe(&mut r, 1.0, &wide, wide_pose());
        r.note_map_update(&m);
        tick(&mut r, &mut m, &wide);
        let mut close = Matrix4::identity();
        close[(2, 3)] = -2.0;
        observe(&mut r, 2.0, &image([180, 80, 20]), close);
        r.dirty.insert((0, 0, 4));
        let stats = r.finish_frame(&mut m, |_| panic!("wide image is already cached"));
        assert_eq!(stats.switched_regions, 0);
        assert_eq!(stats.recolored_regions, 3);
        assert_eq!(stats.largest_group_regions, 3);
        assert!(r.selected.values().all(|s| s.id == 1.0_f64.to_bits()));
    }

    #[test]
    fn new_common_image_merges_clean_neighbor_groups_with_a_dirty_group() {
        let (mut r, mut m) = legacy_wall();
        let wide = image([100, 120, 60]);
        observe(&mut r, 3.0, &wide, wide_pose());
        let stats = r.finish_frame(&mut m, |path| {
            assert_eq!(path, Path::new("3"));
            Ok(wide.clone())
        });
        assert_eq!(stats.recolored_groups, 1);
        assert_eq!(stats.merged_groups, 1);
        assert_eq!(stats.recolored_regions, 3);
        assert_eq!(stats.switched_regions, 3);
        assert!(
            r.assignments()
                .iter()
                .all(|a| a.image_path == Path::new("3"))
        );
    }

    #[test]
    fn unreadable_common_image_never_partially_replaces_old_groups() {
        let (mut r, mut m) = legacy_wall();
        observe(&mut r, 3.0, &image([100, 120, 60]), wide_pose());
        let before: FxHashMap<_, _> = m.voxel_map.iter().map(|(k, c)| (*k, c.rgb)).collect();
        let stats = r.finish_frame(&mut m, |path| match path.to_str().unwrap() {
            "3" => anyhow::bail!("missing common image"),
            "1" => Ok(image([180, 80, 20])),
            "2" => Ok(image([30, 80, 180])),
            _ => unreachable!(),
        });
        assert_eq!(stats.failed_images, 1);
        assert_eq!(stats.switched_regions, 0);
        for (key, cell) in &m.voxel_map {
            assert_eq!(cell.rgb, before[key]);
        }
        assert_eq!(r.selected[&(0, 0, 4)].id, 1.0_f64.to_bits());
        assert_eq!(r.selected[&(2, 0, 4)].id, 2.0_f64.to_bits());
    }

    #[test]
    fn incomplete_full_coverage_rejects_the_entire_previous_group() {
        let mut r = MapRecolorer::new(MapRecolorConfig {
            update_every_frames: 1,
            min_coverage: 1.0,
            max_score_samples_per_region: 1,
            ..Default::default()
        });
        let mut points = wall_points();
        points.push(Point3::new(2.9, 0.2, 4.0));
        let mut m = map(&points, 0.1);
        let wide = image([100, 120, 60]);
        observe(&mut r, 1.0, &wide, wide_pose());
        r.note_map_update(&m);
        tick(&mut r, &mut m, &wide);
        let mut closer = wide_pose();
        closer[(2, 3)] = -1.1;
        observe(&mut r, 2.0, &image([180, 80, 20]), closer);
        r.dirty.insert((0, 0, 4));
        let stats = r.finish_frame(&mut m, |_| {
            panic!("must reject before loading the incomplete image")
        });
        assert_eq!(stats.switched_regions, 0);
        assert_eq!(stats.recolored_regions, 3);
        assert_eq!(stats.colored_points, 4);
        assert!(r.selected.values().all(|s| s.id == 1.0_f64.to_bits()));
    }

    #[test]
    fn retained_neighbor_source_expands_into_a_new_region_outside_the_recent_window() {
        let mut r = MapRecolorer::new(MapRecolorConfig {
            update_every_frames: 1,
            max_recent_images: 1,
            ..Default::default()
        });
        let mut m = map(&wall_points()[..2], 0.1);
        let wide = image([100, 120, 60]);
        observe(&mut r, 1.0, &wide, wide_pose());
        r.note_map_update(&m);
        tick(&mut r, &mut m, &wide);
        let extra = map(&wall_points()[2..], 0.1);
        m.voxel_map.extend(extra.voxel_map);
        let mut behind = Matrix4::identity();
        behind[(2, 3)] = -10.0;
        observe(&mut r, 2.0, &wide, behind);
        r.dirty.insert((2, 0, 4));
        let stats = r.finish_frame(&mut m, |_| {
            panic!("retained neighbor source should be cached")
        });
        assert_eq!(stats.recolored_regions, 3);
        assert_eq!(stats.largest_group_regions, 3);
        assert_eq!(stats.merged_groups, 1);
        assert!(r.selected.values().all(|s| s.id == 1.0_f64.to_bits()));
    }

    #[test]
    fn disconnected_regions_are_separate_groups_even_in_the_same_image() {
        let mut r = instant();
        let p = wall_points();
        let mut m = map(&[p[0], p[2]], 0.1);
        let wide = image([100, 120, 60]);
        observe(&mut r, 1.0, &wide, wide_pose());
        r.note_map_update(&m);
        let stats = tick(&mut r, &mut m, &wide);
        assert_eq!(stats.recolored_groups, 2);
        assert_eq!(stats.largest_group_regions, 1);
    }

    #[test]
    fn broad_coverage_does_not_override_a_large_loss_of_image_resolution() {
        let mut r = instant();
        let mut m = map(&wall_points(), 0.1);
        observe(&mut r, 1.0, &image([180, 80, 20]), narrow_left_pose());
        observe(&mut r, 2.0, &image([30, 80, 180]), narrow_right_pose());
        let mut distant = wide_pose();
        distant[(2, 3)] = 20.0;
        observe(&mut r, 3.0, &image([100, 120, 60]), distant);
        r.note_map_update(&m);
        r.finish_frame(&mut m, |path| {
            assert_ne!(path, Path::new("3"));
            Ok(image([120, 80, 20]))
        });
        assert_eq!(r.selected.len(), 3);
        assert!(r.selected.values().all(|s| s.id != 3.0_f64.to_bits()));
    }

    #[test]
    fn a_better_image_switches_the_whole_continuous_group() {
        let mut r = instant();
        let mut m = map(&wall_points(), 0.1);
        let rgb = image([100, 120, 60]);
        observe(&mut r, 1.0, &rgb, wide_pose());
        r.note_map_update(&m);
        tick(&mut r, &mut m, &rgb);
        let mut closer = wide_pose();
        closer[(2, 3)] = -1.5;
        observe(&mut r, 2.0, &image([180, 80, 20]), closer);
        r.dirty.insert((1, 0, 4));
        let stats = tick(&mut r, &mut m, &image([180, 80, 20]));
        assert_eq!(stats.switched_regions, 3);
        assert_eq!(stats.recolored_groups, 1);
        assert!(r.selected.values().all(|s| s.id == 2.0_f64.to_bits()));
    }

    #[test]
    fn incomplete_best_image_does_not_exclude_a_valid_lower_quality_fallback() {
        let mut r = MapRecolorer::new(MapRecolorConfig {
            update_every_frames: 1,
            min_coverage: 1.0,
            max_score_samples_per_region: 1,
            ..Default::default()
        });
        let mut m = map(
            &[Point3::new(0.2, 0.2, 4.0), Point3::new(0.9, 0.2, 4.0)],
            0.1,
        );
        let rgb = image([100, 120, 60]);
        let mut close = Matrix4::identity();
        close[(2, 3)] = -3.0;
        observe(&mut r, 1.0, &rgb, close);
        observe(&mut r, 2.0, &rgb, Matrix4::identity());
        r.note_map_update(&m);
        let stats = r.finish_frame(&mut m, |path| {
            assert_eq!(path, Path::new("2"));
            Ok(rgb.clone())
        });
        assert_eq!(stats.colored_points, 2);
        assert_eq!(stats.deferred_regions, 0);
        assert!(r.selected.values().all(|s| s.id == 2.0_f64.to_bits()));
    }

    #[test]
    fn unreadable_best_image_does_not_exclude_a_valid_lower_quality_fallback() {
        let mut r = instant();
        let mut m = map(&wall_points()[..1], 0.1);
        let rgb = image([100, 120, 60]);
        let mut close = Matrix4::identity();
        close[(2, 3)] = -2.0;
        observe(&mut r, 1.0, &rgb, close);
        observe(&mut r, 2.0, &rgb, Matrix4::identity());
        r.note_map_update(&m);
        let stats = r.finish_frame(&mut m, |path| {
            if path == Path::new("1") {
                anyhow::bail!("missing best image");
            }
            assert_eq!(path, Path::new("2"));
            Ok(rgb.clone())
        });
        assert_eq!(stats.failed_images, 1);
        assert_eq!(stats.colored_points, 1);
        assert_eq!(stats.deferred_regions, 0);
        assert!(r.selected.values().all(|s| s.id == 2.0_f64.to_bits()));
    }

    #[test]
    fn current_window_image_is_preferred_over_a_more_frontal_old_cached_image() {
        let mut r = instant();
        let point = Point3::new(0.2, 0.2, 4.0);
        let p = point.cast::<f64>();
        let mut m = map(&[point], 0.1);
        set_normal(m.voxel_map.values_mut().next().unwrap(), Vector3::z());
        let rgb = image([100, 120, 60]);
        observe(
            &mut r,
            1.0,
            &rgb,
            camera_pose(p + Vector3::new(0.0, 0.0, -4.0), p),
        );
        r.note_map_update(&m);
        tick(&mut r, &mut m, &rgb);
        let angle = 60.0_f64.to_radians();
        observe(
            &mut r,
            2.0,
            &rgb,
            camera_pose(
                p + Vector3::new(4.0 * angle.sin(), 0.0, -4.0 * angle.cos()),
                p,
            ),
        );
        dirty(&mut m);
        r.note_map_update(&m);
        let stats = r.finish_frame(&mut m, |path| {
            assert_eq!(path, Path::new("2"));
            Ok(rgb.clone())
        });
        assert_eq!(stats.best_oblique_view_groups, 1);
        assert_eq!(stats.retained_image_groups, 0);
        assert!(r.selected.values().all(|s| s.id == 2.0_f64.to_bits()));
    }

    #[test]
    fn unreadable_best_normal_view_falls_back_to_an_oblique_image_without_masking() {
        let mut r = instant();
        let point = Point3::new(0.2, 0.2, 4.0);
        let p = point.cast::<f64>();
        let mut m = map(&[point], 0.1);
        set_normal(m.voxel_map.values_mut().next().unwrap(), Vector3::z());
        let rgb = image([100, 120, 60]);
        observe(
            &mut r,
            1.0,
            &rgb,
            camera_pose(p + Vector3::new(0.0, 0.0, -4.0), p),
        );
        let angle = 60.0_f64.to_radians();
        observe(
            &mut r,
            2.0,
            &rgb,
            camera_pose(
                p + Vector3::new(4.0 * angle.sin(), 0.0, -4.0 * angle.cos()),
                p,
            ),
        );
        r.note_map_update(&m);
        let stats = r.finish_frame(&mut m, |path| {
            if path == Path::new("1") {
                anyhow::bail!("missing frontal image");
            }
            Ok(rgb.clone())
        });
        assert_eq!(stats.failed_images, 1);
        assert_eq!(stats.best_oblique_view_groups, 1);
        assert_eq!(stats.colored_points, 1);
        assert_eq!(stats.cleared_points, 0);
    }

    #[test]
    fn each_frames_normal_is_preserved_instead_of_reusing_only_the_last_normal() {
        let mut r = MapRecolorer::new(MapRecolorConfig {
            update_every_frames: 2,
            ..Default::default()
        });
        let mut m = map(&[Point3::new(0.2, 0.2, 4.0)], 0.1);
        let rgb = image([100, 120, 60]);
        observe(&mut r, 1.0, &rgb, Matrix4::identity());
        set_normal(m.voxel_map.values_mut().next().unwrap(), Vector3::z());
        r.note_map_update(&m);
        tick(&mut r, &mut m, &rgb);
        set_normal(m.voxel_map.values_mut().next().unwrap(), Vector3::x());
        dirty(&mut m);
        r.note_map_update(&m);
        let stats = tick(&mut r, &mut m, &rgb);
        assert_eq!(stats.window_normal_samples, 2);
        assert_eq!(stats.window_mean_normal.unwrap(), [0.5, 0.0, 0.5]);
        assert_eq!(stats.colored_points, 1);
    }

    #[test]
    fn delayed_backfill_only_fills_missing_normals_and_keeps_observed_history() {
        let mut r = MapRecolorer::new(Default::default());
        let mut m = map(&[Point3::new(0.2, 0.2, 4.0)], 0.1);
        let rgb = image([100, 120, 60]);
        observe(&mut r, 1.0, &rgb, Matrix4::identity());
        set_normal(m.voxel_map.values_mut().next().unwrap(), Vector3::z());
        r.note_map_update(&m);
        tick(&mut r, &mut m, &rgb);
        m.voxel_map.values_mut().next().unwrap().surface_status = SurfaceStatus::Unknown;
        dirty(&mut m);
        r.note_map_update(&m);
        tick(&mut r, &mut m, &rgb);
        set_normal(m.voxel_map.values_mut().next().unwrap(), Vector3::x());
        let stats = r.flush(&mut m, |_| Ok(rgb.clone()));
        assert_eq!(stats.window_observations, 2);
        assert_eq!(stats.window_normal_samples, 2);
        assert_eq!(stats.window_mean_normal.unwrap(), [0.5, 0.0, 0.5]);
        assert_eq!(stats.colored_points, 1);
    }

    #[test]
    fn valid_normal_history_remains_usable_when_the_latest_fit_is_unknown() {
        let mut r = MapRecolorer::new(Default::default());
        let mut m = map(&[Point3::new(0.2, 0.2, 4.0)], 0.1);
        let rgb = image([100, 120, 60]);
        observe(&mut r, 1.0, &rgb, Matrix4::identity());
        set_normal(m.voxel_map.values_mut().next().unwrap(), Vector3::z());
        r.note_map_update(&m);
        tick(&mut r, &mut m, &rgb);
        m.voxel_map.values_mut().next().unwrap().surface_status = SurfaceStatus::Unknown;
        dirty(&mut m);
        r.note_map_update(&m);
        tick(&mut r, &mut m, &rgb);
        let stats = r.flush(&mut m, |_| Ok(rgb.clone()));
        assert_eq!(stats.window_observations, 2);
        assert_eq!(stats.window_normal_samples, 1);
        assert_eq!(stats.window_mean_normal.unwrap(), [0.0, 0.0, 1.0]);
        assert_eq!(stats.colored_points, 1);
    }

    #[test]
    fn mean_normal_ranking_still_commits_adjacent_regions_as_one_image_group() {
        let (mut r, mut m) = legacy_wall();
        for cell in m.voxel_map.values_mut() {
            set_normal(cell, Vector3::z());
        }
        let rgb = image([100, 120, 60]);
        observe(&mut r, 3.0, &rgb, wide_pose());
        r.note_map_update(&m);
        let stats = r.finish_frame(&mut m, |path| {
            assert_eq!(path, Path::new("3"));
            Ok(rgb.clone())
        });
        assert_eq!(stats.window_normal_samples, 3);
        assert_eq!(stats.largest_group_regions, 3);
        assert_eq!(stats.recolored_groups, 1);
        assert_eq!(stats.colored_points, 3);
        assert!(r.selected.values().all(|s| s.id == 3.0_f64.to_bits()));
    }

    #[test]
    fn colors_change_only_on_completed_lidar_frames_30_and_60() {
        let mut r = MapRecolorer::new(Default::default());
        let mut m = map(&[Point3::new(0.2, 0.2, 4.0)], 0.1);
        let orange = image([240, 70, 20]);
        observe(&mut r, 1.0, &orange, Matrix4::identity());
        r.note_map_update(&m);
        for frame in 1..=60 {
            let stats = tick(&mut r, &mut m, &orange);
            assert_eq!(stats.ran, frame == 30 || frame == 60);
            assert_eq!(stats.completed_frames, frame);
            assert_eq!(
                m.voxel_map.values().next().unwrap().rgb.is_some(),
                frame >= 30
            );
        }
    }

    #[test]
    fn one_source_replaces_the_whole_region_and_clears_uncovered_old_colors() {
        let mut r = instant();
        let points: Vec<_> = (0..10)
            .map(|i| Point3::new(0.05 + 0.1 * i as f32, 0.05, 4.0))
            .collect();
        let mut m = map(&points, 0.1);
        assert_eq!(m.voxel_map.len(), 10);
        let orange = image([240, 70, 20]);
        observe(&mut r, 1.0, &orange, Matrix4::identity());
        r.note_map_update(&m);
        assert_eq!(tick(&mut r, &mut m, &orange).colored_points, 10);
        let green = image([20, 180, 20]);
        let mut closer = Matrix4::identity();
        closer[(0, 3)] = -0.3;
        closer[(2, 3)] = -3.0;
        observe(&mut r, 2.0, &green, closer);
        dirty(&mut m);
        r.note_map_update(&m);
        let stats = tick(&mut r, &mut m, &green);
        assert_eq!(stats.switched_regions, 1);
        assert_eq!(stats.colored_points, 9);
        assert_eq!(stats.cleared_points, 1);
        assert!(
            m.voxel_map
                .values()
                .all(|c| c.rgb.is_none() || c.rgb == Some([20, 180, 20]))
        );
    }

    #[test]
    fn samples_the_current_mean_and_reprojects_it_without_changing_geometry() {
        let mut r = instant();
        let mut m = map(&[Point3::new(0.1, 0.1, 4.0)], 1.0);
        let rgb = RgbImage::from_fn(96, 96, |x, _| {
            image::Rgb(if x < 54 { [240, 70, 20] } else { [20, 40, 90] })
        });
        observe(&mut r, 1.0, &rgb, Matrix4::identity());
        r.note_map_update(&m);
        tick(&mut r, &mut m, &rgb);
        assert_eq!(
            m.voxel_map.values().next().unwrap().rgb,
            Some([240, 70, 20])
        );
        m.voxel_map.values_mut().next().unwrap().mean.x = 0.8;
        let before = m.voxel_map.values().next().unwrap().clone();
        dirty(&mut m);
        r.note_map_update(&m);
        tick(&mut r, &mut m, &rgb);
        let cell = m.voxel_map.values().next().unwrap();
        assert_eq!(cell.rgb, Some([20, 40, 90]));
        assert_eq!(cell.mean, before.mean);
        assert_eq!(cell.point, before.point);
        assert_eq!(cell.m2, before.m2);
        assert_eq!(cell.observed_frames, before.observed_frames);
        assert_eq!(cell.sample_count, before.sample_count);
    }

    #[test]
    fn final_flush_colors_short_runs_and_is_idempotent() {
        let mut r = MapRecolorer::new(Default::default());
        let mut m = map(&[Point3::new(0.2, 0.2, 4.0)], 0.1);
        let rgb = image([120, 60, 20]);
        observe(&mut r, 1.0, &rgb, Matrix4::identity());
        r.note_map_update(&m);
        for _ in 0..7 {
            assert!(!tick(&mut r, &mut m, &rgb).ran);
        }
        let stats = r.flush(&mut m, |_| Ok(rgb.clone()));
        assert!(stats.final_flush);
        assert_eq!(stats.completed_frames, 7);
        assert_eq!(stats.colored_points, 1);
        let again = r.flush(&mut m, |_| panic!("nothing should be loaded"));
        assert_eq!(again.colored_points, 0);
    }

    #[test]
    fn image_failure_is_transactional_and_uses_the_incumbent() {
        let mut r = instant();
        let mut m = map(&[Point3::new(0.2, 0.2, 4.0)], 0.1);
        let rgb = image([120, 60, 20]);
        observe(&mut r, 1.0, &rgb, Matrix4::identity());
        r.note_map_update(&m);
        tick(&mut r, &mut m, &rgb);
        let mut closer = Matrix4::identity();
        closer[(2, 3)] = -2.0;
        observe(&mut r, 2.0, &image([20, 60, 120]), closer);
        r.note_map_update(&m);
        let stats = r.finish_frame(&mut m, |_| anyhow::bail!("missing image"));
        assert_eq!(stats.failed_images, 1);
        assert_eq!(stats.switched_regions, 0);
        assert_eq!(stats.recolored_regions, 1);
        assert!(m.voxel_map.values().all(|c| c.rgb == Some([120, 60, 20])));
    }

    #[test]
    fn missing_images_defer_regions_and_immature_points_are_not_colored() {
        let mut r = instant();
        let mut m = map(&[Point3::new(0.2, 0.2, 4.0)], 0.1);
        m.voxel_map.values_mut().next().unwrap().observed_frames = 1;
        let rgb = image([120, 60, 20]);
        r.note_map_update(&m);
        assert_eq!(tick(&mut r, &mut m, &rgb).deferred_regions, 1);
        observe(&mut r, 1.0, &rgb, Matrix4::identity());
        assert_eq!(tick(&mut r, &mut m, &rgb).colored_points, 0);
        m.voxel_map.values_mut().next().unwrap().observed_frames = 2;
        assert_eq!(tick(&mut r, &mut m, &rgb).colored_points, 1);
    }

    #[test]
    fn source_hysteresis_rejects_small_distance_improvements() {
        let mut r = instant();
        let mut m = map(&[Point3::new(0.2, 0.2, 4.0)], 0.1);
        let rgb = image([120, 60, 20]);
        observe(&mut r, 1.0, &rgb, Matrix4::identity());
        r.note_map_update(&m);
        tick(&mut r, &mut m, &rgb);
        let mut closer = Matrix4::identity();
        closer[(2, 3)] = -0.1;
        observe(&mut r, 2.0, &image([20, 60, 120]), closer);
        r.note_map_update(&m);
        let stats = r.finish_frame(&mut m, |_| panic!("incumbent should be cached"));
        assert_eq!(stats.switched_regions, 0);
        assert!(m.voxel_map.values().all(|c| c.rgb == Some([120, 60, 20])));
    }

    #[test]
    fn retained_source_survives_recent_window_and_rgb_cache_eviction() {
        let mut r = MapRecolorer::new(MapRecolorConfig {
            update_every_frames: 1,
            max_recent_images: 1,
            max_cached_rgb_images: 1,
            ..Default::default()
        });
        let mut m = map(&[Point3::new(0.2, 0.2, 4.0)], 0.1);
        let rgb = image([120, 60, 20]);
        observe(&mut r, 1.0, &rgb, Matrix4::identity());
        r.note_map_update(&m);
        tick(&mut r, &mut m, &rgb);
        let mut behind = Matrix4::identity();
        behind[(2, 3)] = -10.0;
        observe(&mut r, 2.0, &rgb, behind);
        let newer = Arc::clone(r.recent.front().unwrap());
        r.load_rgb(&newer, &mut |_| Ok(rgb.clone())).unwrap();
        assert_eq!(r.rgb_cache.len(), 1);
        assert_eq!(r.rgb_cache.front().unwrap().0, 2.0_f64.to_bits());
        r.note_map_update(&m);
        let mut reloads = 0;
        let stats = r.finish_frame(&mut m, |path| {
            assert_eq!(path, Path::new("1"));
            reloads += 1;
            Ok(rgb.clone())
        });
        assert_eq!(reloads, 1);
        assert_eq!(stats.colored_points, 1);
        assert_eq!(r.recent.len(), 1);
        assert_eq!(r.selected.values().next().unwrap().id, 1.0_f64.to_bits());
    }

    #[test]
    fn image_quality_rewards_edges_and_penalizes_clipping() {
        let flat = image([100, 100, 100]);
        let sharp = RgbImage::from_fn(96, 96, |x, _| {
            image::Rgb(if x % 3 == 0 { [40; 3] } else { [200; 3] })
        });
        let black = image([0; 3]);
        let a = QualityGrid::new(&flat).weight(48, 48);
        let b = QualityGrid::new(&sharp).weight(48, 48);
        let c = QualityGrid::new(&black).weight(48, 48);
        assert!(b > a * 2.0);
        assert!(c < a * 0.5);
    }

    #[test]
    fn negative_region_boundaries_and_moving_means_have_stable_ownership() {
        let r = instant();
        assert_eq!(
            r.region_key(
                &VoxelKey {
                    ix: -1,
                    iy: 0,
                    iz: 0
                },
                0.25
            ),
            (-1, 0, 0)
        );
        assert_eq!(
            r.region_key(
                &VoxelKey {
                    ix: -4,
                    iy: 0,
                    iz: 0
                },
                0.25
            ),
            (-1, 0, 0)
        );
        assert_eq!(
            r.region_key(
                &VoxelKey {
                    ix: -5,
                    iy: 0,
                    iz: 0
                },
                0.25
            ),
            (-2, 0, 0)
        );
    }
}
