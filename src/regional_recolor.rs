//! 地図の小領域ごとに画像を選び、完成後の座標から再着色する。
//! 画像本体は保持しない。画像時刻の姿勢と、疎な深度キャッシュを画像単位で共有する。

use anyhow::{Context, Result, ensure};
use image::RgbImage;
use nalgebra::{Matrix4, Point3, Vector3};
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::camera::{CameraIntrinsics, project_camera_point_full_image};

#[derive(Debug, Clone, Serialize)]
pub struct RegionalRecolorConfig {
    pub region_size_m: f64,
    pub evaluate_every: usize,
    pub min_points_per_frame_region: usize,
    /// 画面内・法線・遮蔽判定を通る点の割合。
    pub min_visible_fraction: f64,
    pub min_front_cos: f64,
    pub sharpness_scale: f64,
    /// 品質上位N枚に加え、領域全体が画面内に入る候補を1枚保持する。
    pub top_candidates: usize,
    pub occlusion_margin_m: f64,
    pub occlusion_radius_px: u32,
}

impl Default for RegionalRecolorConfig {
    fn default() -> Self {
        Self {
            region_size_m: 0.5,
            evaluate_every: 5,
            min_points_per_frame_region: 3,
            min_visible_fraction: 0.8,
            min_front_cos: 0.25,
            sharpness_scale: 100.0,
            top_candidates: 3,
            occlusion_margin_m: 0.20,
            occlusion_radius_px: 2,
        }
    }
}

impl RegionalRecolorConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.region_size_m.is_finite() && self.region_size_m > 0.0,
            "region_size_m must be finite and positive"
        );
        ensure!(
            self.evaluate_every > 0 && self.min_points_per_frame_region > 0,
            "frame interval and point count must be positive"
        );
        ensure!(
            self.min_visible_fraction.is_finite()
                && self.min_visible_fraction > 0.0
                && self.min_visible_fraction <= 1.0,
            "min_visible_fraction must be in (0, 1]"
        );
        ensure!(
            self.min_front_cos.is_finite() && (0.0..=1.0).contains(&self.min_front_cos),
            "min_front_cos must be in [0, 1]"
        );
        ensure!(
            self.sharpness_scale.is_finite() && self.sharpness_scale > 0.0,
            "sharpness_scale must be positive"
        );
        ensure!(
            (1..=16).contains(&self.top_candidates),
            "top_candidates must be 1..=16"
        );
        ensure!(
            self.occlusion_margin_m.is_finite() && self.occlusion_margin_m >= 0.0,
            "occlusion_margin_m must be finite and non-negative"
        );
        Ok(())
    }

    fn score(
        &self,
        sharpness: f64,
        front_cos: f64,
        distance: f64,
        coverage: f64,
        camera: &CameraIntrinsics,
    ) -> f64 {
        // 平坦で模様のない面も採用可能にするため、鮮明さ項には下限を設ける。
        let sharpness_weight = 0.25 + 0.75 * sharpness / (sharpness + self.sharpness_scale);
        coverage * sharpness_weight * camera.fx * camera.fy * front_cos.powi(2)
            / distance.max(0.1).powi(2)
    }
}

#[derive(Debug, Clone)]
pub struct RegionalRecolorPoint {
    pub position: Point3<f32>,
    /// 世界座標の面法線。符号が未確定でもよい。
    pub normal: Option<Vector3<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct SurfacePatchKey {
    pub region: [i64; 3],
    /// 最大絶対成分の軸。0=X, 1=Y, 2=Z, 3=法線不明。
    pub normal_axis: u8,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    frame_id: usize,
    score: f64,
    full_coverage: bool,
}

#[derive(Default)]
struct CandidatePool {
    ranked: Vec<Candidate>,
    full: Option<Candidate>,
}

impl CandidatePool {
    fn would_keep(&self, candidate: Candidate, top: usize) -> bool {
        self.ranked.len() < top
            || self
                .ranked
                .last()
                .is_some_and(|old| candidate.score > old.score)
            || (candidate.full_coverage
                && self.full.map_or(true, |old| candidate.score > old.score))
    }

    fn insert(&mut self, candidate: Candidate, top: usize) {
        self.ranked.push(candidate);
        self.ranked.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.frame_id.cmp(&b.frame_id))
        });
        self.ranked.truncate(top);
        if candidate.full_coverage && self.full.map_or(true, |old| candidate.score > old.score) {
            self.full = Some(candidate);
        }
    }

    fn frame_ids(&self) -> Vec<usize> {
        let mut ids: Vec<_> = self
            .ranked
            .iter()
            .map(|candidate| candidate.frame_id)
            .collect();
        if let Some(full) = self.full {
            if !ids.contains(&full.frame_id) {
                ids.push(full.frame_id);
            }
        }
        ids.sort_unstable();
        ids
    }
}

struct CachedFrame {
    frame_index: usize,
    timestamp_sec: f64,
    image_path: PathBuf,
    camera_from_world: Matrix4<f64>,
    camera_center: Point3<f64>,
    intrinsics: CameraIntrinsics,
    depth_path: PathBuf,
}

#[derive(Debug, Default)]
pub struct FrameObservationStats {
    pub visible_regions: usize,
    pub retained_candidates: usize,
    pub cached: bool,
}

#[derive(Debug, Default, Serialize)]
pub struct RegionalRecolorStats {
    pub evaluated_frames: usize,
    pub cached_frames: usize,
    pub surface_patches: usize,
    pub selected_patches: usize,
    pub patches_without_candidates: usize,
    pub patches_without_usable_view: usize,
    pub colored_points: usize,
    pub uncolored_points: usize,
    pub unknown_depth_colored_points: usize,
    pub images_loaded_for_scoring: usize,
    pub images_loaded_for_coloring: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PatchSelection {
    pub key: SurfacePatchKey,
    pub frame_id: usize,
    pub frame_index: usize,
    pub score: f64,
    pub coverage: f64,
    pub sharpness: f64,
    pub mean_distance_m: f64,
    pub mean_front_cos: f64,
    pub point_count: usize,
    pub colored_count: usize,
}

pub struct RegionalRecolorResult {
    /// 入力点と同じ順序。採用画像で着色できない点はNone。
    pub colors: Vec<Option<[u8; 3]>>,
    pub stats: RegionalRecolorStats,
    pub selections: Vec<PatchSelection>,
}

pub struct RegionalRecolor {
    config: RegionalRecolorConfig,
    cache_dir: PathBuf,
    frames: Vec<CachedFrame>,
    candidates: HashMap<SurfacePatchKey, CandidatePool>,
    seen_frames: HashSet<usize>,
}

impl RegionalRecolor {
    pub fn new(config: RegionalRecolorConfig, cache_root: impl AsRef<Path>) -> Result<Self> {
        config.validate()?;
        fs::create_dir_all(cache_root.as_ref())?;
        let tick = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let cache_dir = cache_root
            .as_ref()
            .join(format!("run-{}-{tick}", std::process::id()));
        fs::create_dir(&cache_dir)?;
        Ok(Self {
            config,
            cache_dir,
            frames: Vec::new(),
            candidates: HashMap::new(),
            seen_frames: HashSet::new(),
        })
    }

    pub fn should_consider_frame(&self, frame_index: usize) -> bool {
        frame_index % self.config.evaluate_every == 0
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// SLAM中の候補収集。画像は既存と同じload_rgb_imageで読み込んだものを渡す。
    pub fn observe_frame(
        &mut self,
        frame_index: usize,
        timestamp_sec: f64,
        image_path: &Path,
        image: &RgbImage,
        camera_from_world: &Matrix4<f64>,
        world_from_lidar_start: &Matrix4<f64>,
        deskewed_points: &[Point3<f32>],
    ) -> Result<FrameObservationStats> {
        if !self.should_consider_frame(frame_index) {
            return Ok(FrameObservationStats::default());
        }
        ensure!(
            !self.seen_frames.contains(&frame_index),
            "frame {frame_index} was already observed"
        );
        ensure!(timestamp_sec.is_finite(), "non-finite image timestamp");
        ensure!(
            camera_from_world
                .iter()
                .chain(world_from_lidar_start.iter())
                .all(|value| value.is_finite()),
            "non-finite camera/LiDAR pose"
        );
        let intrinsics = CameraIntrinsics::for_image_size(image.width(), image.height())?;
        let world_from_camera = camera_from_world
            .try_inverse()
            .context("singular camera pose")?;
        let camera_center = world_from_camera.transform_point(&Point3::origin());
        let camera_from_lidar = camera_from_world * world_from_lidar_start;
        let mut regions: HashMap<[i64; 3], FrameRegion> = HashMap::new();

        for point in deskewed_points {
            let world = world_from_lidar_start.transform_point(&point.cast::<f64>());
            let Some(region) = spatial_key(&world, self.config.region_size_m) else {
                continue;
            };
            let camera_point = camera_from_lidar.transform_point(&point.cast::<f64>());
            let Some((u, v)) = project_camera_point_full_image(&camera_point, &intrinsics) else {
                continue;
            };
            regions.entry(region).or_default().add(u, v);
        }

        let sharpness = SharpnessIntegral::new(image);
        let frame_id = self.frames.len();
        let mut updates = Vec::new();
        for (&region, observed) in &regions {
            if observed.count < self.config.min_points_per_frame_region {
                continue;
            }
            let center = region_center(region, self.config.region_size_m);
            let view = camera_center - center;
            let distance = view.norm();
            if !distance.is_finite() || distance <= 0.0 {
                continue;
            }
            let direction = view / distance;
            let local_sharpness =
                sharpness.variance(observed.bounds(), image.width(), image.height());
            let full_coverage = cube_in_image(
                region,
                self.config.region_size_m,
                camera_from_world,
                &intrinsics,
            );
            // 床と壁を同じ候補順位にしない。各主要法線軸の近似スコアを使う。
            for normal_axis in 0..4_u8 {
                let cosine = if normal_axis == 3 {
                    1.0
                } else {
                    direction[normal_axis as usize].abs()
                };
                if cosine < self.config.min_front_cos {
                    continue;
                }
                let score = self
                    .config
                    .score(local_sharpness, cosine, distance, 1.0, &intrinsics);
                if !score.is_finite() || score <= 0.0 {
                    continue;
                }
                let key = SurfacePatchKey {
                    region,
                    normal_axis,
                };
                let candidate = Candidate {
                    frame_id,
                    score,
                    full_coverage,
                };
                if self.candidates.get(&key).map_or(true, |pool| {
                    pool.would_keep(candidate, self.config.top_candidates)
                }) {
                    updates.push((key, candidate));
                }
            }
        }

        let mut stats = FrameObservationStats {
            visible_regions: regions.len(),
            ..Default::default()
        };
        if !updates.is_empty() {
            let depth_path = self.cache_dir.join(format!("frame-{frame_index:08}.depth"));
            let depth = DepthRaster::build(deskewed_points, &camera_from_lidar, &intrinsics);
            depth.write_sparse(&depth_path)?;
            // ファイル書き込みが成功してから候補を更新する。
            self.frames.push(CachedFrame {
                frame_index,
                timestamp_sec,
                image_path: image_path.to_path_buf(),
                camera_from_world: *camera_from_world,
                camera_center,
                intrinsics,
                depth_path,
            });
            stats.cached = true;
            stats.retained_candidates = updates.len();
            for (key, candidate) in updates {
                self.candidates
                    .entry(key)
                    .or_default()
                    .insert(candidate, self.config.top_candidates);
            }
        }
        self.seen_frames.insert(frame_index);
        Ok(stats)
    }

    /// 2パスで処理し、同じ画像を領域ごとに読み直すことを避ける。
    /// 第1パスは候補評価、第2パスは採用画像からのRGB取得。
    pub fn recolor_points<F>(
        &self,
        points: &[RegionalRecolorPoint],
        mut load_image: F,
    ) -> Result<RegionalRecolorResult>
    where
        F: FnMut(&Path) -> Result<RgbImage>,
    {
        let mut patches: BTreeMap<SurfacePatchKey, Vec<usize>> = BTreeMap::new();
        for (index, point) in points.iter().enumerate() {
            if let Some(region) =
                spatial_key(&point.position.cast::<f64>(), self.config.region_size_m)
            {
                patches
                    .entry(SurfacePatchKey {
                        region,
                        normal_axis: normal_axis(point.normal.as_ref()),
                    })
                    .or_default()
                    .push(index);
            }
        }
        let patches: Vec<_> = patches.into_iter().collect();
        let mut stats = RegionalRecolorStats {
            evaluated_frames: self.seen_frames.len(),
            cached_frames: self.frames.len(),
            surface_patches: patches.len(),
            ..Default::default()
        };
        let mut by_frame: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (patch_index, (key, _)) in patches.iter().enumerate() {
            let ids = self.candidate_frame_ids(*key);
            if ids.is_empty() {
                stats.patches_without_candidates += 1;
            }
            for id in ids {
                by_frame.entry(id).or_default().push(patch_index);
            }
        }

        let mut winners: Vec<Option<PatchSelection>> = vec![None; patches.len()];
        let scoring_total = by_frame.len();
        for (step, (frame_id, patch_indices)) in by_frame.into_iter().enumerate() {
            if step % 25 == 0 || step + 1 == scoring_total {
                log::info!(
                    "Regional recolor: evaluating image {}/{}",
                    step + 1,
                    scoring_total
                );
            }
            let frame = &self.frames[frame_id];
            let image = load_image(&frame.image_path).with_context(|| {
                format!("loading reference image {}", frame.image_path.display())
            })?;
            ensure!(
                image.dimensions() == (frame.intrinsics.width, frame.intrinsics.height),
                "reference image dimensions changed: {}",
                frame.image_path.display()
            );
            let depth = DepthRaster::read_sparse(&frame.depth_path, &frame.intrinsics)?;
            let sharpness = SharpnessIntegral::new(&image);
            stats.images_loaded_for_scoring += 1;
            for patch_index in patch_indices {
                let (key, indices) = &patches[patch_index];
                let mut visible = FrameRegion::default();
                let mut distance_sum = 0.0;
                let mut cosine_sum = 0.0;
                for &index in indices {
                    if let Some(sample) = self.visible_sample(&points[index], frame, &depth) {
                        visible.add(sample.u, sample.v);
                        distance_sum += sample.distance;
                        cosine_sum += sample.front_cos;
                    }
                }
                let coverage = visible.count as f64 / indices.len() as f64;
                if visible.count == 0 || coverage < self.config.min_visible_fraction {
                    continue;
                }
                let local_sharpness =
                    sharpness.variance(visible.bounds(), image.width(), image.height());
                let distance = distance_sum / visible.count as f64;
                let front_cos = cosine_sum / visible.count as f64;
                let score = self.config.score(
                    local_sharpness,
                    front_cos,
                    distance,
                    coverage,
                    &frame.intrinsics,
                );
                if !score.is_finite() || score <= 0.0 {
                    continue;
                }
                if winners[patch_index].as_ref().map_or(true, |old| {
                    score > old.score || (score == old.score && frame.frame_index < old.frame_index)
                }) {
                    winners[patch_index] = Some(PatchSelection {
                        key: *key,
                        frame_id,
                        frame_index: frame.frame_index,
                        score,
                        coverage,
                        sharpness: local_sharpness,
                        mean_distance_m: distance,
                        mean_front_cos: front_cos,
                        point_count: indices.len(),
                        colored_count: 0,
                    });
                }
            }
        }

        let mut selected_by_frame: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (patch_index, winner) in winners.iter().enumerate() {
            if let Some(winner) = winner {
                selected_by_frame
                    .entry(winner.frame_id)
                    .or_default()
                    .push(patch_index);
            }
        }
        let mut colors = vec![None; points.len()];
        let coloring_total = selected_by_frame.len();
        for (step, (frame_id, patch_indices)) in selected_by_frame.into_iter().enumerate() {
            if step % 25 == 0 || step + 1 == coloring_total {
                log::info!(
                    "Regional recolor: applying image {}/{}",
                    step + 1,
                    coloring_total
                );
            }
            let frame = &self.frames[frame_id];
            let image = load_image(&frame.image_path)?;
            ensure!(
                image.dimensions() == (frame.intrinsics.width, frame.intrinsics.height),
                "reference image dimensions changed: {}",
                frame.image_path.display()
            );
            let depth = DepthRaster::read_sparse(&frame.depth_path, &frame.intrinsics)?;
            stats.images_loaded_for_coloring += 1;
            for patch_index in patch_indices {
                let indices = &patches[patch_index].1;
                let winner = winners[patch_index]
                    .as_mut()
                    .expect("selected patch has a winner");
                for &index in indices {
                    if let Some(sample) = self.visible_sample(&points[index], frame, &depth) {
                        let rgb = image
                            .get_pixel_checked(sample.u, sample.v)
                            .context("reference pixel is outside image")?
                            .0;
                        colors[index] = Some(rgb);
                        stats.colored_points += 1;
                        winner.colored_count += 1;
                        if sample.depth_unknown {
                            stats.unknown_depth_colored_points += 1;
                        }
                    }
                }
            }
        }
        let selections: Vec<_> = winners.into_iter().flatten().collect();
        stats.selected_patches = selections.len();
        stats.patches_without_usable_view =
            patches.len() - stats.patches_without_candidates - selections.len();
        stats.uncolored_points = points.len() - stats.colored_points;
        Ok(RegionalRecolorResult {
            colors,
            stats,
            selections,
        })
    }

    fn visible_sample(
        &self,
        point: &RegionalRecolorPoint,
        frame: &CachedFrame,
        depth: &DepthRaster,
    ) -> Option<VisibleSample> {
        let world = point.position.cast::<f64>();
        let camera = frame.camera_from_world.transform_point(&world);
        let (u, v) = project_camera_point_full_image(&camera, &frame.intrinsics)?;
        let view = frame.camera_center - world;
        let distance = view.norm();
        if !distance.is_finite() || distance <= 0.0 {
            return None;
        }
        let front_cos = unit_normal(point.normal.as_ref())
            .map_or(1.0, |normal| normal.dot(&(view / distance)).abs());
        if front_cos < self.config.min_front_cos {
            return None;
        }
        let state = depth.classify(
            u,
            v,
            camera.z,
            self.config.occlusion_margin_m,
            self.config.occlusion_radius_px,
        );
        if state == DepthState::Occluded {
            return None;
        }
        Some(VisibleSample {
            u,
            v,
            distance,
            front_cos,
            depth_unknown: state == DepthState::Unknown,
        })
    }

    fn candidate_frame_ids(&self, key: SurfacePatchKey) -> Vec<usize> {
        if let Some(pool) = self.candidates.get(&key) {
            return pool.frame_ids();
        }
        let unknown = SurfacePatchKey {
            normal_axis: 3,
            ..key
        };
        if let Some(pool) = self.candidates.get(&unknown) {
            return pool.frame_ids();
        }
        // 最終平均座標が境界をまたいだ場合だけ、隣接領域から候補を拾う。
        let mut ids = Vec::new();
        for x in -1..=1_i64 {
            for y in -1..=1_i64 {
                for z in -1..=1_i64 {
                    let (Some(x), Some(y), Some(z)) = (
                        key.region[0].checked_add(x),
                        key.region[1].checked_add(y),
                        key.region[2].checked_add(z),
                    ) else {
                        continue;
                    };
                    let neighbor = SurfacePatchKey {
                        region: [x, y, z],
                        normal_axis: key.normal_axis,
                    };
                    if let Some(pool) = self.candidates.get(&neighbor).or_else(|| {
                        self.candidates.get(&SurfacePatchKey {
                            normal_axis: 3,
                            ..neighbor
                        })
                    }) {
                        ids.extend(pool.frame_ids());
                    }
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    pub fn write_report(&self, path: &Path, result: &RegionalRecolorResult) -> Result<()> {
        #[derive(Serialize)]
        struct FrameReport<'a> {
            frame_id: usize,
            frame_index: usize,
            timestamp_sec: f64,
            image_path: &'a Path,
            depth_path: &'a Path,
            width: u32,
            height: u32,
            /// nalgebraの列優先順。
            camera_from_world_column_major: [f64; 16],
        }
        let frames: Vec<_> = self
            .frames
            .iter()
            .enumerate()
            .map(|(id, frame)| {
                let mut matrix = [0.0; 16];
                matrix.copy_from_slice(frame.camera_from_world.as_slice());
                FrameReport {
                    frame_id: id,
                    frame_index: frame.frame_index,
                    timestamp_sec: frame.timestamp_sec,
                    image_path: &frame.image_path,
                    depth_path: &frame.depth_path,
                    width: frame.intrinsics.width,
                    height: frame.intrinsics.height,
                    camera_from_world_column_major: matrix,
                }
            })
            .collect();
        #[derive(Serialize)]
        struct Report<'a, T: Serialize> {
            config: &'a RegionalRecolorConfig,
            stats: &'a RegionalRecolorStats,
            frames: T,
            selections: &'a [PatchSelection],
        }
        let report = Report {
            config: &self.config,
            stats: &result.stats,
            frames,
            selections: &result.selections,
        };
        let mut writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(&mut writer, &report)?;
        writer.flush()?;
        Ok(())
    }
}

struct VisibleSample {
    u: u32,
    v: u32,
    distance: f64,
    front_cos: f64,
    depth_unknown: bool,
}

#[derive(Default)]
struct FrameRegion {
    count: usize,
    min_u: u32,
    min_v: u32,
    max_u: u32,
    max_v: u32,
}
impl FrameRegion {
    fn add(&mut self, u: u32, v: u32) {
        if self.count == 0 {
            self.min_u = u;
            self.min_v = v;
        }
        self.min_u = self.min_u.min(u);
        self.min_v = self.min_v.min(v);
        self.max_u = self.max_u.max(u);
        self.max_v = self.max_v.max(v);
        self.count += 1;
    }
    fn bounds(&self) -> [u32; 4] {
        [self.min_u, self.min_v, self.max_u + 1, self.max_v + 1]
    }
}

fn spatial_key(point: &Point3<f64>, size: f64) -> Option<[i64; 3]> {
    let mut key = [0; 3];
    for axis in 0..3 {
        let coordinate = (point[axis] / size).floor();
        if !coordinate.is_finite() || coordinate <= i64::MIN as f64 || coordinate >= i64::MAX as f64
        {
            return None;
        }
        key[axis] = coordinate as i64;
    }
    Some(key)
}

fn region_center(key: [i64; 3], size: f64) -> Point3<f64> {
    Point3::new(
        (key[0] as f64 + 0.5) * size,
        (key[1] as f64 + 0.5) * size,
        (key[2] as f64 + 0.5) * size,
    )
}

fn unit_normal(normal: Option<&Vector3<f32>>) -> Option<Vector3<f64>> {
    let normal = normal?.cast::<f64>();
    let length = normal.norm();
    if !length.is_finite() || length < 1e-6 {
        return None;
    }
    Some(normal / length)
}

fn normal_axis(normal: Option<&Vector3<f32>>) -> u8 {
    let Some(n) = unit_normal(normal) else {
        return 3;
    };
    let mut best = 0;
    for axis in 1..3 {
        if n[axis].abs() > n[best].abs() {
            best = axis;
        }
    }
    best as u8
}

fn cube_in_image(
    key: [i64; 3],
    size: f64,
    camera_from_world: &Matrix4<f64>,
    camera: &CameraIntrinsics,
) -> bool {
    for x in 0..=1 {
        for y in 0..=1 {
            for z in 0..=1 {
                let point = Point3::new(
                    (key[0] as f64 + x as f64) * size,
                    (key[1] as f64 + y as f64) * size,
                    (key[2] as f64 + z as f64) * size,
                );
                if project_camera_point_full_image(
                    &camera_from_world.transform_point(&point),
                    camera,
                )
                .is_none()
                {
                    return false;
                }
            }
        }
    }
    true
}

/// 960x600へ揃えたグレースケールのLaplacian積分画像。領域の分散をO(1)で求める。
struct SharpnessIntegral {
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
}
impl SharpnessIntegral {
    fn new(image: &RgbImage) -> Self {
        let resized =
            image::imageops::resize(image, 960, 600, image::imageops::FilterType::Triangle);
        let gray = image::DynamicImage::ImageRgb8(resized).to_luma8();
        let mut result = Self {
            sum: vec![0.0; 961 * 601],
            sum_sq: vec![0.0; 961 * 601],
        };
        for y in 0..600_u32 {
            let mut row_sum = 0.0;
            let mut row_sq = 0.0;
            for x in 0..960_u32 {
                let value = if x > 0 && x < 959 && y > 0 && y < 599 {
                    gray.get_pixel(x - 1, y).0[0] as f64
                        + gray.get_pixel(x + 1, y).0[0] as f64
                        + gray.get_pixel(x, y - 1).0[0] as f64
                        + gray.get_pixel(x, y + 1).0[0] as f64
                        - 4.0 * gray.get_pixel(x, y).0[0] as f64
                } else {
                    0.0
                };
                row_sum += value;
                row_sq += value * value;
                let index = (y as usize + 1) * 961 + x as usize + 1;
                result.sum[index] = result.sum[index - 961] + row_sum;
                result.sum_sq[index] = result.sum_sq[index - 961] + row_sq;
            }
        }
        result
    }
    fn variance(&self, bounds: [u32; 4], width: u32, height: u32) -> f64 {
        let x0 = ((bounds[0] as f64 * 960.0 / width as f64).floor() as usize).clamp(1, 959);
        let y0 = ((bounds[1] as f64 * 600.0 / height as f64).floor() as usize).clamp(1, 599);
        let x1 = ((bounds[2] as f64 * 960.0 / width as f64).ceil() as usize).clamp(1, 959);
        let y1 = ((bounds[3] as f64 * 600.0 / height as f64).ceil() as usize).clamp(1, 599);
        if x1 <= x0 || y1 <= y0 {
            return 0.0;
        }
        let rect = |array: &[f64]| {
            array[y1 * 961 + x1] + array[y0 * 961 + x0]
                - array[y0 * 961 + x1]
                - array[y1 * 961 + x0]
        };
        let count = ((x1 - x0) * (y1 - y0)) as f64;
        let mean = rect(&self.sum) / count;
        (rect(&self.sum_sq) / count - mean * mean).max(0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DepthState {
    Unknown,
    Clear,
    Occluded,
}
struct DepthRaster {
    width: u32,
    height: u32,
    nearest_z: Vec<f32>,
}
impl DepthRaster {
    fn build(
        points: &[Point3<f32>],
        camera_from_lidar: &Matrix4<f64>,
        camera: &CameraIntrinsics,
    ) -> Self {
        let mut result = Self {
            width: camera.width,
            height: camera.height,
            nearest_z: vec![f32::INFINITY; camera.width as usize * camera.height as usize],
        };
        for point in points {
            let p = camera_from_lidar.transform_point(&point.cast::<f64>());
            if let Some((u, v)) = project_camera_point_full_image(&p, camera) {
                let z = p.z as f32;
                if z.is_finite() && z > 0.0 {
                    let index = v as usize * result.width as usize + u as usize;
                    result.nearest_z[index] = result.nearest_z[index].min(z);
                }
            }
        }
        result
    }
    fn classify(&self, u: u32, v: u32, z: f64, margin: f64, radius: u32) -> DepthState {
        if u >= self.width || v >= self.height || !z.is_finite() || z <= 0.0 {
            return DepthState::Unknown;
        }
        let mut known = false;
        for y in v.saturating_sub(radius)..=v.saturating_add(radius).min(self.height - 1) {
            for x in u.saturating_sub(radius)..=u.saturating_add(radius).min(self.width - 1) {
                let front = self.nearest_z[y as usize * self.width as usize + x as usize];
                if front.is_finite() {
                    known = true;
                    if z > front as f64 + margin {
                        return DepthState::Occluded;
                    }
                }
            }
        }
        if known {
            DepthState::Clear
        } else {
            DepthState::Unknown
        }
    }
    fn write_sparse(&self, path: &Path) -> Result<()> {
        let count = self
            .nearest_z
            .iter()
            .filter(|z| z.is_finite() && **z > 0.0)
            .count() as u64;
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(b"RCDPTH01")?;
        writer.write_all(&self.width.to_le_bytes())?;
        writer.write_all(&self.height.to_le_bytes())?;
        writer.write_all(&count.to_le_bytes())?;
        for (index, &z) in self.nearest_z.iter().enumerate() {
            if z.is_finite() && z > 0.0 {
                writer.write_all(&(index as u32).to_le_bytes())?;
                writer.write_all(&z.to_le_bytes())?;
            }
        }
        writer.flush()?;
        Ok(())
    }
    fn read_sparse(path: &Path, camera: &CameraIntrinsics) -> Result<Self> {
        let mut reader = BufReader::new(
            File::open(path).with_context(|| format!("reading depth cache {}", path.display()))?,
        );
        let mut magic = [0; 8];
        reader.read_exact(&mut magic)?;
        ensure!(&magic == b"RCDPTH01", "invalid depth cache header");
        let mut four = [0; 4];
        reader.read_exact(&mut four)?;
        let width = u32::from_le_bytes(four);
        reader.read_exact(&mut four)?;
        let height = u32::from_le_bytes(four);
        ensure!(
            (width, height) == (camera.width, camera.height),
            "depth cache size mismatch"
        );
        let mut eight = [0; 8];
        reader.read_exact(&mut eight)?;
        let count = u64::from_le_bytes(eight);
        let size = width as usize * height as usize;
        ensure!(count <= size as u64, "invalid depth record count");
        let mut result = Self {
            width,
            height,
            nearest_z: vec![f32::INFINITY; size],
        };
        for _ in 0..count {
            reader.read_exact(&mut four)?;
            let index = u32::from_le_bytes(four) as usize;
            reader.read_exact(&mut four)?;
            let z = f32::from_le_bytes(four);
            ensure!(
                index < size && z.is_finite() && z > 0.0,
                "invalid depth cache record"
            );
            result.nearest_z[index] = result.nearest_z[index].min(z);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST: AtomicU64 = AtomicU64::new(0);
    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let tick = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "regional-recolor-test-{}-{tick}-{}",
                std::process::id(),
                NEXT_TEST.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn config() -> RegionalRecolorConfig {
        RegionalRecolorConfig {
            evaluate_every: 1,
            min_points_per_frame_region: 1,
            occlusion_radius_px: 0,
            ..Default::default()
        }
    }
    fn image(rgb: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(960, 600, image::Rgb(rgb))
    }
    fn point(x: f32, y: f32, z: f32) -> Point3<f32> {
        Point3::new(x, y, z)
    }
    fn surface(position: Point3<f32>) -> RegionalRecolorPoint {
        RegionalRecolorPoint {
            position,
            normal: Some(Vector3::new(0.0, 0.0, 1.0)),
        }
    }
    fn camera_at(x: f64, y: f64, z: f64) -> Matrix4<f64> {
        let mut transform = Matrix4::identity();
        transform[(0, 3)] = -x;
        transform[(1, 3)] = -y;
        transform[(2, 3)] = -z;
        transform
    }
    fn record(
        state: &mut RegionalRecolor,
        index: usize,
        name: &str,
        image: &RgbImage,
        camera: &Matrix4<f64>,
        points: &[Point3<f32>],
    ) {
        state
            .observe_frame(
                index,
                index as f64,
                Path::new(name),
                image,
                camera,
                &Matrix4::identity(),
                points,
            )
            .unwrap();
    }

    #[test]
    fn negative_world_coordinates_use_floor_and_normal_signs_share_a_patch_axis() {
        assert_eq!(
            spatial_key(&Point3::new(-0.01, 0.49, 0.5), 0.5),
            Some([-1, 0, 1])
        );
        assert_eq!(normal_axis(Some(&Vector3::new(-1.0, 0.0, 0.0))), 0);
        assert_eq!(normal_axis(Some(&Vector3::new(1.0, 0.0, 0.0))), 0);
        assert_eq!(normal_axis(Some(&Vector3::zeros())), 3);
        assert_eq!(spatial_key(&Point3::new(f64::NAN, 0.0, 0.0), 0.5), None);
    }

    #[test]
    fn all_points_in_a_surface_patch_use_one_reference_image() {
        let scratch = Scratch::new();
        let mut state = RegionalRecolor::new(config(), &scratch.0).unwrap();
        let raw = [point(0.1, 0.1, 5.1), point(0.2, 0.1, 5.1)];
        record(
            &mut state,
            0,
            "red",
            &image([255, 0, 0]),
            &camera_at(0.0, 0.0, 0.0),
            &raw,
        );
        record(
            &mut state,
            1,
            "green",
            &image([0, 255, 0]),
            &camera_at(0.0, 0.0, 1.0),
            &raw,
        );
        let points: Vec<_> = raw.into_iter().map(surface).collect();
        let result = state
            .recolor_points(&points, |path| {
                Ok(if path == Path::new("red") {
                    image([255, 0, 0])
                } else {
                    image([0, 255, 0])
                })
            })
            .unwrap();
        assert_eq!(result.colors, vec![Some([0, 255, 0]); 2]);
        assert_eq!(result.selections.len(), 1);
        assert_eq!(result.selections[0].frame_index, 1);
    }

    #[test]
    fn distant_regions_can_choose_different_images() {
        let scratch = Scratch::new();
        let mut state = RegionalRecolor::new(config(), &scratch.0).unwrap();
        let a = point(0.1, 0.1, 5.1);
        let b = point(10.1, 0.1, 5.1);
        record(
            &mut state,
            0,
            "a",
            &image([255, 0, 0]),
            &camera_at(0.0, 0.0, 0.0),
            &[a],
        );
        record(
            &mut state,
            1,
            "b",
            &image([0, 255, 0]),
            &camera_at(10.0, 0.0, 0.0),
            &[b],
        );
        let result = state
            .recolor_points(&[surface(a), surface(b)], |path| {
                Ok(image(if path == Path::new("a") {
                    [255, 0, 0]
                } else {
                    [0, 255, 0]
                }))
            })
            .unwrap();
        assert_eq!(result.colors, vec![Some([255, 0, 0]), Some([0, 255, 0])]);
        assert_eq!(result.selections.len(), 2);
    }

    #[test]
    fn wall_and_floor_inside_one_spatial_cell_select_views_independently() {
        let scratch = Scratch::new();
        let mut state = RegionalRecolor::new(config(), &scratch.0).unwrap();
        let p = point(0.1, 0.1, 5.1);
        record(
            &mut state,
            0,
            "side",
            &image([255, 0, 0]),
            &camera_at(3.0, 0.0, 0.0),
            &[p],
        );
        record(
            &mut state,
            1,
            "front",
            &image([0, 255, 0]),
            &camera_at(0.0, 0.0, 0.0),
            &[p],
        );
        let points = [
            RegionalRecolorPoint {
                position: p,
                normal: Some(Vector3::new(1.0, 0.0, 0.0)),
            },
            surface(p),
        ];
        let result = state
            .recolor_points(&points, |path| {
                Ok(image(if path == Path::new("side") {
                    [255, 0, 0]
                } else {
                    [0, 255, 0]
                }))
            })
            .unwrap();
        assert_eq!(result.colors, vec![Some([255, 0, 0]), Some([0, 255, 0])]);
        assert_eq!(result.selections.len(), 2);
    }

    #[test]
    fn a_full_coverage_candidate_survives_higher_scoring_partial_views() {
        let scratch = Scratch::new();
        let mut state = RegionalRecolor::new(
            RegionalRecolorConfig {
                top_candidates: 1,
                ..config()
            },
            &scratch.0,
        )
        .unwrap();
        let raw = [
            point(0.01, 0.1, 5.1),
            point(0.2, 0.1, 5.1),
            point(0.4, 0.1, 5.1),
        ];
        record(
            &mut state,
            0,
            "far",
            &image([255, 0, 0]),
            &camera_at(0.0, 0.0, 0.0),
            &raw,
        );
        for (index, z) in [(1, 4.8), (2, 4.85), (3, 4.9)] {
            record(
                &mut state,
                index,
                "close",
                &image([0, 255, 0]),
                &camera_at(0.0, 0.0, z),
                &raw,
            );
        }
        let points: Vec<_> = raw.into_iter().map(surface).collect();
        let result = state
            .recolor_points(&points, |path| {
                Ok(image(if path == Path::new("far") {
                    [255, 0, 0]
                } else {
                    [0, 255, 0]
                }))
            })
            .unwrap();
        assert_eq!(result.colors, vec![Some([255, 0, 0]); 3]);
        assert_eq!(result.selections[0].frame_index, 0);
        assert_eq!(result.selections[0].coverage, 1.0);
    }

    #[test]
    fn final_coordinates_are_reprojected_instead_of_reusing_observation_colors() {
        let scratch = Scratch::new();
        let mut state = RegionalRecolor::new(config(), &scratch.0).unwrap();
        let p = point(0.1, 0.1, 5.1);
        let final_mean = point(0.12, 0.1, 5.1);
        let camera = CameraIntrinsics::for_image_size(960, 600).unwrap();
        let (u, v) = project_camera_point_full_image(&final_mean.cast::<f64>(), &camera).unwrap();
        let mut img = image([255, 0, 0]);
        img.put_pixel(u, v, image::Rgb([0, 255, 0]));
        record(&mut state, 0, "source", &img, &Matrix4::identity(), &[p]);
        let result = state
            .recolor_points(&[surface(final_mean)], |_| Ok(img.clone()))
            .unwrap();
        assert_eq!(result.colors, vec![Some([0, 255, 0])]);
    }

    #[test]
    fn lidar_start_coordinates_and_world_coordinates_share_the_image_time_pose() {
        let scratch = Scratch::new();
        let mut state = RegionalRecolor::new(config(), &scratch.0).unwrap();
        let raw = point(0.1, 0.1, 5.1);
        let mut world_from_lidar = Matrix4::identity();
        world_from_lidar[(0, 3)] = 2.0;
        state
            .observe_frame(
                0,
                0.0,
                Path::new("source"),
                &image([1, 2, 3]),
                &camera_at(2.0, 0.0, 0.0),
                &world_from_lidar,
                &[raw],
            )
            .unwrap();
        let world = world_from_lidar
            .transform_point(&raw.cast::<f64>())
            .cast::<f32>();
        let result = state
            .recolor_points(&[surface(world)], |_| Ok(image([1, 2, 3])))
            .unwrap();
        assert_eq!(result.colors, vec![Some([1, 2, 3])]);
        assert_eq!(result.selections[0].key.region[0], 4);
    }

    #[test]
    fn background_occlusion_does_not_fall_back_to_another_color() {
        let scratch = Scratch::new();
        let mut state = RegionalRecolor::new(config(), &scratch.0).unwrap();
        let front = point(0.0, 0.0, 3.0);
        let back = point(0.0, 0.0, 8.0);
        record(
            &mut state,
            0,
            "source",
            &image([255, 0, 0]),
            &Matrix4::identity(),
            &[front, back],
        );
        let result = state
            .recolor_points(&[surface(front), surface(back)], |_| Ok(image([255, 0, 0])))
            .unwrap();
        assert_eq!(result.colors, vec![Some([255, 0, 0]), None]);
        assert_eq!(result.stats.patches_without_usable_view, 1);
    }

    #[test]
    fn neighbor_candidates_cover_small_final_mean_changes_across_grid_boundaries() {
        let scratch = Scratch::new();
        let mut state = RegionalRecolor::new(config(), &scratch.0).unwrap();
        record(
            &mut state,
            0,
            "source",
            &image([1, 2, 3]),
            &Matrix4::identity(),
            &[point(0.49, 0.1, 5.1)],
        );
        let result = state
            .recolor_points(&[surface(point(0.51, 0.1, 5.1))], |_| Ok(image([1, 2, 3])))
            .unwrap();
        assert_eq!(result.colors, vec![Some([1, 2, 3])]);
        assert_eq!(result.selections[0].key.region[0], 1);
        assert_eq!(result.stats.unknown_depth_colored_points, 1);
    }

    #[test]
    fn images_are_loaded_per_frame_not_per_patch() {
        let scratch = Scratch::new();
        let mut state = RegionalRecolor::new(config(), &scratch.0).unwrap();
        let raw = [
            point(0.1, 0.1, 5.1),
            point(0.6, 0.1, 5.1),
            point(1.1, 0.1, 5.1),
        ];
        record(
            &mut state,
            0,
            "source",
            &image([1, 2, 3]),
            &Matrix4::identity(),
            &raw,
        );
        let points: Vec<_> = raw.into_iter().map(surface).collect();
        let mut loads = 0;
        let result = state
            .recolor_points(&points, |_| {
                loads += 1;
                Ok(image([1, 2, 3]))
            })
            .unwrap();
        assert_eq!(result.stats.selected_patches, 3);
        assert_eq!(loads, 2);
        assert_eq!(result.stats.images_loaded_for_scoring, 1);
        assert_eq!(result.stats.images_loaded_for_coloring, 1);
    }

    #[test]
    fn sparse_depth_roundtrip_preserves_nearest_depth_and_unknown_pixels() {
        let scratch = Scratch::new();
        let camera = CameraIntrinsics::for_image_size(960, 600).unwrap();
        let depth = DepthRaster::build(
            &[
                point(0.0, 0.0, 8.0),
                point(0.0, 0.0, 3.0),
                point(1.0, 0.0, 3.0),
            ],
            &Matrix4::identity(),
            &camera,
        );
        let file = scratch.0.join("cache.depth");
        depth.write_sparse(&file).unwrap();
        assert_eq!(fs::metadata(&file).unwrap().len(), 24 + 2 * 8);
        let restored = DepthRaster::read_sparse(&file, &camera).unwrap();
        assert_eq!(
            restored.classify(464, 336, 8.0, 0.2, 0),
            DepthState::Occluded
        );
        assert_eq!(restored.classify(464, 336, 3.0, 0.2, 0), DepthState::Clear);
        assert_eq!(restored.classify(0, 0, 3.0, 0.2, 0), DepthState::Unknown);
        fs::write(&file, b"invalid").unwrap();
        assert!(DepthRaster::read_sparse(&file, &camera).is_err());
    }

    #[test]
    fn local_sharpness_distinguishes_crisp_and_blurred_edges() {
        let mut sharp = image([0, 0, 0]);
        for y in 0..600 {
            for x in 0..960 {
                let value = if ((x / 4) + (y / 4)) % 2 == 0 { 0 } else { 255 };
                sharp.put_pixel(x, y, image::Rgb([value; 3]));
            }
        }
        let blurry = image::imageops::blur(&sharp, 2.0);
        let a = SharpnessIntegral::new(&sharp).variance([400, 300, 500, 400], 960, 600);
        let b = SharpnessIntegral::new(&blurry).variance([400, 300, 500, 400], 960, 600);
        assert!(a > b * 4.0, "sharp={a}, blurry={b}");
        assert_eq!(
            SharpnessIntegral::new(&image([100; 3])).variance([400, 300, 500, 400], 960, 600),
            0.0
        );
    }

    #[test]
    fn sharpness_can_prefer_a_sharper_image_over_a_slightly_closer_blurry_one() {
        let scratch = Scratch::new();
        let mut state = RegionalRecolor::new(config(), &scratch.0).unwrap();
        let mut sharp = image([0, 0, 0]);
        for y in 0..600 {
            for x in 0..960 {
                let value = if ((x / 4) + (y / 4)) % 2 == 0 { 0 } else { 255 };
                sharp.put_pixel(x, y, image::Rgb([value; 3]));
            }
        }
        let blurry = image::imageops::blur(&sharp, 4.0);
        let mut raw = Vec::new();
        for x in [0.05_f32, 0.25, 0.45] {
            for y in [0.05_f32, 0.25, 0.45] {
                raw.push(point(x, y, 5.1));
            }
        }
        record(
            &mut state,
            0,
            "sharp",
            &sharp,
            &camera_at(0.0, 0.0, 0.0),
            &raw,
        );
        record(
            &mut state,
            1,
            "blurry",
            &blurry,
            &camera_at(0.0, 0.0, 0.25),
            &raw,
        );
        let points: Vec<_> = raw.into_iter().map(surface).collect();
        let result = state
            .recolor_points(&points, |path| {
                Ok(if path == Path::new("sharp") {
                    sharp.clone()
                } else {
                    blurry.clone()
                })
            })
            .unwrap();
        assert_eq!(result.selections.len(), 1);
        assert_eq!(result.selections[0].frame_index, 0);
        assert_eq!(result.stats.colored_points, points.len());
    }

    #[test]
    fn report_records_selected_images_and_their_camera_poses() {
        let scratch = Scratch::new();
        let mut state = RegionalRecolor::new(config(), &scratch.0).unwrap();
        let p = point(0.1, 0.1, 5.1);
        record(
            &mut state,
            7,
            "source",
            &image([1, 2, 3]),
            &Matrix4::identity(),
            &[p],
        );
        let result = state
            .recolor_points(&[surface(p)], |_| Ok(image([1, 2, 3])))
            .unwrap();
        let path = scratch.0.join("report.json");
        state.write_report(&path, &result).unwrap();
        let json: serde_json::Value = serde_json::from_reader(File::open(path).unwrap()).unwrap();
        assert_eq!(json["selections"][0]["frame_index"], 7);
        assert_eq!(json["frames"][0]["camera_from_world_column_major"][15], 1.0);
        assert_eq!(json["stats"]["colored_points"], 1);
    }

    #[test]
    fn empty_or_invalid_final_points_do_not_load_images() {
        let scratch = Scratch::new();
        let state = RegionalRecolor::new(config(), &scratch.0).unwrap();
        let result = state
            .recolor_points(&[surface(point(f32::NAN, 0.0, 5.0))], |_| {
                panic!("no image needed")
            })
            .unwrap();
        assert_eq!(result.colors, vec![None]);
        assert_eq!(result.stats.uncolored_points, 1);
        assert_eq!(result.stats.surface_patches, 0);
    }

    #[test]
    fn frame_interval_and_duplicate_frame_checks_are_enforced() {
        let scratch = Scratch::new();
        let mut state = RegionalRecolor::new(
            RegionalRecolorConfig {
                evaluate_every: 5,
                ..config()
            },
            &scratch.0,
        )
        .unwrap();
        let p = point(0.1, 0.1, 5.1);
        let img = image([1, 2, 3]);
        record(&mut state, 1, "ignored", &img, &Matrix4::identity(), &[p]);
        assert_eq!(state.frames.len(), 0);
        record(&mut state, 5, "saved", &img, &Matrix4::identity(), &[p]);
        assert!(
            state
                .observe_frame(
                    5,
                    5.0,
                    Path::new("saved"),
                    &img,
                    &Matrix4::identity(),
                    &Matrix4::identity(),
                    &[p]
                )
                .is_err()
        );
    }

    #[test]
    fn failed_cache_write_does_not_commit_dangling_candidates() {
        let scratch = Scratch::new();
        let mut state = RegionalRecolor::new(config(), &scratch.0).unwrap();
        fs::remove_dir(state.cache_dir()).unwrap();
        assert!(
            state
                .observe_frame(
                    0,
                    0.0,
                    Path::new("source"),
                    &image([1, 2, 3]),
                    &Matrix4::identity(),
                    &Matrix4::identity(),
                    &[point(0.1, 0.1, 5.1)]
                )
                .is_err()
        );
        assert!(
            state.frames.is_empty() && state.candidates.is_empty() && state.seen_frames.is_empty()
        );
    }

    #[test]
    fn invalid_configuration_is_rejected_before_creating_a_cache() {
        let mut invalid = config();
        invalid.region_size_m = f64::NAN;
        assert!(invalid.validate().is_err());
        invalid = config();
        invalid.evaluate_every = 0;
        assert!(invalid.validate().is_err());
        invalid = config();
        invalid.min_visible_fraction = 1.01;
        assert!(invalid.validate().is_err());
    }
}
