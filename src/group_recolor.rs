use std::{path::PathBuf, sync::Arc};

use anyhow::{Result, ensure};
use image::RgbImage;
use nalgebra::{Matrix4, Point3, Vector3};
use rustc_hash::FxHashSet;

use crate::{
    camera::{CameraIntrinsics, project_camera_point_subpixel},
    voxel_map::{LOCALMap, SurfaceStatus, VoxelKey, voxel_key},
};

#[derive(Debug, Clone)]
pub struct GroupRecolorConfig {
    pub warmup_frames: usize,
    pub recheck_frames: usize,
    pub min_overlap: f64,
    pub min_coverage: f64,
    pub overlap_voxel_m: f32,
    pub max_candidate_images: usize,
    pub max_group_voxels: usize,

    /// 0.0なら、評価値が改善した時点で交換する。
    pub switch_improvement_ratio: f64,
}

impl Default for GroupRecolorConfig {
    fn default() -> Self {
        Self {
            warmup_frames: 20,
            recheck_frames: 2,
            min_overlap: 0.25,
            min_coverage: 0.90,
            overlap_voxel_m: 0.20,
            max_candidate_images: 24,
            max_group_voxels: 300_000,
            switch_improvement_ratio: 0.0,
        }
    }
}

pub struct ImageCandidate {
    path: PathBuf,
    rgb: Arc<RgbImage>,
    intrinsics: CameraIntrinsics,
    camera_from_world: Matrix4<f64>,
    camera_origin_world: Point3<f64>,
    quality: f64,
}

impl ImageCandidate {
    pub fn new(path: PathBuf, rgb: &RgbImage, camera_from_world: Matrix4<f64>) -> Result<Self> {
        ensure!(
            camera_from_world.iter().all(|v| v.is_finite()),
            "non-finite camera transform"
        );

        let world_from_camera = camera_from_world
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("singular camera transform"))?;

        let camera_origin_world = world_from_camera.transform_point(&Point3::origin());

        ensure!(
            camera_origin_world.coords.iter().all(|v| v.is_finite()),
            "non-finite camera origin"
        );

        Ok(Self {
            path,
            intrinsics: CameraIntrinsics::for_image_size(rgb.width(), rgb.height())?,
            camera_from_world,
            camera_origin_world,
            quality: image_quality(rgb),
            rgb: Arc::new(rgb.clone()),
        })
    }

    /// グループ再着色では画像全体を使う。
    /// 戻り値: 画素座標、カメラからの距離、光軸方向の深度。
    fn project(&self, world_point: &Point3<f32>) -> Option<((u32, u32), f64, f64)> {
        let p = self
            .camera_from_world
            .transform_point(&world_point.cast::<f64>());

        let (pixel, _) = project_camera_point_subpixel(&p, &self.intrinsics)?;

        let u = pixel.x.round();
        let v = pixel.y.round();

        if u < 0.0 || v < 0.0 || u >= self.rgb.width() as f64 || v >= self.rgb.height() as f64 {
            return None;
        }

        let distance = p.coords.norm();
        if !distance.is_finite() || distance <= 0.0 {
            return None;
        }

        Some(((u as u32, v as u32), distance, p.z))
    }

    fn statistics(&self, keys: &FxHashSet<VoxelKey>, map: &LOCALMap) -> (f64, f64) {
        let mut total = 0usize;
        let mut covered = 0usize;
        let mut density_sum = 0.0;

        for key in keys {
            let Some(point) = map_position(map, key) else {
                continue;
            };
            total += 1;

            if let Some((_, _, z)) = self.project(&point) {
                covered += 1;

                // 投影上の画素密度。極端な近距離の影響をlogで抑える。
                density_sum += (self.intrinsics.fx * self.intrinsics.fy / (z * z)).ln_1p();
            }
        }

        let coverage = if total == 0 {
            1.0
        } else {
            covered as f64 / total as f64
        };

        let density = if covered == 0 {
            0.0
        } else {
            density_sum / covered as f64
        };

        (coverage, density)
    }
}

/// 初期選択中は最大20フレーム、継続中は最大2フレームだけ保持する。
/// 各フレーム内でもボクセルキーは重複しない。
struct BufferedFrame {
    number: u64,
    keys: FxHashSet<VoxelKey>,
    image: Option<Arc<ImageCandidate>>,
}

#[derive(Clone, Default)]
struct Group {
    id: u64,
    first_frame: u64,
    last_frame: u64,
    frame_count: u64,
    keys: FxHashSet<VoxelKey>,
    images: Vec<Arc<ImageCandidate>>,
    representative: Option<Arc<ImageCandidate>>,
}

impl Group {
    fn from_frames(frames: &[BufferedFrame]) -> Self {
        let mut group = Self::default();
        group.append(frames);
        group
    }

    fn append(&mut self, frames: &[BufferedFrame]) {
        for frame in frames {
            if self.frame_count == 0 {
                self.first_frame = frame.number;
            }

            self.last_frame = frame.number;
            self.frame_count = self.frame_count.saturating_add(1);
            self.keys.extend(frame.keys.iter().copied());

            if let Some(image) = &frame.image {
                // 同じカメラ画像が複数LiDARフレームに対応しても1候補。
                if !self.images.iter().any(|old| old.path == image.path) {
                    self.images.push(Arc::clone(image));
                }
            }
        }
    }

    fn trim_images(&mut self, limit: usize) {
        while self.images.len() > limit {
            // 代表画像は固定し、それ以外は古い候補から除く。
            let index = self
                .images
                .iter()
                .position(|image| {
                    self.representative
                        .as_ref()
                        .is_none_or(|selected| selected.path != image.path)
                })
                .expect("at least one non-representative image");

            self.images.remove(index);
        }
    }
}

struct ImageEvaluation {
    image: Arc<ImageCandidate>,
    coverage: f64,
    score: f64,
    sufficient: bool,
}

pub struct GroupRecolorer {
    config: GroupRecolorConfig,
    active: Option<Group>,
    buffered: Vec<BufferedFrame>,
    next_group_id: u64,
}

impl GroupRecolorer {
    pub fn new(config: GroupRecolorConfig) -> Self {
        assert!(config.warmup_frames > 0);
        assert!(config.recheck_frames > 0);
        assert!(
            config.max_candidate_images
                >= config
                    .warmup_frames
                    .max(config.recheck_frames.saturating_add(1))
        );
        assert!(config.max_group_voxels > 0);
        assert!(config.overlap_voxel_m.is_finite() && config.overlap_voxel_m > 0.0);
        assert!(config.min_overlap.is_finite() && (0.0..1.0).contains(&config.min_overlap));
        assert!(
            config.min_coverage.is_finite()
                && config.min_coverage > 0.0
                && config.min_coverage <= 1.0
        );
        assert!(
            config.switch_improvement_ratio.is_finite() && config.switch_improvement_ratio >= 0.0
        );

        Self {
            config,
            active: None,
            buffered: Vec::new(),
            next_group_id: 0,
        }
    }

    /// 完了したLiDARフレームごとに1回呼ぶ。
    ///
    /// keysには、このフレームでGlobalMapに実際に採用された
    /// dirty_keysを渡す。ICP不採用時は空集合、imageはNone。
    pub fn push_frame(
        &mut self,
        number: u64,
        mut keys: FxHashSet<VoxelKey>,
        image: Option<ImageCandidate>,
        map: &mut LOCALMap,
    ) {
        keys.retain(|key| map_position(map, key).is_some());

        self.buffered.push(BufferedFrame {
            number,
            keys,
            image: image.map(Arc::new),
        });

        if self.active.is_some() {
            if self.buffered.len() >= self.config.recheck_frames {
                self.recheck(map);
            }
        } else if self.buffered.len() >= self.config.warmup_frames {
            let frames = std::mem::take(&mut self.buffered);
            self.initialize(frames, map, true);
        }
    }

    /// 最終PCD保存前に呼ぶ。20未満・2未満の端数も処理する。
    pub fn flush(&mut self, map: &mut LOCALMap) {
        if self.active.is_some() && !self.buffered.is_empty() {
            self.recheck(map);
        }

        if let Some(group) = self.active.take() {
            self.close(group, map, "end_of_input");
        }

        if !self.buffered.is_empty() {
            let frames = std::mem::take(&mut self.buffered);
            self.initialize(frames, map, false);
        }
    }

    fn allocate_id(&mut self) -> u64 {
        self.next_group_id = self.next_group_id.saturating_add(1);
        self.next_group_id
    }

    fn active_image_limit(&self) -> usize {
        // 次の判定用画像の席を予約する。
        // active + buffered の候補画像を設定上限以内に収める。
        self.config.max_candidate_images - self.config.recheck_frames
    }

    fn recheck(&mut self, map: &mut LOCALMap) {
        let frames = std::mem::take(&mut self.buffered);
        let old = self.active.take().expect("active group required");
        let incoming = Group::from_frames(&frames);

        let overlap = overlap_ratio(&old.keys, &incoming.keys, map, self.config.overlap_voxel_m);

        // 判定前に既存グループを変更しない。
        let mut trial = old.clone();
        trial.append(&frames);

        let within_limit = trial.keys.len() <= self.config.max_group_voxels;
        let overlap_passed = overlap > self.config.min_overlap;

        let selected = if within_limit && overlap_passed {
            self.select(&trial, &frames, map)
        } else {
            None
        };

        if let Some(image) = selected {
            paint_group(&mut trial, image, map, "continue");
            trial.trim_images(self.active_image_limit());
            self.active = Some(trial);

            log::info!("Group recolor: continued overlap={:.3}", overlap,);
        } else {
            let reason = if !within_limit {
                "voxel_limit"
            } else if !overlap_passed {
                "low_overlap"
            } else {
                "insufficient_single_image_coverage"
            };

            self.close(old, map, reason);

            // 保留中のフレームは次グループの先頭へ引き継ぐ。
            self.buffered = frames;

            log::info!(
                "Group recolor: boundary reason={} overlap={:.3} \
                 next_buffered_frames={}",
                reason,
                overlap,
                self.buffered.len(),
            );

            // recheck_framesをwarmup_frames以上に設定した場合にも対応。
            if self.buffered.len() >= self.config.warmup_frames {
                let frames = std::mem::take(&mut self.buffered);
                self.initialize(frames, map, true);
            }
        }
    }

    fn initialize(
        &mut self,
        mut frames: Vec<BufferedFrame>,
        map: &mut LOCALMap,
        keep_tail_active: bool,
    ) {
        let mut group = Group::from_frames(&frames);

        if group.keys.is_empty() {
            return;
        }

        if group.images.is_empty() {
            log::warn!(
                "Group recolor: frames {}..{} have no candidate image; \
                 geometry retained",
                group.first_frame,
                group.last_frame,
            );
            return;
        }

        let selected = if group.keys.len() <= self.config.max_group_voxels {
            self.select(&group, &frames, map)
        } else {
            None
        };

        if let Some(image) = selected {
            group.id = self.allocate_id();
            paint_group(&mut group, image, map, "initial_selection");

            if keep_tail_active {
                group.trim_images(self.active_image_limit());
                self.active = Some(group);
            } else {
                log::info!(
                    "Group recolor: finalized group={} frames={}..{}",
                    group.id,
                    group.first_frame,
                    group.last_frame,
                );
            }

            return;
        }

        if frames.len() > 1 {
            // 初期20フレームですでに1枚では覆えない場合。
            // 時系列を保ったまま小グループへ分割して再選択する。
            let right = frames.split_off(frames.len() / 2);
            self.initialize(frames, map, false);
            self.initialize(right, map, keep_tail_active);
            return;
        }

        // 1フレームでも画角外が多い場合、存在する画像だけでは
        // 必要被覆率を達成できない。最も広く覆える画像で着色し、
        // 継続グループにはしない。点群の形状は保持する。
        //
        // 単一フレーム自体が上限を超える場合も、点を捨てずに分割。
        let keys = std::mem::take(&mut group.keys);
        let mut keys = keys.into_iter();

        loop {
            group.keys = keys.by_ref().take(self.config.max_group_voxels).collect();

            if group.keys.is_empty() {
                break;
            }

            group.id = self.allocate_id();
            self.close(group.clone(), map, "single_frame_fallback");
        }
    }

    fn evaluate(
        &self,
        group: &Group,
        checks: &[BufferedFrame],
        map: &LOCALMap,
    ) -> Vec<ImageEvaluation> {
        let (centroid, normal) = group_view(group, map);

        group
            .images
            .iter()
            .map(|image| {
                let (coverage, density) = image.statistics(&group.keys, map);

                // 累積全体だけでなく、今回の各フレームも十分に
                // 覆えることを確認する。古い範囲による希釈を防ぐ。
                let sufficient = coverage + 1e-12 >= self.config.min_coverage
                    && checks.iter().all(|frame| {
                        image.statistics(&frame.keys, map).0 + 1e-12 >= self.config.min_coverage
                    });

                let view = (image.camera_origin_world - centroid).try_normalize(1e-8);

                let frontal = match (normal.as_ref(), view.as_ref()) {
                    (Some(n), Some(v)) => n.dot(v).abs().clamp(0.0, 1.0),
                    _ => 1.0,
                };

                ImageEvaluation {
                    image: Arc::clone(image),
                    coverage,
                    score: image.quality * density * (0.25 + 0.75 * frontal),
                    sufficient,
                }
            })
            .collect()
    }

    fn select(
        &self,
        group: &Group,
        checks: &[BufferedFrame],
        map: &LOCALMap,
    ) -> Option<Arc<ImageCandidate>> {
        let evaluations = self.evaluate(group, checks, map);

        let best = evaluations
            .iter()
            .filter(|item| item.sufficient)
            .max_by(|a, b| {
                a.score
                    .total_cmp(&b.score)
                    .then_with(|| a.coverage.total_cmp(&b.coverage))
            })?;

        if let Some(current) = &group.representative {
            if let Some(old) = evaluations
                .iter()
                .find(|item| item.sufficient && item.image.path == current.path)
            {
                let required_score = old.score * (1.0 + self.config.switch_improvement_ratio);

                if best.score <= required_score {
                    return Some(Arc::clone(&old.image));
                }
            }
        }

        Some(Arc::clone(&best.image))
    }

    fn close(&self, mut group: Group, map: &mut LOCALMap, reason: &str) {
        let selected = self.select(&group, &[], map).or_else(|| {
            // 十分に覆える画像がない場合の明示的なフォールバック。
            self.evaluate(&group, &[], map)
                .into_iter()
                .filter(|item| item.coverage > 0.0)
                .max_by(|a, b| {
                    a.coverage
                        .total_cmp(&b.coverage)
                        .then_with(|| a.score.total_cmp(&b.score))
                })
                .map(|item| item.image)
        });

        if let Some(image) = selected {
            paint_group(&mut group, image, map, reason);
        } else {
            log::warn!(
                "Group recolor: group={} frames={}..{} has no usable image; \
                 geometry retained",
                group.id,
                group.first_frame,
                group.last_frame,
            );
        }

        log::info!(
            "Group recolor: finalized group={} frames={}..{} reason={}",
            group.id,
            group.first_frame,
            group.last_frame,
            reason,
        );
    }
}

fn map_position(map: &LOCALMap, key: &VoxelKey) -> Option<Point3<f32>> {
    let point = map.voxel_map.get(key)?.mean;
    point.coords.iter().all(|v| v.is_finite()).then_some(point)
}

fn overlap_ratio(
    old: &FxHashSet<VoxelKey>,
    incoming: &FxHashSet<VoxelKey>,
    map: &LOCALMap,
    voxel_size: f32,
) -> f64 {
    let coarse = |keys: &FxHashSet<VoxelKey>| -> FxHashSet<VoxelKey> {
        keys.iter()
            .filter_map(|key| map_position(map, key))
            .map(|point| voxel_key(&point, voxel_size))
            .collect()
    };

    let old = coarse(old);
    let incoming = coarse(incoming);

    if incoming.is_empty() {
        // ICP不採用など、観測のないフレームだけでは区切らない。
        return 1.0;
    }

    incoming.intersection(&old).count() as f64 / incoming.len() as f64
}

fn group_view(group: &Group, map: &LOCALMap) -> (Point3<f64>, Option<Vector3<f64>>) {
    let mut position_sum = Vector3::<f64>::zeros();
    let mut point_count = 0u64;
    let mut normal_sum = Vector3::<f64>::zeros();

    for key in &group.keys {
        let Some(cell) = map.voxel_map.get(key) else {
            continue;
        };

        if !cell.mean.coords.iter().all(|v| v.is_finite()) {
            continue;
        }

        position_sum += cell.mean.coords.cast::<f64>();
        point_count += 1;

        if cell.surface_status != SurfaceStatus::Planar {
            continue;
        }

        let Some(plane) = cell.surface_plane else {
            continue;
        };

        let Some(mut normal) = plane.normal.cast::<f64>().try_normalize(1e-8) else {
            continue;
        };

        if normal_sum.dot(&normal) < 0.0 {
            normal = -normal;
        }
        normal_sum += normal;
    }

    let centroid = if point_count == 0 {
        Point3::origin()
    } else {
        Point3::from(position_sum / point_count as f64)
    };

    (centroid, normal_sum.try_normalize(1e-8))
}

fn paint_group(group: &mut Group, image: Arc<ImageCandidate>, map: &mut LOCALMap, reason: &str) {
    let switched = group
        .representative
        .as_ref()
        .is_some_and(|old| old.path != image.path);

    let mut colored = 0usize;

    for key in &group.keys {
        let Some(cell) = map.voxel_map.get_mut(key) else {
            continue;
        };

        if let Some(((u, v), distance, _)) = image.project(&cell.mean) {
            // グループ全体を同じ画像で上書きする。
            // update_color_if_closer()は使わない。
            cell.rgb = Some(image.rgb.get_pixel(u, v).0);
            cell.color_observation_distance_m = Some(distance);
            colored += 1;
        } else {
            // 前の代表画像の色が部分的に残ることを防ぐ。
            cell.rgb = None;
            cell.color_observation_distance_m = None;
        }
    }

    let coverage = image.statistics(&group.keys, map).0;

    log::info!(
        "Group recolor: group={} frames={}..{} frame_count={} \
         voxels={} colored={} coverage={:.3} switched={} \
         candidates={} reason={} image={}",
        group.id,
        group.first_frame,
        group.last_frame,
        group.frame_count,
        group.keys.len(),
        colored,
        coverage,
        switched,
        group.images.len(),
        reason,
        image.path.display(),
    );

    group.representative = Some(image);
}

fn image_quality(image: &RgbImage) -> f64 {
    if image.width() < 3 || image.height() < 3 {
        return 0.0;
    }

    let gray = |x: u32, y: u32| {
        let p = image.get_pixel(x, y).0;
        0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64
    };

    let mut count = 0.0;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut clipped = 0.0;

    for y in (1..image.height() - 1).step_by(4) {
        for x in (1..image.width() - 1).step_by(4) {
            let center = gray(x, y);
            let laplacian =
                gray(x - 1, y) + gray(x + 1, y) + gray(x, y - 1) + gray(x, y + 1) - 4.0 * center;

            count += 1.0;
            let delta = laplacian - mean;
            mean += delta / count;
            m2 += delta * (laplacian - mean);

            if center <= 5.0 || center >= 250.0 {
                clipped += 1.0;
            }
        }
    }

    let variance = (m2 / count.max(1.0)).max(0.0);
    let usable = (1.0 - clipped / count.max(1.0)).max(0.05);

    variance.ln_1p().max(0.01) * usable * usable
}
