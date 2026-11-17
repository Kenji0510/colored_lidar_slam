//! キーフレーム画像の貼り付け。
//!
//! 視覚アライメントのキーフレーム N, N+1 について、N+1 が確定した時点で N を貼り付け待ちにし、
//! `paste_delay_frames` 後（GlobalMap の平面判定が N の視野に追いついた後）に、N の画像を
//! N がカバーする平面セルへ直接貼る。これを走行中ずっと繰り返す。
//!
//! - 貼り付け先: GlobalMap の Planar セルのコピー（このモジュールが持つ貼り付け用データ）。
//!   GlobalMap 本体（`VoxelCell::rgb` など）は変更しない。GlobalMap は新しい観測が入った
//!   セルの判定をいったん Unknown に戻すが、コピーは最後に Planar と確定したときの点と
//!   法線を持ち続ける。NonPlanar と判定し直されたら貼り付け対象から外す（貼った色は残す）。
//! - 遮蔽: 平面フィルタをかける前の GlobalMap の全セルを N の画像に投影した深度バッファで判定。
//! - 重なり: 同じセルに貼れる画像が複数あれば、面上の解像度 cosθ / 距離²（より近く、
//!   より正面から見ているほど大きい）が高い方で上書きする。
//! - 色: セルの見かけの大きさの範囲をバイリニアで平均する（2.5 cm セルが数画素に写る近距離で、
//!   1 画素だけ拾ってギザギザになるのを防ぐ）。

use std::collections::VecDeque;

use image::RgbImage;
use nalgebra::{Matrix4, Point3, Vector2, Vector3};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::camera::{CameraIntrinsics, project_camera_point_subpixel};
use crate::voxel_map::{
    SurfaceClassificationRecord, SurfacePlane, SurfaceStatus, VoxelKey, VoxelMap,
};

#[derive(Debug, Clone)]
pub struct KeyframePasteConfig {
    /// N+1 の確定から N を貼るまで待つフレーム数（GlobalMap の平面判定の遅れに合わせる）。
    pub paste_delay_frames: u64,
    pub min_depth_m: f64,
    pub max_depth_m: f64,
    /// 画像の縦横それぞれ、中央のこの割合だけを使う（既存の色付けと同じ 75%）。
    pub image_keep_ratio: f64,
    /// 面の法線と視線のなす角の cos の下限（これより斜めに見ている面には貼らない）。
    pub min_view_cosine: f64,
    /// 遮蔽判定用の深度バッファのマス [px]。幅 960 px の画像での値で、画像の幅に比例させる。
    pub depth_buffer_cell_px: f64,
    /// 遮蔽物のセルを深度バッファに描く大きさ（セルの見かけの幅に対する倍率）。
    pub occluder_splat_scale: f64,
    /// 深度バッファの最小深度より `margin_m + ratio·深度 + 面の傾きによる深度差` 以上
    /// 奥にあるセルは遮蔽されているとみなす。
    pub occlusion_margin_m: f64,
    pub occlusion_margin_ratio: f64,
    /// セルの見かけの大きさに合わせて色を平均するときの、1 辺あたりの最大サンプル数。
    pub max_footprint_samples: usize,
    /// 既に色があるセルを上書きするのに必要なスコアの比（1.0 = 少しでも良ければ上書き）。
    pub min_score_ratio: f64,
}

impl Default for KeyframePasteConfig {
    fn default() -> Self {
        Self {
            paste_delay_frames: 5,
            min_depth_m: 0.5,
            max_depth_m: 40.0,
            image_keep_ratio: 0.75,
            min_view_cosine: 0.25,
            depth_buffer_cell_px: 4.0,
            occluder_splat_scale: 1.5,
            occlusion_margin_m: 0.05,
            occlusion_margin_ratio: 0.01,
            max_footprint_samples: 5,
            min_score_ratio: 1.0,
        }
    }
}

impl KeyframePasteConfig {
    fn valid(&self) -> bool {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        let non_negative = |v: f64| v.is_finite() && v >= 0.0;
        positive(self.min_depth_m)
            && self.max_depth_m.is_finite()
            && self.max_depth_m > self.min_depth_m
            && self.image_keep_ratio > 0.0
            && self.image_keep_ratio <= 1.0
            && self.min_view_cosine > 0.0
            && self.min_view_cosine <= 1.0
            && positive(self.depth_buffer_cell_px)
            && positive(self.occluder_splat_scale)
            && non_negative(self.occlusion_margin_m)
            && non_negative(self.occlusion_margin_ratio)
            && self.max_footprint_samples > 0
            && positive(self.min_score_ratio)
    }
}

/// 貼り付けに使うキーフレーム 1 枚分（視覚アライメントのキーフレームと同じ画像・姿勢）。
#[derive(Clone)]
pub struct KeyframeImage {
    pub frame_index: u64,
    /// 画像を撮った時刻のカメラ姿勢（キーフレーム作成時に確定した値）。
    pub camera_from_world: Matrix4<f64>,
    pub camera: CameraIntrinsics,
    pub rgb: RgbImage,
}

/// セルに貼られた色と、その色を貼ったときの見え方。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PastedColor {
    pub rgb: [u8; 3],
    /// 面上の解像度 cosθ / 距離² [1/m²]。大きいほど近く、正面から見ている。
    pub score: f32,
    pub keyframe_frame_index: u64,
}

/// GlobalMap の Planar セル 1 つのコピー。
#[derive(Debug, Clone, Copy)]
struct PasteCell {
    /// 最後に Planar と確定したときのセルの点（その平面上へ射影済み）と法線。
    point: Point3<f32>,
    normal: Vector3<f32>,
    /// 最新の確定判定が Planar か。NonPlanar になったら貼り付け対象から外す。
    planar: bool,
    color: Option<PastedColor>,
}

/// キーフレーム 1 枚を貼ったときの内訳。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PasteStats {
    pub keyframe_frame_index: u64,
    /// 深度バッファに描いた GlobalMap のセル数（平面フィルタ前の全セルのうち視野内）。
    pub occluder_cells: usize,
    /// コピー内の Planar セル数（貼り付け対象の全体）。
    pub target_cells: usize,
    /// 深度範囲内で、画像の中央部（`image_keep_ratio`）に写った対象セル数。
    pub in_view: usize,
    pub too_oblique: usize,
    pub occluded: usize,
    pub sample_failed: usize,
    /// まだ色がなかったセルに貼った数。
    pub pasted_new: usize,
    /// より良い見え方だったので上書きした数。
    pub overwritten: usize,
    /// 既存の色の方が良い見え方だったので残した数。
    pub kept_existing: usize,
}

struct PendingPaste {
    due_frame: u64,
    keyframe: KeyframeImage,
}

pub struct KeyframePaster {
    pub config: KeyframePasteConfig,
    cells: FxHashMap<VoxelKey, PasteCell>,
    /// 今のキーフレーム（次のキーフレームが確定したら貼り付け待ちへ移る）。
    current: Option<KeyframeImage>,
    pending: VecDeque<PendingPaste>,
}

impl KeyframePaster {
    pub fn new(config: KeyframePasteConfig) -> Self {
        Self {
            config,
            cells: FxHashMap::default(),
            current: None,
            pending: VecDeque::new(),
        }
    }

    /// GlobalMap の平面判定の結果をコピーへ反映する（毎フレーム、遅延判定の後に呼ぶ）。
    /// Planar: 点と法線を更新（色は残す）。NonPlanar: 貼り付け対象から外す。
    pub fn sync_planes(&mut self, records: &[SurfaceClassificationRecord]) {
        for record in records {
            self.apply_classification(record.key, &record.mean, record.status, record.plane);
        }
    }

    /// GlobalMap の今の判定を、全セルについてそのままコピーへ反映する。
    /// 走行の最後、全セルの再判定（`classify_all_surface_voxels`）の後で `flush` の前に呼ぶ。
    /// 走行の終わり近くに初めて見えた場所は、遅延判定がまだ届いていないため。
    pub fn sync_from_map(&mut self, map: &VoxelMap) {
        for (key, cell) in map {
            self.apply_classification(*key, &cell.mean, cell.surface_status, cell.surface_plane);
        }
    }

    fn apply_classification(
        &mut self,
        key: VoxelKey,
        mean: &Point3<f32>,
        status: SurfaceStatus,
        plane: Option<SurfacePlane>,
    ) {
        match status {
            SurfaceStatus::Planar => {
                let Some(plane) = plane else {
                    return;
                };
                let norm = plane.normal.norm();
                if !(norm.is_finite() && norm > 1e-6)
                    || !plane.d.is_finite()
                    || !mean.coords.iter().all(|v| v.is_finite())
                {
                    return;
                }
                let normal = plane.normal / norm;
                let offset = plane.d / norm;
                let point = mean - normal * (normal.dot(&mean.coords) + offset);
                self.cells
                    .entry(key)
                    .and_modify(|cell| {
                        cell.point = point;
                        cell.normal = normal;
                        cell.planar = true;
                    })
                    .or_insert(PasteCell {
                        point,
                        normal,
                        planar: true,
                        color: None,
                    });
            }
            SurfaceStatus::NonPlanar => {
                if let Some(cell) = self.cells.get_mut(&key) {
                    cell.planar = false;
                }
            }
            SurfaceStatus::Unknown => {}
        }
    }

    /// 新しいキーフレーム N+1 が確定した。1 つ前のキーフレーム N を
    /// `current_frame + paste_delay_frames` に貼る予約をし、(N のフレーム番号, 予定フレーム) を返す。
    pub fn on_new_keyframe(
        &mut self,
        current_frame: u64,
        keyframe: KeyframeImage,
    ) -> Option<(u64, u64)> {
        let previous = self.current.replace(keyframe)?;
        let due_frame = current_frame.saturating_add(self.config.paste_delay_frames);
        let index = previous.frame_index;
        self.pending.push_back(PendingPaste {
            due_frame,
            keyframe: previous,
        });
        Some((index, due_frame))
    }

    /// 予定フレームに達したキーフレームを、確定順に貼る。
    pub fn paste_due(
        &mut self,
        current_frame: u64,
        map: &VoxelMap,
        voxel_size: f32,
    ) -> Vec<PasteStats> {
        let mut results = Vec::new();
        while self
            .pending
            .front()
            .is_some_and(|pending| pending.due_frame <= current_frame)
        {
            if let Some(pending) = self.pending.pop_front() {
                results.push(self.paste(&pending.keyframe, map, voxel_size));
            }
        }
        results
    }

    /// 走行の最後に呼ぶ。待っているキーフレームと、今のキーフレームを貼る
    /// （以後マップは増えないので、待つ理由がない）。
    pub fn flush(&mut self, map: &VoxelMap, voxel_size: f32) -> Vec<PasteStats> {
        let mut keyframes: Vec<KeyframeImage> = self
            .pending
            .drain(..)
            .map(|pending| pending.keyframe)
            .collect();
        keyframes.extend(self.current.take());
        keyframes
            .iter()
            .map(|keyframe| self.paste(keyframe, map, voxel_size))
            .collect()
    }

    /// キーフレーム 1 枚を、写っている Planar セル（コピー）へ貼る。
    /// `map` は平面フィルタ前の GlobalMap（全セルを遮蔽物として使う）。
    pub fn paste(
        &mut self,
        keyframe: &KeyframeImage,
        map: &VoxelMap,
        voxel_size: f32,
    ) -> PasteStats {
        let config = self.config.clone();
        let camera = &keyframe.camera;
        let voxel_size = voxel_size as f64;
        let mut stats = PasteStats {
            keyframe_frame_index: keyframe.frame_index,
            ..PasteStats::default()
        };
        if !config.valid()
            || !(voxel_size.is_finite() && voxel_size > 0.0)
            || keyframe.rgb.width() != camera.width
            || keyframe.rgb.height() != camera.height
            || camera.width < 2
            || camera.height < 2
            || !(camera.fx > 0.0 && camera.fy > 0.0)
        {
            log::warn!(
                "keyframe paste {}: invalid config, voxel size or image dimensions",
                keyframe.frame_index
            );
            return stats;
        }
        let Some(world_from_camera) = keyframe.camera_from_world.try_inverse() else {
            return stats;
        };
        let start = std::time::Instant::now();

        // 1. 遮蔽物: 平面フィルタをかける前の GlobalMap の全セル
        let depth_buffer = DepthBuffer::build(
            map,
            &keyframe.camera_from_world,
            camera,
            voxel_size,
            &config,
        );
        stats.occluder_cells = depth_buffer.drawn_cells;
        let depth_buffer_time = start.elapsed();

        // 2. 貼り付け対象: コピー内の Planar セル
        let view = View {
            config: &config,
            camera_from_world: &keyframe.camera_from_world,
            camera_center: Point3::from(world_from_camera.fixed_view::<3, 1>(0, 3).into_owned()),
            camera,
            image: &keyframe.rgb,
            crop: Crop::new(camera, config.image_keep_ratio),
            cull: QuickCull::new(
                &keyframe.camera_from_world,
                camera,
                config.min_depth_m,
                config.max_depth_m,
                0.25,
            ),
            depth_buffer: &depth_buffer,
            voxel_size,
        };
        let outcomes: Vec<(VoxelKey, TargetOutcome)> = self
            .cells
            .par_iter()
            .filter(|(_, cell)| cell.planar)
            .filter_map(|(key, cell)| view.evaluate(cell).map(|outcome| (*key, outcome)))
            .collect();
        stats.target_cells = self.planar_cell_count();
        stats.in_view = outcomes.len();
        let targets_time = start.elapsed() - depth_buffer_time;

        // 3. 反映（同じキーフレームの中で同じセルは 1 回だけ出てくる）
        for (key, outcome) in outcomes {
            match outcome {
                TargetOutcome::TooOblique => stats.too_oblique += 1,
                TargetOutcome::Occluded => stats.occluded += 1,
                TargetOutcome::SampleFailed => stats.sample_failed += 1,
                TargetOutcome::Sampled { rgb, score } => {
                    let Some(cell) = self.cells.get_mut(&key) else {
                        continue;
                    };
                    let candidate = PastedColor {
                        rgb,
                        score,
                        keyframe_frame_index: keyframe.frame_index,
                    };
                    match cell.color {
                        None => {
                            cell.color = Some(candidate);
                            stats.pasted_new += 1;
                        }
                        Some(existing)
                            if score as f64 > existing.score as f64 * config.min_score_ratio =>
                        {
                            cell.color = Some(candidate);
                            stats.overwritten += 1;
                        }
                        Some(_) => stats.kept_existing += 1,
                    }
                }
            }
        }
        log::debug!(
            "keyframe paste {} timings [ms]: depth_buffer={:.3} (map cells={}), \
             targets={:.3}, apply={:.3}",
            keyframe.frame_index,
            depth_buffer_time.as_secs_f64() * 1e3,
            map.len(),
            targets_time.as_secs_f64() * 1e3,
            (start.elapsed() - depth_buffer_time - targets_time).as_secs_f64() * 1e3,
        );
        stats
    }

    /// GlobalMap のセル `key` に貼られた色（コピーにない、またはまだ貼られていなければ None）。
    pub fn color(&self, key: &VoxelKey) -> Option<PastedColor> {
        self.cells.get(key).and_then(|cell| cell.color)
    }

    /// コピー内で、最新の確定判定が Planar のセル数。
    pub fn planar_cell_count(&self) -> usize {
        self.cells.values().filter(|cell| cell.planar).count()
    }

    /// 色が貼られたセル数（後から NonPlanar になったセルも含む）。
    pub fn pasted_cell_count(&self) -> usize {
        self.cells
            .values()
            .filter(|cell| cell.color.is_some())
            .count()
    }

    pub fn current_keyframe_index(&self) -> Option<u64> {
        self.current.as_ref().map(|keyframe| keyframe.frame_index)
    }

    /// 貼り付け待ちのキーフレーム（フレーム番号, 予定フレーム）。
    pub fn pending_keyframes(&self) -> Vec<(u64, u64)> {
        self.pending
            .iter()
            .map(|pending| (pending.keyframe.frame_index, pending.due_frame))
            .collect()
    }
}

/// 貼り付け元のキーフレームを見分けるための色（デバッグ出力用）。
pub fn keyframe_debug_color(frame_index: u64) -> [u8; 3] {
    let mut value = frame_index.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    let bytes = value.to_le_bytes();
    [
        64 + bytes[0] % 192,
        64 + bytes[1] % 192,
        64 + bytes[2] % 192,
    ]
}

// ---------------------------------------------------------------------------
// 1 枚分の処理
// ---------------------------------------------------------------------------

enum TargetOutcome {
    TooOblique,
    Occluded,
    SampleFailed,
    Sampled { rgb: [u8; 3], score: f32 },
}

/// 画像の中央部（縦横それぞれ `keep_ratio`）。
struct Crop {
    min: Vector2<f64>,
    max: Vector2<f64>,
}

impl Crop {
    fn new(camera: &CameraIntrinsics, keep_ratio: f64) -> Self {
        let margin = (1.0 - keep_ratio) * 0.5;
        let size = Vector2::new(camera.width as f64, camera.height as f64);
        Self {
            min: size * margin,
            max: size * (1.0 - margin),
        }
    }

    fn contains(&self, pixel: &Vector2<f64>) -> bool {
        pixel.x >= self.min.x
            && pixel.y >= self.min.y
            && pixel.x < self.max.x
            && pixel.y < self.max.y
    }
}

struct View<'a> {
    config: &'a KeyframePasteConfig,
    camera_from_world: &'a Matrix4<f64>,
    camera_center: Point3<f64>,
    camera: &'a CameraIntrinsics,
    image: &'a RgbImage,
    crop: Crop,
    cull: QuickCull,
    depth_buffer: &'a DepthBuffer,
    voxel_size: f64,
}

impl View<'_> {
    /// 対象セル 1 つ。視野外（深度範囲外・画像の中央部の外）なら None。
    fn evaluate(&self, cell: &PasteCell) -> Option<TargetOutcome> {
        if !self.cull.passes(&cell.point) {
            return None;
        }
        let point = cell.point.cast::<f64>();
        let pc = self.camera_from_world.transform_point(&point);
        if pc.z.is_nan()
            || pc.z < self.config.min_depth_m
            || pc.z > self.config.max_depth_m
            || !roughly_in_view(&pc, self.camera, 0.25)
        {
            return None;
        }
        let (pixel, _) = project_camera_point_subpixel(&pc, self.camera)?;
        if !self.crop.contains(&pixel) {
            return None;
        }
        let to_camera = self.camera_center - point;
        let distance = to_camera.norm();
        if !(distance.is_finite() && distance > 0.0) {
            return None;
        }
        let view_cosine = (cell.normal.cast::<f64>().dot(&to_camera) / distance)
            .abs()
            .min(1.0);
        if view_cosine < self.config.min_view_cosine {
            return Some(TargetOutcome::TooOblique);
        }
        let footprint_radius_px = 0.5 * self.camera.fx * self.voxel_size / pc.z;
        // 色を読む範囲（見かけの大きさ＋バイリニアの 1 画素）に手前の物が少しでも
        // かかっていれば貼らない。物の輪郭で手前の色が混ざるのを防ぐ。
        let support_px = footprint_radius_px + 1.0;
        if self
            .depth_buffer
            .is_occluded(&pixel, support_px, pc.z, view_cosine, self.camera.fx)
        {
            return Some(TargetOutcome::Occluded);
        }
        let Some(rgb) = sample_footprint(
            self.image,
            &pixel,
            footprint_radius_px,
            self.config.max_footprint_samples,
        ) else {
            return Some(TargetOutcome::SampleFailed);
        };
        Some(TargetOutcome::Sampled {
            rgb,
            score: (view_cosine / (distance * distance)) as f32,
        })
    }
}

/// 歪みを無視したピンホール投影で、明らかに画像外の点を安く除外する（余白付き）。
fn roughly_in_view(pc: &Point3<f64>, camera: &CameraIntrinsics, margin_ratio: f64) -> bool {
    let margin_x = margin_ratio * camera.width as f64;
    let margin_y = margin_ratio * camera.height as f64;
    let u = camera.fx * pc.x / pc.z + camera.cx;
    let v = camera.fy * pc.y / pc.z + camera.cy;
    u >= -margin_x
        && u <= camera.width as f64 + margin_x
        && v >= -margin_y
        && v <= camera.height as f64 + margin_y
}

/// GlobalMap の全セルを毎回 f64 で変換すると重いので、f32 の深度と視野の判定で
/// 大半を先に落とす（判定は少し広めにし、残った点は f64 で正確に判定し直す）。
struct QuickCull {
    rows: [[f32; 4]; 3],
    min_z: f32,
    max_z: f32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    u_range: (f32, f32),
    v_range: (f32, f32),
}

impl QuickCull {
    fn new(
        camera_from_world: &Matrix4<f64>,
        camera: &CameraIntrinsics,
        min_depth_m: f64,
        max_depth_m: f64,
        margin_ratio: f64,
    ) -> Self {
        let margin_x = (margin_ratio + 0.01) * camera.width as f64;
        let margin_y = (margin_ratio + 0.01) * camera.height as f64;
        Self {
            rows: std::array::from_fn(|r| {
                std::array::from_fn(|c| camera_from_world[(r, c)] as f32)
            }),
            min_z: (min_depth_m - 0.01) as f32,
            max_z: (max_depth_m + 0.01) as f32,
            fx: camera.fx as f32,
            fy: camera.fy as f32,
            cx: camera.cx as f32,
            cy: camera.cy as f32,
            u_range: (-margin_x as f32, (camera.width as f64 + margin_x) as f32),
            v_range: (-margin_y as f32, (camera.height as f64 + margin_y) as f32),
        }
    }

    #[inline]
    fn passes(&self, point: &Point3<f32>) -> bool {
        let row = |r: &[f32; 4]| r[0] * point.x + r[1] * point.y + r[2] * point.z + r[3];
        let z = row(&self.rows[2]);
        if !(z >= self.min_z && z <= self.max_z) {
            return false;
        }
        let u = self.fx * row(&self.rows[0]) / z + self.cx;
        let v = self.fy * row(&self.rows[1]) / z + self.cy;
        u >= self.u_range.0 && u <= self.u_range.1 && v >= self.v_range.0 && v <= self.v_range.1
    }
}

/// GlobalMap の全セルを画像に描き、マスごとの最小深度を記録したもの。
/// セルは見かけの大きさ（`occluder_splat_scale` 倍）の正方形として描くので、
/// 近くの物ほど広く隠す。
struct DepthBuffer {
    cell_px: f64,
    cols: usize,
    rows: usize,
    depth: Vec<f32>,
    drawn_cells: usize,
    splat_scale: f64,
    voxel_size: f64,
    margin_m: f64,
    margin_ratio: f64,
}

impl DepthBuffer {
    fn build(
        map: &VoxelMap,
        camera_from_world: &Matrix4<f64>,
        camera: &CameraIntrinsics,
        voxel_size: f64,
        config: &KeyframePasteConfig,
    ) -> Self {
        let width = camera.width as f64;
        let height = camera.height as f64;
        let cell_px = (config.depth_buffer_cell_px * width / 960.0).max(1.0);
        let cols = ((width / cell_px).ceil() as usize).max(1);
        let rows = ((height / cell_px).ceil() as usize).max(1);
        let max_radius = 0.25 * width.max(height);

        // 投影は並列、描画（最小値の更新）は順番に行う（順序に依らず同じ結果になる）。
        let cull = QuickCull::new(
            camera_from_world,
            camera,
            config.min_depth_m,
            config.max_depth_m,
            0.3,
        );
        let splats: Vec<[f32; 4]> = map
            .par_iter()
            .filter_map(|(_, cell)| {
                if !cull.passes(&cell.mean) {
                    return None;
                }
                let pc = camera_from_world.transform_point(&cell.mean.cast::<f64>());
                if pc.z.is_nan()
                    || pc.z < config.min_depth_m
                    || pc.z > config.max_depth_m
                    || !roughly_in_view(&pc, camera, 0.3)
                {
                    return None;
                }
                let radius = (0.5 * config.occluder_splat_scale * camera.fx * voxel_size / pc.z)
                    .min(max_radius);
                let (pixel, _) = project_camera_point_subpixel(&pc, camera)?;
                if pixel.x + radius < 0.0
                    || pixel.y + radius < 0.0
                    || pixel.x - radius >= width
                    || pixel.y - radius >= height
                {
                    return None;
                }
                Some([pixel.x as f32, pixel.y as f32, pc.z as f32, radius as f32])
            })
            .collect();

        let mut depth = vec![f32::INFINITY; cols * rows];
        let to_cell =
            |value: f64, count: usize| ((value / cell_px).floor().max(0.0) as usize).min(count - 1);
        for &[u, v, z, radius] in &splats {
            let (u, v, radius) = (u as f64, v as f64, radius as f64);
            let (x0, x1) = (to_cell(u - radius, cols), to_cell(u + radius, cols));
            let (y0, y1) = (to_cell(v - radius, rows), to_cell(v + radius, rows));
            for y in y0..=y1 {
                for value in &mut depth[y * cols + x0..=y * cols + x1] {
                    if z < *value {
                        *value = z;
                    }
                }
            }
        }

        Self {
            cell_px,
            cols,
            rows,
            depth,
            drawn_cells: splats.len(),
            splat_scale: config.occluder_splat_scale,
            voxel_size,
            margin_m: config.occlusion_margin_m,
            margin_ratio: config.occlusion_margin_ratio,
        }
    }

    /// 深度 `depth`、視線との cos が `view_cosine` の面上の点を、`pixel` の周り
    /// ±`support_px` で読むとき、その範囲に手前の物が写っているか。
    ///
    /// 範囲内のマスには、同じ面の少し横の点も描かれている。斜めの面ではそれが手前に来るので、
    /// 横にずれうる距離（範囲＋マスの幅＋セルの見かけの大きさ）の分の深度差（面の傾き分）を許す。
    fn is_occluded(
        &self,
        pixel: &Vector2<f64>,
        support_px: f64,
        depth: f64,
        view_cosine: f64,
        fx: f64,
    ) -> bool {
        if !pixel.iter().all(|v| v.is_finite()) || !support_px.is_finite() {
            return false;
        }
        let to_cell = |value: f64, count: usize| -> Option<usize> {
            let index = (value / self.cell_px).floor();
            (index >= 0.0).then(|| (index as usize).min(count - 1))
        };
        let (Some(x0), Some(y0)) = (
            to_cell((pixel.x - support_px).max(0.0), self.cols),
            to_cell((pixel.y - support_px).max(0.0), self.rows),
        ) else {
            return false;
        };
        let (Some(x1), Some(y1)) = (
            to_cell(pixel.x + support_px, self.cols),
            to_cell(pixel.y + support_px, self.rows),
        ) else {
            return false;
        };
        let near = (y0..=y1)
            .flat_map(|y| self.depth[y * self.cols + x0..=y * self.cols + x1].iter())
            .fold(f32::INFINITY, |nearest, &value| nearest.min(value)) as f64;
        if !near.is_finite() {
            return false;
        }
        let splat_radius_px = 0.5 * self.splat_scale * fx * self.voxel_size / depth;
        let lateral_m =
            std::f64::consts::SQRT_2 * (support_px + self.cell_px + splat_radius_px) * depth / fx;
        let slope = (1.0 - view_cosine * view_cosine).max(0.0).sqrt() / view_cosine;
        depth > near + self.margin_m + self.margin_ratio * depth + lateral_m * slope
    }
}

/// サブピクセル位置のバイリニア補間（RGB）。4 近傍が画像内にない位置は None。
fn sample_rgb_bilinear(image: &RgbImage, pixel: &Vector2<f64>) -> Option<[f64; 3]> {
    let (width, height) = (image.width() as usize, image.height() as usize);
    if width < 2
        || height < 2
        || !pixel.iter().all(|v| v.is_finite())
        || pixel.x < 0.0
        || pixel.y < 0.0
        || pixel.x >= (width - 1) as f64
        || pixel.y >= (height - 1) as f64
    {
        return None;
    }
    let x = pixel.x.floor() as u32;
    let y = pixel.y.floor() as u32;
    let dx = pixel.x - x as f64;
    let dy = pixel.y - y as f64;
    let a = image.get_pixel(x, y).0;
    let b = image.get_pixel(x + 1, y).0;
    let c = image.get_pixel(x, y + 1).0;
    let d = image.get_pixel(x + 1, y + 1).0;
    Some(std::array::from_fn(|k| {
        (1.0 - dy) * ((1.0 - dx) * a[k] as f64 + dx * b[k] as f64)
            + dy * ((1.0 - dx) * c[k] as f64 + dx * d[k] as f64)
    }))
}

/// 中心 `center`、半径 `radius_px` の正方形（セルの見かけの大きさ）の平均色。
/// 正方形を n×n（n は見かけの幅の画素数、最大 `max_samples`）に分け、各中心をバイリニアで読む。
fn sample_footprint(
    image: &RgbImage,
    center: &Vector2<f64>,
    radius_px: f64,
    max_samples: usize,
) -> Option<[u8; 3]> {
    if !radius_px.is_finite() || radius_px < 0.0 {
        return None;
    }
    let count = ((2.0 * radius_px).round() as usize).clamp(1, max_samples.max(1));
    let step = 2.0 * radius_px / count as f64;
    let mut sum = [0.0f64; 3];
    let mut used = 0usize;
    for j in 0..count {
        for i in 0..count {
            let offset = Vector2::new(
                -radius_px + (i as f64 + 0.5) * step,
                -radius_px + (j as f64 + 0.5) * step,
            );
            if let Some(rgb) = sample_rgb_bilinear(image, &(center + offset)) {
                for (total, value) in sum.iter_mut().zip(rgb) {
                    *total += value;
                }
                used += 1;
            }
        }
    }
    (used > 0).then(|| sum.map(|total| (total / used as f64).round().clamp(0.0, 255.0) as u8))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::unproject_pixel_at_depth;
    use crate::voxel_map::{VoxelCell, voxel_key};
    use image::Rgb;
    use nalgebra::{Rotation3, Translation3};

    const VOXEL: f32 = 0.025;
    const RED: [u8; 3] = [200, 30, 30];
    const BLUE: [u8; 3] = [30, 30, 200];
    const GREEN: [u8; 3] = [30, 200, 30];
    const YELLOW: [u8; 3] = [220, 220, 40];

    fn camera() -> CameraIntrinsics {
        CameraIntrinsics {
            width: 320,
            height: 240,
            fx: 210.0,
            fy: 212.0,
            cx: 159.3,
            cy: 119.7,
            distortion: [0.0; 5],
        }
    }

    /// 位置 `position` から、+z 方向を見ているカメラ（`yaw` で y 軸回りに回す）。
    fn camera_from_world_at(position: Vector3<f64>, yaw: f64) -> Matrix4<f64> {
        let world_from_camera = Translation3::from(position).to_homogeneous()
            * Rotation3::from_axis_angle(&Vector3::y_axis(), yaw).to_homogeneous();
        world_from_camera.try_inverse().unwrap()
    }

    fn keyframe(
        frame_index: u64,
        camera_from_world: Matrix4<f64>,
        camera: CameraIntrinsics,
        rgb: RgbImage,
    ) -> KeyframeImage {
        KeyframeImage {
            frame_index,
            camera_from_world,
            camera,
            rgb,
        }
    }

    fn uniform(camera: &CameraIntrinsics, color: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(camera.width, camera.height, Rgb(color))
    }

    /// 平面 normal·X = offset に `color_at(ワールド点)` の模様を貼り、`camera_from_world` で撮った画像。
    /// 歪み付きのカメラでも、各画素の光線（歪み補正済み）で平面と交差させる。
    fn render(
        camera: &CameraIntrinsics,
        camera_from_world: &Matrix4<f64>,
        normal: &Vector3<f64>,
        offset: f64,
        color_at: &dyn Fn(&Point3<f64>) -> [u8; 3],
    ) -> RgbImage {
        let world_from_camera = camera_from_world.try_inverse().unwrap();
        let origin: Vector3<f64> = world_from_camera.fixed_view::<3, 1>(0, 3).into_owned();
        let rotation = world_from_camera.fixed_view::<3, 3>(0, 0).into_owned();
        RgbImage::from_fn(camera.width, camera.height, |u, v| {
            let ray = unproject_pixel_at_depth(&Vector2::new(u as f64, v as f64), 1.0, camera)
                .unwrap()
                .coords;
            let direction = rotation * ray;
            let t = (offset - normal.dot(&origin)) / normal.dot(&direction);
            Rgb(color_at(&Point3::from(origin + direction * t)))
        })
    }

    /// 平面上の格子点（origin + u·e1 + v·e2、2.5 cm 間隔）を、`status` のセルにした GlobalMap。
    fn plane_cells(
        map: &mut VoxelMap,
        origin: &Point3<f64>,
        e1: &Vector3<f64>,
        e2: &Vector3<f64>,
        (u_min, u_max, v_min, v_max): (f64, f64, f64, f64),
        status: SurfaceStatus,
    ) {
        let normal = e1.cross(e2).normalize();
        let spacing = VOXEL as f64;
        let mut u = u_min + 0.5 * spacing;
        while u < u_max {
            let mut v = v_min + 0.5 * spacing;
            while v < v_max {
                let point = (origin + e1 * u + e2 * v).cast::<f32>();
                let key = voxel_key(&point, VOXEL);
                let mut cell = VoxelCell::from_key(&key, VOXEL, point, 0);
                cell.sample_count = 3;
                cell.observed_frames = 3;
                cell.surface_status = status;
                cell.surface_plane = (status == SurfaceStatus::Planar).then(|| SurfacePlane {
                    normal: normal.cast(),
                    d: -normal.dot(&point.cast::<f64>().coords) as f32,
                    rmse_m: 0.003,
                    neighbor_count: 25,
                    inlier_count: 25,
                });
                map.insert(key, cell);
                v += spacing;
            }
            u += spacing;
        }
    }

    /// 正対した壁 z = `z`（x: ±half_x, y: ±half_y）。
    fn wall(map: &mut VoxelMap, z: f64, half_x: f64, half_y: f64) {
        plane_cells(
            map,
            &Point3::new(0.0, 0.0, z),
            &Vector3::x(),
            &Vector3::y(),
            (-half_x, half_x, -half_y, half_y),
            SurfaceStatus::Planar,
        );
    }

    /// GlobalMap の Planar セルを、平面判定の記録としてコピーへ渡す。
    fn sync_all(paster: &mut KeyframePaster, map: &VoxelMap) {
        let records: Vec<SurfaceClassificationRecord> = map
            .iter()
            .filter(|(_, cell)| cell.surface_status != SurfaceStatus::Unknown)
            .map(|(key, cell)| SurfaceClassificationRecord {
                key: *key,
                mean: cell.mean,
                status: cell.surface_status,
                plane: cell.surface_plane,
            })
            .collect();
        paster.sync_planes(&records);
    }

    fn pixel_of(camera_from_world: &Matrix4<f64>, point: &Point3<f32>) -> Option<Vector2<f64>> {
        let pc = camera_from_world.transform_point(&point.cast::<f64>());
        project_camera_point_subpixel(&pc, &camera()).map(|(pixel, _)| pixel)
    }

    #[test]
    fn bilinear_and_footprint_sampling() {
        let image = RgbImage::from_fn(4, 4, |u, v| Rgb([(u * 40) as u8, (v * 40) as u8, 7]));
        let rgb = sample_rgb_bilinear(&image, &Vector2::new(1.25, 2.5)).unwrap();
        assert!((rgb[0] - 50.0).abs() < 1e-9 && (rgb[1] - 100.0).abs() < 1e-9);
        assert!(sample_rgb_bilinear(&image, &Vector2::new(3.0, 1.0)).is_none());

        // 1 画素ごとの白黒の市松模様: 1 点だけ読むと白か黒、見かけの大きさで平均すると灰色
        let checker = RgbImage::from_fn(40, 40, |u, v| {
            let value = if (u + v) % 2 == 0 { 0 } else { 255 };
            Rgb([value, value, value])
        });
        let center = Vector2::new(20.0, 20.0);
        let single = sample_footprint(&checker, &center, 0.2, 5).unwrap();
        assert!(single[0] == 0 || single[0] == 255);
        let averaged = sample_footprint(&checker, &center, 2.0, 5).unwrap();
        assert!((averaged[0] as i32 - 128).abs() <= 30, "{averaged:?}");
    }

    #[test]
    fn pastes_keyframe_image_onto_planar_copy_cells_inside_the_central_crop() {
        let camera = camera();
        let mut map = VoxelMap::default();
        wall(&mut map, 5.0, 4.0, 3.0);
        let mut paster = KeyframePaster::new(KeyframePasteConfig::default());
        sync_all(&mut paster, &map);
        assert_eq!(paster.planar_cell_count(), map.len());

        let pose = camera_from_world_at(Vector3::zeros(), 0.0);
        let image = render(&camera, &pose, &Vector3::z(), 5.0, &|p| {
            if p.x < 0.0 { RED } else { BLUE }
        });
        let stats = paster.paste(&keyframe(7, pose, camera.clone(), image), &map, VOXEL);
        assert_eq!(stats.keyframe_frame_index, 7);
        assert_eq!(stats.occluded, 0, "{stats:?}");
        assert_eq!(stats.too_oblique, 0, "{stats:?}");
        assert_eq!(stats.pasted_new, stats.in_view, "{stats:?}");
        assert!(stats.pasted_new > 10_000, "{stats:?}");

        let crop = Crop::new(&camera, 0.75);
        for (key, cell) in &map {
            let pixel = pixel_of(&pose, &cell.mean).unwrap();
            let pasted = paster.color(key);
            assert_eq!(pasted.is_some(), crop.contains(&pixel), "{pixel:?}");
            if let Some(color) = pasted {
                assert_eq!(color.keyframe_frame_index, 7);
                let distance = cell.mean.cast::<f64>().coords.norm();
                let expected_score = (5.0 / distance) / (distance * distance);
                assert!((color.score as f64 - expected_score).abs() < 1e-6);
                if cell.mean.x < -0.05 {
                    assert_eq!(color.rgb, RED);
                } else if cell.mean.x > 0.05 {
                    assert_eq!(color.rgb, BLUE);
                }
            }
            // GlobalMap 本体の色は変えない
            assert!(cell.rgb.is_none());
        }
    }

    #[test]
    fn cells_hidden_behind_non_planar_occluders_are_not_pasted() {
        let camera = camera();
        let mut map = VoxelMap::default();
        wall(&mut map, 5.0, 4.0, 3.0);
        // 3 m 先の板（平面判定は NonPlanar = コピーには入らないが、遮蔽物としては使う）
        let (board_x, board_y) = ((-1.0, -0.2), (-0.5, 0.5));
        plane_cells(
            &mut map,
            &Point3::new(0.0, 0.0, 3.0),
            &Vector3::x(),
            &Vector3::y(),
            (board_x.0, board_x.1, board_y.0, board_y.1),
            SurfaceStatus::NonPlanar,
        );
        let mut paster = KeyframePaster::new(KeyframePasteConfig::default());
        sync_all(&mut paster, &map);

        let pose = camera_from_world_at(Vector3::zeros(), 0.0);
        let board_hit = |p: &Point3<f64>| {
            // 視線を 3 m で切った位置が板の範囲内か
            let at_board = p.coords * (3.0 / p.z);
            (board_x.0..board_x.1).contains(&at_board.x)
                && (board_y.0..board_y.1).contains(&at_board.y)
        };
        let image = render(&camera, &pose, &Vector3::z(), 5.0, &|p| {
            if board_hit(p) { GREEN } else { RED }
        });
        let stats = paster.paste(&keyframe(0, pose, camera.clone(), image), &map, VOXEL);
        assert!(stats.occluded > 500, "{stats:?}");

        let board_pixels = |corner_x: f64, corner_y: f64| {
            pixel_of(&pose, &Point3::new(corner_x as f32, corner_y as f32, 3.0)).unwrap()
        };
        let (top_left, bottom_right) = (
            board_pixels(board_x.0, board_y.0),
            board_pixels(board_x.1, board_y.1),
        );
        let mut hidden = 0;
        for (key, cell) in &map {
            if cell.mean.z < 4.0 {
                assert!(
                    paster.color(key).is_none(),
                    "the board itself is not a target"
                );
                continue;
            }
            if let Some(color) = paster.color(key) {
                assert_eq!(color.rgb, RED, "board color leaked onto the wall");
            }
            let pixel = pixel_of(&pose, &cell.mean).unwrap();
            let inside = pixel.x > top_left.x + 1.0
                && pixel.x < bottom_right.x - 1.0
                && pixel.y > top_left.y + 1.0
                && pixel.y < bottom_right.y - 1.0;
            if inside {
                hidden += 1;
                assert!(
                    paster.color(key).is_none(),
                    "wall cell behind the board was pasted"
                );
            }
        }
        assert!(hidden > 500);
        assert_eq!(
            stats.pasted_new + stats.occluded,
            stats.in_view,
            "{stats:?}"
        );
    }

    /// より近い・より正面から見たキーフレームだけが上書きする（セルごとに判断）。
    #[test]
    fn closer_or_more_frontal_views_overwrite_and_worse_views_are_kept() {
        let camera = camera();
        let mut map = VoxelMap::default();
        wall(&mut map, 5.0, 1.0, 0.6);
        let mut paster = KeyframePaster::new(KeyframePasteConfig::default());
        sync_all(&mut paster, &map);
        let paste = |paster: &mut KeyframePaster, index, position: Vector3<f64>, color| {
            let pose = camera_from_world_at(position, 0.0);
            paster.paste(
                &keyframe(index, pose, camera.clone(), uniform(&camera, color)),
                &map,
                VOXEL,
            )
        };

        let first = paste(&mut paster, 1, Vector3::zeros(), RED);
        assert_eq!(first.pasted_new, map.len(), "{first:?}");

        // 1.5 m 近づいた（距離 3.5 m）→ 全部上書き
        let closer = paste(&mut paster, 2, Vector3::new(0.0, 0.0, 1.5), BLUE);
        assert_eq!(closer.overwritten, closer.in_view, "{closer:?}");
        assert!(closer.in_view > map.len() * 9 / 10, "{closer:?}");

        // 遠い（距離 6 m）→ 何も変えない
        let farther = paste(&mut paster, 3, Vector3::new(0.0, 0.0, -1.0), GREEN);
        assert_eq!(farther.overwritten, 0, "{farther:?}");
        assert_eq!(farther.kept_existing, farther.in_view, "{farther:?}");

        // 横に 1.2 m ずれた位置（距離 3.5 m）: 真正面に来たセルだけ上書き、遠く斜めになるセルは残す
        let side = paste(&mut paster, 4, Vector3::new(1.2, 0.0, 1.5), YELLOW);
        assert!(side.overwritten > 0 && side.kept_existing > 0, "{side:?}");
        let color_near = |x: f32| {
            let key = voxel_key(&Point3::new(x, 0.0125, 5.0), VOXEL);
            paster.color(&key).unwrap()
        };
        assert_eq!(color_near(0.9125).rgb, YELLOW);
        assert_eq!(color_near(0.9125).keyframe_frame_index, 4);
        assert_eq!(color_near(-0.4875).rgb, BLUE);
        assert_eq!(color_near(-0.4875).keyframe_frame_index, 2);
    }

    #[test]
    fn tilted_plane_is_not_hidden_by_its_own_cells_and_grazing_views_are_skipped() {
        let camera = camera();
        let pose = camera_from_world_at(Vector3::zeros(), 0.0);
        let tilted = |tilt_deg: f64| {
            let tilt = tilt_deg.to_radians();
            let mut map = VoxelMap::default();
            plane_cells(
                &mut map,
                &Point3::new(0.0, 0.0, 5.0),
                &Vector3::x(),
                &Vector3::new(0.0, tilt.cos(), tilt.sin()),
                (-3.0, 3.0, -2.0, 2.0),
                SurfaceStatus::Planar,
            );
            let mut paster = KeyframePaster::new(KeyframePasteConfig::default());
            sync_all(&mut paster, &map);
            let stats = paster.paste(
                &keyframe(0, pose, camera.clone(), uniform(&camera, RED)),
                &map,
                VOXEL,
            );
            // 貼ったセルは、実際の視線との角度が下限以上のものだけ
            let normal = Vector3::new(0.0, -tilt.sin(), tilt.cos());
            for (key, cell) in &map {
                let point = cell.mean.cast::<f64>();
                let cosine = normal.dot(&point.coords).abs() / point.coords.norm();
                if paster.color(key).is_some() {
                    assert!(cosine >= paster.config.min_view_cosine - 1e-6, "{cosine}");
                }
            }
            stats
        };

        // 60°傾いた面（cos ≈ 0.5）: 自分の隣のセルに隠されない
        let oblique = tilted(60.0);
        assert!(oblique.pasted_new > 2000, "{oblique:?}");
        assert!(oblique.occluded * 100 <= oblique.in_view, "{oblique:?}");

        // 80°（光軸上で cos ≈ 0.17 < 0.25）: ほとんどは斜めすぎるので貼らない
        // （画像の端の方は視線が傾くので、cos が 0.25 を超える部分だけ貼れる）
        let grazing = tilted(80.0);
        assert!(grazing.too_oblique > 10 * grazing.pasted_new, "{grazing:?}");
        assert_eq!(grazing.occluded, 0, "{grazing:?}");
    }

    #[test]
    fn keyframe_n_is_pasted_delay_frames_after_n_plus_one_and_flush_pastes_the_rest() {
        let camera = camera();
        let mut map = VoxelMap::default();
        wall(&mut map, 5.0, 1.0, 0.6);
        let mut paster = KeyframePaster::new(KeyframePasteConfig {
            paste_delay_frames: 5,
            ..KeyframePasteConfig::default()
        });
        sync_all(&mut paster, &map);
        let make = |index, z: f64, color| {
            keyframe(
                index,
                camera_from_world_at(Vector3::new(0.0, 0.0, z), 0.0),
                camera.clone(),
                uniform(&camera, color),
            )
        };
        let probe = voxel_key(&Point3::new(0.0125, 0.0125, 5.0), VOXEL);

        assert_eq!(paster.on_new_keyframe(0, make(0, 0.0, RED)), None);
        assert_eq!(paster.current_keyframe_index(), Some(0));
        for frame in 0..=3 {
            assert!(paster.paste_due(frame, &map, VOXEL).is_empty());
        }
        // フレーム 3 で N+1 (=3) が確定 → N (=0) はフレーム 8 に貼る
        assert_eq!(paster.on_new_keyframe(3, make(3, 1.0, BLUE)), Some((0, 8)));
        assert_eq!(paster.pending_keyframes(), vec![(0, 8)]);
        for frame in 4..8 {
            assert!(paster.paste_due(frame, &map, VOXEL).is_empty());
            assert!(paster.color(&probe).is_none());
        }
        let pasted = paster.paste_due(8, &map, VOXEL);
        assert_eq!(pasted.len(), 1);
        assert_eq!(pasted[0].keyframe_frame_index, 0);
        assert_eq!(paster.color(&probe).unwrap().rgb, RED);
        assert!(paster.pending_keyframes().is_empty());

        assert_eq!(
            paster.on_new_keyframe(10, make(10, 2.0, GREEN)),
            Some((3, 15))
        );
        // 最後: 待っている 3 と、今のキーフレーム 10 を確定順に貼る
        let flushed = paster.flush(&map, VOXEL);
        assert_eq!(
            flushed
                .iter()
                .map(|s| s.keyframe_frame_index)
                .collect::<Vec<_>>(),
            vec![3, 10]
        );
        assert_eq!(paster.color(&probe).unwrap().rgb, GREEN);
        assert_eq!(paster.current_keyframe_index(), None);
        assert!(paster.flush(&map, VOXEL).is_empty());
    }

    /// 実際の GlobalMap（main.rs と同じ設定）で: 再観測でセルが Unknown に戻っても、
    /// コピーは最後の平面を持っているので貼れる。NonPlanar になったセルは対象から外れる。
    #[test]
    fn copy_keeps_the_last_plane_after_global_map_resets_cells_to_unknown() {
        use crate::voxel_map::{LOCALMap, LocalMapConfig, SurfaceFilterConfig};

        let mut global = LOCALMap::new(LocalMapConfig {
            index_voxel_size: VOXEL,
            max_points_per_voxel: 20,
            min_points_per_voxel: 2,
            min_observed_frames_per_voxel: 2,
            max_frames: 50,
            max_distance: 150.0,
        });
        global.set_surface_classification_log(true);
        let mut scan = Vec::new();
        let mut y = -1.5f32;
        while y < 1.5 {
            let mut x = -2.0f32;
            while x < 2.0 {
                scan.push(Point3::new(x, y, 5.0f32));
                x += 0.02;
            }
            y += 0.02;
        }
        for _ in 0..3 {
            global.update_world_map(&scan, &Matrix4::identity());
        }
        let stats = global.classify_all_surface_voxels(&SurfaceFilterConfig {
            neighbor_radius_voxels: 2,
            min_neighbors: 10,
            min_ransac_inliers: 8,
            min_center_observed_frames: 2,
            ransac_iterations: 48,
            ransac_confidence: 0.999,
            ransac_min_iterations: 8,
            ransac_inlier_distance_m: 0.025,
            min_inlier_ratio: 0.5,
            min_planarity: 0.2,
            max_surface_variation: 0.04,
            max_pca_rmse_m: 0.025,
            max_center_distance_m: 0.025,
        });
        assert!(stats.planar > 10_000, "{stats:?}");

        let mut paster = KeyframePaster::new(KeyframePasteConfig::default());
        paster.sync_planes(&global.take_surface_classification_log());
        assert_eq!(paster.planar_cell_count(), stats.planar);

        // 同じ場所をもう一度観測 → GlobalMap では全セルが Unknown に戻る
        global.update_world_map(&scan, &Matrix4::identity());
        paster.sync_planes(&global.take_surface_classification_log());
        let planar_now = global
            .voxel_map
            .values()
            .filter(|cell| cell.surface_status == SurfaceStatus::Planar)
            .count();
        assert_eq!(planar_now, 0);

        let camera = camera();
        let pose = camera_from_world_at(Vector3::zeros(), 0.0);
        let red = keyframe(0, pose, camera.clone(), uniform(&camera, RED));
        let pasted = paster.paste(&red, &global.voxel_map, VOXEL);
        assert!(pasted.pasted_new > 10_000, "{pasted:?}");
        assert_eq!(pasted.occluded, 0, "{pasted:?}");

        // NonPlanar と判定されたセルは対象から外れる（貼った色は残す）
        let key = voxel_key(&Point3::new(0.0125, 0.0125, 5.0), VOXEL);
        assert_eq!(paster.color(&key).unwrap().rgb, RED);
        paster.sync_planes(&[SurfaceClassificationRecord {
            key,
            mean: Point3::new(0.0125, 0.0125, 5.0),
            status: SurfaceStatus::NonPlanar,
            plane: None,
        }]);
        assert_eq!(paster.planar_cell_count(), stats.planar - 1);
        let closer = keyframe(
            1,
            camera_from_world_at(Vector3::new(0.0, 0.0, 2.0), 0.0),
            camera.clone(),
            uniform(&camera, BLUE),
        );
        paster.paste(&closer, &global.voxel_map, VOXEL);
        assert_eq!(paster.color(&key).unwrap().rgb, RED);
        let neighbor = voxel_key(&Point3::new(0.0375, 0.0125, 5.0), VOXEL);
        assert_eq!(paster.color(&neighbor).unwrap().rgb, BLUE);
    }

    /// 実機と同じ歪み係数のカメラで、セルの位置の模様の色が正しく貼られる。
    #[test]
    fn distorted_camera_pastes_the_color_at_each_cell_position() {
        let scale = 0.25;
        let camera = CameraIntrinsics {
            width: 480,
            height: 300,
            fx: 1175.01946 * scale,
            fy: 1177.56737 * scale,
            cx: 939.43935 * scale,
            cy: 627.06837 * scale,
            distortion: [-0.136349, 0.028462, -0.002188, 0.001479, 0.0],
        };
        let mut map = VoxelMap::default();
        wall(&mut map, 5.0, 4.0, 3.0);
        let mut paster = KeyframePaster::new(KeyframePasteConfig::default());
        sync_all(&mut paster, &map);

        let texture = |p: &Point3<f64>| {
            let value = 128.0 + 70.0 * (1.3 * p.x).sin() * (1.1 * p.y).cos();
            [value.round() as u8, (255.0 - value).round() as u8, 90]
        };
        let pose = camera_from_world_at(Vector3::new(0.3, -0.2, 0.0), 0.08);
        let image = render(&camera, &pose, &Vector3::z(), 5.0, &texture);
        let stats = paster.paste(&keyframe(0, pose, camera.clone(), image), &map, VOXEL);
        assert!(stats.pasted_new > 10_000, "{stats:?}");

        let mut checked = 0;
        for (key, cell) in &map {
            let Some(color) = paster.color(key) else {
                continue;
            };
            let expected = texture(&cell.mean.cast::<f64>());
            for channel in 0..3 {
                assert!(
                    (color.rgb[channel] as i32 - expected[channel] as i32).abs() <= 3,
                    "cell {:?}: {:?} vs {:?}",
                    cell.mean,
                    color.rgb,
                    expected
                );
            }
            checked += 1;
        }
        assert_eq!(checked, stats.pasted_new);
    }

    /// main.rs と同じ順番の走行の再現: 模様のある壁に沿って動くカメラ（= LiDAR）で、
    /// 視覚アライメント → 新しいキーフレームの登録 → GlobalMap 更新 → 遅延平面判定 →
    /// コピーへの反映 → 予定のキーフレームを貼る、を毎フレーム行い、最後に残りを貼る。
    /// 壁は毎フレーム観測されるので GlobalMap ではセルが Unknown に戻り続けるが、
    /// 予定どおりのフレームで貼られ、視野の範囲はほぼすべて正しい色になること。
    #[test]
    #[cfg_attr(debug_assertions, ignore = "slow in debug builds; run with --release")]
    fn simulated_run_pastes_each_keyframe_on_schedule_with_correct_colors() {
        use crate::eskf::{Eskf, EskfConfig, ImuModel, ImuNoise, InitialStd, NavState};
        use crate::visual_alignment::{VisualAligner, VisualAlignmentConfig, VisualFrame};
        use crate::voxel_map::{LOCALMap, LocalMapConfig, SurfaceFilterConfig};
        use nalgebra::UnitQuaternion;

        const MAP_VOXEL: f32 = 0.05;
        const STEP_M: f64 = 0.6;
        const FRAMES: u64 = 26;
        let camera = camera();
        let texture = |p: &Point3<f64>| {
            let value = (128.0
                + 41.0 * (7.1 * p.x + 2.3 * p.y).sin()
                + 34.0 * (3.7 * p.x - 8.2 * p.y).cos()
                + 22.0 * (13.3 * p.x + 11.4 * p.y).sin())
            .round()
            .clamp(0.0, 255.0) as u8;
            [value, value, value]
        };
        let mut global = LOCALMap::new(LocalMapConfig {
            index_voxel_size: MAP_VOXEL,
            max_points_per_voxel: 20,
            min_points_per_voxel: 2,
            min_observed_frames_per_voxel: 2,
            max_frames: 50,
            max_distance: 150.0,
        });
        global.set_surface_classification_log(true);
        let surface_config = SurfaceFilterConfig {
            neighbor_radius_voxels: 2,
            min_neighbors: 10,
            min_ransac_inliers: 8,
            min_center_observed_frames: 2,
            ransac_iterations: 48,
            ransac_confidence: 0.999,
            ransac_min_iterations: 8,
            ransac_inlier_distance_m: 0.025,
            min_inlier_ratio: 0.5,
            min_planarity: 0.2,
            max_surface_variation: 0.04,
            max_pca_rmse_m: 0.025,
            max_center_distance_m: 0.025,
        };
        let mut filter = Eskf::new(
            NavState::default(),
            EskfConfig {
                imu_noise: ImuNoise {
                    gyro_noise_density: 0.002,
                    accel_noise_density: 0.02,
                    gyro_bias_random_walk: 0.00005,
                    accel_bias_random_walk: 0.0005,
                },
                initial_std: InitialStd {
                    attitude_rad: 0.03,
                    position_m: 0.1,
                    velocity_mps: 0.1,
                    gyro_bias_radps: 0.01,
                    accel_bias_mps2: 0.02,
                },
                gravity_mps2: 9.80665,
                lidar_point_sigma_m: 0.05,
            },
            ImuModel::new(UnitQuaternion::identity(), 1.0),
            Vector3::zeros(),
        );
        let mut aligner = VisualAligner::new(VisualAlignmentConfig {
            grid_size_px: 24,
            min_shi_tomasi_score: 1.0,
            ..VisualAlignmentConfig::default()
        });
        let mut paster = KeyframePaster::new(KeyframePasteConfig::default());
        let mut created = Vec::new();
        let mut pasted_at = Vec::new();

        for i in 0..FRAMES {
            let position = Vector3::new(i as f64 * STEP_M, 0.0, 0.0);
            let pose = camera_from_world_at(position, 0.0);
            let image = render(&camera, &pose, &Vector3::z(), 5.0, &texture);
            // LiDAR 点（LiDAR = カメラ座標）: 壁の上を 7 cm 間隔、フレームごとに半分ずらす
            let jitter = if i % 2 == 0 { 0.0 } else { 0.035 };
            let mut scan = Vec::new();
            let mut y = -2.6 + jitter;
            while y < 2.6 {
                let mut x = -3.8 + jitter;
                while x < 3.8 {
                    scan.push(Point3::new(x as f32, y as f32, 5.0));
                    x += 0.07;
                }
                y += 0.07;
            }
            filter.state = NavState {
                pos: position,
                ..NavState::default()
            };

            // 1. 視覚アライメント（GlobalMap はこのフレームの点を入れる前）
            aligner.process_frame(
                &mut filter,
                &VisualFrame {
                    imu: &[],
                    frame_start_time: i as f64 * 0.1,
                    image_time: i as f64 * 0.1,
                    frame_index: i,
                    rgb: &image,
                    camera: &camera,
                    camera_from_lidar: &Matrix4::identity(),
                    frame_points: &scan,
                    planar_map: &global.voxel_map,
                },
            );
            // 2. 新しいキーフレームの登録（main.rs と同じ条件）
            if let Some(keyframe) = aligner.keyframe_pose()
                && keyframe.frame_index == i
                && paster.current_keyframe_index() != Some(keyframe.frame_index)
            {
                created.push(i);
                paster.on_new_keyframe(
                    i,
                    KeyframeImage {
                        frame_index: i,
                        camera_from_world: keyframe.camera_from_world,
                        camera: camera.clone(),
                        rgb: image.clone(),
                    },
                );
            }
            // 3. GlobalMap 更新と遅延平面判定
            let world_from_lidar = Translation3::from(position).to_homogeneous();
            global.update_world_map(&scan, &world_from_lidar);
            global.classify_delayed_surface_voxels(5, &surface_config);
            // 4. コピーへの反映と、予定のキーフレームの貼り付け
            paster.sync_planes(&global.take_surface_classification_log());
            for stats in paster.paste_due(i, &global.voxel_map, MAP_VOXEL) {
                assert!(stats.pasted_new + stats.overwritten > 0, "{stats:?}");
                assert_eq!(stats.occluded, 0, "{stats:?}");
                pasted_at.push((stats.keyframe_frame_index, i));
            }
        }
        // 走行の最後（main.rs と同じ順番）: 全セルを再判定し、その結果をコピーへ反映してから
        // 残りのキーフレームを貼る
        paster.sync_planes(&global.take_surface_classification_log());
        global.set_surface_classification_log(false);
        global.classify_all_surface_voxels(&surface_config);
        let planar_before_final_sync = paster.planar_cell_count();
        paster.sync_from_map(&global.voxel_map);
        assert!(paster.planar_cell_count() > planar_before_final_sync);
        let flushed: Vec<u64> = paster
            .flush(&global.voxel_map, MAP_VOXEL)
            .iter()
            .map(|stats| stats.keyframe_frame_index)
            .collect();

        // キーフレーム N は「N+1 の確定 + 5 フレーム」に貼られ、残りは最後に貼られる
        assert!(created.len() >= 4, "keyframes created at {created:?}");
        for (n, pair) in created.windows(2).enumerate() {
            let due = pair[1] + 5;
            if due < FRAMES {
                assert_eq!(pasted_at[n], (pair[0], due), "{pasted_at:?} {created:?}");
            } else {
                assert!(flushed.contains(&pair[0]), "{flushed:?}");
            }
        }
        assert_eq!(flushed.last(), created.last());

        // 最終判定で Planar のセルのうち、キーフレームの画像の中央部に入った範囲はほぼ全部貼られ、
        // 色は壁の模様と一致する
        let first_x = created[0] as f64 * STEP_M;
        let last_x = *created.last().unwrap() as f64 * STEP_M;
        let (mut covered, mut inside, mut errors) = (0usize, 0usize, Vec::new());
        for (key, cell) in &global.voxel_map {
            if cell.surface_status != SurfaceStatus::Planar {
                continue;
            }
            let (x, y) = (cell.mean.x as f64, cell.mean.y as f64);
            if x > first_x - 2.5 && x < last_x + 2.5 && y.abs() < 1.9 {
                inside += 1;
                covered += usize::from(paster.color(key).is_some());
            }
            if let Some(color) = paster.color(key) {
                let expected = texture(&cell.mean.cast::<f64>())[0] as f64;
                errors.push((color.rgb[0] as f64 - expected).abs());
            }
        }
        assert!(inside > 5000, "{inside}");
        assert!(
            covered as f64 > 0.97 * inside as f64,
            "covered {covered} / {inside}"
        );
        errors.sort_by(f64::total_cmp);
        let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
        let p95 = errors[errors.len() * 95 / 100];
        println!(
            "keyframes {created:?}, pasted at {pasted_at:?}, flushed {flushed:?}; \
             coverage {covered}/{inside}, color error mean {mean_error:.2} p95 {p95:.1}"
        );
        assert!(mean_error < 3.0 && p95 < 8.0, "mean {mean_error} p95 {p95}");
    }

    /// 実行時間の目安（`cargo test --release -- --ignored --nocapture`）。
    #[test]
    #[ignore]
    fn bench_paste_on_a_large_map() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Debug)
            .try_init();
        let camera = CameraIntrinsics {
            width: 960,
            height: 600,
            fx: 587.5,
            fy: 588.8,
            cx: 469.7,
            cy: 313.5,
            distortion: [-0.136349, 0.028462, -0.002188, 0.001479, 0.0],
        };
        let mut map = VoxelMap::default();
        // 視野内の壁（20 m × 6 m）＋視野外の広い地面（マップ全体の大半）
        plane_cells(
            &mut map,
            &Point3::new(0.0, 0.0, 8.0),
            &Vector3::x(),
            &Vector3::y(),
            (-10.0, 10.0, -3.0, 3.0),
            SurfaceStatus::Planar,
        );
        plane_cells(
            &mut map,
            &Point3::new(0.0, 2.0, 0.0),
            &Vector3::x(),
            &Vector3::z(),
            (-20.0, 20.0, -20.0, -1.0),
            SurfaceStatus::NonPlanar,
        );
        let mut paster = KeyframePaster::new(KeyframePasteConfig::default());
        sync_all(&mut paster, &map);
        let pose = camera_from_world_at(Vector3::zeros(), 0.0);
        let image = RgbImage::from_fn(960, 600, |u, v| Rgb([(u % 256) as u8, (v % 256) as u8, 77]));
        let frame = keyframe(0, pose, camera, image);
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let stats = paster.paste(&frame, &map, VOXEL);
            println!(
                "map cells={} planar copy={} paste {:.1} ms: {stats:?}",
                map.len(),
                paster.planar_cell_count(),
                start.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}
