//! キーフレーム画像の貼り付け（走行中は画像と姿勢を貯め、最後にまとめて貼る）。
//!
//! 走行中
//! - 視覚アライメントのキーフレーム N+1 が確定したら、1 つ前の N を貼り付け待ちにし、
//!   「N+1 の確定から `paste_delay_frames` 後」に、N の画像とカメラ姿勢を
//!   貼る画像としてメモリ上に確定する（この時点ではまだ貼らない）。
//! - GlobalMap の数フレームごとの平面判定（遅延判定）の結果を、貼り付け用のコピーへ
//!   反映し続ける。GlobalMap は新しい観測が入ったセルの判定をいったん Unknown に戻すが、
//!   コピーは最後の判定（Planar ならその平面）を持ち続ける。
//!
//! 走行の最後（`finish`。最終の全セル平面判定の前に呼ぶ）
//! - 確定したキーフレームを確定順に、コピー（遅延判定で Planar のセル。位置は最終の
//!   GlobalMap のセル代表点）へ直接貼る。GlobalMap 本体は変更しない。
//! - 重なり: 既に貼られたセルは更新しない（先に貼ったキーフレームを優先する）。
//! - 遮蔽: 平面フィルタをかける前の GlobalMap の全セルを、キーフレームの画像に投影した
//!   深度バッファで判定する。
//! - 色: セルの見かけの大きさの範囲をバイリニアで平均する（2.5 cm セルが数画素に写る
//!   近距離で、1 画素だけ拾ってギザギザになるのを防ぐ）。
//! - 速度: 全セルを空間ブロックにまとめておき、キーフレームごとに視野外のブロックを
//!   まとめて飛ばす（キーフレーム数 × マップ全体を毎回なめない）。

use std::collections::VecDeque;
use std::ops::Range;
use std::time::Instant;

use image::RgbImage;
use nalgebra::{Matrix4, Point3, Vector2, Vector3};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::camera::{CameraIntrinsics, project_camera_point_subpixel};
use crate::color_map::sample_rgb_bilinear;
use crate::voxel_map::{
    SurfaceClassificationRecord, SurfacePlane, SurfaceStatus, VoxelKey, VoxelMap,
};

#[derive(Debug, Clone)]
pub struct KeyframePasteConfig {
    /// N+1 の確定から、N を貼る画像として確定するまで待つフレーム数
    /// （GlobalMap の平面判定の遅れに合わせる）。
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
    /// 視野外をまとめて飛ばすための空間ブロックの大きさ [m]（結果には影響しない）。
    pub block_size_m: f32,
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
            block_size_m: 4.0,
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
            && positive(self.block_size_m as f64)
    }
}

/// 貼る画像 1 枚分（視覚アライメントのキーフレームと同じ画像・姿勢）。
#[derive(Clone)]
pub struct KeyframeImage {
    pub frame_index: u64,
    /// 画像を撮った時刻のカメラ姿勢（キーフレーム作成時に確定した値）。
    pub camera_from_world: Matrix4<f64>,
    pub camera: CameraIntrinsics,
    pub rgb: RgbImage,
}

/// 色が貼られたセル 1 つ（位置は最終の GlobalMap のセル代表点）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PastedPoint {
    pub position: Point3<f32>,
    pub rgb: [u8; 3],
    pub keyframe_frame_index: u64,
}

/// キーフレーム 1 枚を貼ったときの内訳。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PasteStats {
    pub keyframe_frame_index: u64,
    /// 深度バッファに描いた GlobalMap のセル数（平面フィルタ前の全セルのうち視野内）。
    pub occluder_cells: usize,
    /// 深度範囲内で、画像の中央部（`image_keep_ratio`）に写った対象セル数。
    pub in_view: usize,
    /// まだ色がなかったセルに貼った数。
    pub pasted_new: usize,
    /// 既に先のキーフレームで貼られていたので、そのままにした数。
    pub already_pasted: usize,
    pub occluded: usize,
    pub too_oblique: usize,
    pub sample_failed: usize,
    pub elapsed_ms: f64,
}

/// `finish` の結果。
#[derive(Debug, Clone, Default)]
pub struct PasteSummary {
    pub points: Vec<PastedPoint>,
    /// 貼り付け対象（コピーの Planar セルのうち、出力条件を満たすもの）の数。
    pub target_cells: usize,
    /// 遮蔽判定に使った GlobalMap の全セル数。
    pub occluder_cells: usize,
    /// 貼った順のキーフレームごとの内訳。
    pub keyframes: Vec<PasteStats>,
    /// 走行の終了時にまだ確定していなかった（貼り付け待ちと、最後のキーフレーム）ので、
    /// 最後に加えたキーフレームの数。
    pub added_at_end: usize,
    pub elapsed_ms: f64,
}

/// 遅延判定で Planar と確定した平面（単位法線）。
#[derive(Debug, Clone, Copy)]
struct PlaneRecord {
    normal: Vector3<f32>,
    d: f32,
}

impl PlaneRecord {
    fn from_surface(plane: &SurfacePlane) -> Option<Self> {
        let norm = plane.normal.norm();
        (norm.is_finite() && norm > 1e-6 && plane.d.is_finite()).then(|| Self {
            normal: plane.normal / norm,
            d: plane.d / norm,
        })
    }
}

struct PendingKeyframe {
    due_frame: u64,
    keyframe: KeyframeImage,
}

pub struct KeyframePaster {
    pub config: KeyframePasteConfig,
    /// 貼り付け用のコピー: 遅延判定の最新結果が Planar のセルと、その平面。
    planes: FxHashMap<VoxelKey, PlaneRecord>,
    /// 今のキーフレーム（次のキーフレームが確定したら貼り付け待ちへ移る）。
    current: Option<KeyframeImage>,
    pending: VecDeque<PendingKeyframe>,
    /// 貼る画像として確定したキーフレーム（確定順）。
    committed: Vec<KeyframeImage>,
}

impl KeyframePaster {
    pub fn new(config: KeyframePasteConfig) -> Self {
        Self {
            config,
            planes: FxHashMap::default(),
            current: None,
            pending: VecDeque::new(),
            committed: Vec::new(),
        }
    }

    /// GlobalMap の平面判定の結果をコピーへ反映する（毎フレーム、遅延判定の後に呼ぶ）。
    /// Planar: 平面を更新。NonPlanar: コピーから外す。Unknown: 何もしない。
    pub fn sync_planes(&mut self, records: &[SurfaceClassificationRecord]) {
        for record in records {
            match record.status {
                SurfaceStatus::Planar => {
                    match record.plane.as_ref().and_then(PlaneRecord::from_surface) {
                        Some(plane) => {
                            self.planes.insert(record.key, plane);
                        }
                        None => {
                            self.planes.remove(&record.key);
                        }
                    }
                }
                SurfaceStatus::NonPlanar => {
                    self.planes.remove(&record.key);
                }
                SurfaceStatus::Unknown => {}
            }
        }
    }

    /// 新しいキーフレーム N+1 が確定した。1 つ前のキーフレーム N を
    /// `current_frame + paste_delay_frames` に確定する予約をし、(N のフレーム番号, 予定フレーム) を返す。
    pub fn on_new_keyframe(
        &mut self,
        current_frame: u64,
        keyframe: KeyframeImage,
    ) -> Option<(u64, u64)> {
        let previous = self.current.replace(keyframe)?;
        let due_frame = current_frame.saturating_add(self.config.paste_delay_frames);
        let index = previous.frame_index;
        self.pending.push_back(PendingKeyframe {
            due_frame,
            keyframe: previous,
        });
        Some((index, due_frame))
    }

    /// 予定フレームに達したキーフレームを、貼る画像として確定する（まだ貼らない）。
    /// 確定したキーフレームのフレーム番号を返す。
    pub fn commit_due(&mut self, current_frame: u64) -> Vec<u64> {
        let mut committed = Vec::new();
        while self
            .pending
            .front()
            .is_some_and(|pending| pending.due_frame <= current_frame)
        {
            if let Some(pending) = self.pending.pop_front() {
                committed.push(pending.keyframe.frame_index);
                self.committed.push(pending.keyframe);
            }
        }
        committed
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

    /// 貼る画像として確定したキーフレームのフレーム番号（確定順）。
    pub fn committed_keyframes(&self) -> Vec<u64> {
        self.committed.iter().map(|k| k.frame_index).collect()
    }

    /// コピー内の Planar セル数。
    pub fn planar_cell_count(&self) -> usize {
        self.planes.len()
    }

    /// メモリ上に保持している画像（確定済み・貼り付け待ち・今のキーフレーム）の枚数とバイト数。
    pub fn stored_images(&self) -> (usize, usize) {
        let images = self
            .committed
            .iter()
            .chain(self.pending.iter().map(|pending| &pending.keyframe))
            .chain(self.current.iter());
        images.fold((0, 0), |(count, bytes), keyframe| {
            (count + 1, bytes + keyframe.rgb.as_raw().len())
        })
    }

    /// 走行の最後に 1 回呼ぶ。最終の全セル平面判定（`classify_all_surface_voxels`）の前に、
    /// 平面フィルタ前の GlobalMap を渡す。
    ///
    /// まだ確定していないキーフレーム（貼り付け待ちと、最後のキーフレーム）も、
    /// 以後マップは増えないので確定順の後ろに加え、確定順に貼る。既に貼られたセルは更新しない。
    /// 出力の対象は、コピーの Planar セルのうち GlobalMap の出力条件
    /// （`min_samples` / `min_frames`）を満たすもの。
    pub fn finish(
        mut self,
        map: &VoxelMap,
        voxel_size: f32,
        min_samples: u64,
        min_frames: u64,
    ) -> PasteSummary {
        let start = Instant::now();
        let mut summary = PasteSummary::default();
        let mut keyframes = std::mem::take(&mut self.committed);
        let committed_count = keyframes.len();
        keyframes.extend(self.pending.drain(..).map(|pending| pending.keyframe));
        keyframes.extend(self.current.take());
        summary.added_at_end = keyframes.len() - committed_count;
        let config = self.config.clone();
        if keyframes.is_empty() {
            return summary;
        }
        if !config.valid() || !(voxel_size.is_finite() && voxel_size > 0.0) {
            log::warn!("keyframe paste skipped: invalid config or voxel size");
            return summary;
        }

        // 貼り付け対象: コピーの Planar セル（位置は最終の GlobalMap のセル代表点）
        let planes = std::mem::take(&mut self.planes);
        let targets: Vec<Target> = planes
            .par_iter()
            .filter_map(|(key, plane)| {
                let cell = map.get(key)?;
                (cell.sample_count >= min_samples
                    && cell.observed_frames >= min_frames
                    && cell.mean.coords.iter().all(|v| v.is_finite()))
                .then_some(Target {
                    mean: cell.mean,
                    normal: plane.normal,
                    d: plane.d,
                })
            })
            .collect();
        drop(planes);
        // 遮蔽物: 平面フィルタをかける前の全セル
        let occluders: Vec<Point3<f32>> = map
            .par_iter()
            .map(|(_, cell)| cell.mean)
            .filter(|mean| mean.coords.iter().all(|v| v.is_finite()))
            .collect();
        summary.target_cells = targets.len();
        summary.occluder_cells = occluders.len();
        let targets = BlockIndex::build(targets, config.block_size_m, |target| target.mean);
        let occluders = BlockIndex::build(occluders, config.block_size_m, |point| *point);

        let frame_indices: Vec<u64> = keyframes.iter().map(|k| k.frame_index).collect();
        let mut colors: Vec<Option<PastedColor>> = vec![None; targets.items.len()];
        for (slot, keyframe) in keyframes.into_iter().enumerate() {
            // 貼り終えた画像はここで手放す
            let stats = paste_keyframe(
                &config,
                &keyframe,
                slot as u32,
                &occluders,
                &targets,
                &mut colors,
                voxel_size as f64,
            );
            summary.keyframes.push(stats);
        }
        summary.points = targets
            .items
            .iter()
            .zip(&colors)
            .filter_map(|(target, color)| {
                color.map(|color| PastedPoint {
                    position: target.mean,
                    rgb: color.rgb,
                    keyframe_frame_index: frame_indices[color.slot as usize],
                })
            })
            .collect();
        summary.elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        summary
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
// 1 枚分の貼り付け
// ---------------------------------------------------------------------------

/// 貼り付け対象のセル（最終の代表点と、遅延判定の平面）。
#[derive(Debug, Clone, Copy)]
struct Target {
    mean: Point3<f32>,
    normal: Vector3<f32>,
    d: f32,
}

#[derive(Debug, Clone, Copy)]
struct PastedColor {
    rgb: [u8; 3],
    /// 何番目に貼ったキーフレームか。
    slot: u32,
}

enum TargetOutcome {
    AlreadyPasted,
    TooOblique,
    Occluded,
    SampleFailed,
    Sampled([u8; 3]),
}

fn paste_keyframe(
    config: &KeyframePasteConfig,
    keyframe: &KeyframeImage,
    slot: u32,
    occluders: &BlockIndex<Point3<f32>>,
    targets: &BlockIndex<Target>,
    colors: &mut [Option<PastedColor>],
    voxel_size: f64,
) -> PasteStats {
    let start = Instant::now();
    let camera = &keyframe.camera;
    let mut stats = PasteStats {
        keyframe_frame_index: keyframe.frame_index,
        ..PasteStats::default()
    };
    if keyframe.rgb.width() != camera.width
        || keyframe.rgb.height() != camera.height
        || camera.width < 2
        || camera.height < 2
        || !(camera.fx > 0.0 && camera.fy > 0.0)
    {
        log::warn!(
            "keyframe paste {}: image size does not match the camera",
            keyframe.frame_index
        );
        return stats;
    }
    let camera_from_world = &keyframe.camera_from_world;
    let Some(world_from_camera) = camera_from_world.try_inverse() else {
        return stats;
    };
    // ブロックの選別は少し広めに（対象セルは代表点で選別し、平面上の点で判定するため）
    let frustum = Frustum::new(
        camera_from_world,
        camera,
        config.min_depth_m - 0.1,
        config.max_depth_m + 0.1,
        0.3,
    );

    // 1. 遮蔽物（平面フィルタ前の全セル）を深度バッファに描く。
    //    投影は並列、描画（最小値の更新）は順番に行う（順序に依らず同じ結果になる）。
    let occluder_cull = QuickCull::new(
        camera_from_world,
        camera,
        config.min_depth_m,
        config.max_depth_m,
        0.3,
    );
    let max_radius = 0.25 * (camera.width.max(camera.height) as f64);
    let splats: Vec<[f32; 4]> = occluders
        .visible(&frustum)
        .par_iter()
        .flat_map_iter(|range| occluders.items[range.clone()].iter())
        .filter_map(|point| {
            project_occluder(
                point,
                camera_from_world,
                camera,
                &occluder_cull,
                voxel_size,
                config,
                max_radius,
            )
        })
        .collect();
    let mut depth_buffer = DepthBuffer::new(camera, voxel_size, config);
    depth_buffer.draw(&splats);
    stats.occluder_cells = splats.len();

    // 2. 対象セルを判定する（色はまだ書き込まない）
    let view = View {
        config,
        camera_from_world,
        camera_center: Point3::from(world_from_camera.fixed_view::<3, 1>(0, 3).into_owned()),
        camera,
        image: &keyframe.rgb,
        crop: Crop::new(camera, config.image_keep_ratio),
        cull: QuickCull::new(
            camera_from_world,
            camera,
            config.min_depth_m - 0.05,
            config.max_depth_m + 0.05,
            0.25,
        ),
        depth_buffer: &depth_buffer,
        voxel_size,
    };
    let painted: &[Option<PastedColor>] = colors;
    let outcomes: Vec<(usize, TargetOutcome)> = targets
        .visible(&frustum)
        .par_iter()
        .flat_map_iter(|range| range.clone())
        .filter_map(|index| {
            view.evaluate(&targets.items[index], painted[index].is_some())
                .map(|outcome| (index, outcome))
        })
        .collect();
    stats.in_view = outcomes.len();

    // 3. 反映: まだ色のないセルにだけ貼る
    for (index, outcome) in outcomes {
        match outcome {
            TargetOutcome::AlreadyPasted => stats.already_pasted += 1,
            TargetOutcome::TooOblique => stats.too_oblique += 1,
            TargetOutcome::Occluded => stats.occluded += 1,
            TargetOutcome::SampleFailed => stats.sample_failed += 1,
            TargetOutcome::Sampled(rgb) => {
                colors[index] = Some(PastedColor { rgb, slot });
                stats.pasted_new += 1;
            }
        }
    }
    stats.elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
    stats
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
    fn evaluate(&self, target: &Target, already_pasted: bool) -> Option<TargetOutcome> {
        if !self.cull.passes(&target.mean) {
            return None;
        }
        // セルの代表点を、その平面上へ射影した位置で画像を読む
        let normal = target.normal.cast::<f64>();
        let mean = target.mean.cast::<f64>();
        let point = mean - normal * (normal.dot(&mean.coords) + target.d as f64);
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
        if already_pasted {
            return Some(TargetOutcome::AlreadyPasted);
        }
        let to_camera = self.camera_center - point;
        let distance = to_camera.norm();
        if !(distance.is_finite() && distance > 0.0) {
            return None;
        }
        let view_cosine = (normal.dot(&to_camera) / distance).abs().min(1.0);
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
        Some(
            match sample_footprint(
                self.image,
                &pixel,
                footprint_radius_px,
                self.config.max_footprint_samples,
            ) {
                Some(rgb) => TargetOutcome::Sampled(rgb),
                None => TargetOutcome::SampleFailed,
            },
        )
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

/// f32 の深度と視野の判定で、点を安く先に落とす（判定は少し広めにし、残った点は
/// f64 で正確に判定し直す）。
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

/// 視錐台（深度範囲＋余白付きの画像範囲。歪みは無視）。空間ブロックの選別に使う。
struct Frustum {
    camera_from_world: Matrix4<f64>,
    fx: f64,
    fy: f64,
    left: f64,
    right: f64,
    top: f64,
    bottom: f64,
    min_z: f64,
    max_z: f64,
}

impl Frustum {
    fn new(
        camera_from_world: &Matrix4<f64>,
        camera: &CameraIntrinsics,
        min_depth_m: f64,
        max_depth_m: f64,
        margin_ratio: f64,
    ) -> Self {
        let margin_x = margin_ratio * camera.width as f64;
        let margin_y = margin_ratio * camera.height as f64;
        Self {
            camera_from_world: *camera_from_world,
            fx: camera.fx,
            fy: camera.fy,
            // u = fx·x/z + cx が [-margin, width+margin] に入る ⇔ 下の 4 つが非負
            left: camera.cx + margin_x,
            right: camera.width as f64 + margin_x - camera.cx,
            top: camera.cy + margin_y,
            bottom: camera.height as f64 + margin_y - camera.cy,
            min_z: min_depth_m,
            max_z: max_depth_m,
        }
    }

    /// 軸平行な箱 [min, max] が視錐台と交わりうるか（8 隅がどれか 1 つの面の外側に
    /// そろっていれば交わらない。保守的な判定）。
    fn may_contain(&self, min: &Point3<f32>, max: &Point3<f32>) -> bool {
        let corners: [Point3<f64>; 8] = std::array::from_fn(|i| {
            let corner = Point3::new(
                if i & 1 == 0 { min.x } else { max.x },
                if i & 2 == 0 { min.y } else { max.y },
                if i & 4 == 0 { min.z } else { max.z },
            );
            self.camera_from_world
                .transform_point(&corner.cast::<f64>())
        });
        let outside =
            |inside: &dyn Fn(&Point3<f64>) -> f64| corners.iter().all(|c| inside(c) < 0.0);
        !(outside(&|c| c.z - self.min_z)
            || outside(&|c| self.max_z - c.z)
            || outside(&|c| self.fx * c.x + self.left * c.z)
            || outside(&|c| self.right * c.z - self.fx * c.x)
            || outside(&|c| self.fy * c.y + self.top * c.z)
            || outside(&|c| self.bottom * c.z - self.fy * c.y))
    }
}

/// 点を空間ブロックごとにまとめて並べ直したもの（ブロックごとの外接箱付き）。
struct BlockIndex<T> {
    items: Vec<T>,
    blocks: Vec<Block>,
}

struct Block {
    min: Point3<f32>,
    max: Point3<f32>,
    range: Range<usize>,
}

impl<T: Send> BlockIndex<T> {
    fn build(
        items: Vec<T>,
        block_size: f32,
        position: impl Fn(&T) -> Point3<f32> + Sync + Send,
    ) -> Self {
        let block_of = |point: &Point3<f32>| -> [i32; 3] {
            std::array::from_fn(|axis| (point[axis] / block_size).floor() as i32)
        };
        let mut keyed: Vec<([i32; 3], T)> = items
            .into_par_iter()
            .map(|item| (block_of(&position(&item)), item))
            .collect();
        keyed.par_sort_unstable_by_key(|(key, _)| *key);

        let mut blocks: Vec<([i32; 3], Block)> = Vec::new();
        let mut sorted = Vec::with_capacity(keyed.len());
        for (index, (key, item)) in keyed.into_iter().enumerate() {
            let point = position(&item);
            match blocks.last_mut() {
                Some((last_key, block)) if *last_key == key => {
                    block.min = block.min.inf(&point);
                    block.max = block.max.sup(&point);
                    block.range.end = index + 1;
                }
                _ => blocks.push((
                    key,
                    Block {
                        min: point,
                        max: point,
                        range: index..index + 1,
                    },
                )),
            }
            sorted.push(item);
        }
        Self {
            items: sorted,
            blocks: blocks.into_iter().map(|(_, block)| block).collect(),
        }
    }

    /// 視錐台と交わりうるブロックの、`items` 内の範囲。
    fn visible(&self, frustum: &Frustum) -> Vec<Range<usize>> {
        self.blocks
            .iter()
            .filter(|block| frustum.may_contain(&block.min, &block.max))
            .map(|block| block.range.clone())
            .collect()
    }
}

/// 遮蔽物のセル 1 つを画像へ投影する（[u, v, 深度, 描く半径 px]）。
fn project_occluder(
    point: &Point3<f32>,
    camera_from_world: &Matrix4<f64>,
    camera: &CameraIntrinsics,
    cull: &QuickCull,
    voxel_size: f64,
    config: &KeyframePasteConfig,
    max_radius: f64,
) -> Option<[f32; 4]> {
    if !cull.passes(point) {
        return None;
    }
    let pc = camera_from_world.transform_point(&point.cast::<f64>());
    if pc.z.is_nan()
        || pc.z < config.min_depth_m
        || pc.z > config.max_depth_m
        || !roughly_in_view(&pc, camera, 0.3)
    {
        return None;
    }
    let radius =
        (0.5 * config.occluder_splat_scale * camera.fx * voxel_size / pc.z).min(max_radius);
    let (pixel, _) = project_camera_point_subpixel(&pc, camera)?;
    let (width, height) = (camera.width as f64, camera.height as f64);
    if pixel.x + radius < 0.0
        || pixel.y + radius < 0.0
        || pixel.x - radius >= width
        || pixel.y - radius >= height
    {
        return None;
    }
    Some([pixel.x as f32, pixel.y as f32, pc.z as f32, radius as f32])
}

/// GlobalMap の全セルを画像に描き、マスごとの最小深度を記録したもの。
/// セルは見かけの大きさ（`occluder_splat_scale` 倍）の正方形として描くので、
/// 近くの物ほど広く隠す。
struct DepthBuffer {
    cell_px: f64,
    cols: usize,
    rows: usize,
    depth: Vec<f32>,
    splat_scale: f64,
    voxel_size: f64,
    margin_m: f64,
    margin_ratio: f64,
}

impl DepthBuffer {
    fn new(camera: &CameraIntrinsics, voxel_size: f64, config: &KeyframePasteConfig) -> Self {
        let width = camera.width as f64;
        let height = camera.height as f64;
        let cell_px = (config.depth_buffer_cell_px * width / 960.0).max(1.0);
        let cols = ((width / cell_px).ceil() as usize).max(1);
        let rows = ((height / cell_px).ceil() as usize).max(1);
        Self {
            cell_px,
            cols,
            rows,
            depth: vec![f32::INFINITY; cols * rows],
            splat_scale: config.occluder_splat_scale,
            voxel_size,
            margin_m: config.occlusion_margin_m,
            margin_ratio: config.occlusion_margin_ratio,
        }
    }

    fn draw(&mut self, splats: &[[f32; 4]]) {
        let (cell_px, cols, rows) = (self.cell_px, self.cols, self.rows);
        let to_cell =
            |value: f64, count: usize| ((value / cell_px).floor().max(0.0) as usize).min(count - 1);
        for &[u, v, z, radius] in splats {
            let (u, v, radius) = (u as f64, v as f64, radius as f64);
            let (x0, x1) = (to_cell(u - radius, cols), to_cell(u + radius, cols));
            let (y0, y1) = (to_cell(v - radius, rows), to_cell(v + radius, rows));
            for y in y0..=y1 {
                for value in &mut self.depth[y * cols + x0..=y * cols + x1] {
                    if z < *value {
                        *value = z;
                    }
                }
            }
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
                    *total += value as f64;
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

    /// 位置 `position` から +z 方向を見ているカメラ（`yaw` で y 軸回りに回す）。
    fn camera_from_world_at(position: Vector3<f64>, yaw: f64) -> Matrix4<f64> {
        let world_from_camera = Translation3::from(position).to_homogeneous()
            * Rotation3::from_axis_angle(&Vector3::y_axis(), yaw).to_homogeneous();
        world_from_camera.try_inverse().unwrap()
    }

    fn keyframe(
        frame_index: u64,
        camera_from_world: Matrix4<f64>,
        camera: &CameraIntrinsics,
        rgb: RgbImage,
    ) -> KeyframeImage {
        KeyframeImage {
            frame_index,
            camera_from_world,
            camera: camera.clone(),
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

    /// 平面上の格子点（origin + u·e1 + v·e2、2.5 cm 間隔）を、`status` のセルとして GlobalMap に入れる。
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

    /// GlobalMap の今の判定を、平面判定の記録としてコピーへ渡す。
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

    /// キーフレームを順に登録して（フレーム 10 ごと）、最後に貼る。
    fn paste_all(
        map: &VoxelMap,
        keyframes: Vec<KeyframeImage>,
        config: KeyframePasteConfig,
    ) -> PasteSummary {
        let mut paster = KeyframePaster::new(config);
        sync_all(&mut paster, map);
        for (i, keyframe) in keyframes.into_iter().enumerate() {
            paster.on_new_keyframe(i as u64 * 10, keyframe);
        }
        paster.finish(map, VOXEL, 1, 1)
    }

    fn by_key(summary: &PasteSummary) -> FxHashMap<VoxelKey, PastedPoint> {
        summary
            .points
            .iter()
            .map(|point| (voxel_key(&point.position, VOXEL), *point))
            .collect()
    }

    fn pixel_of(camera_from_world: &Matrix4<f64>, point: &Point3<f32>) -> Option<Vector2<f64>> {
        let pc = camera_from_world.transform_point(&point.cast::<f64>());
        project_camera_point_subpixel(&pc, &camera()).map(|(pixel, _)| pixel)
    }

    #[test]
    fn footprint_sampling_averages_over_the_apparent_cell_size() {
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
        assert!(sample_footprint(&checker, &Vector2::new(-10.0, 5.0), 1.0, 5).is_none());
    }

    #[test]
    fn pastes_onto_planar_copy_cells_inside_the_central_crop_only_at_finish() {
        let camera = camera();
        let mut map = VoxelMap::default();
        wall(&mut map, 5.0, 4.0, 3.0);
        let pose = camera_from_world_at(Vector3::zeros(), 0.0);
        let image = render(&camera, &pose, &Vector3::z(), 5.0, &|p| {
            if p.x < 0.0 { RED } else { BLUE }
        });
        let summary = paste_all(
            &map,
            vec![keyframe(7, pose, &camera, image)],
            KeyframePasteConfig::default(),
        );
        assert_eq!(summary.target_cells, map.len());
        assert_eq!(summary.occluder_cells, map.len());
        assert_eq!(summary.added_at_end, 1);
        let stats = summary.keyframes[0];
        assert_eq!(stats.keyframe_frame_index, 7);
        assert_eq!(stats.occluded, 0, "{stats:?}");
        assert_eq!(stats.too_oblique, 0, "{stats:?}");
        assert_eq!(stats.pasted_new, stats.in_view, "{stats:?}");
        assert!(stats.pasted_new > 10_000, "{stats:?}");

        let pasted = by_key(&summary);
        let crop = Crop::new(&camera, 0.75);
        for (key, cell) in &map {
            let pixel = pixel_of(&pose, &cell.mean).unwrap();
            let point = pasted.get(key);
            assert_eq!(point.is_some(), crop.contains(&pixel), "{pixel:?}");
            if let Some(point) = point {
                assert_eq!(point.position, cell.mean);
                assert_eq!(point.keyframe_frame_index, 7);
                if cell.mean.x < -0.05 {
                    assert_eq!(point.rgb, RED);
                } else if cell.mean.x > 0.05 {
                    assert_eq!(point.rgb, BLUE);
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
        let pose = camera_from_world_at(Vector3::zeros(), 0.0);
        let board_hit = |p: &Point3<f64>| {
            let at_board = p.coords * (3.0 / p.z);
            (board_x.0..board_x.1).contains(&at_board.x)
                && (board_y.0..board_y.1).contains(&at_board.y)
        };
        let image = render(&camera, &pose, &Vector3::z(), 5.0, &|p| {
            if board_hit(p) { GREEN } else { RED }
        });
        let summary = paste_all(
            &map,
            vec![keyframe(0, pose, &camera, image)],
            KeyframePasteConfig::default(),
        );
        let stats = summary.keyframes[0];
        assert!(stats.occluded > 500, "{stats:?}");
        assert_eq!(
            stats.pasted_new + stats.occluded,
            stats.in_view,
            "{stats:?}"
        );

        let pasted = by_key(&summary);
        let corner =
            |x: f64, y: f64| pixel_of(&pose, &Point3::new(x as f32, y as f32, 3.0)).unwrap();
        let (top_left, bottom_right) = (corner(board_x.0, board_y.0), corner(board_x.1, board_y.1));
        let mut hidden = 0;
        for (key, cell) in &map {
            let point = pasted.get(key);
            if cell.mean.z < 4.0 {
                assert!(point.is_none(), "the board itself is not a target");
                continue;
            }
            if let Some(point) = point {
                assert_eq!(point.rgb, RED, "board color leaked onto the wall");
            }
            let pixel = pixel_of(&pose, &cell.mean).unwrap();
            if pixel.x > top_left.x + 1.0
                && pixel.x < bottom_right.x - 1.0
                && pixel.y > top_left.y + 1.0
                && pixel.y < bottom_right.y - 1.0
            {
                hidden += 1;
                assert!(point.is_none(), "wall cell behind the board was pasted");
            }
        }
        assert!(hidden > 500);
    }

    /// 重なったところは先に貼ったキーフレームの色のまま（後から近くで撮っても更新しない）。
    #[test]
    fn already_pasted_cells_keep_the_first_keyframe_color() {
        let camera = camera();
        let mut map = VoxelMap::default();
        wall(&mut map, 5.0, 4.0, 1.5);
        let first = camera_from_world_at(Vector3::zeros(), 0.0);
        // 2 枚目は右へ 2.5 m、壁へ 1.5 m 近づいた位置から（もっと良い見え方）
        let second = camera_from_world_at(Vector3::new(2.5, 0.0, 1.5), 0.0);
        let summary = paste_all(
            &map,
            vec![
                keyframe(1, first, &camera, uniform(&camera, RED)),
                keyframe(2, second, &camera, uniform(&camera, BLUE)),
            ],
            KeyframePasteConfig::default(),
        );
        let [one, two] = [summary.keyframes[0], summary.keyframes[1]];
        assert_eq!((one.keyframe_frame_index, two.keyframe_frame_index), (1, 2));
        assert!(one.pasted_new > 0 && one.already_pasted == 0, "{one:?}");
        assert!(two.already_pasted > 0 && two.pasted_new > 0, "{two:?}");
        assert_eq!(two.pasted_new + two.already_pasted, two.in_view, "{two:?}");

        let pasted = by_key(&summary);
        let color_at = |x: f32| pasted[&voxel_key(&Point3::new(x, 0.0125, 5.0), VOXEL)];
        // 両方に写る範囲（x ≈ 1 m）は 1 枚目のまま、2 枚目にしか写らない範囲（x ≈ 3.5 m）は 2 枚目
        assert_eq!(color_at(1.0125).rgb, RED);
        assert_eq!(color_at(1.0125).keyframe_frame_index, 1);
        assert_eq!(color_at(3.5125).rgb, BLUE);
        assert_eq!(color_at(3.5125).keyframe_frame_index, 2);
        assert_eq!(color_at(-2.0125).rgb, RED);
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
            let summary = paste_all(
                &map,
                vec![keyframe(0, pose, &camera, uniform(&camera, RED))],
                KeyframePasteConfig::default(),
            );
            // 貼ったセルは、実際の視線との角度が下限以上のものだけ
            let normal = Vector3::new(0.0, -tilt.sin(), tilt.cos());
            for point in &summary.points {
                let p = point.position.cast::<f64>();
                let cosine = normal.dot(&p.coords).abs() / p.coords.norm();
                assert!(cosine >= 0.25 - 1e-6, "{cosine}");
            }
            summary.keyframes[0]
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
    fn keyframe_n_is_committed_delay_frames_after_n_plus_one_and_pasted_only_at_finish() {
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
                &camera,
                uniform(&camera, color),
            )
        };

        assert_eq!(paster.on_new_keyframe(0, make(0, 0.0, RED)), None);
        assert_eq!(paster.current_keyframe_index(), Some(0));
        for frame in 0..=3 {
            assert!(paster.commit_due(frame).is_empty());
        }
        // フレーム 3 で N+1 (=3) が確定 → N (=0) はフレーム 8 に確定する
        assert_eq!(paster.on_new_keyframe(3, make(3, 1.0, BLUE)), Some((0, 8)));
        assert_eq!(paster.pending_keyframes(), vec![(0, 8)]);
        for frame in 4..8 {
            assert!(paster.commit_due(frame).is_empty());
        }
        assert_eq!(paster.commit_due(8), vec![0]);
        assert_eq!(paster.committed_keyframes(), vec![0]);
        assert!(paster.pending_keyframes().is_empty());
        assert_eq!(
            paster.on_new_keyframe(10, make(10, 2.0, GREEN)),
            Some((3, 15))
        );

        // 画像は 3 枚とも保持している（確定 1・待ち 1・今 1）
        let (count, bytes) = paster.stored_images();
        assert_eq!((count, bytes), (3, 3 * 320 * 240 * 3));

        // 最後: 確定済み 0 → 待ち 3 → 今の 10 の順に貼る。先に貼った 0（赤）が残る
        let summary = paster.finish(&map, VOXEL, 1, 1);
        assert_eq!(summary.added_at_end, 2);
        assert_eq!(
            summary
                .keyframes
                .iter()
                .map(|s| s.keyframe_frame_index)
                .collect::<Vec<_>>(),
            vec![0, 3, 10]
        );
        assert_eq!(summary.points.len(), map.len());
        assert!(summary.points.iter().all(|p| p.rgb == RED));
        assert_eq!(summary.keyframes[1].pasted_new, 0);
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
        assert!(
            global
                .voxel_map
                .values()
                .all(|cell| cell.surface_status != SurfaceStatus::Planar)
        );

        // NonPlanar と判定されたセルはコピーから外れる
        let key = voxel_key(&Point3::new(0.0125, 0.0125, 5.0), VOXEL);
        paster.sync_planes(&[SurfaceClassificationRecord {
            key,
            mean: Point3::new(0.0125, 0.0125, 5.0),
            status: SurfaceStatus::NonPlanar,
            plane: None,
        }]);
        assert_eq!(paster.planar_cell_count(), stats.planar - 1);

        let camera = camera();
        let pose = camera_from_world_at(Vector3::zeros(), 0.0);
        paster.on_new_keyframe(0, keyframe(0, pose, &camera, uniform(&camera, RED)));
        let summary = paster.finish(&global.voxel_map, VOXEL, 2, 2);
        assert_eq!(summary.target_cells, stats.planar - 1);
        assert!(summary.points.len() > 10_000, "{:?}", summary.keyframes);
        assert_eq!(summary.keyframes[0].occluded, 0, "{:?}", summary.keyframes);
        let pasted = by_key(&summary);
        assert!(!pasted.contains_key(&key));
        assert_eq!(
            pasted[&voxel_key(&Point3::new(0.0375, 0.0125, 5.0), VOXEL)].rgb,
            RED
        );
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
        let texture = |p: &Point3<f64>| {
            let value = 128.0 + 70.0 * (1.3 * p.x).sin() * (1.1 * p.y).cos();
            [value.round() as u8, (255.0 - value).round() as u8, 90]
        };
        let pose = camera_from_world_at(Vector3::new(0.3, -0.2, 0.0), 0.08);
        let image = render(&camera, &pose, &Vector3::z(), 5.0, &texture);
        let summary = paste_all(
            &map,
            vec![keyframe(0, pose, &camera, image)],
            KeyframePasteConfig::default(),
        );
        assert!(summary.points.len() > 10_000, "{:?}", summary.keyframes);
        for point in &summary.points {
            let expected = texture(&point.position.cast::<f64>());
            for channel in 0..3 {
                assert!(
                    (point.rgb[channel] as i32 - expected[channel] as i32).abs() <= 3,
                    "cell {:?}: {:?} vs {:?}",
                    point.position,
                    point.rgb,
                    expected
                );
            }
        }
    }

    /// 空間ブロックで視野外を飛ばしても、全部を調べた場合と結果が完全に一致する。
    #[test]
    fn block_culling_gives_exactly_the_same_result_as_scanning_everything() {
        let camera = camera();
        let mut map = VoxelMap::default();
        // 正面の壁、横の壁、床、手前の柱（NonPlanar）、カメラの後ろの壁
        wall(&mut map, 6.0, 6.0, 2.0);
        plane_cells(
            &mut map,
            &Point3::new(-3.0, 0.0, 3.0),
            &Vector3::z(),
            &Vector3::y(),
            (-3.0, 3.0, -2.0, 2.0),
            SurfaceStatus::Planar,
        );
        plane_cells(
            &mut map,
            &Point3::new(0.0, 2.0, 3.0),
            &Vector3::x(),
            &Vector3::z(),
            (-6.0, 6.0, -3.0, 3.0),
            SurfaceStatus::Planar,
        );
        plane_cells(
            &mut map,
            &Point3::new(0.8, 0.0, 2.5),
            &Vector3::x(),
            &Vector3::y(),
            (-0.2, 0.2, -2.0, 2.0),
            SurfaceStatus::NonPlanar,
        );
        wall(&mut map, -4.0, 4.0, 2.0);
        let texture = |p: &Point3<f64>| {
            let value = (128.0 + 60.0 * (2.1 * p.x + 0.7 * p.z).sin() * (1.7 * p.y).cos()) as u8;
            [value, 255 - value, 64]
        };
        let keyframes = || {
            [
                (Vector3::new(0.0, 0.0, 0.0), 0.0),
                (Vector3::new(1.0, 0.1, 0.5), -0.35),
                (Vector3::new(-0.5, -0.2, 1.0), 0.4),
                (Vector3::new(2.0, 0.0, -0.5), 0.15),
            ]
            .into_iter()
            .enumerate()
            .map(|(i, (position, yaw))| {
                let pose = camera_from_world_at(position, yaw);
                // 画像は位置ごとの単純な模様（ブロックの有無で結果が変わらないことだけを見る）
                let image = RgbImage::from_fn(camera.width, camera.height, |u, v| {
                    let p = Point3::new(u as f64 * 0.01 + i as f64, v as f64 * 0.01, 0.0);
                    Rgb(texture(&p))
                });
                keyframe(i as u64, pose, &camera, image)
            })
            .collect::<Vec<_>>()
        };
        let small_blocks = paste_all(
            &map,
            keyframes(),
            KeyframePasteConfig {
                block_size_m: 0.5,
                ..KeyframePasteConfig::default()
            },
        );
        let one_block = paste_all(
            &map,
            keyframes(),
            KeyframePasteConfig {
                block_size_m: 1.0e4,
                ..KeyframePasteConfig::default()
            },
        );
        let sorted = |summary: &PasteSummary| {
            let mut points: Vec<_> = summary
                .points
                .iter()
                .map(|p| (voxel_key(&p.position, VOXEL), p.rgb, p.keyframe_frame_index))
                .collect();
            points.sort_by_key(|(key, _, _)| (key.ix, key.iy, key.iz));
            points
        };
        assert!(small_blocks.points.len() > 20_000);
        assert!(small_blocks.keyframes.iter().any(|s| s.occluded > 0));
        assert!(small_blocks.keyframes.iter().any(|s| s.already_pasted > 0));
        assert_eq!(sorted(&small_blocks), sorted(&one_block));
        for (a, b) in small_blocks.keyframes.iter().zip(&one_block.keyframes) {
            assert_eq!(
                (
                    a.in_view,
                    a.pasted_new,
                    a.already_pasted,
                    a.occluded,
                    a.too_oblique
                ),
                (
                    b.in_view,
                    b.pasted_new,
                    b.already_pasted,
                    b.occluded,
                    b.too_oblique
                )
            );
            assert_eq!(a.occluder_cells, b.occluder_cells);
        }
    }

    /// main.rs と同じ順番の走行の再現: 模様のある壁に沿って動くカメラ（= LiDAR）で、
    /// 視覚アライメント → 新しいキーフレームの登録 → GlobalMap 更新 → 遅延平面判定 →
    /// コピーへの反映 → 予定のキーフレームの確定、を毎フレーム行い、最後に貼る。
    #[test]
    #[cfg_attr(debug_assertions, ignore = "slow in debug builds; run with --release")]
    fn simulated_run_commits_on_schedule_and_pastes_correct_colors_at_the_end() {
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
        let mut committed_at = Vec::new();

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
            // 4. コピーへの反映と、予定のキーフレームの確定（まだ貼らない）
            paster.sync_planes(&global.take_surface_classification_log());
            for index in paster.commit_due(i) {
                committed_at.push((index, i));
            }
        }
        // 走行の最後（main.rs と同じ順番）: 最終の全セル判定の前に貼る
        paster.sync_planes(&global.take_surface_classification_log());
        global.set_surface_classification_log(false);
        let summary = paster.finish(&global.voxel_map, MAP_VOXEL, 2, 2);

        // キーフレーム N は「N+1 の確定 + 5 フレーム」に確定し、残りは最後に加えられる
        assert!(created.len() >= 4, "keyframes created at {created:?}");
        let mut expected_committed = Vec::new();
        for pair in created.windows(2) {
            if pair[1] + 5 < FRAMES {
                expected_committed.push((pair[0], pair[1] + 5));
            }
        }
        assert_eq!(committed_at, expected_committed);
        assert_eq!(
            summary
                .keyframes
                .iter()
                .map(|s| s.keyframe_frame_index)
                .collect::<Vec<_>>(),
            created,
            "pasted in commit order, then the rest"
        );
        assert_eq!(summary.added_at_end, created.len() - committed_at.len());

        // 遅延判定の済んだ範囲は、キーフレームの画像の中央部に入る所がほぼ全部貼られ、
        // 色は壁の模様と一致する（走行の最後に初めて見えた先端部は、遅延判定がまだ
        // 済んでいないので対象外）
        let first_x = created[0] as f64 * STEP_M;
        let settled_x = (FRAMES - 12) as f64 * STEP_M;
        let pasted: FxHashMap<VoxelKey, PastedPoint> = summary
            .points
            .iter()
            .map(|p| (voxel_key(&p.position, MAP_VOXEL), *p))
            .collect();
        let (mut covered, mut inside) = (0usize, 0usize);
        for (key, cell) in &global.voxel_map {
            let (x, y) = (cell.mean.x as f64, cell.mean.y as f64);
            if cell.observed_frames >= 2 && x > first_x - 2.5 && x < settled_x && y.abs() < 1.9 {
                inside += 1;
                covered += usize::from(pasted.contains_key(key));
            }
        }
        let mut errors: Vec<f64> = summary
            .points
            .iter()
            .map(|p| (p.rgb[0] as f64 - texture(&p.position.cast::<f64>())[0] as f64).abs())
            .collect();
        errors.sort_by(f64::total_cmp);
        let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
        let p95 = errors[errors.len() * 95 / 100];
        println!(
            "keyframes {created:?}, committed at {committed_at:?}; targets {}, pasted {}; \
             settled-region coverage {covered}/{inside}; color error mean {mean_error:.2} \
             p95 {p95:.1}; finish {:.1} ms",
            summary.target_cells,
            summary.points.len(),
            summary.elapsed_ms
        );
        assert!(inside > 3000, "{inside}");
        assert!(
            covered as f64 > 0.97 * inside as f64,
            "covered {covered} / {inside}"
        );
        assert!(mean_error < 3.0 && p95 < 8.0, "mean {mean_error} p95 {p95}");
    }

    /// 実行時間の目安（`cargo test --release -- --ignored --nocapture`）。
    /// 大きなマップ（視野外が大半）に、位置をずらしたキーフレームを何枚も貼る。
    #[test]
    #[ignore]
    fn bench_finish_on_a_large_map() {
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
        // 長い壁（200 m × 6 m、平面）と、遠くの広い地面（遮蔽物のみ）
        plane_cells(
            &mut map,
            &Point3::new(0.0, 0.0, 8.0),
            &Vector3::x(),
            &Vector3::y(),
            (-10.0, 190.0, -3.0, 3.0),
            SurfaceStatus::Planar,
        );
        plane_cells(
            &mut map,
            &Point3::new(0.0, 3.0, -20.0),
            &Vector3::x(),
            &Vector3::z(),
            (-10.0, 190.0, -6.0, 0.0),
            SurfaceStatus::NonPlanar,
        );
        let mut paster = KeyframePaster::new(KeyframePasteConfig::default());
        sync_all(&mut paster, &map);
        let keyframe_count = 60u64;
        for k in 0..keyframe_count {
            let pose = camera_from_world_at(Vector3::new(k as f64 * 3.0, 0.0, 0.0), 0.0);
            let image =
                RgbImage::from_fn(960, 600, |u, v| Rgb([(u % 256) as u8, (v % 256) as u8, 77]));
            paster.on_new_keyframe(k * 10, keyframe(k, pose, &camera, image));
        }
        let (images, bytes) = paster.stored_images();
        let summary = paster.finish(&map, VOXEL, 1, 1);
        let per_keyframe: Vec<f64> = summary.keyframes.iter().map(|s| s.elapsed_ms).collect();
        println!(
            "map cells={} targets={} images={images} ({:.0} MB) pasted={} finish {:.0} ms \
             (setup {:.0} ms, per keyframe mean {:.1} ms)",
            map.len(),
            summary.target_cells,
            bytes as f64 / 1e6,
            summary.points.len(),
            summary.elapsed_ms,
            summary.elapsed_ms - per_keyframe.iter().sum::<f64>(),
            per_keyframe.iter().sum::<f64>() / per_keyframe.len() as f64,
        );
    }
}
