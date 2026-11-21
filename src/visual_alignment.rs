//! キーフレーム方式の疎な直接法画像アライメント（LiDAR 更新の後に実行）。
//!
//! 概要
//! 1. キーフレーム K（画像ピラミッド・確定したカメラ姿勢・視覚点のセット）と、
//!    現在フレーム N（点群・画像・LiDAR 更新済みの姿勢）がある。
//! 2. K の視覚点は、GlobalMap の Planar セルの点を先に選び、それが K の画像に写る
//!    ピクセルの周りを 9×9 で切り出したもの。パッチの各画素は「その点の法線を持つ
//!    平面」と K カメラからの光線の交点（3D 点）とみなす。この 3D 点を N の画像に
//!    投影し、輝度が K と一致するように N のフレーム開始時刻の 15 次元状態を補正する
//!    （LiDAR の事後分布を事前分布にした Gauss-Newton。配分は情報行列が決める）。
//!    N の点群から作った深度バッファで、手前の物に隠れたパッチは使わない。
//! 3. K から `max_keyframe_age_frames` 未満、かつ見えているパッチの割合が
//!    `min_visible_ratio` 以上の間は、同じ K を基準にし続ける。
//! 4. 交代条件を満たしたら、位置合わせ済みの現在フレームから新しい K を作る。
//!
//! 交代のルール
//! - 新しい K を作るのは、LiDAR（呼び出し側で保証）と画像の更新がどちらも
//!   採用されたフレームだけ。
//! - 交代後、最低 `min_keyframe_frames` フレームは同じ K を使う。
//! - 画像の更新が `max_consecutive_failures` 回連続で失敗したら、LiDAR が採用された
//!   現在フレームから、画像の更新なしで K を作り直す。
//! - K がまだない場合（起動直後）は、画像の更新なしで K を作る。

use image::RgbImage;
use nalgebra::{Cholesky, Matrix3, Matrix4, Point3, Vector2, Vector3};
use rayon::prelude::*;

use crate::camera::{CameraIntrinsics, project_camera_point_subpixel, unproject_pixel_at_depth};
use crate::eskf::{ERROR_STATE_DIM, Eskf, Matrix15, NavState, Vector15};
use crate::types::{IMU, VisualAlignmentLog};
use crate::voxel_map::{SurfaceStatus, VoxelMap};

pub const PATCH_SIZE: usize = 9;
pub const PATCH_RADIUS: i32 = 4;
pub const PATCH_PIXELS: usize = PATCH_SIZE * PATCH_SIZE;
pub const PYRAMID_LEVELS: usize = 3;

#[derive(Debug, Clone)]
pub struct VisualAlignmentConfig {
    // --- キーフレームの視覚点の選び方 ---
    /// 1 セルに最大 1 点を採用するグリッドの大きさ [px]（レベル 0）。
    pub grid_size_px: usize,
    /// キーフレームが持つ視覚点の上限。
    pub max_keyframe_points: usize,
    /// これ未満の視覚点しか作れない画像からはキーフレームを作らない。
    pub min_keyframe_points: usize,
    /// 9×9 窓の Shi-Tomasi スコア sqrt(λmin / 81)（階調/px）の下限。
    pub min_shi_tomasi_score: f64,
    pub min_depth_m: f64,
    pub max_depth_m: f64,
    /// 法線と視線のなす角の cos の下限（0.25 ≒ 75°より斜めの面は使わない）。
    pub min_view_cosine: f64,
    /// 遮蔽チェック用の深度バッファのマス [px]（レベル 0）。
    pub depth_buffer_cell_px: usize,
    /// 深度バッファの最小深度より `margin_m + ratio * 深度` 以上奥なら遮蔽とみなす。
    pub occlusion_margin_m: f64,
    pub occlusion_margin_ratio: f64,

    // --- キーフレームの交代 ---
    /// K からこのフレーム数経過したら交代する（上限）。
    pub max_keyframe_age_frames: u64,
    /// 交代後、最低このフレーム数は同じ K を使う。
    pub min_keyframe_frames: u64,
    /// 見えているパッチの割合がこれ未満なら交代する。
    pub min_visible_ratio: f64,
    /// 画像の更新がこの回数連続で失敗したら、現在フレームから K を作り直す。
    pub max_consecutive_failures: u32,

    // --- 最適化 ---
    pub min_patches: usize,
    pub max_iterations_per_level: usize,
    /// Stddev of one patch, in grayscale units. Information is averaged over
    /// 81 pixels to avoid treating a correlated patch as 81 independent poses.
    pub patch_sigma: f64,
    pub huber_delta: f64,
    pub max_initial_patch_rmse: f64,
    pub max_final_rmse: f64,
    /// Total visual correction relative to the accepted LiDAR state.
    pub max_translation_correction_m: f64,
    pub max_rotation_correction_deg: f64,
}

impl Default for VisualAlignmentConfig {
    fn default() -> Self {
        Self {
            grid_size_px: 32,
            max_keyframe_points: 300,
            min_keyframe_points: 50,
            min_shi_tomasi_score: 2.0,
            min_depth_m: 0.5,
            max_depth_m: 40.0,
            min_view_cosine: 0.25,
            depth_buffer_cell_px: 8,
            occlusion_margin_m: 0.2,
            occlusion_margin_ratio: 0.05,
            max_keyframe_age_frames: 20,
            min_keyframe_frames: 3,
            min_visible_ratio: 0.6,
            max_consecutive_failures: 5,
            min_patches: 20,
            max_iterations_per_level: 5,
            patch_sigma: 5.0,
            huber_delta: 10.0,
            max_initial_patch_rmse: 45.0,
            max_final_rmse: 30.0,
            max_translation_correction_m: 0.10,
            max_rotation_correction_deg: 1.0,
        }
    }
}

impl VisualAlignmentConfig {
    fn valid(&self) -> bool {
        let finite_non_negative = |v: f64| v.is_finite() && v >= 0.0;
        self.grid_size_px > 0
            && self.depth_buffer_cell_px > 0
            && self.min_patches > 0
            && self.min_keyframe_points >= self.min_patches
            && self.max_keyframe_points >= self.min_keyframe_points
            && self.max_iterations_per_level > 0
            && self.max_keyframe_age_frames > 0
            && self.min_keyframe_frames <= self.max_keyframe_age_frames
            && self.max_consecutive_failures > 0
            && self.min_depth_m.is_finite()
            && self.min_depth_m > 0.0
            && self.max_depth_m.is_finite()
            && self.max_depth_m > self.min_depth_m
            && self.min_visible_ratio > 0.0
            && self.min_visible_ratio <= 1.0
            && self.min_view_cosine > 0.0
            && self.min_view_cosine <= 1.0
            && finite_non_negative(self.min_shi_tomasi_score)
            && finite_non_negative(self.occlusion_margin_m)
            && finite_non_negative(self.occlusion_margin_ratio)
            && [
                self.patch_sigma,
                self.huber_delta,
                self.max_initial_patch_rmse,
                self.max_final_rmse,
                self.max_translation_correction_m,
                self.max_rotation_correction_deg,
            ]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
    }
}

// ---------------------------------------------------------------------------
// 画像
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct GrayImage {
    width: usize,
    height: usize,
    pixels: Vec<f32>,
}

impl GrayImage {
    fn from_rgb(image: &RgbImage) -> Self {
        Self {
            width: image.width() as usize,
            height: image.height() as usize,
            pixels: image
                .pixels()
                .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
                .collect(),
        }
    }

    fn half(&self) -> Self {
        let width = self.width / 2;
        let height = self.height / 2;
        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let i = 2 * y * self.width + 2 * x;
                pixels.push(
                    (self.pixels[i]
                        + self.pixels[i + 1]
                        + self.pixels[i + self.width]
                        + self.pixels[i + self.width + 1])
                        * 0.25,
                );
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    fn mean(&self) -> f64 {
        if self.pixels.is_empty() {
            return 0.0;
        }
        self.pixels.iter().map(|&v| v as f64).sum::<f64>() / self.pixels.len() as f64
    }

    /// Bilinear intensity and its exact within-cell derivative. Subpixel
    /// coordinates are preserved throughout alignment (no integer rounding).
    fn sample(&self, p: &Vector2<f64>) -> Option<(f64, Vector2<f64>)> {
        if self.width < 2
            || self.height < 2
            || !p.iter().all(|v| v.is_finite())
            || p.x < 0.0
            || p.y < 0.0
            || p.x >= (self.width - 1) as f64
            || p.y >= (self.height - 1) as f64
        {
            return None;
        }
        let x = p.x.floor() as usize;
        let y = p.y.floor() as usize;
        let dx = p.x - x as f64;
        let dy = p.y - y as f64;
        let i = y * self.width + x;
        let a = self.pixels[i] as f64;
        let b = self.pixels[i + 1] as f64;
        let c = self.pixels[i + self.width] as f64;
        let d = self.pixels[i + self.width + 1] as f64;
        Some((
            (1.0 - dy) * ((1.0 - dx) * a + dx * b) + dy * ((1.0 - dx) * c + dx * d),
            Vector2::new(
                (1.0 - dy) * (b - a) + dy * (d - c),
                (1.0 - dx) * (c - a) + dx * (d - b),
            ),
        ))
    }

    /// サブピクセル中心の 9×9 パッチ全体を `sample` で読めるか。
    fn patch_fits(&self, center: &Vector2<f64>) -> bool {
        let r = PATCH_RADIUS as f64;
        center.iter().all(|v| v.is_finite())
            && center.x - r >= 0.0
            && center.y - r >= 0.0
            && center.x + r < self.width.saturating_sub(1) as f64
            && center.y + r < self.height.saturating_sub(1) as f64
    }
}

struct ImageLevel {
    image: GrayImage,
    camera: CameraIntrinsics,
}

fn pyramid(rgb: &RgbImage, camera: &CameraIntrinsics) -> Vec<ImageLevel> {
    let mut result = vec![ImageLevel {
        image: GrayImage::from_rgb(rgb),
        camera: camera.clone(),
    }];
    for _ in 1..PYRAMID_LEVELS {
        let last = result.last().unwrap();
        let image = last.image.half();
        let mut camera = last.camera.clone();
        camera.width = image.width as u32;
        camera.height = image.height as u32;
        camera.fx *= 0.5;
        camera.fy *= 0.5;
        // A downsampled pixel represents the center of its 2x2 source block.
        camera.cx = (camera.cx + 0.5) * 0.5 - 0.5;
        camera.cy = (camera.cy + 0.5) * 0.5 - 0.5;
        result.push(ImageLevel { image, camera });
    }
    result
}

/// レベル 0 の画素位置を、ピラミッドのレベル `level` の画素位置へ変換する。
/// `pyramid` の内部パラメータ（fx/2, (cx+0.5)/2-0.5）と厳密に一致する。
fn pixel_at_level(pixel0: &Vector2<f64>, level: usize) -> Vector2<f64> {
    let scale = 0.5f64.powi(level as i32);
    pixel0.map(|v| (v + 0.5) * scale - 0.5)
}

/// レベル 0 の画素 `pixel0` を中心とする 9×9 パッチが、全ピラミッド段で画像内に収まるか。
fn patch_fits_all_levels(images: &[ImageLevel], pixel0: &Vector2<f64>) -> bool {
    images
        .iter()
        .enumerate()
        .all(|(index, level)| level.image.patch_fits(&pixel_at_level(pixel0, index)))
}

/// 歪みを無視したピンホール投影で、明らかに画像外の点を安く除外する（余白付き）。
fn roughly_in_view(pc: &Point3<f64>, camera: &CameraIntrinsics) -> bool {
    let margin_x = 0.25 * camera.width as f64;
    let margin_y = 0.25 * camera.height as f64;
    let u = camera.fx * pc.x / pc.z + camera.cx;
    let v = camera.fy * pc.y / pc.z + camera.cy;
    u >= -margin_x
        && u <= camera.width as f64 + margin_x
        && v >= -margin_y
        && v <= camera.height as f64 + margin_y
}

// ---------------------------------------------------------------------------
// Shi-Tomasi スコア（構造テンソルの積分画像）
// ---------------------------------------------------------------------------

/// 構造テンソル Σ∇I∇Iᵀ の各成分 (gx², gx·gy, gy²) の積分画像。
/// 勾配は中心差分。9×9 窓の和を O(1) で取り出せる。
struct StructureTensorIntegral {
    width: usize,
    height: usize,
    sums: Vec<[f64; 3]>,
}

impl StructureTensorIntegral {
    fn new(image: &GrayImage) -> Self {
        let (w, h) = (image.width, image.height);
        let stride = w + 1;
        let mut sums = vec![[0.0f64; 3]; stride * (h + 1)];
        for y in 0..h {
            let mut row = [0.0f64; 3];
            for x in 0..w {
                let (gx, gy) = if x > 0 && x + 1 < w && y > 0 && y + 1 < h {
                    let i = y * w + x;
                    (
                        0.5 * (image.pixels[i + 1] - image.pixels[i - 1]) as f64,
                        0.5 * (image.pixels[i + w] - image.pixels[i - w]) as f64,
                    )
                } else {
                    (0.0, 0.0)
                };
                row[0] += gx * gx;
                row[1] += gx * gy;
                row[2] += gy * gy;
                let above = sums[y * stride + x + 1];
                sums[(y + 1) * stride + x + 1] =
                    [above[0] + row[0], above[1] + row[1], above[2] + row[2]];
            }
        }
        Self {
            width: w,
            height: h,
            sums,
        }
    }

    /// 最も近い整数画素を中心とする 9×9 窓の Shi-Tomasi スコア sqrt(λmin / 81)。
    /// 単位は勾配の RMS と同じ「階調/px」。窓が画像外にはみ出すなら None。
    fn score_at(&self, pixel: &Vector2<f64>) -> Option<f64> {
        if !pixel.iter().all(|v| v.is_finite()) {
            return None;
        }
        let r = PATCH_RADIUS as i64;
        let cx = pixel.x.round() as i64;
        let cy = pixel.y.round() as i64;
        let (x0, y0, x1, y1) = (cx - r, cy - r, cx + r + 1, cy + r + 1);
        if x0 < 0 || y0 < 0 || x1 > self.width as i64 || y1 > self.height as i64 {
            return None;
        }
        let stride = self.width + 1;
        let at = |x: i64, y: i64| self.sums[y as usize * stride + x as usize];
        let (s11, s01, s10, s00) = (at(x1, y1), at(x0, y1), at(x1, y0), at(x0, y0));
        let total: [f64; 3] = std::array::from_fn(|k| s11[k] - s01[k] - s10[k] + s00[k]);
        let (a, b, c) = (total[0], total[1], total[2]);
        let half_trace = 0.5 * (a + c);
        let lambda_min = half_trace - ((0.5 * (a - c)).powi(2) + b * b).sqrt();
        Some((lambda_min.max(0.0) / PATCH_PIXELS as f64).sqrt())
    }
}

// ---------------------------------------------------------------------------
// 遮蔽チェック用の深度バッファ
// ---------------------------------------------------------------------------

/// フレームの点群を画像に投影し、マスごとの最小深度を記録したもの。
struct DepthBuffer {
    cell_px: f64,
    cols: usize,
    rows: usize,
    depth: Vec<f32>,
}

impl DepthBuffer {
    fn build(
        points: &[Point3<f32>],
        camera_from_points: &Matrix4<f64>,
        camera: &CameraIntrinsics,
        cell_px: usize,
        min_depth_m: f64,
    ) -> Self {
        let cols = (camera.width as usize).div_ceil(cell_px).max(1);
        let rows = (camera.height as usize).div_ceil(cell_px).max(1);
        let mut depth = vec![f32::INFINITY; cols * rows];
        let (width, height) = (camera.width as f64, camera.height as f64);
        for point in points {
            if !point.coords.iter().all(|v| v.is_finite()) {
                continue;
            }
            let pc = camera_from_points.transform_point(&point.cast::<f64>());
            if pc.z.is_nan() || pc.z < min_depth_m {
                continue;
            }
            let Some((pixel, _)) = project_camera_point_subpixel(&pc, camera) else {
                continue;
            };
            if pixel.x < 0.0 || pixel.y < 0.0 || pixel.x >= width || pixel.y >= height {
                continue;
            }
            let cell = (pixel.y as usize / cell_px) * cols + pixel.x as usize / cell_px;
            depth[cell] = depth[cell].min(pc.z as f32);
        }
        Self {
            cell_px: cell_px as f64,
            cols,
            rows,
            depth,
        }
    }

    /// 周囲 3×3 マスの最小深度（点が 1 つもなければ None）。
    fn nearest_depth_around(&self, pixel: &Vector2<f64>) -> Option<f64> {
        if !pixel.iter().all(|v| v.is_finite()) || pixel.x < 0.0 || pixel.y < 0.0 {
            return None;
        }
        let cx = (pixel.x / self.cell_px) as usize;
        let cy = (pixel.y / self.cell_px) as usize;
        let mut nearest = f32::INFINITY;
        for y in cy.saturating_sub(1)..=(cy + 1).min(self.rows.saturating_sub(1)) {
            for x in cx.saturating_sub(1)..=(cx + 1).min(self.cols.saturating_sub(1)) {
                nearest = nearest.min(self.depth[y * self.cols + x]);
            }
        }
        nearest.is_finite().then_some(nearest as f64)
    }

    /// 深度 `depth` の点が、手前の点に隠れているか。点群が届いていない場所は判定できないので
    /// 隠れていない扱いにする。
    fn is_occluded(
        &self,
        pixel: &Vector2<f64>,
        depth: f64,
        config: &VisualAlignmentConfig,
    ) -> bool {
        self.nearest_depth_around(pixel).is_some_and(|near| {
            depth > near + config.occlusion_margin_m + config.occlusion_margin_ratio * near
        })
    }
}

// ---------------------------------------------------------------------------
// キーフレーム
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ReferencePatch {
    /// Raw grayscale values, always 9x9, row-major around the projected point.
    gray: [f32; PATCH_PIXELS],
    /// 各画素の光線と「点の法線を持つ平面」の交点（ワールド座標）。
    /// 中心・法線・K の姿勢から再計算できるが、毎回の計算を省くために保持する。
    world_samples: [Point3<f64>; PATCH_PIXELS],
}

struct KeyframePoint {
    /// GlobalMap の Planar セルの点を、その平面上へ射影した位置（ワールド座標）。
    world_point: Point3<f64>,
    /// 平面の法線（ワールド座標、K のカメラ側を向くように符号をそろえる）。
    normal: Vector3<f64>,
    shi_tomasi_score: f64,
    patches: [ReferencePatch; PYRAMID_LEVELS],
}

struct Keyframe {
    frame_index: u64,
    image_time: f64,
    /// 画像を撮った時刻のカメラ姿勢（確定値。以後変更しない）。
    camera_from_world: Matrix4<f64>,
    mean_intensity: f64,
    /// 方針どおり画像ピラミッドも保持する（点の選び直しや可視化に使える）。
    #[allow(dead_code)]
    pyramid: Vec<ImageLevel>,
    points: Vec<KeyframePoint>,
}

/// K の視覚点の候補（GlobalMap の 1 セル）。
#[derive(Clone, Copy)]
struct PointCandidate {
    world_point: Point3<f64>,
    normal: Vector3<f64>,
    camera_point: Point3<f64>,
    score: f64,
}

impl PointCandidate {
    /// スコアが高い → 手前 → 座標の順で比較（スレッド順に依存しない決定的な選択）。
    fn is_better_than(&self, other: &Self) -> bool {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.camera_point.z.total_cmp(&other.camera_point.z))
            .then_with(|| self.world_point.x.total_cmp(&other.world_point.x))
            .then_with(|| self.world_point.y.total_cmp(&other.world_point.y))
            .then_with(|| self.world_point.z.total_cmp(&other.world_point.z))
            .is_lt()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct KeyframeSelectionStats {
    /// 視野・深度・角度・画像端の条件を満たした Planar セルの数。
    candidates: usize,
    occluded: usize,
    low_score: usize,
}

impl std::ops::Add for KeyframeSelectionStats {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            candidates: self.candidates + other.candidates,
            occluded: self.occluded + other.occluded,
            low_score: self.low_score + other.low_score,
        }
    }
}

/// パッチの各画素を「`center` を通り法線 `normal` を持つ平面」と光線の交点として 3D 化し、
/// 3 段のピラミッドそれぞれで 9×9 を切り出す（すべてカメラ座標で与える）。
fn capture_plane_patches(
    images: &[ImageLevel],
    center: &Point3<f64>,
    normal: &Vector3<f64>,
    world_from_camera: &Matrix4<f64>,
    min_view_cosine: f64,
) -> Option<[ReferencePatch; PYRAMID_LEVELS]> {
    // 平面: n · X = n · center
    let plane_offset = normal.dot(&center.coords);
    let mut patches = Vec::with_capacity(PYRAMID_LEVELS);
    for level in images {
        let (pixel, _) = project_camera_point_subpixel(center, &level.camera)?;
        let mut gray = [0.0f32; PATCH_PIXELS];
        let mut world_samples = [Point3::origin(); PATCH_PIXELS];
        for y in -PATCH_RADIUS..=PATCH_RADIUS {
            for x in -PATCH_RADIUS..=PATCH_RADIUS {
                let i = ((y + PATCH_RADIUS) as usize) * PATCH_SIZE + (x + PATCH_RADIUS) as usize;
                let p = pixel + Vector2::new(x as f64, y as f64);
                let (value, _) = level.image.sample(&p)?;
                // 深度 Z=1 の点 = その画素の光線方向（歪み補正済み）
                let ray = unproject_pixel_at_depth(&p, 1.0, &level.camera)?.coords;
                let denom = normal.dot(&ray);
                if denom.abs() < min_view_cosine * ray.norm() {
                    return None;
                }
                let lambda = plane_offset / denom;
                if !(lambda.is_finite() && lambda > 0.0) {
                    return None;
                }
                gray[i] = value as f32;
                world_samples[i] = world_from_camera.transform_point(&Point3::from(ray * lambda));
            }
        }
        patches.push(ReferencePatch {
            gray,
            world_samples,
        });
    }
    patches.try_into().ok()
}

/// 位置合わせ済みの現在フレームから K を作る。
///
/// 1. 視野内（深度 min..max）の GlobalMap の Planar セルの点を K の画像に投影する
/// 2. 遮蔽チェック：K の点群の深度バッファより明らかに奥の点は捨てる
/// 3. 斜めすぎる面、レベル 2 で 9×9 が収まらない点は捨てる
/// 4. セルごとに Shi-Tomasi スコアが最も高い点を 1 つ採用（閾値未満は採用しない）
/// 5. スコア順に最大 `max_keyframe_points` 点、法線付き平面でパッチを切り出す
fn build_keyframe(
    config: &VisualAlignmentConfig,
    images: Vec<ImageLevel>,
    camera_from_world: Matrix4<f64>,
    frame_index: u64,
    image_time: f64,
    planar_map: &VoxelMap,
    depth_buffer: &DepthBuffer,
) -> Option<(Keyframe, KeyframeSelectionStats)> {
    let world_from_camera = camera_from_world.try_inverse()?;
    let camera_center = camera_center(&camera_from_world)?;
    let camera_rotation: Matrix3<f64> = camera_from_world.fixed_view::<3, 3>(0, 0).into_owned();
    let level0 = &images[0];
    let tensor = StructureTensorIntegral::new(&level0.image);
    let cols = level0.image.width.div_ceil(config.grid_size_px);
    let rows = level0.image.height.div_ceil(config.grid_size_px);

    let empty = || {
        (
            vec![None::<PointCandidate>; cols * rows],
            KeyframeSelectionStats::default(),
        )
    };
    let (best_per_cell, stats) = planar_map
        .par_iter()
        .fold(empty, |(mut best, mut stats), (_, cell)| {
            if cell.surface_status != SurfaceStatus::Planar {
                return (best, stats);
            }
            let Some(plane) = cell.surface_plane else {
                return (best, stats);
            };
            let mean = cell.mean.cast::<f64>();
            // まずセルの代表点で、深度と視野を安く判定する（大半のセルはここで落ちる）。
            // 平面への射影で動くのは数 cm なので、余白を少し取っておく。
            let mean_camera = camera_from_world.transform_point(&mean);
            if !(mean_camera.z >= config.min_depth_m - 0.1
                && mean_camera.z <= config.max_depth_m + 0.1
                && roughly_in_view(&mean_camera, &level0.camera))
            {
                return (best, stats);
            }
            let raw_normal = plane.normal.cast::<f64>();
            let norm = raw_normal.norm();
            if !(norm.is_finite() && norm > 1e-9) {
                return (best, stats);
            }
            let mut normal = raw_normal / norm;
            let offset = plane.d as f64 / norm;
            // セルの点を平面上へ射影する（深度をより正確にするため）。
            let world_point = mean - normal * (normal.dot(&mean.coords) + offset);
            let to_camera = camera_center - world_point;
            let distance = to_camera.norm();
            if distance.is_nan() || distance <= 0.0 {
                return (best, stats);
            }
            if normal.dot(&to_camera) < 0.0 {
                normal = -normal;
            }
            if normal.dot(&to_camera) < config.min_view_cosine * distance {
                return (best, stats);
            }
            let camera_point = camera_from_world.transform_point(&world_point);
            if camera_point.z < config.min_depth_m
                || camera_point.z > config.max_depth_m
                || !roughly_in_view(&camera_point, &level0.camera)
            {
                return (best, stats);
            }
            let Some((pixel, _)) = project_camera_point_subpixel(&camera_point, &level0.camera)
            else {
                return (best, stats);
            };
            if !patch_fits_all_levels(&images, &pixel) {
                return (best, stats);
            }
            stats.candidates += 1;
            if depth_buffer.is_occluded(&pixel, camera_point.z, config) {
                stats.occluded += 1;
                return (best, stats);
            }
            let score = tensor.score_at(&pixel).unwrap_or(0.0);
            if score < config.min_shi_tomasi_score {
                stats.low_score += 1;
                return (best, stats);
            }
            let candidate = PointCandidate {
                world_point,
                normal,
                camera_point,
                score,
            };
            let index = (pixel.y as usize / config.grid_size_px) * cols
                + pixel.x as usize / config.grid_size_px;
            if best[index]
                .as_ref()
                .is_none_or(|current| candidate.is_better_than(current))
            {
                best[index] = Some(candidate);
            }
            (best, stats)
        })
        .reduce(empty, |(mut left, left_stats), (right, right_stats)| {
            for (l, r) in left.iter_mut().zip(right) {
                if let Some(r) = r
                    && l.as_ref().is_none_or(|current| r.is_better_than(current))
                {
                    *l = Some(r);
                }
            }
            (left, left_stats + right_stats)
        });

    let mut selected: Vec<PointCandidate> = best_per_cell.into_iter().flatten().collect();
    selected.sort_by(|a, b| {
        if a.is_better_than(b) {
            std::cmp::Ordering::Less
        } else if b.is_better_than(a) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });

    // スコア順に、足りない分だけを並列に切り出す（切り出しに失敗した点は次の候補で補う）。
    let mut points = Vec::with_capacity(config.max_keyframe_points.min(selected.len()));
    let mut remaining = selected.as_slice();
    while points.len() < config.max_keyframe_points && !remaining.is_empty() {
        let batch_len = (config.max_keyframe_points - points.len()).min(remaining.len());
        let (batch, rest) = remaining.split_at(batch_len);
        remaining = rest;
        let captured: Vec<Option<KeyframePoint>> = batch
            .par_iter()
            .map(|candidate| {
                let patches = capture_plane_patches(
                    &images,
                    &candidate.camera_point,
                    &(camera_rotation * candidate.normal),
                    &world_from_camera,
                    config.min_view_cosine,
                )?;
                Some(KeyframePoint {
                    world_point: candidate.world_point,
                    normal: candidate.normal,
                    shi_tomasi_score: candidate.score,
                    patches,
                })
            })
            .collect();
        points.extend(captured.into_iter().flatten());
    }

    let mean_intensity = images[0].image.mean();
    Some((
        Keyframe {
            frame_index,
            image_time,
            camera_from_world,
            mean_intensity,
            pyramid: images,
            points,
        },
        stats,
    ))
}

struct Visibility {
    visible: Vec<usize>,
    occluded: usize,
}

/// 姿勢 `camera_from_world` のカメラ中心（ワールド座標）。
fn camera_center(camera_from_world: &Matrix4<f64>) -> Option<Point3<f64>> {
    let world_from_camera = camera_from_world.try_inverse()?;
    Some(Point3::from(
        world_from_camera.fixed_view::<3, 1>(0, 3).into_owned(),
    ))
}

/// K の視覚点のうち、`camera_from_world` の姿勢で今の画像（レベル 0）に 9×9 が収まり、
/// 面を斜めすぎない角度で表側から見ていて、今の点群で遮蔽されていないもの。
fn keyframe_visibility(
    config: &VisualAlignmentConfig,
    keyframe: &Keyframe,
    camera_from_world: &Matrix4<f64>,
    level0: &ImageLevel,
    depth_buffer: &DepthBuffer,
) -> Visibility {
    let mut visible = Vec::with_capacity(keyframe.points.len());
    let mut occluded = 0;
    let Some(center) = camera_center(camera_from_world) else {
        return Visibility { visible, occluded };
    };
    for (index, point) in keyframe.points.iter().enumerate() {
        let pc = camera_from_world.transform_point(&point.world_point);
        if pc.z < config.min_depth_m {
            continue;
        }
        let to_camera = center - point.world_point;
        if point.normal.dot(&to_camera) < config.min_view_cosine * to_camera.norm() {
            continue;
        }
        let Some((pixel, _)) = project_camera_point_subpixel(&pc, &level0.camera) else {
            continue;
        };
        if !level0.image.patch_fits(&pixel) {
            continue;
        }
        if depth_buffer.is_occluded(&pixel, pc.z, config) {
            occluded += 1;
            continue;
        }
        visible.push(index);
    }
    Visibility { visible, occluded }
}

// ---------------------------------------------------------------------------
// 観測モデル（画像時刻のカメラ姿勢と、その 15 次元状態に対する微分）
// ---------------------------------------------------------------------------

struct CameraDerivative {
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
}

struct CameraLinearization {
    pose: Matrix4<f64>,
    derivatives: [CameraDerivative; ERROR_STATE_DIM],
}

/// The filter's state stays at t_start. Only a temporary state is integrated to
/// t_image. This avoids applying an image-time correction as a frame-start pose.
struct ImageObservation<'a> {
    filter: &'a Eskf,
    imu: &'a [IMU],
    t_start: f64,
    t_image: f64,
    camera_from_lidar: &'a Matrix4<f64>,
}

impl ImageObservation<'_> {
    fn camera_pose(&self, state: &NavState) -> Option<Matrix4<f64>> {
        let mut temporary = self.filter.clone();
        temporary.state = *state;
        let at_image = temporary.predict_nominal(self.imu, self.t_start, self.t_image);
        let world_from_lidar = temporary.lidar_pose_of(&at_image);
        let pose = self.camera_from_lidar * world_from_lidar.try_inverse()?;
        pose.iter().all(|v| v.is_finite()).then_some(pose)
    }

    fn linearize(&self, state: &NavState) -> Option<CameraLinearization> {
        let pose = self.camera_pose(state)?;
        let mut derivatives = std::array::from_fn(|_| CameraDerivative {
            rotation: Matrix3::zeros(),
            translation: Vector3::zeros(),
        });
        // Only the small IMU/pose mapping is differentiated numerically (30
        // integrations per iteration). Per-pixel projection and image derivatives
        // are analytic. The same perturbation convention as NavState::boxplus
        // includes velocity, biases and the IMU-camera lever arm correctly.
        for (col, derivative) in derivatives.iter_mut().enumerate() {
            let epsilon = 1e-5;
            let mut dx = Vector15::zeros();
            dx[col] = epsilon;
            let plus = self.camera_pose(&state.boxplus(&dx))?;
            let minus = self.camera_pose(&state.boxplus(&(-dx)))?;
            let difference = (plus - minus) / (2.0 * epsilon);
            derivative.rotation = difference.fixed_view::<3, 3>(0, 0).into_owned();
            derivative.translation = difference.fixed_view::<3, 1>(0, 3).into_owned();
        }
        Some(CameraLinearization { pose, derivatives })
    }
}

struct PatchEvaluation {
    residuals: [f64; PATCH_PIXELS],
    jacobians: [Vector15; PATCH_PIXELS],
}

/// K のパッチの 81 個の 3D 点を今の姿勢で投影し、その位置の輝度との差を残差にする。
/// 斜めの面では、投影した 81 点の並びが今の画像上で自動的に歪む。
///
/// Remove a per-patch constant brightness offset from both residuals AND
/// Jacobians. This tolerates additive brightness shifts, not arbitrary exposure
/// gain, white balance or clipped pixels.
fn evaluate_patch(
    patch: &ReferencePatch,
    level: &ImageLevel,
    pose: &Matrix4<f64>,
    linearization: Option<&CameraLinearization>,
) -> Option<PatchEvaluation> {
    let mut residuals = [0.0; PATCH_PIXELS];
    let mut jacobians = [Vector15::zeros(); PATCH_PIXELS];
    let mut mean_r = 0.0;
    let mut mean_j = Vector15::zeros();
    for i in 0..PATCH_PIXELS {
        let world = patch.world_samples[i];
        let camera_point = pose.transform_point(&world);
        let (pixel, projection_jacobian) =
            project_camera_point_subpixel(&camera_point, &level.camera)?;
        let (gray, gradient) = level.image.sample(&pixel)?;
        residuals[i] = gray - patch.gray[i] as f64;
        mean_r += residuals[i];
        if let Some(linearization) = linearization {
            let image_wrt_point = projection_jacobian.transpose() * gradient;
            for (col, derivative) in linearization.derivatives.iter().enumerate() {
                let dp = derivative.rotation * world.coords + derivative.translation;
                jacobians[i][col] = image_wrt_point.dot(&dp);
            }
            mean_j += jacobians[i];
        }
    }
    mean_r /= PATCH_PIXELS as f64;
    mean_j /= PATCH_PIXELS as f64;
    for i in 0..PATCH_PIXELS {
        residuals[i] -= mean_r;
        jacobians[i] -= mean_j;
    }
    Some(PatchEvaluation {
        residuals,
        jacobians,
    })
}

fn huber(r: f64, delta: f64) -> (f64, f64) {
    if r.abs() <= delta {
        (1.0, r * r)
    } else {
        (delta / r.abs(), 2.0 * delta * r.abs() - delta * delta)
    }
}

struct VisualSystem {
    information: Matrix15,
    rhs: Vector15,
    cost: f64,
    squared_error: f64,
    pixels: usize,
}

impl VisualSystem {
    fn rmse(&self) -> f64 {
        (self.squared_error / self.pixels.max(1) as f64).sqrt()
    }
}

fn build_system(
    config: &VisualAlignmentConfig,
    keyframe: &Keyframe,
    indices: &[usize],
    level_index: usize,
    level: &ImageLevel,
    pose: &Matrix4<f64>,
    linearization: Option<&CameraLinearization>,
) -> Option<VisualSystem> {
    let mut system = VisualSystem {
        information: Matrix15::zeros(),
        rhs: Vector15::zeros(),
        cost: 0.0,
        squared_error: 0.0,
        pixels: 0,
    };
    let information_per_pixel = 1.0 / (config.patch_sigma.powi(2) * PATCH_PIXELS as f64);
    // A fixed patch/pixel set is used across each level and its line search.
    // If a candidate loses a patch, reject it instead of rewarding lost data.
    for &index in indices {
        let evaluation = evaluate_patch(
            &keyframe.points[index].patches[level_index],
            level,
            pose,
            linearization,
        )?;
        for i in 0..PATCH_PIXELS {
            let residual = evaluation.residuals[i];
            let (weight, loss) = huber(residual, config.huber_delta);
            system.cost += loss * information_per_pixel;
            system.squared_error += residual * residual;
            system.pixels += 1;
            if linearization.is_some() {
                let j = evaluation.jacobians[i];
                let w = weight * information_per_pixel;
                system.rhs -= j * (w * residual);
                for row in 0..ERROR_STATE_DIM {
                    for col in row..ERROR_STATE_DIM {
                        system.information[(row, col)] += w * j[row] * j[col];
                    }
                }
            }
        }
    }
    for row in 1..ERROR_STATE_DIM {
        for col in 0..row {
            system.information[(row, col)] = system.information[(col, row)];
        }
    }
    Some(system)
}

/// K の見えているパッチに今の画像を合わせ、採用できれば ESKF を更新する。
/// 返り値は「画像の更新が採用された（または補正不要だった）か」。
fn optimize(
    config: &VisualAlignmentConfig,
    observation: &ImageObservation<'_>,
    images: &[ImageLevel],
    keyframe: &Keyframe,
    indices: &[usize],
    filter: &mut Eskf,
    report: &mut VisualAlignmentLog,
) -> bool {
    let prior_state = observation.filter.state;
    let prior_cov = (observation.filter.cov + observation.filter.cov.transpose()) * 0.5;
    let Some(prior_cholesky) = Cholesky::new(prior_cov) else {
        report.status = "invalid_prior_covariance".into();
        return false;
    };
    let prior_info = prior_cholesky.inverse();
    let prior_cost = |state: &NavState| {
        let difference = state.boxminus(&prior_state);
        difference.dot(&(prior_info * difference))
    };
    let mut state = prior_state;
    let mut finest_indices = Vec::new();
    for level_index in (0..PYRAMID_LEVELS).rev() {
        let level = &images[level_index];
        let Some(pose) = observation.camera_pose(&state) else {
            continue;
        };
        let active: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|&i| {
                evaluate_patch(&keyframe.points[i].patches[level_index], level, &pose, None)
                    .is_some_and(|p| {
                        (p.residuals.iter().map(|r| r * r).sum::<f64>() / PATCH_PIXELS as f64)
                            .sqrt()
                            <= config.max_initial_patch_rmse
                    })
            })
            .collect();
        if active.len() < config.min_patches {
            continue;
        }
        if level_index == 0 {
            finest_indices = active.clone();
        }
        for _ in 0..config.max_iterations_per_level {
            let Some(linearization) = observation.linearize(&state) else {
                break;
            };
            let Some(system) = build_system(
                config,
                keyframe,
                &active,
                level_index,
                level,
                &linearization.pose,
                Some(&linearization),
            ) else {
                break;
            };
            let current_cost = system.cost + prior_cost(&state);
            let a = system.information + prior_info;
            let rhs = system.rhs - prior_info * state.boxminus(&prior_state);
            let Some(cholesky) = Cholesky::new((a + a.transpose()) * 0.5) else {
                break;
            };
            let dx = cholesky.solve(&rhs);
            if !dx.iter().all(|v| v.is_finite()) {
                break;
            }
            let mut accepted = None;
            for backtrack in 0..8 {
                let step = dx * 0.5f64.powi(backtrack);
                let candidate = state.boxplus(&step);
                let (translation, rotation) =
                    correction(&prior_state, &candidate, observation.filter);
                if translation > config.max_translation_correction_m
                    || rotation > config.max_rotation_correction_deg
                {
                    continue;
                }
                let Some(candidate_pose) = observation.camera_pose(&candidate) else {
                    continue;
                };
                let Some(candidate_system) = build_system(
                    config,
                    keyframe,
                    &active,
                    level_index,
                    level,
                    &candidate_pose,
                    None,
                ) else {
                    continue;
                };
                let cost = candidate_system.cost + prior_cost(&candidate);
                if cost.is_finite() && cost <= current_cost + 1e-10 {
                    accepted = Some((candidate, step));
                    break;
                }
            }
            let Some((candidate, step)) = accepted else {
                break;
            };
            state = candidate;
            report.iterations += 1;
            if step.fixed_rows::<3>(0).norm() < 1e-6 && step.fixed_rows::<3>(3).norm() < 1e-5 {
                break;
            }
        }
    }
    if finest_indices.len() < config.min_patches {
        report.status = "insufficient_fine_patches".into();
        return false;
    }
    let Some(initial_pose) = observation.camera_pose(&prior_state) else {
        report.status = "invalid_camera_pose".into();
        return false;
    };
    let Some(final_linearization) = observation.linearize(&state) else {
        report.status = "invalid_camera_pose".into();
        return false;
    };
    let Some(initial) = build_system(
        config,
        keyframe,
        &finest_indices,
        0,
        &images[0],
        &initial_pose,
        None,
    ) else {
        report.status = "reference_support_lost".into();
        return false;
    };
    let Some(final_system) = build_system(
        config,
        keyframe,
        &finest_indices,
        0,
        &images[0],
        &final_linearization.pose,
        Some(&final_linearization),
    ) else {
        report.status = "reference_support_lost".into();
        return false;
    };
    report.matched_patches = finest_indices.len();
    report.residual_pixels = final_system.pixels;
    report.initial_rmse = Some(initial.rmse());
    report.final_rmse = Some(final_system.rmse());
    let (translation, rotation) = correction(&prior_state, &state, observation.filter);
    // 棄却理由をステータスに残す（どのしきい値が効いているかをログから判断するため）。
    let rejection_reason = if !final_system.cost.is_finite() {
        Some("non_finite_cost")
    } else if final_system.rmse() > config.max_final_rmse {
        Some("final_rmse")
    } else if final_system.cost + prior_cost(&state) > initial.cost + 1e-9 {
        Some("cost_increased")
    } else if final_system.squared_error > initial.squared_error + 1e-6 {
        Some("error_increased")
    } else if translation > config.max_translation_correction_m {
        Some("translation_limit")
    } else if rotation > config.max_rotation_correction_deg {
        Some("rotation_limit")
    } else {
        None
    };
    if let Some(reason) = rejection_reason {
        report.status = format!("rejected:{reason}");
        return false;
    }
    if state.boxminus(&prior_state).norm() < 1e-10 {
        // In particular, don't shrink P again on an exact self-consistent
        // replay with no accepted correction.
        report.status = "already_aligned".into();
        return true;
    }
    let a = final_system.information + prior_info;
    let Some(cholesky) = Cholesky::new((a + a.transpose()) * 0.5) else {
        report.status = "invalid_posterior_covariance".into();
        return false;
    };
    let covariance = cholesky.inverse();
    if !covariance.iter().all(|v| v.is_finite()) {
        report.status = "invalid_posterior_covariance".into();
        return false;
    }
    // Pyramids revisit the same evidence. Commit state and covariance ONCE,
    // using only the final full-resolution information and the LiDAR prior.
    filter.accept_update(state, covariance);
    report.update_applied = true;
    report.translation_correction_m = translation;
    report.rotation_correction_deg = rotation;
    report.status = "accepted".into();
    true
}

fn correction(prior: &NavState, state: &NavState, filter: &Eskf) -> (f64, f64) {
    let prior_pose = filter.lidar_pose_of(prior);
    let pose = filter.lidar_pose_of(state);
    let translation = (pose.fixed_view::<3, 1>(0, 3) - prior_pose.fixed_view::<3, 1>(0, 3)).norm();
    let rotation = (state.rot * prior.rot.inverse()).angle().to_degrees();
    (translation, rotation)
}

// ---------------------------------------------------------------------------
// フレームごとの処理
// ---------------------------------------------------------------------------

/// 1 フレーム分の入力。
pub struct VisualFrame<'a> {
    pub imu: &'a [IMU],
    /// LiDAR フレームの開始時刻（ESKF の状態の時刻）。
    pub frame_start_time: f64,
    pub image_time: f64,
    pub frame_index: u64,
    pub rgb: &'a RgbImage,
    pub camera: &'a CameraIntrinsics,
    pub camera_from_lidar: &'a Matrix4<f64>,
    /// このフレームの点群（deskew 済み、フレーム開始時刻の LiDAR 座標）。遮蔽チェック用。
    pub frame_points: &'a [Point3<f32>],
    /// 視覚点の候補にする GlobalMap（Planar セルだけを使う）。
    pub planar_map: &'a VoxelMap,
}

#[derive(Default)]
pub struct VisualAligner {
    pub config: VisualAlignmentConfig,
    keyframe: Option<Keyframe>,
    consecutive_failures: u32,
    last_image_time: Option<f64>,
}

impl VisualAligner {
    pub fn new(config: VisualAlignmentConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }

    /// 今のキーフレームを作ったフレーム番号。
    pub fn keyframe_frame_index(&self) -> Option<u64> {
        self.keyframe.as_ref().map(|keyframe| keyframe.frame_index)
    }

    /// LiDAR 更新が採用されたフレームでだけ呼ぶ。色付けとマップ更新より前に呼ぶ。
    ///
    /// 順番: 今の K に対して位置合わせ → 状態を更新 → 交代の判定 →
    /// 交代するなら同じ画像と更新後の姿勢で新しい K を作る。
    pub fn process_frame(
        &mut self,
        filter: &mut Eskf,
        frame: &VisualFrame<'_>,
    ) -> VisualAlignmentLog {
        let config = self.config.clone();
        let mut report = VisualAlignmentLog::default();
        if !config.valid() {
            report.status = "invalid_config".into();
            return report;
        }
        if self
            .last_image_time
            .is_some_and(|last| frame.image_time <= last + 1e-9)
        {
            report.status = "reused_or_older_image".into();
            return report;
        }
        let imu = frame.imu;
        if !frame.frame_start_time.is_finite()
            || !frame.image_time.is_finite()
            || frame.image_time < frame.frame_start_time
            || (frame.image_time > frame.frame_start_time
                && !(imu
                    .first()
                    .is_some_and(|s| s.timestamp <= frame.frame_start_time)
                    && imu.last().is_some_and(|s| s.timestamp >= frame.image_time)))
            || frame.rgb.width() != frame.camera.width
            || frame.rgb.height() != frame.camera.height
            || frame.rgb.width() < 48
            || frame.rgb.height() < 48
        {
            report.status = "invalid_image_time_or_dimensions".into();
            return report;
        }
        self.last_image_time = Some(frame.image_time);

        let images = pyramid(frame.rgb, frame.camera);
        let prior = filter.clone();
        let observation = ImageObservation {
            filter: &prior,
            imu,
            t_start: frame.frame_start_time,
            t_image: frame.image_time,
            camera_from_lidar: frame.camera_from_lidar,
        };
        let Some(initial_camera_pose) = observation.camera_pose(&prior.state) else {
            report.status = "invalid_camera_pose".into();
            return report;
        };

        // --- 1. 今の K に対する位置合わせ ---
        let mut visual_ok = false;
        if let Some(keyframe) = self.keyframe.as_ref() {
            report.keyframe_frame_index = Some(keyframe.frame_index);
            report.keyframe_age_frames = frame.frame_index.saturating_sub(keyframe.frame_index);
            report.keyframe_points = keyframe.points.len();
            if let (Some(keyframe_center), Some(current_center)) = (
                camera_center(&keyframe.camera_from_world),
                camera_center(&initial_camera_pose),
            ) {
                report.keyframe_baseline_m = Some((current_center - keyframe_center).norm());
            }
            let current_mean = images[0].image.mean();
            if current_mean > 0.0 && keyframe.mean_intensity > 0.0 {
                report.brightness_log_ratio = Some((current_mean / keyframe.mean_intensity).ln());
            }
            // N の点群で遮蔽チェックするための深度バッファ（LiDAR 更新後の姿勢）。
            let depth_buffer = DepthBuffer::build(
                frame.frame_points,
                &(initial_camera_pose * prior.lidar_pose()),
                &images[0].camera,
                config.depth_buffer_cell_px,
                config.min_depth_m,
            );
            let visibility = keyframe_visibility(
                &config,
                keyframe,
                &initial_camera_pose,
                &images[0],
                &depth_buffer,
            );
            report.visible_patches = visibility.visible.len();
            report.occluded_patches = visibility.occluded;
            report.matched_patches = visibility.visible.len();
            if visibility.visible.len() >= config.min_patches {
                visual_ok = optimize(
                    &config,
                    &observation,
                    &images,
                    keyframe,
                    &visibility.visible,
                    filter,
                    &mut report,
                );
            } else {
                report.status = "insufficient_patches".into();
            }
            // 交代判定用の「見えている割合」は位置合わせ後の姿勢で数える。
            let final_pose = observation
                .camera_pose(&filter.state)
                .unwrap_or(initial_camera_pose);
            let after =
                keyframe_visibility(&config, keyframe, &final_pose, &images[0], &depth_buffer);
            if !keyframe.points.is_empty() {
                report.visible_ratio =
                    Some(after.visible.len() as f64 / keyframe.points.len() as f64);
            }
            if visual_ok {
                self.consecutive_failures = 0;
            } else {
                self.consecutive_failures = self.consecutive_failures.saturating_add(1);
            }
        } else {
            report.status = "no_keyframe".into();
        }

        // --- 2. 交代の判定 ---
        let reason = match self.keyframe.as_ref() {
            None => Some("bootstrap"),
            Some(_) => {
                let age = report.keyframe_age_frames;
                if self.consecutive_failures >= config.max_consecutive_failures {
                    Some("reset_after_failures")
                } else if visual_ok && age >= config.min_keyframe_frames {
                    if age >= config.max_keyframe_age_frames {
                        Some("max_age")
                    } else if report
                        .visible_ratio
                        .is_some_and(|ratio| ratio < config.min_visible_ratio)
                    {
                        Some("low_visibility")
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        };

        // --- 3. 新しい K を作る（同じ画像・更新後の状態から求めた画像時刻の姿勢） ---
        if let Some(reason) = reason {
            match observation.camera_pose(&filter.state) {
                Some(camera_from_world) => {
                    let depth_buffer = DepthBuffer::build(
                        frame.frame_points,
                        &(camera_from_world * filter.lidar_pose()),
                        &images[0].camera,
                        config.depth_buffer_cell_px,
                        config.min_depth_m,
                    );
                    match build_keyframe(
                        &config,
                        images,
                        camera_from_world,
                        frame.frame_index,
                        frame.image_time,
                        frame.planar_map,
                        &depth_buffer,
                    ) {
                        Some((keyframe, stats)) => {
                            report.keyframe_candidates = stats.candidates;
                            report.keyframe_candidates_occluded = stats.occluded;
                            report.new_keyframe_points = keyframe.points.len();
                            if keyframe.points.len() >= config.min_keyframe_points {
                                log::debug!(
                                    "visual keyframe {} ({reason}): points={}, candidates={}, \
                                     occluded={}, low_score={}, image_time={:.6}, \
                                     mean_shi_tomasi={:.2}",
                                    keyframe.frame_index,
                                    keyframe.points.len(),
                                    stats.candidates,
                                    stats.occluded,
                                    stats.low_score,
                                    keyframe.image_time,
                                    keyframe
                                        .points
                                        .iter()
                                        .map(|p| p.shi_tomasi_score)
                                        .sum::<f64>()
                                        / keyframe.points.len() as f64,
                                );
                                report.keyframe_event = format!("created:{reason}");
                                self.keyframe = Some(keyframe);
                                self.consecutive_failures = 0;
                            } else {
                                report.keyframe_event = format!("skipped:{reason}:too_few_points");
                            }
                        }
                        None => {
                            report.keyframe_event = format!("skipped:{reason}:invalid_camera_pose");
                        }
                    }
                }
                None => report.keyframe_event = format!("skipped:{reason}:invalid_camera_pose"),
            }
        }
        report.consecutive_failures = self.consecutive_failures;
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eskf::{EskfConfig, ImuModel, ImuNoise, InitialStd};
    use crate::voxel_map::{SurfacePlane, VoxelCell, voxel_key};
    use image::Rgb;
    use nalgebra::UnitQuaternion;

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

    fn filter() -> Eskf {
        Eskf::new(
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
        )
    }

    fn texture(x: f64, y: f64) -> u8 {
        (128.0
            + 41.0 * (7.1 * x + 2.3 * y).sin()
            + 34.0 * (3.7 * x - 8.2 * y).cos()
            + 22.0 * (13.3 * x + 11.4 * y).sin())
        .round()
        .clamp(0.0, 255.0) as u8
    }

    /// ワールドの平面 z=5 に貼った模様を、`pose`（world_from_camera）から撮った画像。
    fn render_plane(camera: &CameraIntrinsics, pose: Matrix4<f64>, offset: i16) -> RgbImage {
        RgbImage::from_fn(camera.width, camera.height, |u, v| {
            let direction_camera = Vector3::new(
                (u as f64 - camera.cx) / camera.fx,
                (v as f64 - camera.cy) / camera.fy,
                1.0,
            );
            let direction = pose.fixed_view::<3, 3>(0, 0) * direction_camera;
            let origin = pose.fixed_view::<3, 1>(0, 3);
            let t = (5.0 - origin.z) / direction.z;
            let point = origin + direction * t;
            let value = (texture(point.x, point.y) as i16 + offset).clamp(0, 255) as u8;
            Rgb([value, value, value])
        })
    }

    /// カメラ（= LiDAR）座標で深度 `depth` に並ぶ点群。
    fn frame_points(camera: &CameraIntrinsics, depth: f64) -> Vec<Point3<f32>> {
        let mut result = Vec::new();
        for v in (0..camera.height).step_by(4) {
            for u in (0..camera.width).step_by(4) {
                result.push(
                    unproject_pixel_at_depth(
                        &Vector2::new(u as f64 + 0.2, v as f64 + 0.3),
                        depth,
                        camera,
                    )
                    .unwrap()
                    .cast(),
                );
            }
        }
        result
    }

    /// ワールド平面 z=`z` 上の、成熟した Planar セルだけからなる GlobalMap。
    /// 法線はわざとカメラと逆向き (+z) にしておき、符号合わせも確認する。
    fn planar_map(z: f64, half_x: f64, half_y: f64, spacing: f64) -> VoxelMap {
        let mut map = VoxelMap::default();
        let nx = (half_x / spacing).round() as i32;
        let ny = (half_y / spacing).round() as i32;
        for iy in -ny..ny {
            for ix in -nx..nx {
                let point = Point3::new(
                    ((ix as f64 + 0.5) * spacing) as f32,
                    ((iy as f64 + 0.5) * spacing) as f32,
                    z as f32,
                );
                let key = voxel_key(&point, spacing as f32);
                let mut cell = VoxelCell::from_key(&key, spacing as f32, point, 0);
                cell.sample_count = 3;
                cell.observed_frames = 3;
                cell.surface_status = SurfaceStatus::Planar;
                cell.surface_plane = Some(SurfacePlane {
                    normal: Vector3::new(0.0, 0.0, 1.0),
                    d: -z as f32,
                    rmse_m: 0.005,
                    neighbor_count: 25,
                    inlier_count: 25,
                });
                map.insert(key, cell);
            }
        }
        map
    }

    /// ワールドの平面 normal·X = offset に貼った模様を `pose`（world_from_camera）から撮った画像。
    fn render_world_plane(
        camera: &CameraIntrinsics,
        pose: &Matrix4<f64>,
        normal: &Vector3<f64>,
        offset: f64,
    ) -> RgbImage {
        RgbImage::from_fn(camera.width, camera.height, |u, v| {
            let direction_camera = Vector3::new(
                (u as f64 - camera.cx) / camera.fx,
                (v as f64 - camera.cy) / camera.fy,
                1.0,
            );
            let direction = pose.fixed_view::<3, 3>(0, 0) * direction_camera;
            let origin: Vector3<f64> = pose.fixed_view::<3, 1>(0, 3).into_owned();
            let t = (offset - normal.dot(&origin)) / normal.dot(&direction);
            let point = origin + direction * t;
            let value = texture(point.x, point.y + 0.7 * point.z);
            Rgb([value, value, value])
        })
    }

    /// 同じ平面を LiDAR（= カメラ）座標で観測した点群。
    fn world_plane_points(
        camera: &CameraIntrinsics,
        pose: &Matrix4<f64>,
        normal: &Vector3<f64>,
        offset: f64,
    ) -> Vec<Point3<f32>> {
        let camera_from_world = pose.try_inverse().unwrap();
        let mut result = Vec::new();
        for v in (0..camera.height).step_by(4) {
            for u in (0..camera.width).step_by(4) {
                let direction_camera = Vector3::new(
                    (u as f64 - camera.cx) / camera.fx,
                    (v as f64 - camera.cy) / camera.fy,
                    1.0,
                );
                let direction = pose.fixed_view::<3, 3>(0, 0) * direction_camera;
                let origin: Vector3<f64> = pose.fixed_view::<3, 1>(0, 3).into_owned();
                let t = (offset - normal.dot(&origin)) / normal.dot(&direction);
                let world = Point3::from(origin + direction * t);
                result.push(camera_from_world.transform_point(&world).cast());
            }
        }
        result
    }

    /// 平面上の格子点（origin + u·e1 + v·e2）を Planar セルにした GlobalMap。
    /// セルに記録する法線は `stored_normal`（わざと間違った向きも渡せる）。
    fn plane_map(
        origin: &Point3<f64>,
        e1: &Vector3<f64>,
        e2: &Vector3<f64>,
        (u_min, u_max, v_min, v_max): (f64, f64, f64, f64),
        spacing: f64,
        stored_normal: &Vector3<f64>,
    ) -> VoxelMap {
        let mut map = VoxelMap::default();
        let mut u = u_min;
        while u < u_max {
            let mut v = v_min;
            while v < v_max {
                let p = origin + e1 * u + e2 * v;
                let point = p.cast::<f32>();
                let key = voxel_key(&point, 0.02);
                let mut cell = VoxelCell::from_key(&key, 0.02, point, 0);
                cell.sample_count = 3;
                cell.observed_frames = 3;
                cell.surface_status = SurfaceStatus::Planar;
                cell.surface_plane = Some(SurfacePlane {
                    normal: stored_normal.cast(),
                    d: -stored_normal.dot(&p.coords) as f32,
                    rmse_m: 0.005,
                    neighbor_count: 25,
                    inlier_count: 25,
                });
                map.insert(key, cell);
                v += spacing;
            }
            u += spacing;
        }
        map
    }

    fn config() -> VisualAlignmentConfig {
        VisualAlignmentConfig {
            grid_size_px: 24,
            min_shi_tomasi_score: 1.0,
            max_translation_correction_m: 0.15,
            max_rotation_correction_deg: 2.0,
            max_iterations_per_level: 8,
            ..Default::default()
        }
    }

    /// 交代ロジックだけを確認するテスト用（点数・寿命・失敗回数を小さくする）。
    fn small_logic_config() -> VisualAlignmentConfig {
        VisualAlignmentConfig {
            max_keyframe_points: 60,
            min_keyframe_points: 30,
            max_keyframe_age_frames: 6,
            min_keyframe_frames: 2,
            max_consecutive_failures: 3,
            max_iterations_per_level: 4,
            ..config()
        }
    }

    /// テスト用に 1 フレーム分を処理する（カメラ = LiDAR、レバーアームなし）。
    fn run(
        aligner: &mut VisualAligner,
        filter: &mut Eskf,
        frame_index: u64,
        time: f64,
        image: &RgbImage,
        points: &[Point3<f32>],
        map: &VoxelMap,
    ) -> VisualAlignmentLog {
        let camera = camera();
        aligner.process_frame(
            filter,
            &VisualFrame {
                imu: &[],
                frame_start_time: time,
                image_time: time,
                frame_index,
                rgb: image,
                camera: &camera,
                camera_from_lidar: &Matrix4::identity(),
                frame_points: points,
                planar_map: map,
            },
        )
    }

    fn pose_of(state: &NavState) -> Matrix4<f64> {
        state.lidar_pose(&Vector3::zeros())
    }

    #[test]
    fn grayscale_uses_rgb_weights_and_subpixel_derivative() {
        let rgb = RgbImage::from_pixel(2, 2, Rgb([200, 100, 50]));
        let gray = GrayImage::from_rgb(&rgb);
        assert!((gray.pixels[0] as f64 - 124.2).abs() < 1e-4);
        let image = GrayImage {
            width: 2,
            height: 2,
            pixels: vec![10.0, 30.0, 40.0, 90.0],
        };
        let p = Vector2::new(0.3, 0.4);
        let (value, g) = image.sample(&p).unwrap();
        assert!((value - 31.6).abs() < 1e-9);
        for axis in 0..2 {
            let mut d = Vector2::zeros();
            d[axis] = 1e-5;
            let numerical =
                (image.sample(&(p + d)).unwrap().0 - image.sample(&(p - d)).unwrap().0) / 2e-5;
            assert!((numerical - g[axis]).abs() < 1e-7);
        }
    }

    #[test]
    fn level_pixel_conversion_matches_projection_with_level_intrinsics() {
        let mut camera = camera();
        camera.distortion = [-0.13, 0.03, -0.002, 0.001, 0.0];
        let images = pyramid(&RgbImage::new(320, 240), &camera);
        for pc in [Point3::new(0.4, -0.3, 3.0), Point3::new(-1.2, 0.8, 7.5)] {
            let (pixel0, _) = project_camera_point_subpixel(&pc, &images[0].camera).unwrap();
            for (index, level) in images.iter().enumerate() {
                let (expected, _) = project_camera_point_subpixel(&pc, &level.camera).unwrap();
                assert!((pixel_at_level(&pixel0, index) - expected).norm() < 1e-9);
            }
        }
    }

    #[test]
    fn shi_tomasi_is_low_on_an_edge_high_on_a_corner_and_matches_brute_force() {
        let make = |f: &dyn Fn(usize, usize) -> f32| GrayImage {
            width: 40,
            height: 40,
            pixels: (0..40)
                .flat_map(|y| (0..40).map(move |x| (x, y)))
                .map(|(x, y)| f(x, y))
                .collect(),
        };
        // 縦のエッジ: 横方向にしか拘束しないので λmin ≈ 0
        let edge = make(&|x, _| if x < 20 { 50.0 } else { 200.0 });
        let edge_score = StructureTensorIntegral::new(&edge)
            .score_at(&Vector2::new(20.0, 20.0))
            .unwrap();
        assert!(edge_score < 1e-6, "edge score {edge_score}");
        // 角: 両方向に変化するので高得点
        let corner = make(&|x, y| if x < 20 && y < 20 { 50.0 } else { 200.0 });
        let corner_score = StructureTensorIntegral::new(&corner)
            .score_at(&Vector2::new(20.0, 20.0))
            .unwrap();
        assert!(corner_score > 5.0, "corner score {corner_score}");
        // 画像端で 9×9 がはみ出すなら None
        assert!(
            StructureTensorIntegral::new(&corner)
                .score_at(&Vector2::new(2.0, 20.0))
                .is_none()
        );

        // 積分画像の結果が素直な 81 画素の和と一致する
        let textured = make(&|x, y| texture(x as f64 / 7.0, y as f64 / 7.0) as f32);
        let tensor = StructureTensorIntegral::new(&textured);
        for (cx, cy) in [(10usize, 12usize), (25, 17), (31, 31)] {
            let (mut a, mut b, mut c) = (0.0f64, 0.0f64, 0.0f64);
            for y in cy - 4..=cy + 4 {
                for x in cx - 4..=cx + 4 {
                    let i = y * 40 + x;
                    let gx = 0.5 * (textured.pixels[i + 1] - textured.pixels[i - 1]) as f64;
                    let gy = 0.5 * (textured.pixels[i + 40] - textured.pixels[i - 40]) as f64;
                    a += gx * gx;
                    b += gx * gy;
                    c += gy * gy;
                }
            }
            let lambda_min = 0.5 * (a + c) - ((0.5 * (a - c)).powi(2) + b * b).sqrt();
            let expected = (lambda_min / 81.0).sqrt();
            let actual = tensor
                .score_at(&Vector2::new(cx as f64 + 0.3, cy as f64 - 0.4))
                .unwrap();
            assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
        }
    }

    #[test]
    fn plane_patches_lie_on_the_tilted_plane_and_reduce_to_constant_depth_when_facing() {
        let camera = camera();
        let images = pyramid(&render_plane(&camera, Matrix4::identity(), 0), &camera);
        let center = Point3::new(0.3, -0.2, 5.0);
        let tilt = 0.6f64;
        let normal = Vector3::new(0.0, tilt.sin(), -tilt.cos());
        let patches =
            capture_plane_patches(&images, &center, &normal, &Matrix4::identity(), 0.25).unwrap();
        for patch in &patches {
            assert!((patch.world_samples[40] - center).norm() < 1e-8);
            for sample in &patch.world_samples {
                assert!(normal.dot(&(sample - center)).abs() < 1e-9);
            }
            // 斜めの面なので、深度は一定ではない
            assert!((patch.world_samples[0].z - patch.world_samples[80].z).abs() > 1e-3);
        }

        // 正対した面では、従来の「深度一定」と同じ点になる
        let facing = capture_plane_patches(
            &images,
            &center,
            &Vector3::new(0.0, 0.0, -1.0),
            &Matrix4::identity(),
            0.25,
        )
        .unwrap();
        let (pixel, _) = project_camera_point_subpixel(&center, &images[0].camera).unwrap();
        let corner = unproject_pixel_at_depth(
            &(pixel + Vector2::new(-4.0, -4.0)),
            center.z,
            &images[0].camera,
        )
        .unwrap();
        assert!((facing[0].world_samples[0] - corner).norm() < 1e-9);

        // 面をかすめる角度なら作らない
        let grazing = Vector3::new(1.0, 0.0, -0.05).normalize();
        assert!(
            capture_plane_patches(&images, &center, &grazing, &Matrix4::identity(), 0.25).is_none()
        );
    }

    #[test]
    fn bootstrap_builds_keyframe_from_planar_map_without_updating_the_filter() {
        let camera = camera();
        let map = planar_map(5.0, 5.0, 4.0, 0.05);
        let image = render_plane(&camera, Matrix4::identity(), 0);
        let points = frame_points(&camera, 5.0);
        let mut filter = filter();
        let (state, cov) = (filter.state, filter.cov);
        let mut aligner = VisualAligner::new(config());

        let report = run(&mut aligner, &mut filter, 0, 0.0, &image, &points, &map);
        assert_eq!(report.status, "no_keyframe");
        assert_eq!(report.keyframe_event, "created:bootstrap", "{report:?}");
        assert!(!report.update_applied);
        assert_eq!(filter.state, state);
        assert_eq!(filter.cov, cov);

        let keyframe = aligner.keyframe.as_ref().unwrap();
        assert_eq!(keyframe.frame_index, 0);
        assert!(keyframe.points.len() >= aligner.config.min_keyframe_points);
        assert!(keyframe.points.len() <= aligner.config.max_keyframe_points);
        let cols = (camera.width as usize).div_ceil(aligner.config.grid_size_px);
        let mut used_cells = std::collections::HashSet::new();
        for point in &keyframe.points {
            assert!((point.world_point.z - 5.0).abs() < 1e-6);
            // 法線は K のカメラ側（-z）を向くようにそろえられている
            assert!(point.normal.z < -0.99);
            assert!(point.shi_tomasi_score >= aligner.config.min_shi_tomasi_score);
            for patch in &point.patches {
                assert!((patch.world_samples[40] - point.world_point).norm() < 1e-8);
                assert!(patch.world_samples.iter().all(|s| (s.z - 5.0).abs() < 1e-8));
            }
            // グリッドのセルごとに最大 1 点
            let pc = keyframe
                .camera_from_world
                .transform_point(&point.world_point);
            let (pixel, _) = project_camera_point_subpixel(&pc, &camera).unwrap();
            let cell = (pixel.y as usize / aligner.config.grid_size_px) * cols
                + pixel.x as usize / aligner.config.grid_size_px;
            assert!(used_cells.insert(cell), "two points in cell {cell}");
        }

        let again = run(&mut aligner, &mut filter, 1, 0.0, &image, &points, &map);
        assert_eq!(again.status, "reused_or_older_image");
        assert_eq!(filter.state, state);
    }

    #[test]
    fn map_points_hidden_behind_the_keyframe_point_cloud_are_not_selected() {
        let camera = camera();
        let images = || pyramid(&render_plane(&camera, Matrix4::identity(), 0), &camera);
        // マップの平面は 8 m 先。点群は 5 m に壁がある → 全部隠れている
        let map = planar_map(8.0, 7.0, 5.0, 0.05);
        let cfg = config();
        let wall = DepthBuffer::build(
            &frame_points(&camera, 5.0),
            &Matrix4::identity(),
            &camera,
            cfg.depth_buffer_cell_px,
            cfg.min_depth_m,
        );
        let (hidden, stats) =
            build_keyframe(&cfg, images(), Matrix4::identity(), 0, 0.0, &map, &wall).unwrap();
        assert!(hidden.points.is_empty());
        assert!(stats.occluded > 0 && stats.occluded == stats.candidates);

        // 点群も 8 m なら遮蔽はない
        let open = DepthBuffer::build(
            &frame_points(&camera, 8.0),
            &Matrix4::identity(),
            &camera,
            cfg.depth_buffer_cell_px,
            cfg.min_depth_m,
        );
        let (visible, stats) =
            build_keyframe(&cfg, images(), Matrix4::identity(), 0, 0.0, &map, &open).unwrap();
        assert_eq!(stats.occluded, 0);
        assert!(visible.points.len() >= cfg.min_keyframe_points);
    }

    #[test]
    fn all_15_image_jacobians_match_residual_finite_differences_with_imu_delay_and_lever_arm() {
        let mut filter = filter();
        let mut camera = camera();
        camera.distortion = [-0.13, 0.03, -0.002, 0.001, 0.0];
        filter.state.rot = UnitQuaternion::from_euler_angles(0.03, -0.02, 0.05);
        filter.state.pos = Vector3::new(0.1, -0.2, 0.05);
        filter.state.vel = Vector3::new(0.4, -0.2, 0.1);
        filter.state.bg = Vector3::new(0.01, -0.015, 0.005);
        filter.state.ba = Vector3::new(0.02, -0.03, 0.01);
        filter.imu_origin_in_lidar = Vector3::new(-0.04165, -0.02326, 0.0284);
        let imu: Vec<_> = (0..=20)
            .map(|i| IMU {
                timestamp: i as f64 * 0.01,
                angular_velocity: [0.15, -0.04, 0.2],
                linear_acceleration: [0.3, 0.2, 9.8],
            })
            .collect();
        let extrinsic =
            crate::camera::build_camera_from_lidar(Vector3::new(0.04, 0.02, 0.05), 0.2, -0.3, 0.1);
        let observation = ImageObservation {
            filter: &filter,
            imu: &imu,
            t_start: 0.0,
            t_image: 0.073,
            camera_from_lidar: &extrinsic,
        };
        let lin = observation.linearize(&filter.state).unwrap();
        let world_from_camera = lin.pose.try_inverse().unwrap();
        let image = GrayImage {
            width: 320,
            height: 240,
            pixels: (0..240)
                .flat_map(|y| {
                    (0..320).map(move |x| texture(x as f64 / 20.0, y as f64 / 20.0) as f32)
                })
                .collect(),
        };
        let level = ImageLevel { image, camera };
        let mut patch = ReferencePatch {
            gray: [0.0; 81],
            world_samples: [Point3::origin(); 81],
        };
        for y in -4..=4 {
            for x in -4..=4 {
                let i = ((y + 4) * 9 + x + 4) as usize;
                let p = Vector2::new(173.23 + x as f64, 103.37 + y as f64);
                patch.world_samples[i] = world_from_camera
                    .transform_point(&unproject_pixel_at_depth(&p, 4.3, &level.camera).unwrap());
            }
        }
        let actual = evaluate_patch(&patch, &level, &lin.pose, Some(&lin)).unwrap();
        for col in 0..15 {
            let mut dx = Vector15::zeros();
            dx[col] = 1e-5;
            let plus = observation.camera_pose(&filter.state.boxplus(&dx)).unwrap();
            let minus = observation
                .camera_pose(&filter.state.boxplus(&(-dx)))
                .unwrap();
            let rp = evaluate_patch(&patch, &level, &plus, None).unwrap();
            let rm = evaluate_patch(&patch, &level, &minus, None).unwrap();
            for i in 0..81 {
                let numeric = (rp.residuals[i] - rm.residuals[i]) / 2e-5;
                let analytic = actual.jacobians[i][col];
                assert!(
                    (numeric - analytic).abs() < 1e-3 + 1e-4 * numeric.abs(),
                    "col={col} pixel={i} numeric={numeric} analytic={analytic}"
                );
            }
        }
    }

    #[test]
    fn direct_alignment_to_keyframe_recovers_synthetic_translation_and_rotation() {
        let camera = camera();
        let map = planar_map(5.0, 5.0, 4.0, 0.05);
        let points = frame_points(&camera, 5.0);
        let mut filter = filter();
        let mut aligner = VisualAligner::new(config());
        let reference = render_plane(&camera, Matrix4::identity(), 0);
        run(&mut aligner, &mut filter, 0, 0.0, &reference, &points, &map);
        assert_eq!(aligner.keyframe_frame_index(), Some(0));

        let truth = NavState {
            pos: Vector3::new(0.027, -0.018, 0.009),
            rot: UnitQuaternion::from_euler_angles(0.0015, -0.002, 0.003),
            ..NavState::default()
        };
        let current = render_plane(&camera, pose_of(&truth), 0);
        // LiDAR estimate is close, but still has a measurable subpixel error.
        filter.state.pos = Vector3::new(0.01, -0.005, 0.002);
        filter.state.rot = UnitQuaternion::from_euler_angles(0.0005, -0.001, 0.001);
        let before = (filter.state.pos - truth.pos).norm();
        let original_state = filter.state;
        let original_cov = filter.cov;

        let report = run(&mut aligner, &mut filter, 1, 0.1, &current, &points, &map);
        assert!(report.update_applied, "{report:?}");
        assert_eq!(report.keyframe_frame_index, Some(0));
        assert_eq!(report.keyframe_event, "none", "K must not change at age 1");
        let after = (filter.state.pos - truth.pos).norm();
        assert!(
            after < before * 0.6,
            "before={before} after={after} report={report:?}"
        );
        assert!(
            report.final_rmse.unwrap() < report.initial_rmse.unwrap() * 0.6,
            "{report:?}"
        );
        assert!(Cholesky::new(filter.cov).is_some());
        assert!(filter.cov.trace() < original_cov.trace());
        assert!(
            (filter.state.rot * truth.rot.inverse()).angle()
                < (original_state.rot * truth.rot.inverse()).angle()
        );
    }

    /// 斜めの面を、K とは違う位置から見るとパッチが歪む。法線付き平面で 3D 化した
    /// パッチはその歪みを正しく予測できるので、正対平面（深度一定）より良く合う。
    #[test]
    fn plane_normals_improve_alignment_on_a_tilted_plane_seen_from_a_new_viewpoint() {
        let camera = camera();
        let tilt = 40f64.to_radians();
        let normal = Vector3::new(0.0, tilt.sin(), -tilt.cos());
        let origin = Point3::new(0.0, 0.0, 5.0);
        let offset = normal.dot(&origin.coords);
        let e1 = Vector3::x();
        let e2 = Vector3::new(0.0, tilt.cos(), tilt.sin());
        let extent = (-8.0, 8.0, -3.0, 8.0);

        let truth = NavState {
            pos: Vector3::new(0.35, 0.25, 0.6),
            rot: UnitQuaternion::from_euler_angles(0.03, -0.04, 0.02),
            ..NavState::default()
        };
        let mut estimate = truth;
        estimate.pos += Vector3::new(0.02, -0.015, 0.012);
        estimate.rot = UnitQuaternion::from_euler_angles(0.002, -0.001, 0.0015) * truth.rot;
        let before = (estimate.pos - truth.pos).norm();

        let keyframe_image = render_world_plane(&camera, &Matrix4::identity(), &normal, offset);
        let keyframe_points = world_plane_points(&camera, &Matrix4::identity(), &normal, offset);
        let current_image = render_world_plane(&camera, &pose_of(&truth), &normal, offset);
        let current_points = world_plane_points(&camera, &pose_of(&truth), &normal, offset);

        let align_with = |stored_normal: Vector3<f64>| {
            let map = plane_map(&origin, &e1, &e2, extent, 0.05, &stored_normal);
            let mut filter = filter();
            let mut aligner = VisualAligner::new(config());
            let bootstrap = run(
                &mut aligner,
                &mut filter,
                0,
                0.0,
                &keyframe_image,
                &keyframe_points,
                &map,
            );
            assert_eq!(
                bootstrap.keyframe_event, "created:bootstrap",
                "{bootstrap:?}"
            );
            filter.state = estimate;
            let report = run(
                &mut aligner,
                &mut filter,
                1,
                0.1,
                &current_image,
                &current_points,
                &map,
            );
            ((filter.state.pos - truth.pos).norm(), report)
        };

        let (with_normal_error, with_normal) = align_with(normal);
        let (fronto_error, fronto) = align_with(Vector3::new(0.0, 0.0, -1.0));
        assert!(with_normal.update_applied, "{with_normal:?}");
        assert!(
            with_normal_error < before * 0.5,
            "before={before} after={with_normal_error} {with_normal:?}"
        );
        // 同じ初期値・同じ画像でも、法線ありの方が残差が小さく、位置も正確になる
        let fronto_rmse = fronto.final_rmse.unwrap_or(f64::INFINITY);
        assert!(
            with_normal.final_rmse.unwrap() < 0.8 * fronto_rmse,
            "with normal {:?} vs fronto {:?}",
            with_normal.final_rmse,
            fronto.final_rmse
        );
        assert!(
            with_normal_error < fronto_error,
            "with normal {with_normal_error} vs fronto {fronto_error}"
        );
    }

    /// 手作りの Planar セルではなく、実際の GlobalMap の更新・平面判定（main.rs と同じ設定）を
    /// 通したマップからキーフレームを作り、位置合わせできること。
    #[test]
    fn keyframe_from_real_global_map_plane_classification_aligns() {
        use crate::voxel_map::{LOCALMap, LocalMapConfig, SurfaceFilterConfig};

        let mut global = LOCALMap::new(LocalMapConfig {
            index_voxel_size: 0.025,
            max_points_per_voxel: 20,
            min_points_per_voxel: 2,
            min_observed_frames_per_voxel: 2,
            max_frames: 50,
            max_distance: 150.0,
        });
        let mut scan = Vec::new();
        let mut y = -3.5f32;
        while y < 3.5 {
            let mut x = -4.5f32;
            while x < 4.5 {
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

        let camera = camera();
        let points = frame_points(&camera, 5.0);
        let mut filter = filter();
        let mut aligner = VisualAligner::new(config());
        let bootstrap = run(
            &mut aligner,
            &mut filter,
            0,
            0.0,
            &render_plane(&camera, Matrix4::identity(), 0),
            &points,
            &global.voxel_map,
        );
        assert_eq!(
            bootstrap.keyframe_event, "created:bootstrap",
            "{bootstrap:?}"
        );

        let truth = NavState {
            pos: Vector3::new(0.02, 0.015, -0.01),
            ..NavState::default()
        };
        filter.state.pos = Vector3::new(0.005, 0.0, 0.0);
        let before = (filter.state.pos - truth.pos).norm();
        let report = run(
            &mut aligner,
            &mut filter,
            1,
            0.1,
            &render_plane(&camera, pose_of(&truth), 0),
            &points,
            &global.voxel_map,
        );
        assert!(report.update_applied, "{report:?}");
        assert!(
            (filter.state.pos - truth.pos).norm() < before * 0.5,
            "{report:?}"
        );
    }

    #[test]
    fn current_point_cloud_hides_occluded_keyframe_patches() {
        let camera = camera();
        let map = planar_map(5.0, 5.0, 4.0, 0.05);
        let mut filter = filter();
        let mut aligner = VisualAligner::new(config());
        let image = render_plane(&camera, Matrix4::identity(), 0);
        run(
            &mut aligner,
            &mut filter,
            0,
            0.0,
            &image,
            &frame_points(&camera, 5.0),
            &map,
        );
        let total = aligner.keyframe.as_ref().unwrap().points.len();

        // 今のフレームでは、画像の左半分の手前 2 m に物がある
        let mut points: Vec<Point3<f32>> = frame_points(&camera, 5.0)
            .into_iter()
            .filter(|p| p.x >= 0.0)
            .collect();
        points.extend(frame_points(&camera, 2.0).into_iter().filter(|p| p.x < 0.0));
        let report = run(&mut aligner, &mut filter, 1, 0.1, &image, &points, &map);
        assert!(report.occluded_patches > total / 4, "{report:?}");
        assert!(report.visible_patches + report.occluded_patches <= total);
        assert!(report.visible_patches < total);
    }

    #[test]
    fn keyframe_is_kept_until_max_age_then_replaced() {
        let camera = camera();
        let map = planar_map(5.0, 5.0, 4.0, 0.05);
        let points = frame_points(&camera, 5.0);
        let image = render_plane(&camera, Matrix4::identity(), 0);
        let mut filter = filter();
        // 交代ロジックの確認なので、点数と寿命を小さくして実行時間を抑える
        let mut aligner = VisualAligner::new(small_logic_config());
        run(&mut aligner, &mut filter, 0, 0.0, &image, &points, &map);
        let max_age = aligner.config.max_keyframe_age_frames;
        for frame_index in 1..max_age {
            let report = run(
                &mut aligner,
                &mut filter,
                frame_index,
                frame_index as f64 * 0.1,
                &image,
                &points,
                &map,
            );
            assert!(
                matches!(report.status.as_str(), "accepted" | "already_aligned"),
                "{report:?}"
            );
            assert_eq!(report.keyframe_frame_index, Some(0));
            assert!(report.visible_ratio.unwrap() > 0.99);
            assert_eq!(report.keyframe_event, "none");
        }
        let report = run(
            &mut aligner,
            &mut filter,
            max_age,
            max_age as f64 * 0.1,
            &image,
            &points,
            &map,
        );
        assert_eq!(report.keyframe_age_frames, max_age);
        assert_eq!(report.keyframe_event, "created:max_age", "{report:?}");
        assert_eq!(aligner.keyframe_frame_index(), Some(max_age));
    }

    #[test]
    fn low_visibility_switches_keyframe_after_minimum_frames() {
        let camera = camera();
        let map = planar_map(5.0, 9.0, 4.0, 0.05);
        let points = frame_points(&camera, 5.0);
        let mut filter = filter();
        let mut aligner = VisualAligner::new(config());
        run(
            &mut aligner,
            &mut filter,
            0,
            0.0,
            &render_plane(&camera, Matrix4::identity(), 0),
            &points,
            &map,
        );

        // 横に 3 m 動いた → K のパッチは半分程度しか見えない（LiDAR は正しい姿勢を出している）
        let truth = NavState {
            pos: Vector3::new(3.0, 0.0, 0.0),
            ..NavState::default()
        };
        let shifted = render_plane(&camera, pose_of(&truth), 0);
        let min_frames = aligner.config.min_keyframe_frames;
        for frame_index in 1..=min_frames {
            filter.state = truth;
            let report = run(
                &mut aligner,
                &mut filter,
                frame_index,
                frame_index as f64 * 0.1,
                &shifted,
                &points,
                &map,
            );
            assert!(
                matches!(report.status.as_str(), "accepted" | "already_aligned"),
                "{report:?}"
            );
            assert!(report.visible_ratio.unwrap() < aligner.config.min_visible_ratio);
            if frame_index < min_frames {
                assert_eq!(report.keyframe_event, "none", "{report:?}");
                assert_eq!(aligner.keyframe_frame_index(), Some(0));
            } else {
                assert_eq!(
                    report.keyframe_event, "created:low_visibility",
                    "{report:?}"
                );
                assert_eq!(aligner.keyframe_frame_index(), Some(min_frames));
            }
        }
    }

    #[test]
    fn rejected_updates_keep_keyframe_until_consecutive_failures_force_a_rebuild() {
        let camera = camera();
        let map = planar_map(5.0, 5.0, 4.0, 0.05);
        let points = frame_points(&camera, 5.0);
        let mut filter = filter();
        let mut aligner = VisualAligner::new(small_logic_config());
        run(
            &mut aligner,
            &mut filter,
            0,
            0.0,
            &render_plane(&camera, Matrix4::identity(), 0),
            &points,
            &map,
        );
        aligner.config.max_final_rmse = 0.001;
        let truth = NavState {
            pos: Vector3::new(0.025, -0.015, 0.01),
            ..NavState::default()
        };
        let current = render_plane(&camera, pose_of(&truth), 0);
        let (state, cov) = (filter.state, filter.cov);
        let max_failures = aligner.config.max_consecutive_failures as u64;

        for frame_index in 1..max_failures {
            let report = run(
                &mut aligner,
                &mut filter,
                frame_index,
                frame_index as f64 * 0.1,
                &current,
                &points,
                &map,
            );
            assert_eq!(report.status, "rejected:final_rmse", "{report:?}");
            assert_eq!(report.consecutive_failures, frame_index as u32);
            assert_eq!(report.keyframe_event, "none");
            assert_eq!(filter.state, state);
            assert_eq!(filter.cov, cov);
            assert_eq!(aligner.keyframe_frame_index(), Some(0));
        }
        let report = run(
            &mut aligner,
            &mut filter,
            max_failures,
            max_failures as f64 * 0.1,
            &current,
            &points,
            &map,
        );
        assert_eq!(report.status, "rejected:final_rmse");
        assert_eq!(
            report.keyframe_event, "created:reset_after_failures",
            "{report:?}"
        );
        assert_eq!(report.consecutive_failures, 0);
        assert_eq!(aligner.keyframe_frame_index(), Some(max_failures));
        // 作り直しでも、画像の更新はしていない
        assert_eq!(filter.state, state);
    }

    #[test]
    fn textureless_image_or_invalid_time_never_changes_filter_or_creates_keyframe() {
        let camera = camera();
        let map = planar_map(5.0, 5.0, 4.0, 0.05);
        let points = frame_points(&camera, 5.0);
        let mut filter = filter();
        let (state, cov) = (filter.state, filter.cov);
        let mut aligner = VisualAligner::new(config());
        let blank = RgbImage::from_pixel(320, 240, Rgb([120, 120, 120]));
        let report = run(&mut aligner, &mut filter, 0, 0.0, &blank, &points, &map);
        assert_eq!(report.keyframe_event, "skipped:bootstrap:too_few_points");
        assert_eq!(report.new_keyframe_points, 0);
        assert!(aligner.keyframe.is_none());

        let invalid = aligner.process_frame(
            &mut filter,
            &VisualFrame {
                imu: &[],
                frame_start_time: 1.0,
                image_time: 0.9,
                frame_index: 1,
                rgb: &blank,
                camera: &camera,
                camera_from_lidar: &Matrix4::identity(),
                frame_points: &points,
                planar_map: &map,
            },
        );
        assert_eq!(invalid.status, "invalid_image_time_or_dimensions");
        assert_eq!(filter.state, state);
        assert_eq!(filter.cov, cov);
    }

    #[test]
    fn candidate_cannot_reduce_cost_by_moving_patches_out_of_image() {
        let camera = camera();
        let map = planar_map(5.0, 5.0, 4.0, 0.05);
        let mut filter = filter();
        let mut aligner = VisualAligner::new(config());
        let rgb = render_plane(&camera, Matrix4::identity(), 0);
        let images = pyramid(&rgb, &camera);
        run(
            &mut aligner,
            &mut filter,
            0,
            0.0,
            &rgb,
            &frame_points(&camera, 5.0),
            &map,
        );
        let keyframe = aligner.keyframe.as_ref().unwrap();
        let indices: Vec<usize> = (0..keyframe.points.len()).collect();
        assert!(
            build_system(
                &aligner.config,
                keyframe,
                &indices,
                0,
                &images[0],
                &Matrix4::identity(),
                None
            )
            .is_some()
        );
        let mut moved = Matrix4::identity();
        moved[(0, 3)] = 100.0;
        assert!(
            build_system(
                &aligner.config,
                keyframe,
                &indices,
                0,
                &images[0],
                &moved,
                None
            )
            .is_none()
        );
    }
}
