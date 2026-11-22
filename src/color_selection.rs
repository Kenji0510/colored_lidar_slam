//! GlobalMap に付ける色の選び方（ベストビュー方式）。
//!
//! 1 つのセルには、これまでで最もスコアの高い 1 回の観測の色だけを残す（平均しない）。
//!
//! ```text
//! スコア = 距離の項 × 角度の項 × 露光の項 × 鮮鋭さの項
//!   距離の項   1 / (1 + (Z / Z0)²)      近いほど良い（Z: カメラからの距離）
//!   角度の項   |cos θ|                  正面から見ているほど良い（θ: 面の法線と視線の角度）
//!   露光の項   min(g, 1)²               地図より暗く写ったフレームを減点
//!   鮮鋭さの項 1 / (1 + (b / b0)²)      予測ブレ量 b [px] が小さいほど良い
//! ```
//!
//! g はフレームの露光比で、「このフレームで読んだ輝度 / 同じセルに地図がすでに持つ色の輝度」
//! の中央値。自動露光は画像全体の平均を一定に保とうとするので、画像全体の明るさではなく
//! 同じ場所の明るさで比べる。g > 1（地図の色の方が暗く写った観測）のときは、
//! 既存の色のスコアを (1/g)² 倍してから比べる。どちらの向きでも「暗く写った側を減点する」。
//!
//! b は、IMU の角速度と ESKF の速度から求めた、露光時間のあいだに点が画像上を動く量。
//! 画像の幅 `reference_width_px` 相当の画素数に換算して扱う。
//!
//! フレームごとの判定（どれか 1 つでも当てはまれば、そのフレームでは色を一切付けない）
//! - 露光比 g が `min_exposure_ratio` 未満（暗すぎる）
//! - 白飛び画素の割合が `max_saturated_ratio` 超
//! - 予測ブレ量の中央値が `max_frame_blur_px` 超（ブレすぎ）
//!
//! 点ごとの判定（その観測だけ使わない）
//! - どれかのチャネルが `saturation_level` 以上（白飛び）
//! - 面を `min_view_cosine` より斜めに見ている
//! - 予測ブレ量が `max_point_blur_px` 超
//! - 深度が範囲外、または画像の中央 75% の外

use image::RgbImage;
use nalgebra::{Matrix3, Matrix4, Point3, Vector3};
use rayon::prelude::*;

use crate::camera::{CameraIntrinsics, project_camera_point, project_camera_point_subpixel};
use crate::eskf::{ImuModel, NavState};
use crate::types::{ColoringLog, IMU};
use crate::voxel_map::{ColorObservation, VoxelMap, voxel_key};

#[derive(Debug, Clone)]
pub struct ColorSelectionConfig {
    // --- 距離 ---
    /// 距離の項 1/(1+(Z/Z0)²) の Z0 [m]。Z0 の距離で 0.5 になる。
    pub reference_distance_m: f64,
    pub min_depth_m: f64,
    pub max_depth_m: f64,

    // --- 角度 ---
    /// 法線と視線のなす角の cos の下限。これより斜めの観測は使わない（0.3 ≒ 73°）。
    pub min_view_cosine: f64,
    /// 法線が分からない点（最初のフレームなど）に使う角度の項。
    pub unknown_view_cosine: f64,

    // --- 露光 ---
    /// どれかのチャネルがこの値以上なら白飛びとみなし、その画素は使わない。
    pub saturation_level: u8,
    /// フレームの露光比 g がこれ未満なら、そのフレームでは色を付けない。
    pub min_exposure_ratio: f64,
    /// 白飛び画素の割合がこれを超えるフレームでは色を付けない。
    pub max_saturated_ratio: f64,
    /// 露光比の推定に使う輝度の範囲。両端は応答が非線形なので比を取らない。
    pub exposure_min_luma: f64,
    pub exposure_max_luma: f64,
    /// 露光比の推定に必要な、地図と重なった点の最小数。
    pub min_exposure_samples: usize,
    /// 露光比が推定できないフレーム（地図にまだ色がない場所など）の露光の項。
    /// 後から露光比の分かるフレームが来れば置き換わるよう、低めにしておく。
    pub unknown_exposure_term: f64,

    // --- ブレ ---
    /// ブレ量の予測に使う露光時間 [s]。カメラの実際の露光時間が分かれば置き換える。
    pub assumed_exposure_time_s: f64,
    /// ブレ量は、この幅の画像での画素数に換算して扱う（960×600 と 1920×1200 で同じ基準）。
    pub reference_width_px: f64,
    /// 鮮鋭さの項 1/(1+(b/b0)²) の b0 [px]。
    pub blur_scale_px: f64,
    /// 予測ブレ量がこれを超える観測は使わない [px]。
    pub max_point_blur_px: f64,
    /// 予測ブレ量の中央値がこれを超えるフレームでは色を付けない [px]。
    pub max_frame_blur_px: f64,
}

impl Default for ColorSelectionConfig {
    fn default() -> Self {
        Self {
            reference_distance_m: 3.0,
            min_depth_m: 0.3,
            max_depth_m: 40.0,
            min_view_cosine: 0.3,
            unknown_view_cosine: 0.5,
            saturation_level: 250,
            min_exposure_ratio: 0.6,
            max_saturated_ratio: 0.3,
            exposure_min_luma: 20.0,
            exposure_max_luma: 235.0,
            min_exposure_samples: 50,
            unknown_exposure_term: 0.25,
            assumed_exposure_time_s: 0.010,
            reference_width_px: 960.0,
            blur_scale_px: 1.0,
            max_point_blur_px: 3.0,
            max_frame_blur_px: 2.0,
        }
    }
}

impl ColorSelectionConfig {
    fn valid(&self) -> bool {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        let unit = |v: f64| v.is_finite() && v > 0.0 && v <= 1.0;
        positive(self.reference_distance_m)
            && positive(self.min_depth_m)
            && self.max_depth_m.is_finite()
            && self.max_depth_m > self.min_depth_m
            && unit(self.min_view_cosine)
            && unit(self.unknown_view_cosine)
            && self.saturation_level > 0
            && self.min_exposure_ratio.is_finite()
            && self.min_exposure_ratio >= 0.0
            && self.max_saturated_ratio.is_finite()
            && (0.0..=1.0).contains(&self.max_saturated_ratio)
            && self.exposure_min_luma.is_finite()
            && self.exposure_min_luma > 0.0
            && self.exposure_max_luma.is_finite()
            && self.exposure_max_luma > self.exposure_min_luma
            && unit(self.unknown_exposure_term)
            && self.assumed_exposure_time_s.is_finite()
            && self.assumed_exposure_time_s >= 0.0
            && positive(self.reference_width_px)
            && positive(self.blur_scale_px)
            && positive(self.max_point_blur_px)
            && positive(self.max_frame_blur_px)
    }
}

// ---------------------------------------------------------------------------
// カメラの動き（ブレ量の予測用）
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraMotion {
    /// カメラの角速度（カメラ座標）[rad/s]。
    pub angular_velocity: Vector3<f64>,
    /// カメラ中心の速度（カメラ座標）[m/s]。
    pub linear_velocity: Vector3<f64>,
}

impl CameraMotion {
    /// カメラ座標の静止点 `point` が、画像上を動く速さ [px/s]（歪みを含む投影の微分）。
    ///
    /// 静止点のカメラ座標の時間微分は ṗ = −ω × p − v。
    pub fn pixel_speed(&self, point: &Point3<f64>, camera: &CameraIntrinsics) -> Option<f64> {
        let (_, jacobian) = project_camera_point_subpixel(point, camera)?;
        let point_velocity = -self.angular_velocity.cross(&point.coords) - self.linear_velocity;
        let speed = (jacobian * point_velocity).norm();
        speed.is_finite().then_some(speed)
    }
}

/// 画像時刻のカメラの角速度と速度（どちらもカメラ座標）。
///
/// 角速度は `time ± half_window_s` の IMU の平均からジャイロバイアスを引いたもの
/// （区間に IMU がなければ最も近い 1 サンプル）。速度は ESKF の IMU 原点の速度に、
/// IMU 原点からカメラまでのレバーアームによる回転分を足したもの。
/// `state` は画像時刻まで予測した名目状態、ボディ軸 = LiDAR 軸。
pub fn camera_motion_at(
    imu: &[IMU],
    model: &ImuModel,
    state: &NavState,
    imu_origin_in_lidar: &Vector3<f64>,
    camera_from_lidar: &Matrix4<f64>,
    time: f64,
    half_window_s: f64,
) -> Option<CameraMotion> {
    if imu.is_empty() || !time.is_finite() || !half_window_s.is_finite() || half_window_s < 0.0 {
        return None;
    }
    let begin = imu.partition_point(|s| s.timestamp < time - half_window_s);
    let end = imu.partition_point(|s| s.timestamp <= time + half_window_s);
    let samples: &[IMU] = if end > begin {
        &imu[begin..end]
    } else {
        let next = imu.partition_point(|s| s.timestamp < time);
        let nearest = [next.checked_sub(1), (next < imu.len()).then_some(next)]
            .into_iter()
            .flatten()
            .min_by(|&a, &b| {
                (imu[a].timestamp - time)
                    .abs()
                    .total_cmp(&(imu[b].timestamp - time).abs())
            })?;
        &imu[nearest..=nearest]
    };
    let omega_sum = samples.iter().fold(Vector3::zeros(), |sum, sample| {
        sum + model.measurement(sample).0
    });
    let omega_body = omega_sum / samples.len() as f64 - state.bg;

    let camera_rotation: Matrix3<f64> = camera_from_lidar.fixed_view::<3, 3>(0, 0).into_owned();
    let camera_translation: Vector3<f64> = camera_from_lidar.fixed_view::<3, 1>(0, 3).into_owned();
    let camera_in_lidar = -(camera_rotation.transpose() * camera_translation);
    let lever = camera_in_lidar - imu_origin_in_lidar;
    let world_from_body = state.rot.to_rotation_matrix().into_inner();
    let camera_velocity_world = state.vel + world_from_body * omega_body.cross(&lever);
    let camera_from_world_rotation = camera_rotation * world_from_body.transpose();

    let motion = CameraMotion {
        angular_velocity: camera_rotation * omega_body,
        linear_velocity: camera_from_world_rotation * camera_velocity_world,
    };
    (motion.angular_velocity.iter().all(|v| v.is_finite())
        && motion.linear_velocity.iter().all(|v| v.is_finite()))
    .then_some(motion)
}

// ---------------------------------------------------------------------------
// フレームの色の選択
// ---------------------------------------------------------------------------

pub struct FrameColorInput<'a> {
    pub image: &'a RgbImage,
    pub camera: &'a CameraIntrinsics,
    /// フレーム開始時刻の LiDAR 座標 → 画像時刻のカメラ座標。
    pub camera_from_lidar_start: &'a Matrix4<f64>,
    /// フレーム開始時刻の LiDAR 座標 → ワールド座標。
    pub world_from_lidar_start: &'a Matrix4<f64>,
    /// 色を付ける点（フレーム開始時刻の LiDAR 座標）。
    pub points: &'a [Point3<f32>],
    /// 各点の面の法線（ワールド座標）。`points` と同じ並び、分からなければ None。
    pub normals: &'a [Option<Vector3<f32>>],
    /// 画像時刻のカメラの動き。None ならブレ量は評価しない。
    pub motion: Option<&'a CameraMotion>,
    /// 露光比の推定に使う地図（各セルにすでに付いている色）。
    pub map: &'a VoxelMap,
    pub map_voxel_size: f32,
}

pub struct FrameColorSelection {
    /// 色付けするフレームなら、点ごとの観測（`points` と同じ並び）。しないなら None。
    pub observations: Option<Vec<Option<ColorObservation>>>,
    pub report: ColoringLog,
}

/// 画像に写った 1 点の情報（ゲート前）。
struct Candidate {
    rgb: [u8; 3],
    world: Point3<f32>,
    distance_m: f64,
    /// 法線と視線のなす角の |cos|（法線が分からなければ None）。
    view_cosine: Option<f64>,
    /// 予測ブレ量 [px]（基準幅に換算済み。動きが分からなければ None）。
    blur_px: Option<f64>,
    saturated: bool,
}

#[inline]
fn luma(rgb: [u8; 3]) -> f64 {
    0.299 * rgb[0] as f64 + 0.587 * rgb[1] as f64 + 0.114 * rgb[2] as f64
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mid = values.len() / 2;
    let (_, upper, _) = values.select_nth_unstable_by(mid, f64::total_cmp);
    let upper = *upper;
    if values.len() % 2 == 1 {
        return Some(upper);
    }
    let lower = values[..mid]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    Some(0.5 * (lower + upper))
}

/// 距離・角度・鮮鋭さの項の積（露光の項を除いたスコア）。
fn geometric_score(config: &ColorSelectionConfig, candidate: &Candidate) -> f64 {
    let distance_term = 1.0 / (1.0 + (candidate.distance_m / config.reference_distance_m).powi(2));
    let angle_term = candidate
        .view_cosine
        .unwrap_or(config.unknown_view_cosine)
        .min(1.0);
    let sharpness_term = candidate.blur_px.map_or(1.0, |blur| {
        1.0 / (1.0 + (blur / config.blur_scale_px).powi(2))
    });
    distance_term * angle_term * sharpness_term
}

/// 1 フレーム分の色の観測を作る。フレームごとの判定で不合格なら観測は返さない。
pub fn select_frame_colors(
    config: &ColorSelectionConfig,
    input: &FrameColorInput<'_>,
) -> FrameColorSelection {
    let mut report = ColoringLog::default();
    let reject = |mut report: ColoringLog, status: &str| {
        report.status = status.into();
        FrameColorSelection {
            observations: None,
            report,
        }
    };
    if !config.valid() {
        return reject(report, "invalid_config");
    }
    if input.image.width() != input.camera.width || input.image.height() != input.camera.height {
        return reject(report, "invalid_image_dimensions");
    }
    let Some(world_from_camera) = input
        .camera_from_lidar_start
        .try_inverse()
        .map(|lidar_start_from_camera| input.world_from_lidar_start * lidar_start_from_camera)
    else {
        return reject(report, "invalid_camera_pose");
    };
    let camera_center: Vector3<f64> = world_from_camera.fixed_view::<3, 1>(0, 3).into_owned();
    report.angular_speed_deg_s = input
        .motion
        .map(|motion| motion.angular_velocity.norm().to_degrees());
    let blur_scale =
        config.assumed_exposure_time_s * config.reference_width_px / input.camera.width as f64;

    // --- 1. 画像に写る点を集める ---
    let candidates: Vec<Option<Candidate>> = input
        .points
        .par_iter()
        .enumerate()
        .map(|(index, point)| {
            if !point.coords.iter().all(|v| v.is_finite()) {
                return None;
            }
            let point64 = point.cast::<f64>();
            let camera_point = input.camera_from_lidar_start.transform_point(&point64);
            if !(camera_point.z >= config.min_depth_m && camera_point.z <= config.max_depth_m) {
                return None;
            }
            let (u, v) = project_camera_point(&camera_point, input.camera)?;
            let rgb = input.image.get_pixel_checked(u, v)?.0;
            let distance_m = camera_point.coords.norm();
            let world = input.world_from_lidar_start.transform_point(&point64);
            let view_cosine = input
                .normals
                .get(index)
                .copied()
                .flatten()
                .and_then(|normal| {
                    let normal = normal.cast::<f64>();
                    let ray = world.coords - camera_center;
                    let cosine = normal.dot(&ray).abs() / (normal.norm() * ray.norm());
                    cosine.is_finite().then_some(cosine)
                });
            let blur_px = input.motion.and_then(|motion| {
                motion
                    .pixel_speed(&camera_point, input.camera)
                    .map(|speed| speed * blur_scale)
            });
            Some(Candidate {
                rgb,
                world: world.cast::<f32>(),
                distance_m,
                view_cosine,
                blur_px,
                saturated: rgb.iter().any(|&c| c >= config.saturation_level),
            })
        })
        .collect();

    let total = candidates.iter().flatten().count();
    report.candidates = total;
    if total == 0 {
        return reject(report, "no_candidates");
    }

    // --- 2. フレーム全体の指標（白飛び率・ブレ量の中央値・露光比） ---
    let saturated = candidates.iter().flatten().filter(|c| c.saturated).count();
    report.saturated_ratio = Some(saturated as f64 / total as f64);
    let mut blurs: Vec<f64> = candidates
        .iter()
        .flatten()
        .filter_map(|c| c.blur_px)
        .collect();
    report.median_blur_px = median(&mut blurs);

    let in_luma_range =
        |value: f64| value >= config.exposure_min_luma && value <= config.exposure_max_luma;
    let mut ratios: Vec<f64> = candidates
        .iter()
        .flatten()
        .filter(|c| !c.saturated)
        .filter_map(|c| {
            let frame_luma = luma(c.rgb);
            if !in_luma_range(frame_luma) {
                return None;
            }
            let stored = input
                .map
                .get(&voxel_key(&c.world, input.map_voxel_size))?
                .rgb?;
            if stored.iter().any(|&v| v >= config.saturation_level) {
                return None;
            }
            let map_luma = luma(stored);
            in_luma_range(map_luma).then_some(frame_luma / map_luma)
        })
        .collect();
    report.exposure_samples = ratios.len();
    let exposure_ratio = if ratios.len() >= config.min_exposure_samples {
        median(&mut ratios)
    } else {
        None
    };
    report.exposure_ratio = exposure_ratio;

    // --- 3. フレームごとの判定 ---
    if exposure_ratio.is_some_and(|g| g < config.min_exposure_ratio) {
        return reject(report, "too_dark");
    }
    if report
        .saturated_ratio
        .is_some_and(|ratio| ratio > config.max_saturated_ratio)
    {
        return reject(report, "too_saturated");
    }
    if report
        .median_blur_px
        .is_some_and(|blur| blur > config.max_frame_blur_px)
    {
        return reject(report, "too_blurry");
    }

    // --- 4. 点ごとの判定とスコア ---
    let exposure_term = exposure_ratio.map_or(config.unknown_exposure_term, |g| g.min(1.0).powi(2));
    // g > 1: 地図の色の方が暗く写った観測なので、既存のスコアを (1/g)² 倍して比べる。
    let existing_score_scale = exposure_ratio.map_or(1.0, |g| (1.0 / g).min(1.0).powi(2)) as f32;
    let mut observations = Vec::with_capacity(candidates.len());
    for candidate in &candidates {
        let Some(candidate) = candidate else {
            observations.push(None);
            continue;
        };
        if candidate.saturated {
            report.rejected_saturated += 1;
            observations.push(None);
            continue;
        }
        if candidate
            .view_cosine
            .is_some_and(|cosine| cosine < config.min_view_cosine)
        {
            report.rejected_oblique += 1;
            observations.push(None);
            continue;
        }
        if candidate
            .blur_px
            .is_some_and(|blur| blur > config.max_point_blur_px)
        {
            report.rejected_blurry += 1;
            observations.push(None);
            continue;
        }
        let score = (geometric_score(config, candidate) * exposure_term) as f32;
        if !(score.is_finite() && score > 0.0) {
            observations.push(None);
            continue;
        }
        report.observations += 1;
        observations.push(Some(ColorObservation {
            rgb: candidate.rgb,
            distance_m: candidate.distance_m,
            score,
            existing_score_scale,
        }));
    }
    report.status = "colored".into();
    FrameColorSelection {
        observations: Some(observations),
        report,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::build_camera_from_lidar;
    use crate::voxel_map::VoxelCell;
    use image::Rgb;
    use nalgebra::UnitQuaternion;

    const VOXEL: f32 = 0.025;

    fn camera() -> CameraIntrinsics {
        CameraIntrinsics {
            width: 320,
            height: 240,
            fx: 200.0,
            fy: 200.0,
            cx: 160.0,
            cy: 120.0,
            distortion: [0.0; 5],
        }
    }

    /// カメラ座標（= LiDAR 座標 = ワールド座標）の z=`depth` の平面上に並んだ点。
    fn plane_points(depth: f32) -> Vec<Point3<f32>> {
        let mut points = Vec::new();
        for iy in -10..10 {
            for ix in -15..15 {
                let x = (ix as f32 + 0.5) * 0.05 * depth / 5.0;
                let y = (iy as f32 + 0.5) * 0.05 * depth / 5.0;
                points.push(Point3::new(x, y, depth));
            }
        }
        points
    }

    fn facing_normals(count: usize) -> Vec<Option<Vector3<f32>>> {
        vec![Some(Vector3::new(0.0, 0.0, -1.0)); count]
    }

    /// 各点のセルに色 `rgb` を持つ地図。
    fn colored_map(points: &[Point3<f32>], rgb: [u8; 3]) -> VoxelMap {
        let mut map = VoxelMap::default();
        for point in points {
            let key = voxel_key(point, VOXEL);
            let mut cell = VoxelCell::from_key(&key, VOXEL, *point, 0);
            cell.rgb = Some(rgb);
            map.insert(key, cell);
        }
        map
    }

    struct Scene {
        image: RgbImage,
        points: Vec<Point3<f32>>,
        normals: Vec<Option<Vector3<f32>>>,
        map: VoxelMap,
        motion: Option<CameraMotion>,
    }

    impl Scene {
        fn new(value: u8, map_value: Option<u8>) -> Self {
            let points = plane_points(5.0);
            let normals = facing_normals(points.len());
            let map = map_value.map_or_else(VoxelMap::default, |v| colored_map(&points, [v; 3]));
            Self {
                image: RgbImage::from_pixel(320, 240, Rgb([value; 3])),
                points,
                normals,
                map,
                motion: None,
            }
        }

        fn select(&self, config: &ColorSelectionConfig) -> FrameColorSelection {
            let camera = camera();
            select_frame_colors(
                config,
                &FrameColorInput {
                    image: &self.image,
                    camera: &camera,
                    camera_from_lidar_start: &Matrix4::identity(),
                    world_from_lidar_start: &Matrix4::identity(),
                    points: &self.points,
                    normals: &self.normals,
                    motion: self.motion.as_ref(),
                    map: &self.map,
                    map_voxel_size: VOXEL,
                },
            )
        }
    }

    fn accepted(selection: &FrameColorSelection) -> Vec<ColorObservation> {
        selection
            .observations
            .as_ref()
            .expect("frame must be colored")
            .iter()
            .flatten()
            .copied()
            .collect()
    }

    #[test]
    fn pixel_speed_of_a_pan_is_fx_times_rate_at_the_image_center() {
        let motion = CameraMotion {
            angular_velocity: Vector3::new(0.0, 0.2, 0.0),
            linear_velocity: Vector3::zeros(),
        };
        let speed = motion
            .pixel_speed(&Point3::new(0.0, 0.0, 5.0), &camera())
            .unwrap();
        assert!((speed - 200.0 * 0.2).abs() < 1e-9, "speed={speed}");

        // 数値微分: dt 後のカメラから見た同じ点の画素位置の変化と一致する
        let mut distorted = camera();
        distorted.distortion = [-0.13, 0.03, -0.002, 0.001, 0.0];
        let motion = CameraMotion {
            angular_velocity: Vector3::new(0.3, -0.5, 0.2),
            linear_velocity: Vector3::new(0.4, -0.1, 0.6),
        };
        let point = Point3::new(0.7, -0.4, 3.0);
        let dt = 1e-6;
        let moved = Point3::from(
            point.coords
                + (-motion.angular_velocity.cross(&point.coords) - motion.linear_velocity) * dt,
        );
        let (before, _) = project_camera_point_subpixel(&point, &distorted).unwrap();
        let (after, _) = project_camera_point_subpixel(&moved, &distorted).unwrap();
        let numeric = (after - before).norm() / dt;
        let analytic = motion.pixel_speed(&point, &distorted).unwrap();
        assert!(
            (numeric - analytic).abs() < 1e-3 * analytic,
            "{numeric} vs {analytic}"
        );
    }

    #[test]
    fn camera_motion_converts_gyro_to_camera_axes_and_adds_lever_arm_velocity() {
        // 静止した状態で LiDAR の z 軸まわりに 1 rad/s。カメラは LiDAR の +y 0.2 m。
        let imu: Vec<IMU> = (0..=20)
            .map(|i| IMU {
                timestamp: i as f64 * 0.005,
                angular_velocity: [0.0, 0.0, 1.0],
                linear_acceleration: [0.0, 0.0, 1.0],
            })
            .collect();
        let model = ImuModel::new(UnitQuaternion::identity(), 9.80665);
        let state = NavState::default();
        let extrinsic = build_camera_from_lidar(Vector3::new(0.0, 0.2, 0.0), 0.0, 0.0, 0.0);
        let motion = camera_motion_at(
            &imu,
            &model,
            &state,
            &Vector3::zeros(),
            &extrinsic,
            0.05,
            0.01,
        )
        .unwrap();
        // カメラ座標: x = -y_lidar, y = -z_lidar, z = x_lidar
        assert!((motion.angular_velocity - Vector3::new(0.0, -1.0, 0.0)).norm() < 1e-12);
        // カメラ中心の速度 = ω × r = (0,0,1) × (0,0.2,0) = (-0.2,0,0)（LiDAR）→ カメラの z = -0.2
        assert!(
            (motion.linear_velocity - Vector3::new(0.0, 0.0, -0.2)).norm() < 1e-12,
            "{:?}",
            motion.linear_velocity
        );

        // ジャイロバイアスは引かれる
        let biased = NavState {
            bg: Vector3::new(0.0, 0.0, 1.0),
            ..NavState::default()
        };
        let still = camera_motion_at(
            &imu,
            &model,
            &biased,
            &Vector3::zeros(),
            &extrinsic,
            0.05,
            0.01,
        )
        .unwrap();
        assert!(still.angular_velocity.norm() < 1e-12);
        assert!(still.linear_velocity.norm() < 1e-12);
    }

    #[test]
    fn frame_darker_than_the_map_is_not_colored() {
        // 地図の壁は 150、このフレームでは 60 → g = 0.4（暗すぎ）
        let selection = Scene::new(60, Some(150)).select(&ColorSelectionConfig::default());
        assert_eq!(selection.report.status, "too_dark");
        assert!(selection.observations.is_none());
        let g = selection.report.exposure_ratio.unwrap();
        assert!((g - 0.4).abs() < 1e-9, "g={g}");
    }

    #[test]
    fn exposure_ratio_sets_the_exposure_term_and_rescales_existing_colors() {
        let config = ColorSelectionConfig::default();
        // 同じ明るさ: g = 1
        let same = Scene::new(150, Some(150)).select(&config);
        let same_obs = accepted(&same);
        assert_eq!(same.report.status, "colored");
        assert!(
            same_obs
                .iter()
                .all(|o| (o.existing_score_scale - 1.0).abs() < 1e-6)
        );

        // 少し暗い（g = 0.8）: 露光の項 0.64
        let darker = accepted(&Scene::new(120, Some(150)).select(&config));
        let ratio = darker[0].score / same_obs[0].score;
        assert!((ratio - 0.64).abs() < 1e-4, "ratio={ratio}");

        // 地図の方が暗い（g = 2）: 新しい観測は減点せず、既存のスコアを 1/4 にして比べる
        let brighter = accepted(&Scene::new(200, Some(100)).select(&config));
        assert!((brighter[0].score - same_obs[0].score).abs() < 1e-6);
        assert!(
            brighter
                .iter()
                .all(|o| (o.existing_score_scale - 0.25).abs() < 1e-6)
        );
    }

    #[test]
    fn unknown_exposure_is_colored_with_a_low_provisional_term() {
        let config = ColorSelectionConfig::default();
        let unknown = Scene::new(150, None).select(&config);
        assert_eq!(unknown.report.status, "colored");
        assert_eq!(unknown.report.exposure_ratio, None);
        let known = Scene::new(150, Some(150)).select(&config);
        let ratio = accepted(&unknown)[0].score / accepted(&known)[0].score;
        assert!((ratio - config.unknown_exposure_term as f32).abs() < 1e-5);
    }

    #[test]
    fn saturated_pixels_are_rejected_and_a_mostly_saturated_frame_is_skipped() {
        let config = ColorSelectionConfig::default();
        // 点が写る範囲（u = 131..189）の左 7 列分だけ白飛び（約 23%）
        let mut scene = Scene::new(150, Some(150));
        for (x, _, pixel) in scene.image.enumerate_pixels_mut() {
            if x < 145 {
                *pixel = Rgb([255, 180, 120]);
            }
        }
        let partial = scene.select(&config);
        assert_eq!(partial.report.status, "colored");
        assert!(partial.report.rejected_saturated > 0);
        assert!(accepted(&partial).iter().all(|o| o.rgb == [150; 3]));

        let all = Scene::new(255, Some(150)).select(&config);
        assert_eq!(all.report.status, "too_saturated");
        assert!(all.observations.is_none());
    }

    #[test]
    fn fast_rotation_skips_the_frame_and_moderate_rotation_lowers_the_score() {
        let config = ColorSelectionConfig::default();
        let still = accepted(&Scene::new(150, Some(150)).select(&config));

        // 幅 320 の画像で fx = 200。基準幅 960 換算のブレ量 = 200·ω·0.01·3 = 6ω [px]
        let mut scene = Scene::new(150, Some(150));
        scene.motion = Some(CameraMotion {
            angular_velocity: Vector3::new(0.0, 0.5, 0.0), // 中心で 3 px
            linear_velocity: Vector3::zeros(),
        });
        let fast = scene.select(&config);
        assert_eq!(fast.report.status, "too_blurry");
        assert!(fast.report.median_blur_px.unwrap() > config.max_frame_blur_px);
        assert!(fast.observations.is_none());

        scene.motion = Some(CameraMotion {
            angular_velocity: Vector3::new(0.0, 0.1, 0.0), // 中心で 0.6 px
            linear_velocity: Vector3::zeros(),
        });
        let moderate = scene.select(&config);
        assert_eq!(moderate.report.status, "colored");
        let blur = moderate.report.median_blur_px.unwrap();
        assert!(blur > 0.5 && blur < 0.8, "blur={blur}");
        let moderate_obs = accepted(&moderate);
        assert!(
            moderate_obs
                .iter()
                .zip(&still)
                .all(|(m, s)| m.score < s.score)
        );
    }

    #[test]
    fn closer_and_more_frontal_views_score_higher_and_grazing_views_are_rejected() {
        let config = ColorSelectionConfig::default();
        let far = accepted(&Scene::new(150, Some(150)).select(&config));

        let mut near = Scene::new(150, Some(150));
        near.points = plane_points(2.0);
        near.normals = facing_normals(near.points.len());
        near.map = colored_map(&near.points, [150; 3]);
        let near = accepted(&near.select(&config));
        assert!(near[0].score > far[0].score * 1.5);

        // 法線を 60° 傾ける: |cos| ≈ 0.5 前後 → スコアが下がる
        let mut tilted = Scene::new(150, Some(150));
        let tilt = 60f32.to_radians();
        tilted.normals =
            vec![Some(Vector3::new(tilt.sin(), 0.0, -tilt.cos())); tilted.points.len()];
        let tilted_obs = accepted(&tilted.select(&config));
        assert!(tilted_obs.iter().all(|o| o.score < far[0].score * 0.7));

        // 80° 傾ける: ほぼかすめる角度なので使わない
        let grazing = 80f32.to_radians();
        tilted.normals =
            vec![Some(Vector3::new(grazing.sin(), 0.0, -grazing.cos())); tilted.points.len()];
        let grazing = tilted.select(&config);
        assert!(grazing.report.rejected_oblique > 0);
        assert!(grazing.report.observations < grazing.report.candidates);
    }

    /// 選択 → GlobalMap 更新を数フレーム続ける。暗いフレームの色は地図に一切入らず、
    /// 露光比が分からず仮のスコアで付いた色は、後の普通のフレームで置き換わる。
    #[test]
    fn dark_frames_never_reach_the_map_and_later_good_frames_replace_provisional_colors() {
        use crate::voxel_map::{LOCALMap, LocalMapConfig, WorldMapUpdateFilterConfig};

        let config = ColorSelectionConfig::default();
        let camera = camera();
        let points = plane_points(5.0);
        let normals = facing_normals(points.len());
        let mut map = LOCALMap::new(LocalMapConfig {
            index_voxel_size: VOXEL,
            max_points_per_voxel: 20,
            min_points_per_voxel: 2,
            min_observed_frames_per_voxel: 2,
            max_frames: 50,
            max_distance: 150.0,
        });
        let filter = WorldMapUpdateFilterConfig {
            mature_plane_search_radius_voxels: 2,
            min_mature_observed_frames: 3,
            accept_distance_m: 0.02,
            pending_distance_m: 0.10,
            project_accepted_points: true,
            pending_max_age_frames: 30,
        };
        let run_frame = |map: &mut LOCALMap, value: u8| {
            let image = RgbImage::from_pixel(320, 240, Rgb([value; 3]));
            let selection = select_frame_colors(
                &config,
                &FrameColorInput {
                    image: &image,
                    camera: &camera,
                    camera_from_lidar_start: &Matrix4::identity(),
                    world_from_lidar_start: &Matrix4::identity(),
                    points: &points,
                    normals: &normals,
                    motion: None,
                    map: &map.voxel_map,
                    map_voxel_size: VOXEL,
                },
            );
            let stats = map.update_world_map_filtered_with_rgb(
                &points,
                selection.observations.as_deref(),
                &Matrix4::identity(),
                &filter,
            );
            (selection.report, stats.color_updates)
        };
        let map_colors = |map: &LOCALMap| -> Vec<[u8; 3]> {
            map.voxel_map.values().filter_map(|cell| cell.rgb).collect()
        };

        let (first, updates) = run_frame(&mut map, 150);
        assert_eq!(first.status, "colored");
        assert_eq!(first.exposure_ratio, None);
        assert_eq!(updates, points.len());

        let (dark, updates) = run_frame(&mut map, 60);
        assert_eq!(dark.status, "too_dark");
        assert_eq!(updates, 0);
        assert!(map_colors(&map).iter().all(|rgb| *rgb == [150; 3]));

        let (good, updates) = run_frame(&mut map, 160);
        assert_eq!(good.status, "colored");
        assert!((good.exposure_ratio.unwrap() - 160.0 / 150.0).abs() < 1e-9);
        assert_eq!(updates, points.len());
        let colors = map_colors(&map);
        assert_eq!(colors.len(), points.len());
        assert!(colors.iter().all(|rgb| *rgb == [160; 3]));

        // 同じ条件の次のフレームでは置き換わらない（スコアが同じで、平均もしない）
        let (_, updates) = run_frame(&mut map, 160);
        assert_eq!(updates, 0);
    }

    #[test]
    fn points_outside_the_central_crop_or_depth_range_are_not_candidates() {
        let config = ColorSelectionConfig::default();
        let mut scene = Scene::new(150, Some(150));
        scene.points = vec![
            Point3::new(0.0, 0.0, 5.0),  // 中心
            Point3::new(3.9, 0.0, 5.0),  // u = 316: 中央 75% の外
            Point3::new(0.0, 0.0, 0.1),  // 近すぎ
            Point3::new(0.0, 0.0, -2.0), // カメラの後ろ
            Point3::new(f32::NAN, 0.0, 5.0),
        ];
        scene.normals = facing_normals(scene.points.len());
        let selection = scene.select(&config);
        assert_eq!(selection.report.candidates, 1);
        let observations = selection.observations.unwrap();
        assert_eq!(observations.len(), 5);
        assert!(observations[0].is_some());
        assert!(observations[1..].iter().all(Option::is_none));
    }
}
