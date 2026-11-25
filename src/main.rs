use anyhow::{Result, bail};
use nalgebra::{Matrix3, Matrix4, Point3, Quaternion, UnitQuaternion, Vector3};
use re_lidar_slam::{
    camera::{
        CameraIntrinsics, build_camera_from_lidar, colorize_world_points, project_camera_point,
    },
    color_selection::{
        ColorSelectionConfig, FrameColorInput, camera_motion_at, select_frame_colors,
    },
    dense_color_map::{DenseColorMap, DenseColorMapConfig},
    deskew_points::deskew_points,
    eskf::{
        Eskf, EskfConfig, ImuModel, ImuNoise, InitialStd, NavState, STANDARD_GRAVITY_MPS2,
        StaticInitConfig, estimate_static_init,
    },
    file_handler::{
        find_nearest_image_frame, load_image_meta, load_imu_data, load_pcd_files, load_pcd_xyzit,
        load_rgb_image, save_pcd_xyz, save_pcd_xyzrgb, save_pcd_xyzrgb_frame,
    },
    find_nearest_points::pickup_valid_source_points,
    icp::{
        Vector6f, build_robust_point_to_plane_system, compute_rmse, compute_robust_cost,
        solve_icp_delta_observable,
    },
    predict_pose_by_imu::{align_imu_timestamps, build_rotation_trajectory},
    types::{
        ColoredPoint, ColoringLog, CurrentFrameInfo, FrameLog, FrameRegistrationResult, FrameTiming, IMU,
        ImageFrameMetadata, ImageRegistrationResult, LidarRegistrationResult, PointXYZ,
        PointXYZRGB, RegistrationPose, SLAMMap, VisualAlignmentLog,
    },
    visual_alignment::{VisualAligner, VisualAlignmentConfig, VisualFrame},
    voxel_map::{
        ColorObservation, LOCALMap, LocalMapConfig, SurfaceFilterConfig, SurfaceStatus,
        WorldMapUpdateFilterConfig,
    },
    voxelization::{
        voxel_downsample_colored_points, voxel_downsample_points,
        voxel_downsample_points_and_maps_dual,
    },
};
use std::{
    ffi::OsString,
    time::{Duration, Instant},
};

const DATASET_DIR: &str = "/mnt/nas/share/avia/11222026/01";
// const DATASET_DIR: &str = "/mnt/nas/share/airy96/06212026/park05";
const SAVE_ROOT_DIR: &str = "data/output/11222026";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LidarModel {
    Mid70,
    Airy96,
    Avia,
}

impl LidarModel {
    fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "mid70" | "mid-70" => Ok(Self::Mid70),
            "airy96" | "airy-96" | "airy" => Ok(Self::Airy96),
            "avia" | "livox-avia" => Ok(Self::Avia),
            _ => bail!("unsupported LiDAR model '{value}'; expected 'mid70', 'airy96', or 'avia'"),
        }
    }

    fn input_subdir(self) -> &'static str {
        match self {
            Self::Mid70 => "mid-70",
            Self::Airy96 => "",
            Self::Avia => "avia",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Mid70 => "mid70",
            Self::Airy96 => "airy96",
            Self::Avia => "avia",
        }
    }

    fn imu_to_lidar_rotation(self) -> UnitQuaternion<f64> {
        match self {
            Self::Mid70 => make_imu_to_mid70_rotation(),
            Self::Airy96 => make_imu_to_airy96_rotation(),
            Self::Avia => make_imu_to_avia_rotation(),
        }
    }

    /// IMU原点をLiDAR座標系で表した位置。IMU積分位置からLiDAR原点位置への
    /// レバーアーム補正（predict_pose_by_imu）に使う。既知の値がないセンサーは
    /// ゼロ（IMUとLiDARが同一原点とみなす、従来どおりの挙動）とする。
    fn imu_origin_in_lidar_frame_m(self) -> Vector3<f64> {
        match self {
            Self::Mid70 | Self::Airy96 => Vector3::zeros(),
            Self::Avia => Vector3::new(
                IMU_TO_AVIA_TRANSLATION_X_M,
                IMU_TO_AVIA_TRANSLATION_Y_M,
                IMU_TO_AVIA_TRANSLATION_Z_M,
            ),
        }
    }
}

/// 画像タイムスタンプに足す時刻オフセットの既定値 [s]。
/// FAST-LIVO2 の `time_offset/img_time_offset` と同じく、画像ヘッダ時刻 + offset を
/// LiDAR/IMU 時刻系での撮像時刻とみなす。
const DEFAULT_IMAGE_TIME_OFFSET_SEC: f64 = 0.1;
/// 誤入力（ms を秒として渡す等）を弾くための上限 [s]。
const MAX_ABS_IMAGE_TIME_OFFSET_SEC: f64 = 1.0;

#[derive(Debug, Clone, PartialEq)]
struct RunOptions {
    lidar_model: LidarModel,
    visual_alignment_enabled: bool,
    /// 画像タイムスタンプに足す時刻オフセット [s]。
    image_time_offset_sec: f64,
    /// 出力先を `<SAVE_ROOT_DIR>/<lidar>_<tag>` に分ける（比較実行で上書きしないため）。
    output_tag: Option<String>,
}

enum CommandLineAction {
    Run(RunOptions),
    Help,
}

fn parse_image_time_offset(value: &str) -> Result<f64> {
    let offset: f64 = value
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("--image-time-offset requires seconds, e.g. 0.1 or -0.02"))?;
    if !offset.is_finite() || offset.abs() > MAX_ABS_IMAGE_TIME_OFFSET_SEC {
        bail!(
            "--image-time-offset must be finite and within ±{MAX_ABS_IMAGE_TIME_OFFSET_SEC} s \
             (got {value})"
        );
    }
    Ok(offset)
}

fn parse_output_tag(value: &str) -> Result<String> {
    let valid = !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '+'));
    if !valid || value.starts_with('.') {
        bail!("--output-tag must be non-empty and use only [A-Za-z0-9._+-] (got '{value}')");
    }
    Ok(value.to_string())
}

fn parse_command_line<I>(args: I) -> Result<CommandLineAction>
where
    I: IntoIterator<Item = OsString>,
{
    let mut lidar_model = LidarModel::Mid70;
    let mut visual_alignment_enabled = true;
    let mut image_time_offset_sec = DEFAULT_IMAGE_TIME_OFFSET_SEC;
    let mut output_tag = None;
    let mut args = args.into_iter();
    let mut next_value = |args: &mut I::IntoIter, name: &str| -> Result<String> {
        args.next()
            .ok_or_else(|| anyhow::anyhow!("{name} requires a value"))?
            .into_string()
            .map_err(|_| anyhow::anyhow!("{name} value must be valid UTF-8"))
    };

    while let Some(argument) = args.next() {
        let argument = argument
            .into_string()
            .map_err(|_| anyhow::anyhow!("command-line arguments must be valid UTF-8"))?;

        match argument.as_str() {
            "-h" | "--help" => return Ok(CommandLineAction::Help),
            "--no-visual-alignment" => visual_alignment_enabled = false,
            "--lidar" => {
                let value = next_value(&mut args, "--lidar")
                    .map_err(|_| anyhow::anyhow!("--lidar requires 'mid70', 'airy96', or 'avia'"))?;
                lidar_model = LidarModel::parse(&value)?;
            }
            _ if argument.starts_with("--lidar=") => {
                lidar_model = LidarModel::parse(&argument["--lidar=".len()..])?;
            }
            "--image-time-offset" => {
                let value = next_value(&mut args, "--image-time-offset")?;
                image_time_offset_sec = parse_image_time_offset(&value)?;
            }
            _ if argument.starts_with("--image-time-offset=") => {
                image_time_offset_sec =
                    parse_image_time_offset(&argument["--image-time-offset=".len()..])?;
            }
            "--output-tag" => {
                output_tag = Some(parse_output_tag(&next_value(&mut args, "--output-tag")?)?);
            }
            _ if argument.starts_with("--output-tag=") => {
                output_tag = Some(parse_output_tag(&argument["--output-tag=".len()..])?);
            }
            _ => bail!("unknown argument '{argument}'; use --help for usage"),
        }
    }

    Ok(CommandLineAction::Run(RunOptions {
        lidar_model,
        visual_alignment_enabled,
        image_time_offset_sec,
        output_tag,
    }))
}

fn print_usage() {
    println!(
        "Usage: re_lidar_slam [--lidar <mid70|airy96|avia>] [--no-visual-alignment] \
         [--image-time-offset <sec>] [--output-tag <name>]"
    );
    println!("  --lidar  Select the point-cloud sensor (default: mid70)");
    println!("  --no-visual-alignment  Disable the keyframe-based 9x9 visual ESIKF update");
    println!(
        "  --image-time-offset <sec>  Add this offset to every image timestamp before image \
         selection, coloring and visual alignment (default: {DEFAULT_IMAGE_TIME_OFFSET_SEC:+} s)"
    );
    println!(
        "  --output-tag <name>  Save results under <output>/<lidar>_<name> instead of <lidar>"
    );
}

// Mid-70 sparse-cloud preset.
// The upper range matches the range used by the existing Mid-70 datasets.
const MIN_DIST: f32 = 0.5;
const MAX_DIST: f32 = 200.0;

// Airy-96内蔵IMU座標からAiry-96 LiDAR座標への外部回転。
// Quaternion (x, y, z, w): -0.705437, 0.708767, -0.00246579, 0.00097028
// Translation (x, y, z)  : 0.00425, 0.00418, -0.00446  [m]
const IMU_TO_AIRY96_QUAT_X: f64 = -0.705437;
const IMU_TO_AIRY96_QUAT_Y: f64 = 0.708767;
const IMU_TO_AIRY96_QUAT_Z: f64 = -0.00246579;
const IMU_TO_AIRY96_QUAT_W: f64 = 0.00097028;

// Mid-70原点をAiry-96座標で表した位置。回転変換の導出には回転成分のみを使う。
const MID70_ORIGIN_IN_AIRY96_X_M: f64 = 0.0;
const MID70_ORIGIN_IN_AIRY96_Y_M: f64 = 0.0;
const MID70_ORIGIN_IN_AIRY96_Z_M: f64 = -0.06;

// Livox Avia内蔵IMU座標からAvia LiDAR座標への外部回転・並進。
// IMU原点は、LiDAR点群原点から見て以下の位置にある：
// Translation (x, y, z) : -0.04165, -0.02326, 0.02840  [m]
// 回転軸はLiDARと一致しているため回転は単位回転になる。この並進はIMU積分位置を
// LiDAR原点位置へ変換するレバーアーム補正に使う（predict_pose_by_imu参照）。
const IMU_TO_AVIA_TRANSLATION_X_M: f64 = -0.04165;
const IMU_TO_AVIA_TRANSLATION_Y_M: f64 = -0.02326;
const IMU_TO_AVIA_TRANSLATION_Z_M: f64 = 0.02840;

// Mid-70 is sparser than Airy-96. Keep enough spatial support in each local-map
// cell for stable pose estimation; the stricter filters below are used to keep
// wall/ground boundary points out of the global map.
const DOWNSAMPLE_VOXEL_SIZE: f32 = 0.25; // m
const LOCAL_MAP_VOXEL_SIZE: f32 = 0.25; // m
const GLOBAL_MAP_VOXEL_SIZE: f32 = 0.025; // m
/// フレームごとに保存する色付き点群のボクセルサイズ（GlobalMap と同じ解像度）。
const FRAME_COLORED_CLOUD_VOXEL_SIZE: f32 = GLOBAL_MAP_VOXEL_SIZE; // m
/// フレームごとの色付き点群の保存先（save_dir からの相対パス）。
const FRAME_COLORED_CLOUD_SUBDIR: &str = "frames";
// このセンサー構成では、走行全体で 1 フレームあたり約 1800 個の GlobalMap
// voxel が増える。入力フレーム数から最終容量を先に確保し、実行途中の巨大な
// HashMap 再配置（数百 ms のスパイク）を避ける。
const GLOBAL_MAP_EXPECTED_NEW_VOXELS_PER_FRAME: usize = 2048;

const LOCAL_KNN_K: usize = 5;
const GLOBAL_KNN_K: usize = 5;
// 1.0 m cells x 3 cells gives a 3.0 m search radius.
const SEARCH_RANGE: i32 = 3;
const MAX_DIST_FACTOR: f32 = 3.0;
// k近傍点が推定平面から離れてよい最大距離 [m]
const LOCAL_PLANE_POINT_DISTANCE_THRESHOLD_M: f32 = 0.25;
const GLOBAL_PLANE_POINT_DISTANCE_THRESHOLD_M: f32 = 0.20;
const LOCAL_SOURCE_PLANE_SCORE_THRESHOLD: f32 = 0.85;
const GLOBAL_SOURCE_PLANE_SCORE_THRESHOLD: f32 = 0.75;
// Source点と推定平面との最大距離 [m]
const GLOBAL_SOURCE_TO_PLANE_MAX_DISTANCE_M: f32 = 0.08;
// Wall/ground edges tend to form line-like or mixed neighborhoods. Reject them
// during global-map insertion, while retaining sparse Mid-70 observations.
const GLOBAL_MIN_PLANARITY: f32 = 0.10;

// Global map の累積点から局所平面を確定するための遅延・5x5x5 RANSAC/PCA 設定。
// 成熟平面を後続フレームの挿入ゲートに使うため、従来の30フレームより早く確定する。
const SURFACE_CLASSIFICATION_DELAY_FRAMES: u64 = 5;
const SURFACE_NEIGHBOR_RADIUS_VOXELS: i32 = 2;
const SURFACE_MIN_NEIGHBORS: usize = 10;
const SURFACE_MIN_RANSAC_INLIERS: usize = 8;
const SURFACE_MIN_CENTER_OBSERVED_FRAMES: u64 = 2;
const SURFACE_RANSAC_ITERATIONS: usize = 48;
const SURFACE_RANSAC_CONFIDENCE: f64 = 0.999;
const SURFACE_RANSAC_MIN_ITERATIONS: usize = 8;
const SURFACE_RANSAC_INLIER_DISTANCE_M: f32 = 0.025;
const SURFACE_MIN_INLIER_RATIO: f32 = 0.50;
const SURFACE_MIN_PLANARITY: f32 = 0.20;
const SURFACE_MAX_VARIATION: f32 = 0.04;
const SURFACE_MAX_PCA_RMSE_M: f32 = 0.025;
const SURFACE_MAX_CENTER_DISTANCE_M: f32 = 0.025;

// 成熟した GlobalMap 平面に対する新規観測の更新ゲート。
// 2 cm以内は同一面として平面へ射影し、2～10 cmは二重壁候補として保留する。
const WORLD_UPDATE_PLANE_SEARCH_RADIUS_VOXELS: i32 = 2;
const WORLD_UPDATE_MIN_MATURE_OBSERVED_FRAMES: u64 = 3;
const WORLD_UPDATE_ACCEPT_DISTANCE_M: f32 = 0.020;
const WORLD_UPDATE_PENDING_DISTANCE_M: f32 = 0.10;
const WORLD_UPDATE_PENDING_MAX_AGE_FRAMES: u64 = 30;

const ICP_ITERATIONS: usize = 8;
const ICP_HUBER_DELTA_M: f32 = 0.08;
const ICP_MIN_CORRESPONDENCES: usize = 100;
const ICP_MIN_CORRESPONDENCE_RATIO: f32 = 0.05;
const ICP_MIN_OBSERVABLE_RANK: usize = 3;
const ICP_RELATIVE_EIGENVALUE_THRESHOLD: f32 = 0.02;
const ICP_DAMPING: f32 = 1e-6;
const ICP_MAX_FINAL_RMSE_M: f32 = 0.15;
// Limits apply to the ICP correction relative to the IMU prediction, not to
// the vehicle's total frame-to-frame motion.
const ICP_MAX_TRANSLATION_CORRECTION_M: f32 = 0.10;
const ICP_MAX_ROTATION_CORRECTION_DEG: f32 = 2.0;
const ICP_RMSE_CHANGE_THRESHOLD_M: f32 = 1e-4;
const ICP_TRANSLATION_DELTA_THRESHOLD_M: f32 = 0.001;
const ICP_ROTATION_DELTA_THRESHOLD_DEG: f32 = 0.01;
const LOCAL_SOURCE_TO_PLANE_MAX_DISTANCE_M: f32 = 0.15;
const LOCAL_MIN_PLANARITY: f32 = 0.10;
// If normal matching fails after a large IMU Z drift, retry once with a wider
// neighborhood and looser source-to-plane gates. Only the world-Z component of
// this coarse solve is applied; normal ICP must then validate and refine it.
const ICP_VERTICAL_RECOVERY_SEARCH_RANGE: i32 = 6;
const ICP_VERTICAL_RECOVERY_MAX_DIST_FACTOR: f32 = 6.0;
const ICP_VERTICAL_RECOVERY_SOURCE_PLANE_SCORE_THRESHOLD: f32 = 0.0;
const ICP_VERTICAL_RECOVERY_SOURCE_TO_PLANE_MAX_DISTANCE_M: f32 = 0.60;
const ICP_VERTICAL_RECOVERY_MAX_TRANSLATION_CORRECTION_M: f32 = 0.60;
const MIN_IMU_SAMPLES_PER_POINT_CLOUD_FRAME: usize = 5;

const MAX_DIST_FOR_VOXEL_MAP: f32 = 150.0;

// --- GlobalMap の色付け（ベストビュー方式、color_selection.rs） ---
// 画像時刻のカメラ角速度は、この幅（±）の IMU の平均で求める [s]。
const CAMERA_MOTION_HALF_WINDOW_S: f64 = 0.005;

// --- 最終の色付き点群（リアルタイム表示と同じ細かさ、dense_color_map.rs） ---
// フレームごとの色付き点群（frames/cloud_N.pcd と同じ点）を貯めておき、最後に区画ごとに
// 連続したフレームの窓（1 回の訪問）を 1 つ選んで、その点だけを出力する（周回を混ぜない）。
// 選ぶのは、その区画に入る点が最も多い（＝最も細かく写った）訪問。
const DENSE_COLOR_TILE_SIZE_M: f32 = 0.5;
const DENSE_COLOR_VISIT_WINDOW_FRAMES: u32 = 20;
// 一番良い訪問で点のなかったセルだけ、2 番目以降の訪問で埋める（1 なら埋めない）。
const DENSE_COLOR_MAX_VISITS_PER_TILE: usize = 3;
// ブレすぎと判定されたフレーム（color_selection の too_blurry）の点は貯めない。
const DENSE_COLOR_SKIP_BLURRY_FRAMES: bool = true;
// 点ごとのフレーム番号を入れた確認用のファイルも保存する。
const DENSE_COLOR_SAVE_FRAME_INDEX: bool = true;

// --- ESKF ---
// IMU ノイズ密度。データシート値（例: BMI088 はジャイロ約 2.4e-4 rad/s/√Hz、
// 加速度約 1.7e-3 m/s²/√Hz）に、車体振動などの余裕として 5〜10 倍を掛けた値から始める。
const IMU_GYRO_NOISE_DENSITY: f64 = 2.0e-3; // rad/s/√Hz
const IMU_ACCEL_NOISE_DENSITY: f64 = 2.0e-2; // m/s²/√Hz
const IMU_GYRO_BIAS_RANDOM_WALK: f64 = 5.0e-5; // rad/s²/√Hz
const IMU_ACCEL_BIAS_RANDOM_WALK: f64 = 5.0e-4; // m/s³/√Hz
// point-to-plane 残差 1点あたりの標準偏差。IMU ノイズとの「比」が挙動を決める。
const LIDAR_POINT_SIGMA_M: f64 = 0.05;
// 初期共分散。加速度バイアスは最初は小さめにして、p・R・v・b_g の安定を優先する。
const INIT_ATTITUDE_STD_DEG: f64 = 0.5;
const INIT_POSITION_STD_M: f64 = 0.01;
const INIT_VELOCITY_STD_MPS: f64 = 0.05;
const INIT_GYRO_BIAS_STD_RADPS: f64 = 0.005;
const INIT_ACCEL_BIAS_STD_MPS2: f64 = 0.02;
// 先頭の静止区間からの初期化（重力方向・ジャイロバイアス）。
const STATIC_INIT_WINDOW_S: f64 = 1.0;
const STATIC_INIT_MIN_SAMPLES: usize = 50;
const STATIC_INIT_MAX_GYRO_STD_RADPS: f64 = 0.02;
const STATIC_INIT_MAX_ACCEL_STD_MPS2: f64 = 0.15;
const STATIC_INIT_NORMALIZE_ACCEL_MAGNITUDE: bool = true;

fn make_eskf_config() -> EskfConfig {
    EskfConfig {
        imu_noise: ImuNoise {
            gyro_noise_density: IMU_GYRO_NOISE_DENSITY,
            accel_noise_density: IMU_ACCEL_NOISE_DENSITY,
            gyro_bias_random_walk: IMU_GYRO_BIAS_RANDOM_WALK,
            accel_bias_random_walk: IMU_ACCEL_BIAS_RANDOM_WALK,
        },
        initial_std: InitialStd {
            attitude_rad: INIT_ATTITUDE_STD_DEG.to_radians(),
            position_m: INIT_POSITION_STD_M,
            velocity_mps: INIT_VELOCITY_STD_MPS,
            gyro_bias_radps: INIT_GYRO_BIAS_STD_RADPS,
            accel_bias_mps2: INIT_ACCEL_BIAS_STD_MPS2,
        },
        gravity_mps2: STANDARD_GRAVITY_MPS2,
        lidar_point_sigma_m: LIDAR_POINT_SIGMA_M,
    }
}

/// 最初のフレーム開始時刻で ESKF を作る。
///
/// 静止区間が取れれば、重力方向に合わせた初期姿勢（マップの z 軸が真上になる）と
/// ジャイロバイアスで初期化する。取れなければ従来どおり「水平・バイアス 0」とする。
fn initialize_eskf(
    imu_data: &[IMU],
    imu_to_lidar: UnitQuaternion<f64>,
    imu_origin_in_lidar_frame: Vector3<f64>,
    first_frame_start_time: f64,
) -> Eskf {
    let config = make_eskf_config();
    // IMU の加速度は g 単位（従来の predict_pose_by_imu と同じ前提）。
    let mut imu_model = ImuModel::new(imu_to_lidar, STANDARD_GRAVITY_MPS2);
    let mut initial_state = NavState::default();

    let static_init_config = StaticInitConfig {
        window_s: STATIC_INIT_WINDOW_S,
        min_samples: STATIC_INIT_MIN_SAMPLES,
        max_gyro_std_radps: STATIC_INIT_MAX_GYRO_STD_RADPS,
        max_accel_std_mps2: STATIC_INIT_MAX_ACCEL_STD_MPS2,
        normalize_accel_magnitude: STATIC_INIT_NORMALIZE_ACCEL_MAGNITUDE,
    };

    match estimate_static_init(
        imu_data,
        &imu_model,
        first_frame_start_time,
        &static_init_config,
        config.gravity_mps2,
    ) {
        Ok(init) => {
            let (roll, pitch, _) = init.rotation.euler_angles();
            log::info!(
                "ESKF static init: samples={}, roll={:.3} deg, pitch={:.3} deg, \
                 gyro_bias=[{:.5}, {:.5}, {:.5}] rad/s, |a|={:.4} m/s^2 \
                 (scale correction {:.5}), gyro_std={:.4}, accel_std={:.4}",
                init.sample_count,
                roll.to_degrees(),
                pitch.to_degrees(),
                init.gyro_bias.x,
                init.gyro_bias.y,
                init.gyro_bias.z,
                init.mean_accel_norm_mps2,
                init.accel_scale_correction,
                init.gyro_std_radps,
                init.accel_std_mps2,
            );
            imu_model.accel_scale *= init.accel_scale_correction;
            initial_state.rot = init.rotation;
            initial_state.bg = init.gyro_bias;
        }
        Err(error) => {
            log::warn!(
                "ESKF static init failed: {error}; assuming a level sensor with zero bias \
                 (gravity direction errors will leak into velocity)"
            );
        }
    }

    // 状態の原点は IMU 原点。最初の LiDAR 原点がワールド原点に来るようにずらしておく。
    initial_state.pos = initial_state.rot * imu_origin_in_lidar_frame;

    Eskf::new(initial_state, config, imu_model, imu_origin_in_lidar_frame)
}

fn apply_image_time_offset(frames: &mut [ImageFrameMetadata], offset_sec: f64) {
    for frame in frames {
        frame.timestamp_sec += offset_sec;
    }
}

fn count_imu_samples_in_time_range(imu_data: &[IMU], start_time: f64, end_time: f64) -> usize {
    if !start_time.is_finite() || !end_time.is_finite() || start_time > end_time {
        return 0;
    }

    let start_idx = imu_data.partition_point(|sample| sample.timestamp < start_time);
    let end_idx = imu_data.partition_point(|sample| sample.timestamp <= end_time);

    end_idx.saturating_sub(start_idx)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    let RunOptions {
        lidar_model,
        visual_alignment_enabled,
        image_time_offset_sec,
        output_tag,
    } = match parse_command_line(std::env::args_os().skip(1))? {
        CommandLineAction::Run(options) => options,
        CommandLineAction::Help => {
            print_usage();
            return Ok(());
        }
    };
    let load_dir = format!("{DATASET_DIR}/{}", lidar_model.input_subdir());
    let save_dir = match &output_tag {
        Some(tag) => format!("{SAVE_ROOT_DIR}/{}_{tag}", lidar_model.name()),
        None => format!("{SAVE_ROOT_DIR}/{}", lidar_model.name()),
    };
    let imu_to_lidar = lidar_model.imu_to_lidar_rotation();
    let imu_origin_in_lidar_frame = lidar_model.imu_origin_in_lidar_frame_m();
    log::info!(
        "LiDAR model={}, input={}, output={}",
        lidar_model.name(),
        load_dir,
        save_dir,
    );

    // <--- Loading each data --->
    let pcd_dir = format!("{load_dir}/pcd");
    let pcd_files = load_pcd_files(&pcd_dir)?;

    log::debug!(
        "Found {} PCD files in directory: {}",
        pcd_files.len(),
        pcd_dir
    );

    let imu_dir = format!("{load_dir}/imu");
    let imu_file = format!("{}/imu_data.json", imu_dir);
    let imu_data = load_imu_data(&imu_file)?;
    let imu_data = align_imu_timestamps(&imu_data); // Align IMU timestamps to seconds

    let image_meta_dir = format!("{DATASET_DIR}/camera");
    let image_meta_file = format!("{}/frames.json", image_meta_dir);
    let mut image_meta = load_image_meta(&image_meta_file)?;
    // 画像ヘッダ時刻 + offset を撮像時刻とみなす。以後の画像選択・色付け・画像位置合わせ・
    // キーフレームはすべてこの補正後の時刻を使う。
    apply_image_time_offset(&mut image_meta.frames, image_time_offset_sec);
    log::info!(
        "Image time offset = {:+.6} s ({:+.3} ms) applied to {} image timestamps",
        image_time_offset_sec,
        image_time_offset_sec * 1000.0,
        image_meta.frames.len(),
    );
    // <--- Loading each data --->

    // <--- Initialize current frame info --->
    // current_frame_info は ESKF の状態から毎フレーム作り直す（マップ更新・色付けが参照する）。
    let mut current_frame_info = CurrentFrameInfo {
        current_global_pose: Matrix4::<f64>::identity(),
        current_velocity: Vector3::<f64>::zeros(),
    };
    // ESKF は最初のフレーム開始時刻が分かってから初期化する。
    let mut eskf: Option<Eskf> = None;
    // <--- Initialize current frame info --->

    // <--- Initialize SLAM map --->
    let local_map_config = LocalMapConfig {
        index_voxel_size: LOCAL_MAP_VOXEL_SIZE,
        max_points_per_voxel: 20,
        min_points_per_voxel: 5,
        min_observed_frames_per_voxel: 3,
        max_frames: 50,
        max_distance: MAX_DIST_FOR_VOXEL_MAP,
    };

    let global_map_config = LocalMapConfig {
        index_voxel_size: GLOBAL_MAP_VOXEL_SIZE,
        max_points_per_voxel: 20,
        min_points_per_voxel: 2,
        min_observed_frames_per_voxel: 2,
        max_frames: 50,
        max_distance: MAX_DIST_FOR_VOXEL_MAP,
    };
    let expected_global_voxel_capacity = pcd_files
        .len()
        .saturating_mul(GLOBAL_MAP_EXPECTED_NEW_VOXELS_PER_FRAME);
    let mut slam_map = SLAMMap {
        global_voxel_map: LOCALMap::with_voxel_capacity(
            global_map_config,
            expected_global_voxel_capacity,
        ),
        local_voxel_map: LOCALMap::new(local_map_config),
    };
    let surface_filter_config = SurfaceFilterConfig {
        neighbor_radius_voxels: SURFACE_NEIGHBOR_RADIUS_VOXELS,
        min_neighbors: SURFACE_MIN_NEIGHBORS,
        min_ransac_inliers: SURFACE_MIN_RANSAC_INLIERS,
        min_center_observed_frames: SURFACE_MIN_CENTER_OBSERVED_FRAMES,
        ransac_iterations: SURFACE_RANSAC_ITERATIONS,
        ransac_confidence: SURFACE_RANSAC_CONFIDENCE,
        ransac_min_iterations: SURFACE_RANSAC_MIN_ITERATIONS,
        ransac_inlier_distance_m: SURFACE_RANSAC_INLIER_DISTANCE_M,
        min_inlier_ratio: SURFACE_MIN_INLIER_RATIO,
        min_planarity: SURFACE_MIN_PLANARITY,
        max_surface_variation: SURFACE_MAX_VARIATION,
        max_pca_rmse_m: SURFACE_MAX_PCA_RMSE_M,
        max_center_distance_m: SURFACE_MAX_CENTER_DISTANCE_M,
    };
    let world_map_update_filter_config = WorldMapUpdateFilterConfig {
        mature_plane_search_radius_voxels: WORLD_UPDATE_PLANE_SEARCH_RADIUS_VOXELS,
        min_mature_observed_frames: WORLD_UPDATE_MIN_MATURE_OBSERVED_FRAMES,
        accept_distance_m: WORLD_UPDATE_ACCEPT_DISTANCE_M,
        pending_distance_m: WORLD_UPDATE_PENDING_DISTANCE_M,
        project_accepted_points: true,
        pending_max_age_frames: WORLD_UPDATE_PENDING_MAX_AGE_FRAMES,
    };
    // <--- Initialize SLAM map --->

    let mut prev_frame_start_time: f64 = 0.0;
    let mut frame_logs: Vec<FrameLog> = Vec::new();
    // 各フレームの位置合わせ結果（LiDAR / 画像の姿勢と RMSE）。最後に JSON で保存する。
    let mut registration_results: Vec<FrameRegistrationResult> = Vec::with_capacity(pcd_files.len());
    let frame_cloud_dir = format!("{save_dir}/{FRAME_COLORED_CLOUD_SUBDIR}");
    std::fs::create_dir_all(&frame_cloud_dir)?;
    let mut visual_aligner = VisualAligner::new(VisualAlignmentConfig::default());

    let camera_from_lidar = build_camera_from_lidar(
        Vector3::new(0.01, 0.2, 0.035), // カメラ位置 [m]
        -1.0,                           // roll [deg]（未確定）
        -0.95,                          // pitch [deg] 0.18
        -0.28,                          // yaw [deg]
    );
    // GlobalMap の各セルは、スコア（距離 × 角度 × 露光 × 鮮鋭さ）が最も高い 1 回の観測の色を持つ。
    // 暗すぎる・ブレすぎる・白飛びの多い画像のフレームでは色を付けない。
    let color_selection_config = ColorSelectionConfig::default();
    log::info!("Color selection config: {color_selection_config:?}");
    // 最終の色付き点群（区画ごとに 1 回の訪問の点だけを使う）。
    let mut dense_color_map = DenseColorMap::new(DenseColorMapConfig {
        cell_size: FRAME_COLORED_CLOUD_VOXEL_SIZE,
        tile_size: DENSE_COLOR_TILE_SIZE_M,
        visit_window_frames: DENSE_COLOR_VISIT_WINDOW_FRAMES,
        max_visits_per_tile: DENSE_COLOR_MAX_VISITS_PER_TILE,
    });
    log::info!(
        "Dense color map config: {:?}, skip_blurry_frames={}",
        dense_color_map.config,
        DENSE_COLOR_SKIP_BLURRY_FRAMES,
    );

    //
    for (i, pcd_path) in pcd_files.iter().enumerate() {
        let frame_start = Instant::now();
        log::info!("Processing frame {}: {}", i, pcd_path.to_string_lossy());

        let load_pcd_start = Instant::now();
        let source_pcd = load_pcd_xyzit(&pcd_path.to_string_lossy())?;
        let load_pcd_time = load_pcd_start.elapsed();
        // Main per-frame processing excludes file I/O. The full elapsed time is
        // measured separately from frame_start and reported alongside it.
        let frame_processing_start = Instant::now();

        let timestamp_start = Instant::now();
        let current_frame_start_time = source_pcd
            .iter()
            .map(|p| p.timestamp)
            .fold(f64::INFINITY, f64::min);
        let current_frame_end_time = source_pcd
            .iter()
            .map(|p| p.timestamp)
            .fold(f64::NEG_INFINITY, f64::max);
        let timestamp_time = timestamp_start.elapsed();

        let current_frame_mid_time =
            current_frame_start_time + (current_frame_end_time - current_frame_start_time) * 0.5;

        // 中央時刻との差が50 ms以内の画像を採用する。
        let max_image_time_diff_sec = 0.050;

        let coloring_image_select_start = Instant::now();
        let selected_image = find_nearest_image_frame(
            &image_meta.frames,
            current_frame_mid_time,
            max_image_time_diff_sec,
        );
        let coloring_image_select_time = coloring_image_select_start.elapsed();

        if let Some(image) = selected_image {
            let image_path = std::path::Path::new(&image_meta_dir).join(&image.file_name);

            let time_diff_ms = (image.timestamp_sec - current_frame_mid_time) * 1000.0;

            log::info!(
                "Frame {i}: image={} lidar_mid={:.9} image_time={:.9} delta={:+.3} ms",
                image_path.display(),
                current_frame_mid_time,
                image.timestamp_sec,
                time_diff_ms,
            );
        } else {
            log::warn!(
                "Frame {i}: no matching image near lidar_mid={:.9}",
                current_frame_mid_time,
            );
        }

        let coloring_image_load_start = Instant::now();
        let selected_rgb_image: Option<image::RgbImage> = if let Some(metadata) = selected_image {
            let image_path = std::path::Path::new(&image_meta_dir).join(&metadata.file_name);

            let rgb_image = load_rgb_image(&image_path)?;

            log::info!(
                "Frame {i}: loaded RGB image {}x{} \
             (metadata {}x{})",
                rgb_image.width(),
                rgb_image.height(),
                metadata.width,
                metadata.height,
            );

            Some(rgb_image)
        } else {
            None
        };
        let coloring_image_load_time = coloring_image_load_start.elapsed();

        let imu_sample_count = count_imu_samples_in_time_range(
            &imu_data,
            current_frame_start_time,
            current_frame_end_time,
        );
        if imu_sample_count < MIN_IMU_SAMPLES_PER_POINT_CLOUD_FRAME {
            log::warn!(
                "Frame {i}: only {imu_sample_count} IMU samples in point-cloud interval \
                 [{current_frame_start_time:.6}, {current_frame_end_time:.6}] s \
                 (minimum {MIN_IMU_SAMPLES_PER_POINT_CLOUD_FRAME}); \
                 IMU/LiDAR timestamps may be misaligned"
            );
        }

        if i == 0 {
            prev_frame_start_time = current_frame_start_time;
        }

        let filter = eskf.get_or_insert_with(|| {
            let filter = initialize_eskf(
                &imu_data,
                imu_to_lidar,
                imu_origin_in_lidar_frame,
                current_frame_start_time,
            );
            current_frame_info.current_global_pose = filter.lidar_pose();
            current_frame_info.current_velocity = filter.state.vel;
            filter
        });

        // <--- Predict pose by IMU (ESKF propagation) --->
        // 名目状態（バイアスを引いて積分）と共分散 P を前フレーム開始 → 今フレーム開始へ進め、
        // その結果を LiDAR 反復更新の事前分布として固定する。
        let predict_pose_start = Instant::now();
        let imu_steps =
            filter.propagate(&imu_data, prev_frame_start_time, current_frame_start_time);
        let predicted_state = filter.state;
        let predicted_pose = filter.lidar_pose();
        let mut lidar_update = filter.begin_lidar_update();
        if lidar_update.is_none() {
            log::warn!(
                "Frame {i}: ESKF covariance is not positive definite; LiDAR update disabled"
            );
        }
        let predict_pose_time = predict_pose_start.elapsed();
        log::debug!(
            "Frame {i}: ESKF propagated {imu_steps} IMU steps over {:.3} ms",
            (current_frame_start_time - prev_frame_start_time) * 1000.0,
        );
        // <--- Predict pose by IMU (ESKF propagation) --->

        // <--- Build rotation trajectory --->
        // deskew にも推定済みのジャイロバイアスを反映する。
        let rotation_trajectory_start = Instant::now();
        let rotation_traj = build_rotation_trajectory(
            &imu_data,
            current_frame_start_time,
            current_frame_end_time,
            &imu_to_lidar,
            &filter.state.bg,
        );
        let rotation_trajectory_time = rotation_trajectory_start.elapsed();
        // <--- Build rotation trajectory --->

        // --- Deskew source pcd ---
        let deskew_start = Instant::now();
        let deskewed_points = deskew_points(
            &source_pcd,
            &rotation_traj,
            &imu_to_lidar,
            current_frame_start_time,
            MIN_DIST,
            MAX_DIST,
        );
        let deskew_time = deskew_start.elapsed();
        // --- Deskew source pcd ---

        // --- Downsample deskewed points ---
        let voxel_start = Instant::now();
        let (local_source, global_source) = voxel_downsample_points_and_maps_dual(
            &deskewed_points,
            DOWNSAMPLE_VOXEL_SIZE,
            GLOBAL_MAP_VOXEL_SIZE,
        );
        let voxel_end = voxel_start.elapsed();
        let downsampled_source_points_for_local = local_source.points;
        let downsampled_source_points_for_global = global_source.points;
        log::debug!(
            "Frame {i}: Downsampled {} points → {} points in {:.2?}",
            deskewed_points.len(),
            downsampled_source_points_for_local.len(),
            voxel_end
        );
        // --- Downsample deskewed points ---

        // The source voxel maps were populated while emitting the centroids.
        let source_voxel_map = local_source.voxel_map;
        let source_voxel_map_for_global = global_source.voxel_map;
        let build_map_end = Duration::ZERO;

        // --- ICP (Point to Plane) ---
        // IMU 予測姿勢を初期値として (R, t) を取り出す
        let pred_pose = predicted_pose.cast::<f32>();
        let pred_r: Matrix3<f32> = pred_pose.fixed_view::<3, 3>(0, 0).into();
        let pred_t: Vector3<f32> = pred_pose.fixed_view::<3, 1>(0, 3).into();
        let mut r_mat = pred_r;
        let mut t_vec = pred_t;
        // ESKF の反復点（15次元）。ICP の (r_mat, t_vec) は常にこの状態から導出する。
        let mut iter_state = predicted_state;
        let mut eskf_update_applied = false;
        let mut eskf_prior_cost: Option<f64> = None;

        let local_map_was_empty = slam_map.local_voxel_map.voxel_map.is_empty();
        let source_point_count = source_voxel_map.len();
        let min_correspondences = ICP_MIN_CORRESPONDENCES
            .max((source_point_count as f32 * ICP_MIN_CORRESPONDENCE_RATIO).ceil() as usize);
        let mut previous_rmse: Option<f32> = None;
        let mut final_rmse: Option<f32> = None;
        let mut final_correspondence_count = 0usize;
        let mut final_correspondence_ratio = 0.0f32;
        let mut final_observable_rank = 0usize;
        let mut final_eigenvalue_ratio: Option<f32> = None;
        let mut icp_ok = false; // ICP が有効な解を得られたか
        let mut vertical_recovery_attempted = false;
        let mut use_vertical_recovery_correspondences = false;
        let mut vertical_recovery_pose = false;
        let mut icp_correspondence_search_time = Duration::ZERO;
        let mut icp_linear_system_time = Duration::ZERO;
        let mut icp_solver_time = Duration::ZERO;

        let loop_start = Instant::now();

        if local_map_was_empty {
            log::debug!("Frame {i}: local map empty, skipping ICP");
        } else if let Some(update) = lidar_update.as_mut() {
            for _iter in 0..ICP_ITERATIONS {
                // 対応点をピックアップ
                // - source はローカル座標、target (local_voxel_map) はワールド座標
                // - 現在の (R,t) 推定値で source をワールド変換してから近傍探索
                let recovery_iteration = use_vertical_recovery_correspondences;
                let (
                    search_range,
                    max_dist_factor,
                    source_plane_score_threshold,
                    source_to_plane_max_distance_m,
                ) = if recovery_iteration {
                    (
                        ICP_VERTICAL_RECOVERY_SEARCH_RANGE,
                        ICP_VERTICAL_RECOVERY_MAX_DIST_FACTOR,
                        ICP_VERTICAL_RECOVERY_SOURCE_PLANE_SCORE_THRESHOLD,
                        ICP_VERTICAL_RECOVERY_SOURCE_TO_PLANE_MAX_DISTANCE_M,
                    )
                } else {
                    (
                        SEARCH_RANGE,
                        MAX_DIST_FACTOR,
                        LOCAL_SOURCE_PLANE_SCORE_THRESHOLD,
                        LOCAL_SOURCE_TO_PLANE_MAX_DISTANCE_M,
                    )
                };
                let pickup_start = Instant::now();
                let correspondences = pickup_valid_source_points::<LOCAL_KNN_K>(
                    &source_voxel_map,
                    &slam_map.local_voxel_map.voxel_map,
                    slam_map.local_voxel_map.config.index_voxel_size,
                    search_range,
                    max_dist_factor,
                    LOCAL_PLANE_POINT_DISTANCE_THRESHOLD_M,
                    source_plane_score_threshold,
                    Some(source_to_plane_max_distance_m),
                    Some(LOCAL_MIN_PLANARITY),
                    &r_mat,
                    &t_vec,
                );
                let correspondence_ratio = if source_point_count > 0 {
                    correspondences.len() as f32 / source_point_count as f32
                } else {
                    0.0
                };
                let pickup_end = pickup_start.elapsed();
                icp_correspondence_search_time += pickup_end;
                log::debug!(
                    "ICP iter {_iter}{}: Picked up {} correspondences ({:.1}%) in {:.2?}",
                    if recovery_iteration {
                        " [vertical recovery]"
                    } else {
                        ""
                    },
                    correspondences.len(),
                    correspondence_ratio * 100.0,
                    pickup_end
                );

                // A coarse Z estimate only needs a stable planar subset. The
                // recovered pose is still required to pass the normal 5% gate.
                let required_correspondences = if recovery_iteration {
                    ICP_MIN_CORRESPONDENCES
                } else {
                    min_correspondences
                };
                if correspondences.len() < required_correspondences {
                    log::warn!(
                        "ICP iter {_iter}: insufficient correspondences: {} < {}",
                        correspondences.len(),
                        required_correspondences,
                    );
                    if !vertical_recovery_attempted {
                        vertical_recovery_attempted = true;
                        use_vertical_recovery_correspondences = true;
                        log::warn!(
                            "ICP iter {_iter}: retrying with world-Z-only recovery matching"
                        );
                        continue;
                    }
                    break;
                }

                // 線形システム構築
                let system_start = Instant::now();
                let system = build_robust_point_to_plane_system(
                    &correspondences,
                    &r_mat,
                    &t_vec,
                    ICP_HUBER_DELTA_M,
                );
                let system_end = system_start.elapsed();
                icp_linear_system_time += system_end;

                let solve_start = Instant::now();
                let solve_result = solve_icp_delta_observable(
                    &system,
                    ICP_DAMPING,
                    ICP_RELATIVE_EIGENVALUE_THRESHOLD,
                );
                let solve_end = solve_start.elapsed();
                icp_solver_time += solve_end;
                let Some(solve_result) = solve_result else {
                    log::warn!("ICP iter {_iter}: observable solve failed");
                    break;
                };
                if solve_result.observable_rank < ICP_MIN_OBSERVABLE_RANK {
                    log::warn!(
                        "ICP iter {_iter}: degenerate geometry (observable rank {} < {})",
                        solve_result.observable_rank,
                        ICP_MIN_OBSERVABLE_RANK,
                    );
                    break;
                }

                // ESKF の反復更新: ICP の 6×6 系に IMU 予測（事前分布）を足して 15次元で解く。
                // 上の observable solve は退化判定とログ用で、通常反復の更新量には使わない。
                // 退化方向は事前項が予測側へ寄せるので、固有値の切り捨ては不要になる。
                let candidate_state = if recovery_iteration {
                    // 粗い Z 補正: ICP 単体の解のワールド Z 並進だけを使い、反復の開始点をずらす。
                    // 予測の Z が信用できなかったことを事前分布にも反映し、
                    // 後続の通常反復で事前項が予測の Z へ引き戻さないようにする。
                    let dz = retain_world_z_translation(&solve_result.delta)[5] as f64;
                    update.inflate_position_variance(2, dz.abs());
                    let mut shifted = iter_state;
                    shifted.pos.z += dz;
                    shifted
                } else {
                    let eskf_solve_start = Instant::now();
                    let step = update.solve_step(&iter_state, &system);
                    icp_solver_time += eskf_solve_start.elapsed();
                    let Some(dx) = step else {
                        log::warn!("ICP iter {_iter}: ESKF iterated update solve failed");
                        break;
                    };
                    iter_state.boxplus(&dx)
                };
                let (candidate_r, candidate_t) = update.lidar_rt_f32(&candidate_state);

                let (delta_translation_m, delta_rotation_deg) =
                    pose_correction_magnitudes(&r_mat, &t_vec, &candidate_r, &candidate_t);
                let (total_translation_correction_m, total_rotation_correction_deg) =
                    pose_correction_magnitudes(&pred_r, &pred_t, &candidate_r, &candidate_t);
                let max_translation_correction_m = if recovery_iteration || vertical_recovery_pose {
                    ICP_VERTICAL_RECOVERY_MAX_TRANSLATION_CORRECTION_M
                } else {
                    ICP_MAX_TRANSLATION_CORRECTION_M
                };

                if total_translation_correction_m > max_translation_correction_m
                    || total_rotation_correction_deg > ICP_MAX_ROTATION_CORRECTION_DEG
                {
                    log::warn!(
                        "ICP iter {_iter}: correction exceeds prediction gate: \
                         translation={total_translation_correction_m:.4} m, \
                         rotation={total_rotation_correction_deg:.3} deg",
                    );
                    break;
                }

                // 目的関数 = 点のロバストコスト/σ² + 事前項。両方を含めて改善を確認する。
                let candidate_cost = compute_robust_cost(
                    &correspondences,
                    &candidate_r,
                    &candidate_t,
                    ICP_HUBER_DELTA_M,
                );
                let current_total_cost = update.total_cost(&iter_state, system.cost);
                let candidate_total_cost = update.total_cost(&candidate_state, candidate_cost);
                if !candidate_total_cost.is_finite()
                    || candidate_total_cost > current_total_cost * (1.0 + 1e-5) + 1e-8
                {
                    log::warn!(
                        "ICP iter {_iter}: total cost did not improve ({:.6} -> {:.6}; \
                         robust {:.6} -> {:.6})",
                        current_total_cost,
                        candidate_total_cost,
                        system.cost,
                        candidate_cost,
                    );
                    break;
                }

                let applied_dz = candidate_state.pos.z - iter_state.pos.z;
                iter_state = candidate_state;
                r_mat = candidate_r;
                t_vec = candidate_t;
                icp_ok = true;
                if recovery_iteration {
                    vertical_recovery_pose = true;
                    use_vertical_recovery_correspondences = false;
                    previous_rmse = None;
                    log::warn!(
                        "ICP iter {_iter}: applied coarse world-Z recovery dz={applied_dz:.4} m; \
                         returning to normal matching for validation",
                    );
                }
                log::debug!(
                    "ICP iter {_iter}: point-to-plane system={:.2?}, solve={:.2?}",
                    system_end,
                    solve_end,
                );

                // RMSE を計算して収束チェック
                let rmse = compute_rmse(&correspondences, &r_mat, &t_vec);
                final_rmse = Some(rmse);
                final_correspondence_count = correspondences.len();
                final_correspondence_ratio = correspondence_ratio;
                final_observable_rank = solve_result.observable_rank;
                final_eigenvalue_ratio = Some(solve_result.min_observable_eigenvalue_ratio);
                log::debug!(
                    "ICP iter {_iter}: used={}, robust_cost={:.6}->{:.6}, rmse={:.6}, \
                     rank={}, min_eigen_ratio={:.3e}, delta_t={:.4} m, delta_r={:.3} deg",
                    system.used_count,
                    system.cost,
                    candidate_cost,
                    rmse,
                    solve_result.observable_rank,
                    solve_result.min_observable_eigenvalue_ratio,
                    delta_translation_m,
                    delta_rotation_deg,
                );

                let rmse_converged = previous_rmse
                    .is_some_and(|previous| (previous - rmse).abs() < ICP_RMSE_CHANGE_THRESHOLD_M);
                previous_rmse = Some(rmse);
                if recovery_iteration {
                    continue;
                }
                if rmse_converged
                    && delta_translation_m < ICP_TRANSLATION_DELTA_THRESHOLD_M
                    && delta_rotation_deg < ICP_ROTATION_DELTA_THRESHOLD_DEG
                {
                    log::debug!(
                        "ICP converged at iter {_iter}: delta_t={delta_translation_m:.3e} m, \
                         delta_r={delta_rotation_deg:.3e} deg",
                    );
                    break;
                }
            }

            // Rebuild correspondences at the accepted pose so stale matches cannot
            // make a bad final pose look valid.
            if icp_ok {
                let final_pickup_start = Instant::now();
                let correspondences = pickup_valid_source_points::<LOCAL_KNN_K>(
                    &source_voxel_map,
                    &slam_map.local_voxel_map.voxel_map,
                    slam_map.local_voxel_map.config.index_voxel_size,
                    SEARCH_RANGE,
                    MAX_DIST_FACTOR,
                    LOCAL_PLANE_POINT_DISTANCE_THRESHOLD_M,
                    LOCAL_SOURCE_PLANE_SCORE_THRESHOLD,
                    Some(LOCAL_SOURCE_TO_PLANE_MAX_DISTANCE_M),
                    Some(LOCAL_MIN_PLANARITY),
                    &r_mat,
                    &t_vec,
                );
                icp_correspondence_search_time += final_pickup_start.elapsed();
                final_correspondence_count = correspondences.len();
                final_correspondence_ratio = if source_point_count > 0 {
                    correspondences.len() as f32 / source_point_count as f32
                } else {
                    0.0
                };
                final_rmse = Some(compute_rmse(&correspondences, &r_mat, &t_vec));

                let final_system_start = Instant::now();
                let final_system = build_robust_point_to_plane_system(
                    &correspondences,
                    &r_mat,
                    &t_vec,
                    ICP_HUBER_DELTA_M,
                );
                icp_linear_system_time += final_system_start.elapsed();
                let final_solve_start = Instant::now();
                let final_solve = solve_icp_delta_observable(
                    &final_system,
                    ICP_DAMPING,
                    ICP_RELATIVE_EIGENVALUE_THRESHOLD,
                );
                icp_solver_time += final_solve_start.elapsed();
                final_observable_rank = 0;
                final_eigenvalue_ratio = None;
                if let Some(result) = &final_solve {
                    final_observable_rank = result.observable_rank;
                    final_eigenvalue_ratio = Some(result.min_observable_eigenvalue_ratio);
                }

                let final_quality_ok = final_correspondence_count >= min_correspondences
                    && final_observable_rank >= ICP_MIN_OBSERVABLE_RANK
                    && final_rmse
                        .is_some_and(|rmse| rmse.is_finite() && rmse <= ICP_MAX_FINAL_RMSE_M);
                if !final_quality_ok {
                    log::warn!(
                        "Frame {i}: final ICP quality rejected: correspondences={} (min {}), \
                         rank={}, rmse={:?}",
                        final_correspondence_count,
                        min_correspondences,
                        final_observable_rank,
                        final_rmse,
                    );
                    icp_ok = false;
                }

                if icp_ok {
                    // 収束点で作り直した最終系から事後共分散を求め、ESKF に反映する。
                    // ここで初めて filter の状態が書き換わる（失敗時は予測のまま残る）。
                    match update.posterior_covariance(&iter_state, &final_system) {
                        Some(posterior_cov) => {
                            eskf_prior_cost = Some(update.prior_cost(&iter_state));
                            filter.accept_update(iter_state, posterior_cov);
                            eskf_update_applied = true;
                        }
                        None => {
                            log::warn!(
                                "Frame {i}: ESKF posterior covariance is not positive definite"
                            );
                            icp_ok = false;
                        }
                    }
                }
            }

            if !icp_ok {
                // ESKF は更新せず、予測状態と膨らんだ共分散のまま次フレームへ進む。
                log::warn!("Frame {i}: ICP failed, using IMU prediction");
                r_mat = pred_r;
                t_vec = pred_t;
            }
        } else {
            log::warn!("Frame {i}: ESKF update unavailable, using IMU prediction");
        }
        let loop_end = loop_start.elapsed();
        log::debug!(
            "Frame {i}: ICP loop finished in {:.2?}, final RMSE={:?}",
            loop_end,
            final_rmse,
        );
        // --- ICP (Point to Plane) ---

        // ICP diagnostics describe the accepted LiDAR solution, before visual refinement.
        let (icp_translation_correction_m, icp_rotation_correction_deg) = if icp_ok {
            pose_correction_magnitudes(&pred_r, &pred_t, &r_mat, &t_vec)
        } else {
            (0.0, 0.0)
        };
        let map_update_allowed = local_map_was_empty || icp_ok;
        // LiDAR 点群による位置合わせの結果姿勢（画像位置合わせで補正される前）。
        let lidar_registered_pose = filter.lidar_pose();
        let lidar_registration_status = if local_map_was_empty || lidar_update.is_none() {
            "skipped"
        } else if icp_ok {
            "ok"
        } else {
            "rejected"
        };

        // --- Direct image alignment: keyframe 9x9 patches -> current image ---
        // キーフレーム K のパッチ（GlobalMap の Planar 点＋法線付き平面）を今の画像へ投影し、
        // 輝度が一致するように状態を補正する。残差は画像時刻で評価するが、更新するのは
        // LiDAR フレーム開始時刻の 15 次元状態。以後の処理（色付け・マップ更新・次の IMU
        // 予測）はすべてこの更新後の状態を使う。GlobalMap はこのフレームの点を入れる前の状態。
        let visual_alignment_start = Instant::now();
        let mut visual_alignment_log = VisualAlignmentLog::default();
        if !visual_alignment_enabled {
            visual_alignment_log.status = "disabled".into();
        } else if !map_update_allowed {
            visual_alignment_log.status = "lidar_rejected".into();
        } else if let (Some(metadata), Some(rgb)) = (selected_image, selected_rgb_image.as_ref()) {
            let camera = CameraIntrinsics::for_image_size(rgb.width(), rgb.height())?;
            visual_alignment_log = visual_aligner.process_frame(
                filter,
                &VisualFrame {
                    imu: &imu_data,
                    frame_start_time: current_frame_start_time,
                    image_time: metadata.timestamp_sec,
                    frame_index: i as u64,
                    rgb,
                    camera: &camera,
                    camera_from_lidar: &camera_from_lidar,
                    frame_points: &deskewed_points,
                    planar_map: &slam_map.global_voxel_map.voxel_map,
                },
            );
        } else {
            visual_alignment_log.status = "no_image".into();
        }
        let visual_alignment_time = visual_alignment_start.elapsed();
        log::info!(
            "Frame {i}: visual status={} applied={} patches={} (visible={} occluded={}) \
             rmse={:?}->{:?} delta_t={:.4} m delta_r={:.4} deg | keyframe={:?} age={} \
             points={} visible_ratio={:?} failures={} event={} new_points={} time={:.3} ms",
            visual_alignment_log.status,
            visual_alignment_log.update_applied,
            visual_alignment_log.matched_patches,
            visual_alignment_log.visible_patches,
            visual_alignment_log.occluded_patches,
            visual_alignment_log.initial_rmse,
            visual_alignment_log.final_rmse,
            visual_alignment_log.translation_correction_m,
            visual_alignment_log.rotation_correction_deg,
            visual_alignment_log.keyframe_frame_index,
            visual_alignment_log.keyframe_age_frames,
            visual_alignment_log.keyframe_points,
            visual_alignment_log.visible_ratio,
            visual_alignment_log.consecutive_failures,
            visual_alignment_log.keyframe_event,
            visual_alignment_log.new_keyframe_points,
            duration_ms(visual_alignment_time),
        );
        // 画像位置合わせと LiDAR 点群位置合わせ (ICP) の RMSE を横並びで表示する。
        log::info!(
            "Frame {i}: RMSE | {} | {}",
            format_visual_rmse(&visual_alignment_log),
            format_lidar_rmse(icp_ok, final_rmse),
        );
        // Keep geometric filtering and map insertion at the SAME final pose.
        let final_lidar_pose = filter.lidar_pose().cast::<f32>();
        r_mat = final_lidar_pose.fixed_view::<3, 3>(0, 0).into_owned();
        t_vec = final_lidar_pose.fixed_view::<3, 1>(0, 3).into_owned();

        // --- Update current frame info ---
        let pose_update_start = Instant::now();
        let prev_pos = current_frame_info
            .current_global_pose
            .fixed_view::<3, 1>(0, 3)
            .into_owned();
        let prev_r: Matrix3<f64> = current_frame_info
            .current_global_pose
            .fixed_view::<3, 3>(0, 0)
            .into_owned();

        // 姿勢・速度は ESKF の状態から取る（更新成功時は事後推定、失敗時は IMU 予測）。
        // 速度は位置差分ではなく推定値なので、2 m/s の頭打ちも不要になった。
        let new_global_pose = filter.lidar_pose();
        let r64: Matrix3<f64> = new_global_pose.fixed_view::<3, 3>(0, 0).into_owned();
        let new_pos = new_global_pose.fixed_view::<3, 1>(0, 3).into_owned();
        let new_velocity = filter.state.vel;
        let covariance_summary = filter.covariance_summary();
        log::debug!(
            "Frame {i}: ESKF update={}, v=[{:.3}, {:.3}, {:.3}] m/s, \
             bg=[{:.5}, {:.5}, {:.5}] rad/s, ba=[{:.4}, {:.4}, {:.4}] m/s^2, \
             pos_std=[{:.4}, {:.4}, {:.4}] m, prior_cost={:?}",
            eskf_update_applied,
            new_velocity.x,
            new_velocity.y,
            new_velocity.z,
            filter.state.bg.x,
            filter.state.bg.y,
            filter.state.bg.z,
            filter.state.ba.x,
            filter.state.ba.y,
            filter.state.ba.z,
            covariance_summary.position_m[0],
            covariance_summary.position_m[1],
            covariance_summary.position_m[2],
            eskf_prior_cost,
        );

        current_frame_info.current_global_pose = new_global_pose;
        current_frame_info.current_velocity = new_velocity;
        let pose_state_update_time = pose_update_start.elapsed();

        // --- Colorize the registered frame ---
        let coloring_image_pose_start = Instant::now();
        // 画像時刻の名目状態と LiDAR 姿勢（状態はカメラの動き＝ブレ量の予測にも使う）。
        let image_state_and_pose: Option<(NavState, Matrix4<f64>)> =
            if let (Some(metadata), Some(_)) = (selected_image, selected_rgb_image.as_ref()) {
                let image_time = metadata.timestamp_sec;

                let imu_covers_interval = imu_data
                    .first()
                    .is_some_and(|s| s.timestamp <= current_frame_start_time)
                    && imu_data.last().is_some_and(|s| s.timestamp >= image_time);

                if image_time < current_frame_start_time || !imu_covers_interval {
                    log::warn!("Frame {i}: cannot predict image pose at {:.9}", image_time,);
                    None
                } else {
                    // 共分散は触らず、名目状態（バイアス補正済み）だけを画像時刻まで進める。
                    let image_state =
                        filter.predict_nominal(&imu_data, current_frame_start_time, image_time);
                    let pose = filter.lidar_pose_of(&image_state);

                    log::info!(
                        "Frame {i}: image pose dt={:.3} ms, \
                 position=[{:.3}, {:.3}, {:.3}]",
                        (image_time - current_frame_start_time) * 1000.0,
                        pose[(0, 3)],
                        pose[(1, 3)],
                        pose[(2, 3)],
                    );

                    Some((image_state, pose))
                }
            } else {
                None
            };
        let lidar_pose_at_image_time: Option<Matrix4<f64>> =
            image_state_and_pose.as_ref().map(|(_, pose)| *pose);
        // 画像時刻のカメラの角速度・速度（カメラ座標）。色付けのブレ量の予測に使う。
        let camera_motion = match (image_state_and_pose.as_ref(), selected_image) {
            (Some((image_state, _)), Some(metadata)) => camera_motion_at(
                &imu_data,
                &filter.imu_model,
                image_state,
                &filter.imu_origin_in_lidar,
                &camera_from_lidar,
                metadata.timestamp_sec,
                CAMERA_MOTION_HALF_WINDOW_S,
            ),
            _ => None,
        };

        let camera_from_world_at_image: Option<Matrix4<f64>> = lidar_pose_at_image_time
            .as_ref()
            .and_then(|world_from_lidar| {
                world_from_lidar
                    .try_inverse()
                    .map(|lidar_from_world| camera_from_lidar * lidar_from_world)
            });

        if let Some(transform) = &camera_from_world_at_image {
            log::debug!("Frame {i}: camera_from_world_at_image=\n{}", transform,);
        }
        let coloring_image_pose_time = coloring_image_pose_start.elapsed();

        let coloring_projection_start = Instant::now();
        let projected_pixels = if let (Some(rgb_image), Some(camera_from_world)) = (
            selected_rgb_image.as_ref(),
            camera_from_world_at_image.as_ref(),
        ) {
            let intrinsics =
                CameraIntrinsics::for_image_size(rgb_image.width(), rgb_image.height())?;

            // 開始時刻基準の点群 → 世界座標 → 画像時刻のカメラ座標。
            let camera_from_lidar_start =
                camera_from_world * current_frame_info.current_global_pose;

            let pixels: Vec<Option<(u32, u32)>> = deskewed_points
                .iter()
                .map(|point| {
                    let camera_point =
                        camera_from_lidar_start.transform_point(&point.cast::<f64>());

                    project_camera_point(&camera_point, &intrinsics)
                })
                .collect();

            log::info!(
                "Frame {i}: projection candidates={}/{} image={}x{}",
                pixels.iter().flatten().count(),
                pixels.len(),
                intrinsics.width,
                intrinsics.height,
            );

            Some(pixels)
        } else {
            None
        };
        let coloring_projection_time = coloring_projection_start.elapsed();

        let coloring_colorize_start = Instant::now();
        let colored_world_points = colorize_world_points(
            &deskewed_points,
            &current_frame_info.current_global_pose,
            selected_rgb_image.as_ref(),
            projected_pixels.as_deref(),
        )?;

        let colored_count = colored_world_points
            .iter()
            .filter(|point| point.rgb.is_some())
            .count();

        log::info!(
            "Frame {i}: colored={} uncolored={} total={}",
            colored_count,
            colored_world_points.len() - colored_count,
            colored_world_points.len(),
        );
        let coloring_colorize_time = coloring_colorize_start.elapsed();
        // --- Colorize the registered frame ---
        // --- Update current frame info ---

        // --- Record frame log ---
        let frame_log_record_start = Instant::now();
        let translation_m = (new_pos - prev_pos).norm();
        let delta_r = r64 * prev_r.transpose();
        let rotation_deg = ((delta_r.trace() - 1.0) / 2.0)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();

        // --- Record frame log ---
        // Coloring is reported separately, so pose_update covers only the pose
        // state update and the frame-log bookkeeping around it.
        let pose_update_time = pose_state_update_time + frame_log_record_start.elapsed();

        // --- Filter valid source points, then update the WorldMap ---
        // ローカルマップが空（初回フレーム）の場合はフィルタなしで全点追加。
        // それ以外は pickup_valid_source_points で平面に乗っている点だけ抽出し、
        // ICP 収束後の最終姿勢 (r_mat, t_vec) でワールドマップに追加する。
        let global_filter_start = Instant::now();
        // 点と、その点の面の法線（ワールド座標。LocalMap の近傍平面）。法線は色付けの
        // 角度の項に使う。初回フレームは平面対応がないので法線なし。
        let (global_source_points, global_source_normals): (
            Vec<Point3<f32>>,
            Vec<Option<Vector3<f32>>>,
        ) = if !map_update_allowed {
            log::warn!("Frame {i}: map update skipped because ICP quality is insufficient");
            (Vec::new(), Vec::new())
        } else if local_map_was_empty {
            (
                downsampled_source_points_for_global.clone(),
                vec![None; downsampled_source_points_for_global.len()],
            )
        } else {
            let valid = pickup_valid_source_points::<GLOBAL_KNN_K>(
                &source_voxel_map_for_global,
                &slam_map.local_voxel_map.voxel_map,
                slam_map.local_voxel_map.config.index_voxel_size,
                SEARCH_RANGE,
                MAX_DIST_FACTOR,
                GLOBAL_PLANE_POINT_DISTANCE_THRESHOLD_M,
                GLOBAL_SOURCE_PLANE_SCORE_THRESHOLD,
                Some(GLOBAL_SOURCE_TO_PLANE_MAX_DISTANCE_M),
                Some(GLOBAL_MIN_PLANARITY),
                &r_mat,
                &t_vec,
            );
            log::debug!(
                "Frame {i}: {} / {} source points passed plane filter for global map",
                valid.len(),
                source_voxel_map_for_global.len(),
            );
            valid
                .into_iter()
                .map(|c| (c.src_point, Some(c.plane_normal)))
                .unzip()
        };
        let global_filter_time = global_filter_start.elapsed();

        // --- GlobalMap に付ける色の選択（ベストビュー方式） ---
        // 画像が選べないフレーム（画像なし・姿勢なし・暗すぎ・白飛び・ブレすぎ）は
        // LiDAR の位置合わせとマップ更新だけ行い、色は一切付けない。
        let coloring_global_rgb_start = Instant::now();
        let mut coloring_log = ColoringLog::default();
        let global_source_rgb: Option<Vec<Option<ColorObservation>>> = match (
            map_update_allowed,
            selected_rgb_image.as_ref(),
            camera_from_world_at_image.as_ref(),
        ) {
            (false, _, _) => {
                coloring_log.status = "lidar_rejected".into();
                None
            }
            (true, None, _) => {
                coloring_log.status = "no_image".into();
                None
            }
            (true, Some(_), None) => {
                coloring_log.status = "no_image_pose".into();
                None
            }
            (true, Some(rgb_image), Some(camera_from_world)) => {
                let intrinsics =
                    CameraIntrinsics::for_image_size(rgb_image.width(), rgb_image.height())?;

                // LiDARフレーム開始時の座標 → 画像撮影時のカメラ座標
                let camera_from_lidar_start =
                    camera_from_world * current_frame_info.current_global_pose;

                let selection = select_frame_colors(
                    &color_selection_config,
                    &FrameColorInput {
                        image: rgb_image,
                        camera: &intrinsics,
                        camera_from_lidar_start: &camera_from_lidar_start,
                        world_from_lidar_start: &current_frame_info.current_global_pose,
                        points: &global_source_points,
                        normals: &global_source_normals,
                        motion: camera_motion.as_ref(),
                        map: &slam_map.global_voxel_map.voxel_map,
                        map_voxel_size: slam_map.global_voxel_map.config.index_voxel_size,
                    },
                );
                coloring_log = selection.report;
                selection.observations
            }
        };
        let coloring_global_rgb_time = coloring_global_rgb_start.elapsed();

        let global_map_update_start = Instant::now();
        let global_update_stats = if map_update_allowed {
            slam_map
                .global_voxel_map
                .update_world_map_filtered_with_rgb(
                    &global_source_points,
                    global_source_rgb.as_deref(),
                    &current_frame_info.current_global_pose,
                    &world_map_update_filter_config,
                )
        } else {
            Default::default()
        };
        let global_map_update_time = global_map_update_start.elapsed();
        log::debug!(
            "Frame {i}: GlobalMap update input={}, provisional={}, projected={}, \
             pending={}, pending_voxels={}, non_finite={}",
            global_update_stats.input_points,
            global_update_stats.inserted_provisional,
            global_update_stats.projected_to_mature_plane,
            global_update_stats.held_pending,
            global_update_stats.pending_voxels,
            global_update_stats.rejected_non_finite,
        );
        coloring_log.color_updates = global_update_stats.color_updates;
        log::info!(
            "Frame {i}: coloring status={} exposure_ratio={} (samples={}) median_blur={} px \
             |w|={} deg/s saturated={} | candidates={} observations={} \
             (rejected: saturated={} oblique={} blurry={}) color_updates={}",
            coloring_log.status,
            format_optional(coloring_log.exposure_ratio, 3),
            coloring_log.exposure_samples,
            format_optional(coloring_log.median_blur_px, 2),
            format_optional(coloring_log.angular_speed_deg_s, 1),
            format_optional(coloring_log.saturated_ratio, 3),
            coloring_log.candidates,
            coloring_log.observations,
            coloring_log.rejected_saturated,
            coloring_log.rejected_oblique,
            coloring_log.rejected_blurry,
            coloring_log.color_updates,
        );

        let delayed_surface_start = Instant::now();
        let delayed_surface_stats = slam_map.global_voxel_map.classify_delayed_surface_voxels(
            SURFACE_CLASSIFICATION_DELAY_FRAMES,
            &surface_filter_config,
        );
        let delayed_surface_time = delayed_surface_start.elapsed();
        if delayed_surface_stats.evaluated > 0 {
            log::debug!(
                "Frame {i}: delayed surface classification evaluated={}, planar={}, \
                 non_planar={}, unknown={} in {:.2?}",
                delayed_surface_stats.evaluated,
                delayed_surface_stats.planar,
                delayed_surface_stats.non_planar,
                delayed_surface_stats.unknown,
                delayed_surface_time,
            );
        }
        // --- Filter valid source points, then update the WorldMap ---

        // --- Update the LocalMap with the new frame's points ---
        let local_map_update_start = Instant::now();
        if map_update_allowed {
            slam_map.local_voxel_map.update_with_new_frame(
                &downsampled_source_points_for_local,
                &current_frame_info.current_global_pose,
            );
        }
        let local_map_update_time = local_map_update_start.elapsed();
        // --- Update the LocalMap with the new frame's points ---

        prev_frame_start_time = current_frame_start_time;

        let frame_processing_time = frame_processing_start.elapsed();
        let frame_total_with_file_io = frame_start.elapsed();
        let measured_icp_detail =
            icp_correspondence_search_time + icp_linear_system_time + icp_solver_time;
        let icp_other_time = loop_end.saturating_sub(measured_icp_detail);
        let coloring_time = coloring_image_select_time
            + coloring_image_load_time
            + coloring_image_pose_time
            + coloring_projection_time
            + coloring_colorize_time
            + coloring_global_rgb_time;
        let timing = FrameTiming {
            load_pcd_ms: duration_ms(load_pcd_time),
            timestamps_ms: duration_ms(timestamp_time),
            imu_predict_ms: duration_ms(predict_pose_time),
            rotation_trajectory_ms: duration_ms(rotation_trajectory_time),
            deskew_ms: duration_ms(deskew_time),
            downsample_ms: duration_ms(voxel_end),
            build_source_maps_ms: duration_ms(build_map_end),
            icp_ms: duration_ms(loop_end),
            visual_alignment_ms: duration_ms(visual_alignment_time),
            pose_update_ms: duration_ms(pose_update_time),
            global_filter_ms: duration_ms(global_filter_time),
            global_map_update_ms: duration_ms(global_map_update_time),
            delayed_surface_ms: duration_ms(delayed_surface_time),
            local_map_update_ms: duration_ms(local_map_update_time),
            coloring_ms: duration_ms(coloring_time),
            processing_total_ms: duration_ms(frame_processing_time),
            total_with_file_io_ms: duration_ms(frame_total_with_file_io),
            icp_correspondence_search_ms: duration_ms(icp_correspondence_search_time),
            icp_linear_system_ms: duration_ms(icp_linear_system_time),
            icp_solver_ms: duration_ms(icp_solver_time),
            icp_other_ms: duration_ms(icp_other_time),
            coloring_image_select_ms: duration_ms(coloring_image_select_time),
            coloring_image_load_ms: duration_ms(coloring_image_load_time),
            coloring_image_pose_ms: duration_ms(coloring_image_pose_time),
            coloring_projection_ms: duration_ms(coloring_projection_time),
            coloring_colorize_ms: duration_ms(coloring_colorize_time),
            coloring_global_rgb_ms: duration_ms(coloring_global_rgb_time),
        };
        log::debug!(
            "Frame {i} timings [ms]: load_pcd={:.3} ms, timestamps={:.3} ms, \
             imu_predict={:.3} ms, rotation_trajectory={:.3} ms, deskew={:.3} ms, \
             downsample={:.3} ms, build_source_maps={:.3} ms, icp={:.3} ms, \
             pose_update={:.3} ms, global_filter={:.3} ms, global_map_update={:.3} ms, \
             delayed_surface={:.3} ms, local_map_update={:.3} ms, coloring={:.3} ms, \
             total={:.3} ms (with_file_io={:.3} ms)",
            duration_ms(load_pcd_time),
            duration_ms(timestamp_time),
            duration_ms(predict_pose_time),
            duration_ms(rotation_trajectory_time),
            duration_ms(deskew_time),
            duration_ms(voxel_end),
            duration_ms(build_map_end),
            duration_ms(loop_end),
            duration_ms(pose_update_time),
            duration_ms(global_filter_time),
            duration_ms(global_map_update_time),
            duration_ms(delayed_surface_time),
            duration_ms(local_map_update_time),
            duration_ms(coloring_time),
            duration_ms(frame_processing_time),
            duration_ms(frame_total_with_file_io),
        );
        log::debug!(
            "Frame {i} coloring breakdown [ms]: image_select={:.3} ms, image_load={:.3} ms, \
             image_pose={:.3} ms, projection={:.3} ms, colorize={:.3} ms, \
             global_rgb={:.3} ms, total={:.3} ms",
            timing.coloring_image_select_ms,
            timing.coloring_image_load_ms,
            timing.coloring_image_pose_ms,
            timing.coloring_projection_ms,
            timing.coloring_colorize_ms,
            timing.coloring_global_rgb_ms,
            timing.coloring_ms,
        );
        log::debug!(
            "Frame {i} ICP breakdown [ms]: correspondence_search={:.3}, \
             linear_system={:.3}, solver={:.3}, other={:.3}, total={:.3}",
            timing.icp_correspondence_search_ms,
            timing.icp_linear_system_ms,
            timing.icp_solver_ms,
            timing.icp_other_ms,
            timing.icp_ms,
        );
        // --- Save the voxelized colored frame cloud (same file name as the input PCD) ---
        // 点はこのフレームの最終姿勢で変換したワールド座標。色のない点は含めない。
        let pcd_file_name = pcd_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| format!("cloud_{i}.pcd"));
        let frame_colored_cloud =
            voxel_downsample_colored_points(&colored_world_points, FRAME_COLORED_CLOUD_VOXEL_SIZE);
        let colored_cloud_file = if frame_colored_cloud.is_empty() {
            log::info!("Frame {i}: colored frame cloud not saved (no colored points)");
            None
        } else {
            let relative_path = format!("{FRAME_COLORED_CLOUD_SUBDIR}/{pcd_file_name}");
            let path = format!("{save_dir}/{relative_path}");
            save_pcd_xyzrgb(&frame_colored_cloud, &path)?;
            log::debug!(
                "Frame {i}: saved colored frame cloud {} → {} voxels → {}",
                colored_count,
                frame_colored_cloud.len(),
                path,
            );
            Some(relative_path)
        };

        // --- 最終の色付き点群の候補として貯める（使う点は最後に区画ごとに選ぶ） ---
        // LiDAR の位置合わせが採用され、画像の質の判定（暗すぎ・白飛び・ブレすぎ）を通った
        // フレームだけを使う。白飛びした画素の点は入れない。
        let dense_frame_ok = map_update_allowed
            && match coloring_log.status.as_str() {
                "colored" | "no_candidates" => true,
                "too_blurry" => !DENSE_COLOR_SKIP_BLURRY_FRAMES,
                _ => false,
            };
        let dense_points_stored = match (dense_frame_ok, camera_from_world_at_image.as_ref()) {
            (true, Some(_)) => {
                let unsaturated: Vec<ColoredPoint> = colored_world_points
                    .iter()
                    .filter(|point| {
                        point.rgb.is_some_and(|rgb| {
                            rgb.iter()
                                .all(|&value| value < color_selection_config.saturation_level)
                        })
                    })
                    .cloned()
                    .collect();
                let points =
                    voxel_downsample_colored_points(&unsaturated, FRAME_COLORED_CLOUD_VOXEL_SIZE);
                dense_color_map.insert_frame(i as u64, &points)
            }
            _ => 0,
        };
        coloring_log.dense_points_stored = dense_points_stored;
        log::debug!(
            "Frame {i}: dense color map ok={dense_frame_ok} stored={dense_points_stored}"
        );

        registration_results.push(FrameRegistrationResult {
            frame_index: i,
            pcd_file_name,
            timestamp: current_frame_start_time,
            lidar: LidarRegistrationResult {
                status: lidar_registration_status.into(),
                rmse_m: final_rmse.filter(|rmse| rmse.is_finite()),
                correspondence_count: final_correspondence_count,
                pose: RegistrationPose::from_matrix(&lidar_registered_pose),
            },
            image: ImageRegistrationResult {
                status: visual_alignment_log.status.clone(),
                update_applied: visual_alignment_log.update_applied,
                initial_rmse: visual_alignment_log.initial_rmse,
                final_rmse: visual_alignment_log.final_rmse,
                matched_patches: visual_alignment_log.matched_patches,
                image_file_name: selected_image.map(|metadata| metadata.file_name.clone()),
                image_timestamp_sec: selected_image.map(|metadata| metadata.timestamp_sec),
                pose: RegistrationPose::from_matrix(&new_global_pose),
            },
            colored_cloud_file,
            colored_cloud_points: frame_colored_cloud.len(),
        });
        // --- Save the voxelized colored frame cloud (same file name as the input PCD) ---

        frame_logs.push(FrameLog {
            frame_index: i,
            timestamp: current_frame_start_time,
            icp_ok,
            rmse: final_rmse.filter(|rmse| rmse.is_finite()),
            correspondence_count: final_correspondence_count,
            correspondence_ratio: final_correspondence_ratio,
            observable_rank: final_observable_rank,
            min_observable_eigenvalue_ratio: final_eigenvalue_ratio,
            icp_translation_correction_m,
            icp_rotation_correction_deg,
            map_updated: map_update_allowed,
            translation_m,
            rotation_deg,
            velocity_m_s: new_velocity.norm(),
            pose_x: new_pos.x,
            pose_y: new_pos.y,
            pose_z: new_pos.z,
            timing,
            visual_alignment: visual_alignment_log,
            eskf_update_applied,
            gyro_bias_radps: filter.state.bg.into(),
            accel_bias_mps2: filter.state.ba.into(),
            velocity_mps: new_velocity.into(),
            attitude_std_deg: covariance_summary.attitude_deg,
            position_std_m: covariance_summary.position_m,
            velocity_std_mps: covariance_summary.velocity_mps,
            gyro_bias_std_radps: covariance_summary.gyro_bias_radps,
            accel_bias_std_mps2: covariance_summary.accel_bias_mps2,
            eskf_prior_cost,
            coloring: coloring_log,
        });
    }

    log_timing_summary(&frame_logs);
    log_rmse_summary(&frame_logs, image_time_offset_sec);
    log_coloring_summary(&frame_logs);

    // --- Save the global voxel maps before and after final plane classification ---
    let min_samples = slam_map.global_voxel_map.config.min_points_per_voxel as u64;

    let min_frames = slam_map
        .global_voxel_map
        .config
        .min_observed_frames_per_voxel as u64;

    // 平面処理前: 従来条件を満たす全観測セルを別ファイルへ保存する。
    let world_map_points_before_plane_filter: Vec<Point3<f32>> = slam_map
        .global_voxel_map
        .voxel_map
        .values()
        .filter(|cell| cell.sample_count >= min_samples && cell.observed_frames >= min_frames)
        .map(|cell| Point3::new(cell.mean.x, cell.mean.y, cell.mean.z))
        .collect();

    let downsampled_world_map_points_before_plane_filter =
        voxel_downsample_points(&world_map_points_before_plane_filter, GLOBAL_MAP_VOXEL_SIZE);
    let world_map_points_before_plane_filter_xyz: Vec<PointXYZ> =
        downsampled_world_map_points_before_plane_filter
            .iter()
            .map(|point| PointXYZ {
                x: point.x,
                y: point.y,
                z: point.z,
            })
            .collect();

    std::fs::create_dir_all(&save_dir)?;
    let world_map_before_plane_filter_path = format!(
        "{}/voxel-{}_world_map_before_plane_filter.pcd",
        save_dir, GLOBAL_MAP_VOXEL_SIZE
    );
    save_pcd_xyz(
        &world_map_points_before_plane_filter_xyz,
        &world_map_before_plane_filter_path,
    )?;
    log::info!(
        "Saved world map before plane filter: {} → {} points → {}",
        world_map_points_before_plane_filter.len(),
        world_map_points_before_plane_filter_xyz.len(),
        world_map_before_plane_filter_path,
    );

    // 平面処理前の色付き点群: before_plane_filter と同じセル条件で、色を持つセルだけを保存する。
    let world_map_rgb_before_plane_filter: Vec<PointXYZRGB> = slam_map
        .global_voxel_map
        .voxel_map
        .values()
        .filter(|cell| {
            cell.sample_count >= min_samples
                && cell.observed_frames >= min_frames
                && cell.mean.coords.iter().all(|value| value.is_finite())
        })
        .filter_map(|cell| cell.rgb.map(|rgb| PointXYZRGB::new(cell.mean, rgb)))
        .collect();

    let world_map_rgb_before_plane_filter_path = format!(
        "{}/voxel-{}_world_map_rgb_before_plane_filter.pcd",
        save_dir, GLOBAL_MAP_VOXEL_SIZE,
    );

    if world_map_rgb_before_plane_filter.is_empty() {
        log::info!("RGB export before plane filter skipped: no colored points");
    } else {
        save_pcd_xyzrgb(
            &world_map_rgb_before_plane_filter,
            &world_map_rgb_before_plane_filter_path,
        )?;
        log::info!(
            "Saved colored world map before plane filter: {} / {} points → {}",
            world_map_rgb_before_plane_filter.len(),
            world_map_points_before_plane_filter.len(),
            world_map_rgb_before_plane_filter_path,
        );
    }

    // 終端処理: 遅延キューの状態に依存せず、現在の全累積点で全セルを再判定する。
    let final_surface_start = Instant::now();
    let final_surface_stats = slam_map
        .global_voxel_map
        .classify_all_surface_voxels(&surface_filter_config);
    log::info!(
        "Final surface classification: evaluated={}, planar={}, non_planar={}, unknown={} in {:.2?}",
        final_surface_stats.evaluated,
        final_surface_stats.planar,
        final_surface_stats.non_planar,
        final_surface_stats.unknown,
        final_surface_start.elapsed(),
    );

    let planar_world_map_rgb: Vec<PointXYZRGB> = slam_map
        .global_voxel_map
        .voxel_map
        .values()
        .filter(|cell| {
            cell.sample_count >= min_samples
                && cell.observed_frames >= min_frames
                && cell.surface_status == SurfaceStatus::Planar
                && cell.mean.coords.iter().all(|value| value.is_finite())
        })
        .filter_map(|cell| cell.rgb.map(|rgb| PointXYZRGB::new(cell.mean, rgb)))
        .collect();

    let planar_world_map_rgb_path = format!(
        "{}/voxel-{}_world_map_rgb.pcd",
        save_dir, GLOBAL_MAP_VOXEL_SIZE,
    );

    if planar_world_map_rgb.is_empty() {
        log::info!("RGB export skipped: no colored planar points");
    } else {
        save_pcd_xyzrgb(&planar_world_map_rgb, &planar_world_map_rgb_path)?;

        log::info!(
            "Saved colored planar world map: {} points → {}",
            planar_world_map_rgb.len(),
            planar_world_map_rgb_path,
        );
    }

    // 最終の色付き点群（リアルタイム表示と同じ細かさ。区画ごとに 1 回の訪問の点だけ）。
    let dense_build_start = Instant::now();
    let dense_result = dense_color_map.build();
    let dense_summary = dense_result.summary;
    let dense_world_map_rgb_path = format!(
        "{}/voxel-{}_world_map_rgb_dense.pcd",
        save_dir, GLOBAL_MAP_VOXEL_SIZE,
    );
    if dense_result.points.is_empty() {
        log::info!("Dense RGB export skipped: no colored points");
    } else {
        let dense_world_map_rgb: Vec<PointXYZRGB> = dense_result
            .points
            .iter()
            .map(|point| PointXYZRGB {
                x: point.x,
                y: point.y,
                z: point.z,
                rgb: point.rgb,
            })
            .collect();
        save_pcd_xyzrgb(&dense_world_map_rgb, &dense_world_map_rgb_path)?;
        log::info!(
            "Saved dense colored world map: {} points ({} from the best visit of each tile, \
             {} filling cells missing from it) in {} cells, {} tiles; buffered {} points from \
             {} frames; built in {:.2?} → {}",
            dense_summary.points,
            dense_summary.primary_points,
            dense_summary.fill_points,
            dense_summary.cells,
            dense_summary.tiles,
            dense_summary.buffered_points,
            dense_summary.frames,
            dense_build_start.elapsed(),
            dense_world_map_rgb_path,
        );
        if DENSE_COLOR_SAVE_FRAME_INDEX {
            let path = format!(
                "{}/voxel-{}_world_map_rgb_dense_frames.pcd",
                save_dir, GLOBAL_MAP_VOXEL_SIZE,
            );
            save_pcd_xyzrgb_frame(&dense_result.points, &path)?;
            log::info!("Saved dense colored world map with frame indices → {path}");
        }
    }

    let planar_world_map_points: Vec<Point3<f32>> = slam_map
        .global_voxel_map
        .voxel_map
        .values()
        .filter(|cell| {
            cell.sample_count >= min_samples
                && cell.observed_frames >= min_frames
                && cell.surface_status == SurfaceStatus::Planar
        })
        .map(|cell| Point3::new(cell.mean.x, cell.mean.y, cell.mean.z))
        .collect();

    let downsampled_planar_world_map_points =
        voxel_downsample_points(&planar_world_map_points, GLOBAL_MAP_VOXEL_SIZE);
    let planar_world_map_points_xyz: Vec<PointXYZ> = downsampled_planar_world_map_points
        .iter()
        .map(|point| PointXYZ {
            x: point.x,
            y: point.y,
            z: point.z,
        })
        .collect();

    // 既存ファイル名は平面処理後の最終マップとして維持する。
    let planar_world_map_path =
        format!("{}/voxel-{}_world_map.pcd", save_dir, GLOBAL_MAP_VOXEL_SIZE);
    save_pcd_xyz(&planar_world_map_points_xyz, &planar_world_map_path)?;
    log::info!(
        "Saved planar world map: {} → {} points → {}",
        planar_world_map_points.len(),
        planar_world_map_points_xyz.len(),
        planar_world_map_path,
    );
    // --- Save the global voxel maps before and after final plane classification ---

    // --- Save per-frame ICP logs to JSON ---
    let frame_logs_path = format!("{}/frame_logs.json", save_dir);
    let frame_logs_json = serde_json::to_string_pretty(&frame_logs)?;
    std::fs::write(&frame_logs_path, &frame_logs_json)?;
    log::info!(
        "Saved frame logs: {} frames → {}",
        frame_logs.len(),
        frame_logs_path
    );
    // --- Save per-frame ICP logs to JSON ---

    // --- Save per-frame registration results (LiDAR / image pose and RMSE) to JSON ---
    let registration_results_path = format!("{}/registration_results.json", save_dir);
    let registration_results_json = serde_json::to_string_pretty(&registration_results)?;
    std::fs::write(&registration_results_path, &registration_results_json)?;
    log::info!(
        "Saved registration results: {} frames → {} (colored frame clouds → {})",
        registration_results.len(),
        registration_results_path,
        frame_cloud_dir,
    );
    // --- Save per-frame registration results (LiDAR / image pose and RMSE) to JSON ---

    Ok(())
}

#[inline]
fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

/// 画像位置合わせの RMSE（輝度 0..255、パッチ平均除去後）を `初期->最終 (status)` で表す。
fn format_visual_rmse(log: &VisualAlignmentLog) -> String {
    match (log.initial_rmse, log.final_rmse) {
        (Some(initial), Some(final_rmse)) => format!(
            "image RMSE = {initial:>7.3} -> {final_rmse:>7.3} [gray] ({}{})",
            log.status,
            if log.update_applied { ", applied" } else { "" },
        ),
        _ => format!("image RMSE = {:>18} ({})", "-", log.status),
    }
}

/// LiDAR 点群位置合わせ (Point-to-Plane ICP) の最終 RMSE [m] を表す。
fn format_lidar_rmse(icp_ok: bool, rmse: Option<f32>) -> String {
    let status = if icp_ok { "ok" } else { "rejected" };
    match rmse.filter(|value| value.is_finite()) {
        Some(value) => format!("LiDAR RMSE = {value:>8.5} [m] ({status})"),
        None => format!("LiDAR RMSE = {:>8} ({status})", "-"),
    }
}

#[derive(Debug, Clone, Copy)]
struct ValueStats {
    mean: f64,
    median: f64,
    max: f64,
}

/// 値列の mean / median / max を返す（空なら None）。
fn value_stats(values: &[f64]) -> Option<ValueStats> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    };
    Some(ValueStats {
        mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
        median,
        max: sorted[sorted.len() - 1],
    })
}

fn format_value_stats(values: &[f64], precision: usize) -> String {
    match value_stats(values) {
        Some(ValueStats { mean, median, max }) => format!(
            "n={:<5} mean={mean:.precision$} median={median:.precision$} max={max:.precision$}",
            values.len()
        ),
        None => format!("n={:<5} (no data)", values.len()),
    }
}

/// 全フレームの画像位置合わせ RMSE と LiDAR 位置合わせ RMSE を横並びで集計表示する。
/// 最後の `Image time offset evaluation:` 行は、オフセット掃引で比較するための 1 行要約。
fn log_rmse_summary(frame_logs: &[FrameLog], image_time_offset_sec: f64) {
    if frame_logs.is_empty() {
        return;
    }
    let collect = |applied_only: bool, get: fn(&VisualAlignmentLog) -> Option<f64>| -> Vec<f64> {
        frame_logs
            .iter()
            .map(|log| &log.visual_alignment)
            .filter(|visual| !applied_only || visual.update_applied)
            .filter_map(get)
            .filter(|value| value.is_finite())
            .collect()
    };
    let visual_initial = collect(false, |v| v.initial_rmse);
    let visual_final = collect(false, |v| v.final_rmse);
    let visual_applied_initial = collect(true, |v| v.initial_rmse);
    let visual_applied_final = collect(true, |v| v.final_rmse);
    let translation = collect(true, |v| Some(v.translation_correction_m));
    let rotation = collect(true, |v| Some(v.rotation_correction_deg));
    let lidar_all: Vec<f64> = frame_logs
        .iter()
        .filter_map(|log| log.rmse)
        .map(f64::from)
        .collect();
    let lidar_ok: Vec<f64> = frame_logs
        .iter()
        .filter(|log| log.icp_ok)
        .filter_map(|log| log.rmse)
        .map(f64::from)
        .collect();

    let rows = [
        (
            "initial (all evaluated)",
            format_value_stats(&visual_initial, 3),
            "final (all evaluated)",
            format_value_stats(&lidar_all, 5),
        ),
        (
            "final (all evaluated)",
            format_value_stats(&visual_final, 3),
            "final (icp_ok)",
            format_value_stats(&lidar_ok, 5),
        ),
        (
            "initial (update applied)",
            format_value_stats(&visual_applied_initial, 3),
            "",
            String::new(),
        ),
        (
            "final (update applied)",
            format_value_stats(&visual_applied_final, 3),
            "",
            String::new(),
        ),
    ];
    log::info!(
        "RMSE summary for {} frames (image time offset {:+.3} ms): \
         image alignment [gray 0..255] | LiDAR ICP [m]",
        frame_logs.len(),
        image_time_offset_sec * 1000.0,
    );
    for (visual_label, visual, lidar_label, lidar) in rows {
        log::info!(
            "  image {:<24} {:<55} | LiDAR {:<21} {}",
            visual_label,
            visual,
            lidar_label,
            lidar,
        );
    }

    let mut status_counts: std::collections::BTreeMap<&str, usize> = Default::default();
    for log in frame_logs {
        *status_counts
            .entry(log.visual_alignment.status.as_str())
            .or_default() += 1;
    }
    let status_text = status_counts
        .iter()
        .map(|(status, count)| format!("{status}={count}"))
        .collect::<Vec<_>>()
        .join(", ");
    log::info!("  image status: {status_text}");
    log::info!(
        "  image correction (applied): translation [m] {} | rotation [deg] {}",
        format_value_stats(&translation, 5),
        format_value_stats(&rotation, 4),
    );

    // オフセット掃引用の 1 行要約（key=value, 欠損は nan）。
    let mean = |values: &[f64]| value_stats(values).map_or(f64::NAN, |s| s.mean);
    let median = |values: &[f64]| value_stats(values).map_or(f64::NAN, |s| s.median);
    log::info!(
        "Image time offset evaluation: offset_ms={:.3} lidar_rmse_mean_m={:.6} \
         lidar_rmse_median_m={:.6} image_initial_rmse_mean={:.4} image_initial_rmse_median={:.4} \
         image_final_rmse_mean={:.4} image_evaluated={} image_applied={} \
         correction_t_median_m={:.6} correction_t_mean_m={:.6} \
         correction_r_median_deg={:.5} correction_r_mean_deg={:.5}",
        image_time_offset_sec * 1000.0,
        mean(&lidar_ok),
        median(&lidar_ok),
        mean(&visual_initial),
        median(&visual_initial),
        mean(&visual_final),
        visual_initial.len(),
        visual_applied_final.len(),
        median(&translation),
        mean(&translation),
        median(&rotation),
        mean(&rotation),
    );
}

fn format_optional(value: Option<f64>, precision: usize) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:.precision$}"))
}

/// 色付けしたフレーム数と、色付けしなかった理由の内訳、露光比・ブレ量の分布を表示する。
fn log_coloring_summary(frame_logs: &[FrameLog]) {
    if frame_logs.is_empty() {
        return;
    }
    let mut status_counts: std::collections::BTreeMap<&str, usize> = Default::default();
    for log in frame_logs {
        *status_counts
            .entry(log.coloring.status.as_str())
            .or_default() += 1;
    }
    let colored = status_counts.get("colored").copied().unwrap_or(0);
    let status_text = status_counts
        .iter()
        .map(|(status, count)| format!("{status}={count}"))
        .collect::<Vec<_>>()
        .join(", ");
    log::info!(
        "Coloring summary: colored {colored} / {} frames ({:.1}%) | {status_text}",
        frame_logs.len(),
        colored as f64 / frame_logs.len() as f64 * 100.0,
    );
    let collect = |get: fn(&ColoringLog) -> Option<f64>| -> Vec<f64> {
        frame_logs
            .iter()
            .filter_map(|log| get(&log.coloring))
            .filter(|value| value.is_finite())
            .collect()
    };
    log::info!(
        "  exposure ratio (all evaluated): {} | median blur [px]: {} | |w| [deg/s]: {}",
        format_value_stats(&collect(|c| c.exposure_ratio), 3),
        format_value_stats(&collect(|c| c.median_blur_px), 2),
        format_value_stats(&collect(|c| c.angular_speed_deg_s), 1),
    );
    let updates: usize = frame_logs
        .iter()
        .map(|log| log.coloring.color_updates)
        .sum();
    log::info!("  GlobalMap color updates (total over frames): {updates}");
    let total = |get: fn(&ColoringLog) -> usize| -> usize {
        frame_logs.iter().map(|log| get(&log.coloring)).sum()
    };
    log::info!(
        "  dense color map: buffered {} points from {} frames",
        total(|c| c.dense_points_stored),
        frame_logs
            .iter()
            .filter(|log| log.coloring.dense_points_stored > 0)
            .count(),
    );
}

#[derive(Debug)]
struct TimingStats {
    name: &'static str,
    mean_ms: f64,
    p95_ms: f64,
    max_ms: f64,
}

fn log_timing_summary(frame_logs: &[FrameLog]) {
    if frame_logs.is_empty() {
        return;
    }

    let stages: [(&str, fn(&FrameTiming) -> f64); 15] = [
        ("load_pcd", |t| t.load_pcd_ms),
        ("timestamps", |t| t.timestamps_ms),
        ("imu_predict", |t| t.imu_predict_ms),
        ("rotation_trajectory", |t| t.rotation_trajectory_ms),
        ("deskew", |t| t.deskew_ms),
        ("downsample", |t| t.downsample_ms),
        ("build_source_maps", |t| t.build_source_maps_ms),
        ("icp", |t| t.icp_ms),
        ("visual_alignment", |t| t.visual_alignment_ms),
        ("pose_update", |t| t.pose_update_ms),
        ("global_filter", |t| t.global_filter_ms),
        ("global_map_update", |t| t.global_map_update_ms),
        ("delayed_surface", |t| t.delayed_surface_ms),
        ("local_map_update", |t| t.local_map_update_ms),
        ("coloring", |t| t.coloring_ms),
    ];
    let mut summaries: Vec<TimingStats> = stages
        .into_iter()
        .map(|(name, get)| timing_stats(name, frame_logs, get))
        .collect();
    summaries.sort_by(|left, right| right.mean_ms.total_cmp(&left.mean_ms));

    let processing_total = timing_stats("processing_total", frame_logs, |t| t.processing_total_ms);
    let wall_total = timing_stats("total_with_file_io", frame_logs, |t| {
        t.total_with_file_io_ms
    });
    log::info!(
        "Timing summary for {} frames [ms]: processing mean={:.3}, p95={:.3}, max={:.3}; \
         with_file_io mean={:.3}, p95={:.3}, max={:.3}",
        frame_logs.len(),
        processing_total.mean_ms,
        processing_total.p95_ms,
        processing_total.max_ms,
        wall_total.mean_ms,
        wall_total.p95_ms,
        wall_total.max_ms,
    );
    for (rank, summary) in summaries.iter().enumerate() {
        let share = if wall_total.mean_ms > 0.0 {
            summary.mean_ms / wall_total.mean_ms * 100.0
        } else {
            0.0
        };
        log::info!(
            "Timing rank {:>2}: {:<20} mean={:>9.3} ms ({:>5.1}%), p95={:>9.3} ms, max={:>9.3} ms",
            rank + 1,
            summary.name,
            summary.mean_ms,
            share,
            summary.p95_ms,
            summary.max_ms,
        );
    }

    let icp_details: [(&str, fn(&FrameTiming) -> f64); 4] = [
        ("correspondence_search", |t| t.icp_correspondence_search_ms),
        ("linear_system", |t| t.icp_linear_system_ms),
        ("solver", |t| t.icp_solver_ms),
        ("other", |t| t.icp_other_ms),
    ];
    let mut icp_summaries: Vec<TimingStats> = icp_details
        .into_iter()
        .map(|(name, get)| timing_stats(name, frame_logs, get))
        .collect();
    icp_summaries.sort_by(|left, right| right.mean_ms.total_cmp(&left.mean_ms));
    for summary in icp_summaries {
        let share = if summaries
            .iter()
            .find(|summary| summary.name == "icp")
            .is_some_and(|summary| summary.mean_ms > 0.0)
        {
            let icp_mean = summaries
                .iter()
                .find(|summary| summary.name == "icp")
                .map_or(0.0, |summary| summary.mean_ms);
            summary.mean_ms / icp_mean * 100.0
        } else {
            0.0
        };
        log::info!(
            "ICP timing: {:<21} mean={:>9.3} ms ({:>5.1}% of ICP), p95={:>9.3} ms, max={:>9.3} ms",
            summary.name,
            summary.mean_ms,
            share,
            summary.p95_ms,
            summary.max_ms,
        );
    }

    let coloring_details: [(&str, fn(&FrameTiming) -> f64); 6] = [
        ("image_select", |t| t.coloring_image_select_ms),
        ("image_load", |t| t.coloring_image_load_ms),
        ("image_pose", |t| t.coloring_image_pose_ms),
        ("projection", |t| t.coloring_projection_ms),
        ("colorize", |t| t.coloring_colorize_ms),
        ("global_rgb", |t| t.coloring_global_rgb_ms),
    ];
    let coloring_mean = summaries
        .iter()
        .find(|summary| summary.name == "coloring")
        .map_or(0.0, |summary| summary.mean_ms);
    let mut coloring_summaries: Vec<TimingStats> = coloring_details
        .into_iter()
        .map(|(name, get)| timing_stats(name, frame_logs, get))
        .collect();
    coloring_summaries.sort_by(|left, right| right.mean_ms.total_cmp(&left.mean_ms));
    for summary in coloring_summaries {
        let share = if coloring_mean > 0.0 {
            summary.mean_ms / coloring_mean * 100.0
        } else {
            0.0
        };
        log::info!(
            "Coloring timing: {:<16} mean={:>9.3} ms ({:>5.1}% of coloring), p95={:>9.3} ms, max={:>9.3} ms",
            summary.name,
            summary.mean_ms,
            share,
            summary.p95_ms,
            summary.max_ms,
        );
    }
}

fn timing_stats(
    name: &'static str,
    frame_logs: &[FrameLog],
    get: fn(&FrameTiming) -> f64,
) -> TimingStats {
    let mut values: Vec<f64> = frame_logs
        .iter()
        .map(|frame| get(&frame.timing))
        .filter(|value| value.is_finite())
        .collect();
    values.sort_by(f64::total_cmp);
    let mean_ms = values.iter().sum::<f64>() / values.len().max(1) as f64;
    let p95_index = ((values.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(values.len().saturating_sub(1));

    TimingStats {
        name,
        mean_ms,
        p95_ms: values.get(p95_index).copied().unwrap_or_default(),
        max_ms: values.last().copied().unwrap_or_default(),
    }
}

fn pose_correction_magnitudes(
    reference_r: &Matrix3<f32>,
    reference_t: &Vector3<f32>,
    candidate_r: &Matrix3<f32>,
    candidate_t: &Vector3<f32>,
) -> (f32, f32) {
    let translation_m = (candidate_t - reference_t).norm();
    let delta_r = candidate_r * reference_r.transpose();
    let rotation_deg = ((delta_r.trace() - 1.0) * 0.5)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    (translation_m, rotation_deg)
}

fn retain_world_z_translation(delta: &Vector6f) -> Vector6f {
    let mut vertical_delta = Vector6f::zeros();
    vertical_delta[5] = delta[5];
    vertical_delta
}

/// Mid-70座標の点をAiry-96座標へ写す外部変換 `T_airy96_from_mid70`。
fn make_airy96_from_mid70_extrinsic() -> Matrix4<f64> {
    let rotation = Matrix3::<f64>::new(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let translation = Vector3::<f64>::new(
        MID70_ORIGIN_IN_AIRY96_X_M,
        MID70_ORIGIN_IN_AIRY96_Y_M,
        MID70_ORIGIN_IN_AIRY96_Z_M,
    );

    let mut transform = Matrix4::<f64>::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);
    transform
}

/// Airy-96内蔵IMU座標からMid-70座標への回転を返す。
///
/// `R_mid70_from_imu = R_mid70_from_airy96 * R_airy96_from_imu`
fn make_imu_to_mid70_rotation() -> UnitQuaternion<f64> {
    let imu_to_airy96 = make_imu_to_airy96_rotation();
    let airy96_from_mid70 = make_airy96_from_mid70_extrinsic();
    let airy96_from_mid70_rotation =
        UnitQuaternion::from_matrix(&airy96_from_mid70.fixed_view::<3, 3>(0, 0).into_owned());

    airy96_from_mid70_rotation.inverse() * imu_to_airy96
}

/// Airy-96内蔵IMU座標からAiry-96 LiDAR座標への回転を返す。
fn make_imu_to_airy96_rotation() -> UnitQuaternion<f64> {
    UnitQuaternion::new_normalize(Quaternion::new(
        IMU_TO_AIRY96_QUAT_W,
        IMU_TO_AIRY96_QUAT_X,
        IMU_TO_AIRY96_QUAT_Y,
        IMU_TO_AIRY96_QUAT_Z,
    ))
}

/// Livox Avia内蔵IMU座標からLiDAR座標への回転を返す。
/// Aviaの内蔵IMUとLiDARの座標軸は同じ向きなので、回転は単位回転になる
/// （IMU-LiDAR間の並進はファイル冒頭のコメントを参照）。
fn make_imu_to_avia_rotation() -> UnitQuaternion<f64> {
    UnitQuaternion::identity()
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use nalgebra::{Point3, Vector3};

    use super::{
        CommandLineAction, DEFAULT_IMAGE_TIME_OFFSET_SEC, IMU, ImageFrameMetadata, LidarModel,
        MID70_ORIGIN_IN_AIRY96_Z_M, RunOptions, apply_image_time_offset, value_stats,
        count_imu_samples_in_time_range, make_airy96_from_mid70_extrinsic,
        make_imu_to_airy96_rotation, make_imu_to_avia_rotation, make_imu_to_mid70_rotation,
        parse_command_line, retain_world_z_translation,
    };

    fn imu_sample(timestamp: f64) -> IMU {
        IMU {
            timestamp,
            angular_velocity: [0.0; 3],
            linear_acceleration: [0.0; 3],
        }
    }

    fn parse_args(args: &[&str]) -> anyhow::Result<CommandLineAction> {
        parse_command_line(args.iter().map(OsString::from))
    }

    #[test]
    fn vertical_recovery_discards_rotation_and_xy_translation() {
        let delta = re_lidar_slam::icp::Vector6f::new(1.0, 2.0, 3.0, 4.0, 5.0, -0.45);

        assert_eq!(
            retain_world_z_translation(&delta),
            re_lidar_slam::icp::Vector6f::new(0.0, 0.0, 0.0, 0.0, 0.0, -0.45)
        );
    }

    #[test]
    fn command_line_defaults_to_mid70() {
        let CommandLineAction::Run(options) = parse_args(&[]).unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(options.lidar_model, LidarModel::Mid70);
        assert!(options.visual_alignment_enabled);
        assert_eq!(options.image_time_offset_sec, DEFAULT_IMAGE_TIME_OFFSET_SEC);
        assert_eq!(options.output_tag, None);
    }

    #[test]
    fn command_line_can_disable_visual_alignment_for_comparison() {
        let CommandLineAction::Run(options) =
            parse_args(&["--lidar", "avia", "--no-visual-alignment"]).unwrap()
        else {
            panic!("expected run action");
        };
        assert_eq!(options.lidar_model, LidarModel::Avia);
        assert!(!options.visual_alignment_enabled);
    }

    #[test]
    fn command_line_selects_airy96() {
        let CommandLineAction::Run(RunOptions {
            lidar_model: model, ..
        }) = parse_args(&["--lidar", "airy96"]).unwrap()
        else {
            panic!("expected run action");
        };
        assert_eq!(model, LidarModel::Airy96);

        let CommandLineAction::Run(RunOptions {
            lidar_model: model, ..
        }) = parse_args(&["--lidar=airy"]).unwrap()
        else {
            panic!("expected run action");
        };
        assert_eq!(model, LidarModel::Airy96);
    }

    #[test]
    fn command_line_selects_avia() {
        let CommandLineAction::Run(RunOptions {
            lidar_model: model, ..
        }) = parse_args(&["--lidar", "avia"]).unwrap()
        else {
            panic!("expected run action");
        };
        assert_eq!(model, LidarModel::Avia);
        assert_eq!(model.input_subdir(), "avia");
        assert_eq!(model.name(), "avia");

        let CommandLineAction::Run(RunOptions {
            lidar_model: model, ..
        }) = parse_args(&["--lidar=livox-avia"]).unwrap()
        else {
            panic!("expected run action");
        };
        assert_eq!(model, LidarModel::Avia);
    }

    #[test]
    fn command_line_sets_image_time_offset_and_output_tag() {
        let CommandLineAction::Run(options) = parse_args(&[
            "--image-time-offset",
            "-0.02",
            "--output-tag",
            "offset_m20ms",
        ])
        .unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(options.image_time_offset_sec, -0.02);
        assert_eq!(options.output_tag.as_deref(), Some("offset_m20ms"));

        let CommandLineAction::Run(options) =
            parse_args(&["--image-time-offset=0", "--output-tag=a+10ms"]).unwrap()
        else {
            panic!("expected run action");
        };
        assert_eq!(options.image_time_offset_sec, 0.0);
        assert_eq!(options.output_tag.as_deref(), Some("a+10ms"));
    }

    #[test]
    fn command_line_rejects_invalid_image_time_offset_and_output_tag() {
        for args in [
            &["--image-time-offset"][..],
            &["--image-time-offset", "abc"],
            &["--image-time-offset", "NaN"],
            &["--image-time-offset", "100"],
            &["--output-tag", "../x"],
            &["--output-tag", ""],
        ] {
            assert!(parse_args(args).is_err(), "{args:?} must fail");
        }
    }

    #[test]
    fn image_time_offset_shifts_every_timestamp() {
        let frame = |timestamp_sec: f64| -> ImageFrameMetadata {
            serde_json::from_value(serde_json::json!({
                "frame_index": 0, "file_name": "x.png", "timestamp_ns": 0,
                "timestamp_sec": timestamp_sec, "stamp": {"sec": 0, "nanosec": 0},
                "frame_id": "camera", "width": 960, "height": 600,
                "source_encoding": "rgb8", "source_step": 2880
            }))
            .unwrap()
        };
        let mut frames = vec![frame(1.0), frame(2.5)];
        apply_image_time_offset(&mut frames, 0.1);
        assert!((frames[0].timestamp_sec - 1.1).abs() < 1e-12);
        assert!((frames[1].timestamp_sec - 2.6).abs() < 1e-12);
    }

    #[test]
    fn value_stats_reports_mean_median_and_max() {
        let stats = value_stats(&[3.0, 1.0, 2.0, 10.0]).unwrap();
        assert!((stats.mean - 4.0).abs() < 1e-12);
        assert!((stats.median - 2.5).abs() < 1e-12);
        assert_eq!(stats.max, 10.0);
        assert!(value_stats(&[]).is_none());
    }

    #[test]
    fn command_line_rejects_unknown_lidar() {
        let error = match parse_args(&["--lidar", "unknown"]) {
            Ok(_) => panic!("unknown LiDAR model must fail"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("expected 'mid70', 'airy96', or 'avia'")
        );
    }

    #[test]
    fn counts_imu_samples_inside_inclusive_point_cloud_interval() {
        let imu_data = [
            imu_sample(0.9),
            imu_sample(1.0),
            imu_sample(1.5),
            imu_sample(2.0),
            imu_sample(2.1),
        ];

        assert_eq!(count_imu_samples_in_time_range(&imu_data, 1.0, 2.0), 3);
        assert_eq!(count_imu_samples_in_time_range(&imu_data, 3.0, 4.0), 0);
        assert_eq!(count_imu_samples_in_time_range(&imu_data, 2.0, 1.0), 0);
    }

    #[test]
    fn mid70_extrinsic_maps_axes_into_airy96_coordinates() {
        let airy96_from_mid70 = make_airy96_from_mid70_extrinsic();
        let mid70_x = Point3::new(1.0, 0.0, 0.0);
        let airy96_point = airy96_from_mid70.transform_point(&mid70_x);

        assert!((airy96_point.x - 0.0).abs() < 1e-12);
        assert!((airy96_point.y + 1.0).abs() < 1e-12);
        assert!((airy96_point.z - MID70_ORIGIN_IN_AIRY96_Z_M).abs() < 1e-12);
    }

    #[test]
    fn imu_to_mid70_rotation_composes_back_to_airy96_rotation() {
        let imu_vector = Vector3::new(0.3, -0.4, 0.5);
        let imu_to_mid70 = make_imu_to_mid70_rotation();
        let airy96_from_mid70 = make_airy96_from_mid70_extrinsic();
        let airy96_from_mid70_rotation = airy96_from_mid70.fixed_view::<3, 3>(0, 0).into_owned();

        let via_mid70 = airy96_from_mid70_rotation * (imu_to_mid70 * imu_vector);
        let imu_to_airy96 = make_imu_to_airy96_rotation();
        let direct = imu_to_airy96 * imu_vector;

        assert!((via_mid70 - direct).norm() < 1e-12);
    }

    #[test]
    fn imu_to_avia_rotation_preserves_axes() {
        let imu_vector = Vector3::new(0.3, -0.4, 0.5);
        assert!((make_imu_to_avia_rotation() * imu_vector - imu_vector).norm() < 1e-12);
    }
}
