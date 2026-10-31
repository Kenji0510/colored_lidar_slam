use anyhow::{Result, bail};
use nalgebra::{Matrix3, Matrix4, Point3, Quaternion, UnitQuaternion, Vector3};
use re_lidar_slam::{
    camera::{
        CameraIntrinsics, ColorDepthMap, SingleImageColorReference, build_camera_from_lidar, color_image_sharpness,
        colorize_world_points, project_camera_point,
    },
    deskew_points::deskew_points,
    eskf::{
        Eskf, EskfConfig, ImuModel, ImuNoise, InitialStd, NavState, STANDARD_GRAVITY_MPS2,
        StaticInitConfig, estimate_static_init,
    },
    file_handler::{
        find_nearest_image_frame, load_image_meta, load_imu_data, load_pcd_files, load_pcd_xyzit,
        load_rgb_image, save_pcd_xyz, save_pcd_xyzrgb,
    },
    find_nearest_points::pickup_valid_source_points,
    icp::{
        Vector6f, build_robust_point_to_plane_system, compute_rmse, compute_robust_cost,
        solve_icp_delta_observable,
    },
    predict_pose_by_imu::{align_imu_timestamps, build_rotation_trajectory},
    regional_recolor::{RegionalRecolor, RegionalRecolorConfig, RegionalRecolorPoint},
    types::{CurrentFrameInfo, FrameLog, FrameTiming, IMU, PointXYZ, PointXYZRGB, SLAMMap},
    voxel_map::{
        ColorObservation, LOCALMap, LocalMapConfig, SurfaceFilterConfig, SurfaceStatus,
        WorldMapUpdateFilterConfig,
    },
    voxelization::{voxel_downsample_points, voxel_downsample_points_and_maps_dual},
};
use std::{
    ffi::OsString,
    time::{Duration, Instant},
};

const DATASET_DIR: &str = "/mnt/nas/share/avia/10242026/02";
// const DATASET_DIR: &str = "/mnt/nas/share/airy96/06212026/park05";
const SAVE_ROOT_DIR: &str = "data/output/10242026";

// 遮蔽の色更新ゲート。まずはこの値で確認する。
const COLOR_OCCLUSION_MARGIN_M: f64 = 0.20;
const COLOR_OCCLUSION_RADIUS_PX: u32 = 2;

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

#[derive(Debug)]
struct RunOptions {
    lidar_model: LidarModel,
    /// PCD処理ループの0始まりインデックス。画像のframe_indexとは別。
    recolor_frame: Option<usize>,
    /// 選ばれた元画像上の[x_min, y_min, x_max, y_max]。
    recolor_roi: Option<[u32; 4]>,
    recolor_auto: bool,
    recolor_region_size_m: f64,
    recolor_every: usize,
}

enum CommandLineAction {
    Run(RunOptions),
    Help,
}

fn parse_command_line<I>(args: I) -> Result<CommandLineAction>
where
    I: IntoIterator<Item = OsString>,
{
    let mut lidar_model = LidarModel::Mid70;
    let mut recolor_frame = None;
    let mut recolor_roi = None;
    let regional_defaults = RegionalRecolorConfig::default();
    let mut recolor_auto = false;
    let mut recolor_region_size_m = regional_defaults.region_size_m;
    let mut recolor_every = regional_defaults.evaluate_every;
    let mut regional_option_given = false;
    let mut args = args.into_iter();

    while let Some(argument) = args.next() {
        let argument = argument
            .into_string()
            .map_err(|_| anyhow::anyhow!("command-line arguments must be valid UTF-8"))?;

        match argument.as_str() {
            "-h" | "--help" => return Ok(CommandLineAction::Help),
            "--lidar" => {
                let value = args
                    .next()
                    .ok_or_else(|| {
                        anyhow::anyhow!("--lidar requires 'mid70', 'airy96', or 'avia'")
                    })?
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("LiDAR model must be valid UTF-8"))?;
                lidar_model = LidarModel::parse(&value)?;
            }
            _ if argument.starts_with("--lidar=") => {
                lidar_model = LidarModel::parse(&argument["--lidar=".len()..])?;
            }
            "--recolor-frame" => {
                let value = next_command_line_value(&mut args, "--recolor-frame")?;
                recolor_frame = Some(parse_recolor_frame(&value)?);
            }
            _ if argument.starts_with("--recolor-frame=") => {
                recolor_frame = Some(parse_recolor_frame(&argument["--recolor-frame=".len()..])?);
            }
            "--recolor-roi" => {
                let value = next_command_line_value(&mut args, "--recolor-roi")?;
                recolor_roi = Some(parse_recolor_roi(&value)?);
            }
            _ if argument.starts_with("--recolor-roi=") => {
                recolor_roi = Some(parse_recolor_roi(&argument["--recolor-roi=".len()..])?);
            }
            "--recolor-auto" => recolor_auto = true,
            "--recolor-region-size" => {
                let value = next_command_line_value(&mut args, "--recolor-region-size")?;
                recolor_region_size_m = parse_recolor_region_size(&value)?;
                regional_option_given = true;
            }
            _ if argument.starts_with("--recolor-region-size=") => {
                recolor_region_size_m = parse_recolor_region_size(&argument["--recolor-region-size=".len()..])?;
                regional_option_given = true;
            }
            "--recolor-every" => {
                let value = next_command_line_value(&mut args, "--recolor-every")?;
                recolor_every = parse_recolor_every(&value)?;
                regional_option_given = true;
            }
            _ if argument.starts_with("--recolor-every=") => {
                recolor_every = parse_recolor_every(&argument["--recolor-every=".len()..])?;
                regional_option_given = true;
            }
            _ => bail!("unknown argument '{argument}'; use --help for usage"),
        }
    }

    anyhow::ensure!(
        recolor_roi.is_none() || recolor_frame.is_some(),
        "--recolor-roi requires --recolor-frame"
    );

    anyhow::ensure!(
        !regional_option_given || recolor_auto,
        "--recolor-region-size and --recolor-every require --recolor-auto"
    );

    Ok(CommandLineAction::Run(RunOptions {
        lidar_model,
        recolor_frame,
        recolor_roi,
        recolor_auto,
        recolor_region_size_m,
        recolor_every,
    }))
}

fn next_command_line_value(
    args: &mut impl Iterator<Item = OsString>,
    flag: &str,
) -> Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))?
        .into_string()
        .map_err(|_| anyhow::anyhow!("{flag} must be valid UTF-8"))
}

fn parse_recolor_frame(value: &str) -> Result<usize> {
    value.parse::<usize>().map_err(|_| {
        anyhow::anyhow!("--recolor-frame must be a zero-based non-negative PCD frame index")
    })
}

fn parse_recolor_roi(value: &str) -> Result<[u32; 4]> {
    let coordinates: Vec<u32> = value
        .split(',')
        .map(|part| part.trim().parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| anyhow::anyhow!("--recolor-roi must be x_min,y_min,x_max,y_max"))?;
    anyhow::ensure!(coordinates.len() == 4, "--recolor-roi needs four coordinates");
    let roi = [coordinates[0], coordinates[1], coordinates[2], coordinates[3]];
    anyhow::ensure!(
        roi[0] < roi[2] && roi[1] < roi[3],
        "--recolor-roi requires x_min < x_max and y_min < y_max"
    );
    Ok(roi)
}

fn parse_recolor_region_size(value: &str) -> Result<f64> {
    let size = value.parse::<f64>()
        .map_err(|_| anyhow::anyhow!("--recolor-region-size must be a positive number in meters"))?;
    anyhow::ensure!(size.is_finite() && size > 0.0,
        "--recolor-region-size must be finite and positive");
    Ok(size)
}

fn parse_recolor_every(value: &str) -> Result<usize> {
    let every = value.parse::<usize>()
        .map_err(|_| anyhow::anyhow!("--recolor-every must be a positive frame interval"))?;
    anyhow::ensure!(every > 0, "--recolor-every must be greater than zero");
    Ok(every)
}

fn print_usage() {
    println!("Usage: re_lidar_slam [--lidar <mid70|airy96|avia>] [--recolor-frame N] [--recolor-roi x0,y0,x1,y1] [--recolor-auto] [--recolor-region-size M] [--recolor-every N]");
    println!("  --lidar  Select the point-cloud sensor (default: mid70)");
    println!("  --recolor-frame  Save a separate final map colored from the image selected at PCD frame N (zero-based)");
    println!("  --recolor-roi    Optional rectangle for --recolor-frame; right/bottom edges are exclusive");
    println!("  --recolor-auto   Select one image per world-space surface region and recolor the completed map");
    println!("  --recolor-region-size  Region size in meters (default: 0.5)");
    println!("  --recolor-every        Candidate frame interval (default: 5)");
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
const GLOBAL_MAP_VOXEL_SIZE: f32 = 0.01; // m
// このセンサー構成では、0.025 m voxel で走行全体で 1 フレームあたり約 1800 個の
// GlobalMap voxel が増える。点群は主に面なので、voxel 数は (0.025 / size)^2 で
// 増えるとみなして換算する。入力フレーム数から最終容量を先に確保し、実行途中の
// 巨大な HashMap 再配置（数百 ms のスパイク）を避ける。
const GLOBAL_MAP_EXPECTED_NEW_VOXELS_PER_FRAME: usize = (2048.0
    * (0.025 / GLOBAL_MAP_VOXEL_SIZE)
    * (0.025 / GLOBAL_MAP_VOXEL_SIZE)) as usize;
// GlobalMap の近傍探索半径を voxel サイズに依存しない物理長で指定し、voxel 数へ換算する。
const fn global_map_radius_voxels(radius_m: f32) -> i32 {
    (radius_m / GLOBAL_MAP_VOXEL_SIZE).ceil() as i32
}

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

// Global map の累積点から局所平面を確定するための遅延 RANSAC/PCA 設定。
// 成熟平面を後続フレームの挿入ゲートに使うため、従来の30フレームより早く確定する。
const SURFACE_CLASSIFICATION_DELAY_FRAMES: u64 = 5;
// 近傍は中心から ±5 cm（0.025 m voxel で 5x5x5、0.01 m voxel で 11x11x11）。
// voxel 数で固定すると細かい voxel では近傍が数 cm に縮み、LiDAR ノイズと
// 同程度の範囲で平面判定することになって大半が NonPlanar/Unknown になる。
const SURFACE_NEIGHBOR_RADIUS_M: f32 = 0.05;
const SURFACE_NEIGHBOR_RADIUS_VOXELS: i32 = global_map_radius_voxels(SURFACE_NEIGHBOR_RADIUS_M);
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
const WORLD_UPDATE_PLANE_SEARCH_RADIUS_M: f32 = 0.05;
const WORLD_UPDATE_PLANE_SEARCH_RADIUS_VOXELS: i32 =
    global_map_radius_voxels(WORLD_UPDATE_PLANE_SEARCH_RADIUS_M);
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

const COLOR_MIN_LAPLACIAN_VARIANCE: f64 = 80.0;

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

    let run_options = match parse_command_line(std::env::args_os().skip(1))? {
        CommandLineAction::Run(options) => options,
        CommandLineAction::Help => {
            print_usage();
            return Ok(());
        }
    };
    let lidar_model = run_options.lidar_model;
    let load_dir = format!("{DATASET_DIR}/{}", lidar_model.input_subdir());
    let save_dir = format!("{SAVE_ROOT_DIR}/{}", lidar_model.name());
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
    if let Some(frame) = run_options.recolor_frame {
        anyhow::ensure!(
            frame < pcd_files.len(),
            "--recolor-frame {} is outside the {} PCD frames in this dataset",
            frame,
            pcd_files.len(),
        );
    }

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
    let image_meta = load_image_meta(&image_meta_file)?;
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

    let camera_from_lidar = build_camera_from_lidar(
        Vector3::new(0.04, 0.0, 0.05), // カメラ位置 [m]
        0.0,                           // roll [deg]
        -1.8,                          // pitch [deg]
        -2.2,                          // yaw [deg]
    );

    let mut single_image_color_reference: Option<SingleImageColorReference> = None;
    let mut regional_recolor = if run_options.recolor_auto {
        let config = RegionalRecolorConfig {
            region_size_m: run_options.recolor_region_size_m,
            evaluate_every: run_options.recolor_every,
            occlusion_margin_m: COLOR_OCCLUSION_MARGIN_M,
            occlusion_radius_px: COLOR_OCCLUSION_RADIUS_PX,
            ..Default::default()
        };
        let recolor = RegionalRecolor::new(
            config,
            std::path::Path::new(&save_dir).join("regional_recolor_cache"),
        )?;
        log::info!("Regional recolor: size={} m, every={} frames, cache={}",
            run_options.recolor_region_size_m, run_options.recolor_every,
            recolor.cache_dir().display());
        Some(recolor)
    } else {
        None
    };

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

            let sharpness = color_image_sharpness(&rgb_image);
            let use_for_coloring = sharpness >= COLOR_MIN_LAPLACIAN_VARIANCE;

            log::info!(
                "Frame {i}: image={} size={}x{} sharpness={:.2} \
             threshold={:.2} use_for_coloring={}",
                image_path.display(),
                rgb_image.width(),
                rgb_image.height(),
                sharpness,
                COLOR_MIN_LAPLACIAN_VARIANCE,
                use_for_coloring,
            );

            if use_for_coloring {
                Some(rgb_image)
            } else {
                log::info!("Frame {i}: coloring skipped: image sharpness is too low");

                None
            }
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
        let lidar_pose_at_image_time: Option<Matrix4<f64>> =
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

                    Some(pose)
                }
            } else {
                None
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

        // 選択した1フレームだけ、元画像・画像時刻の姿勢・深度を固定して保持する。
        if run_options.recolor_frame == Some(i) {
            anyhow::ensure!(
                local_map_was_empty || icp_ok,
                "Cannot use recolor frame {i}: ICP quality is insufficient"
            );
            let metadata = selected_image.ok_or_else(|| {
                anyhow::anyhow!("Cannot use recolor frame {i}: no matching image")
            })?;
            let image = selected_rgb_image.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "Cannot use recolor frame {i}: image failed the sharpness gate; inspect its sharpness log"
                )
            })?;
            let camera_from_world = camera_from_world_at_image.as_ref().ok_or_else(|| {
                anyhow::anyhow!("Cannot use recolor frame {i}: image-time camera pose is unavailable")
            })?;

            single_image_color_reference = Some(SingleImageColorReference::capture(
                image,
                camera_from_world,
                &current_frame_info.current_global_pose,
                &deskewed_points,
                run_options.recolor_roi,
            )?);
            log::info!(
                "Single-image reference: PCD frame={} image={} image_time={:.9} roi={:?}",
                i, metadata.file_name, metadata.timestamp_sec, run_options.recolor_roi,
            );
        }
        let coloring_image_pose_time = coloring_image_pose_start.elapsed();

        let coloring_projection_start = Instant::now();
        // 自動再着色の候補収集。実際のRGB取得は全フレーム処理後に行う。
        if let Some(recolor) = regional_recolor.as_mut() {
            if recolor.should_consider_frame(i) && (local_map_was_empty || icp_ok) {
                if let (Some(metadata), Some(image), Some(camera_from_world)) = (
                    selected_image,
                    selected_rgb_image.as_ref(),
                    camera_from_world_at_image.as_ref(),
                ) {
                    let prepare_start = Instant::now();
                    let image_path = std::path::Path::new(&image_meta_dir).join(&metadata.file_name);
                    let observed = recolor.observe_frame(
                        i, metadata.timestamp_sec, &image_path, image,
                        camera_from_world,
                        &current_frame_info.current_global_pose,
                        &deskewed_points,
                    )?;
                    log::info!(
                        "Regional candidates: frame={} regions={} retained={} cached={} elapsed={:.2?}",
                        i, observed.visible_regions, observed.retained_candidates,
                        observed.cached, prepare_start.elapsed(),
                    );
                }
            }
        }
        let (projected_pixels, color_depth_map) =
            if let (Some(rgb_image), Some(camera_from_world)) = (
                selected_rgb_image.as_ref(),
                camera_from_world_at_image.as_ref(),
            ) {
                let intrinsics =
                    CameraIntrinsics::for_image_size(rgb_image.width(), rgb_image.height())?;

                // 点群と深度画像を、同じ画像撮影時刻のカメラ座標へ揃える。
                let camera_from_lidar_start =
                    camera_from_world * current_frame_info.current_global_pose;

                // 平面フィルター・間引き前の点群から作る。
                let depth_map =
                    ColorDepthMap::build(&deskewed_points, &camera_from_lidar_start, &intrinsics);

                let mut occluded_count = 0usize;
                let pixels: Vec<Option<(u32, u32)>> = deskewed_points
                    .iter()
                    .map(|point| {
                        let camera_point =
                            camera_from_lidar_start.transform_point(&point.cast::<f64>());
                        let (u, v) = project_camera_point(&camera_point, &intrinsics)?;

                        if depth_map.is_occluded(
                            u,
                            v,
                            camera_point.z,
                            COLOR_OCCLUSION_MARGIN_M,
                            COLOR_OCCLUSION_RADIUS_PX,
                        ) {
                            occluded_count += 1;
                            return None;
                        }

                        Some((u, v))
                    })
                    .collect();

                log::info!(
                    "Frame {i}: color candidates={}/{} occluded={} image={}x{}",
                    pixels.iter().flatten().count(),
                    pixels.len(),
                    occluded_count,
                    intrinsics.width,
                    intrinsics.height,
                );

                (Some(pixels), Some(depth_map))
            } else {
                (None, None)
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
        let (icp_translation_correction_m, icp_rotation_correction_deg) = if icp_ok {
            pose_correction_magnitudes(&pred_r, &pred_t, &r_mat, &t_vec)
        } else {
            (0.0, 0.0)
        };
        let map_update_allowed = local_map_was_empty || icp_ok;
        // --- Record frame log ---
        // Coloring is reported separately, so pose_update covers only the pose
        // state update and the frame-log bookkeeping around it.
        let pose_update_time = pose_state_update_time + frame_log_record_start.elapsed();

        // --- Filter valid source points, then update the WorldMap ---
        // ローカルマップが空（初回フレーム）の場合はフィルタなしで全点追加。
        // それ以外は pickup_valid_source_points で平面に乗っている点だけ抽出し、
        // ICP 収束後の最終姿勢 (r_mat, t_vec) でワールドマップに追加する。
        let global_filter_start = Instant::now();
        let global_source_points: Vec<Point3<f32>> = if !map_update_allowed {
            log::warn!("Frame {i}: map update skipped because ICP quality is insufficient");
            Vec::new()
        } else if local_map_was_empty {
            downsampled_source_points_for_global.clone()
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
            valid.into_iter().map(|c| c.src_point).collect()
        };
        let global_filter_time = global_filter_start.elapsed();

        let coloring_global_rgb_start = Instant::now();
        let global_source_rgb: Option<Vec<Option<ColorObservation>>> = match (
            map_update_allowed,
            selected_rgb_image.as_ref(),
            camera_from_world_at_image.as_ref(),
        ) {
            (true, Some(rgb_image), Some(camera_from_world)) => {
                let intrinsics =
                    CameraIntrinsics::for_image_size(rgb_image.width(), rgb_image.height())?;

                // LiDARフレーム開始時の座標 → 画像撮影時のカメラ座標
                let camera_from_lidar_start =
                    camera_from_world * current_frame_info.current_global_pose;

                let colors: Vec<Option<ColorObservation>> = global_source_points
                    .iter()
                    .map(|point| {
                        let camera_point =
                            camera_from_lidar_start.transform_point(&point.cast::<f64>());

                        let (u, v) = project_camera_point(&camera_point, &intrinsics)?;

                        // GlobalMap用の色にも同じ遮蔽判定を適用する。
                        if color_depth_map.as_ref().is_some_and(|depth_map| {
                            depth_map.is_occluded(
                                u,
                                v,
                                camera_point.z,
                                COLOR_OCCLUSION_MARGIN_M,
                                COLOR_OCCLUSION_RADIUS_PX,
                            )
                        }) {
                            return None;
                        }

                        let rgb = rgb_image.get_pixel_checked(u, v)?.0;

                        // カメラ座標の原点は、画像撮影時のカメラ位置。
                        // この点の長さが、カメラから対象点までの距離になる。
                        let distance_m = camera_point.coords.norm();

                        if !distance_m.is_finite() || distance_m <= 0.0 {
                            return None;
                        }

                        Some(ColorObservation { rgb, distance_m })
                    })
                    .collect();

                let colored_count = colors.iter().filter(|color| color.is_some()).count();

                log::debug!(
                    "Frame {i}: GlobalMap input RGB: {} / {} points",
                    colored_count,
                    global_source_points.len(),
                );

                Some(colors)
            }
            _ => None,
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
        });
    }

    log_timing_summary(&frame_logs);

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

    // 通常RGBの保存が終わったので、広域再着色の前に一時メモリを解放する。
    drop(planar_world_map_rgb);

    // 完成した形状の最終座標から、指定した1枚の画像のRGBだけで別PCDを作る。
    // 過去のcell.rgbを使わず、参照画像で着色できない点はこの比較用PCDに含めない。
    if let Some(reference) = single_image_color_reference.as_ref() {
        let frame = run_options.recolor_frame.expect("reference has a frame index");
        let single_image_points: Vec<PointXYZRGB> = slam_map
            .global_voxel_map
            .voxel_map
            .values()
            .filter(|cell| {
                cell.sample_count >= min_samples
                    && cell.observed_frames >= min_frames
                    && cell.surface_status == SurfaceStatus::Planar
                    && cell.mean.coords.iter().all(|value| value.is_finite())
            })
            .filter_map(|cell| {
                let rgb = reference.color_for_world_point(
                    &cell.mean,
                    COLOR_OCCLUSION_MARGIN_M,
                    COLOR_OCCLUSION_RADIUS_PX,
                )?;
                Some(PointXYZRGB::new(cell.mean, rgb))
            })
            .collect();

        let single_image_path = format!(
            "{}/voxel-{}_world_map_rgb_single_image_frame-{}.pcd",
            save_dir, GLOBAL_MAP_VOXEL_SIZE, frame,
        );
        if single_image_points.is_empty() {
            log::warn!(
                "Single-image RGB export skipped: no final planar points passed projection/ROI/occlusion; check the selected frame and ROI"
            );
        } else {
            save_pcd_xyzrgb(&single_image_points, &single_image_path)?;
            log::info!(
                "Saved single-image final map: frame={} points={} path={}",
                frame, single_image_points.len(), single_image_path,
            );
        }
    }

    // 領域単位で画像を選び、完成後のcell.meanへRGBを付与する。
    if let Some(recolor) = regional_recolor.as_ref() {
        let regional_start = Instant::now();
        let points: Vec<RegionalRecolorPoint> = slam_map
            .global_voxel_map.voxel_map.values()
            .filter(|cell| {
                cell.sample_count >= min_samples
                    && cell.observed_frames >= min_frames
                    && cell.surface_status == SurfaceStatus::Planar
                    && cell.mean.coords.iter().all(|value| value.is_finite())
            })
            .map(|cell| RegionalRecolorPoint {
                position: cell.mean,
                normal: cell.surface_plane.as_ref().map(|plane| plane.normal),
            })
            .collect();
        // 通常の処理と同じ画像ローダーを使い、画像の向き・形式を揃える。
        let result = recolor.recolor_points(&points, |path| Ok(load_rgb_image(&path.to_path_buf())?))?;
        let colored: Vec<PointXYZRGB> = points.iter().zip(&result.colors)
            .filter_map(|(point, rgb)| rgb.map(|rgb| PointXYZRGB::new(point.position, rgb)))
            .collect();
        let output_path = format!("{}/voxel-{}_world_map_rgb_regional.pcd",
            save_dir, GLOBAL_MAP_VOXEL_SIZE);
        if colored.is_empty() {
            log::warn!("Regional RGB export skipped: no valid reference views; inspect the report and image/pose/quality logs");
        } else {
            save_pcd_xyzrgb(&colored, &output_path)?;
        }
        let report_path = std::path::Path::new(&save_dir).join("regional_recolor_report.json");
        recolor.write_report(&report_path, &result)?;
        log::info!(
            "Regional recolor: input={} colored={} uncolored={} patches={} selected={} no_candidates={} no_usable_view={} unknown_depth={} scoring_images={} coloring_images={} elapsed={:.2?} output={} report={}",
            points.len(), result.stats.colored_points, result.stats.uncolored_points,
            result.stats.surface_patches, result.stats.selected_patches,
            result.stats.patches_without_candidates, result.stats.patches_without_usable_view,
            result.stats.unknown_depth_colored_points,
            result.stats.images_loaded_for_scoring, result.stats.images_loaded_for_coloring,
            regional_start.elapsed(), output_path, report_path.display(),
        );
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

    Ok(())
}

#[inline]
fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
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

    let stages: [(&str, fn(&FrameTiming) -> f64); 14] = [
        ("load_pcd", |t| t.load_pcd_ms),
        ("timestamps", |t| t.timestamps_ms),
        ("imu_predict", |t| t.imu_predict_ms),
        ("rotation_trajectory", |t| t.rotation_trajectory_ms),
        ("deskew", |t| t.deskew_ms),
        ("downsample", |t| t.downsample_ms),
        ("build_source_maps", |t| t.build_source_maps_ms),
        ("icp", |t| t.icp_ms),
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
        CommandLineAction, IMU, LidarModel, MID70_ORIGIN_IN_AIRY96_Z_M,
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
        let CommandLineAction::Run(super::RunOptions { lidar_model: model, .. }) = parse_args(&[]).unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(model, LidarModel::Mid70);
    }

    #[test]
    fn command_line_selects_airy96() {
        let CommandLineAction::Run(super::RunOptions { lidar_model: model, .. }) = parse_args(&["--lidar", "airy96"]).unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(model, LidarModel::Airy96);

        let CommandLineAction::Run(super::RunOptions { lidar_model: model, .. }) = parse_args(&["--lidar=airy"]).unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(model, LidarModel::Airy96);
    }

    #[test]
    fn command_line_selects_avia() {
        let CommandLineAction::Run(super::RunOptions { lidar_model: model, .. }) = parse_args(&["--lidar", "avia"]).unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(model, LidarModel::Avia);
        assert_eq!(model.input_subdir(), "avia");
        assert_eq!(model.name(), "avia");

        let CommandLineAction::Run(super::RunOptions { lidar_model: model, .. }) = parse_args(&["--lidar=livox-avia"]).unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(model, LidarModel::Avia);
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
    fn command_line_recolor_defaults_are_disabled() {
        let CommandLineAction::Run(options) = parse_args(&[]).unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(options.recolor_frame, None);
        assert_eq!(options.recolor_roi, None);
    }

    #[test]
    fn command_line_recolor_accepts_frame_zero_and_roi() {
        let CommandLineAction::Run(options) = parse_args(&[
            "--lidar", "avia", "--recolor-frame", "0",
            "--recolor-roi", "120, 150,820,520",
        ]).unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(options.lidar_model, LidarModel::Avia);
        assert_eq!(options.recolor_frame, Some(0));
        assert_eq!(options.recolor_roi, Some([120, 150, 820, 520]));
    }

    #[test]
    fn command_line_recolor_accepts_equals_syntax() {
        let CommandLineAction::Run(options) = parse_args(&[
            "--recolor-roi=0,0,960,600", "--recolor-frame=42",
        ]).unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(options.recolor_frame, Some(42));
        assert_eq!(options.recolor_roi, Some([0, 0, 960, 600]));
    }

    #[test]
    fn command_line_recolor_rejects_missing_values_and_invalid_frames() {
        for args in [
            vec!["--recolor-frame"], vec!["--recolor-frame", "-1"],
            vec!["--recolor-frame", "abc"], vec!["--recolor-roi"],
        ] {
            assert!(parse_args(&args).is_err(), "arguments: {args:?}");
        }
    }

    #[test]
    fn command_line_recolor_rejects_invalid_or_unpaired_roi() {
        assert!(parse_args(&["--recolor-roi", "0,0,100,100"]).is_err());
        for roi in ["0,0,10", "0,0,10,10,20", "10,0,0,10", "0,5,10,5", "-1,0,10,10"] {
            assert!(parse_args(&["--recolor-frame", "1", "--recolor-roi", roi]).is_err());
        }
    }

    #[test]
    fn command_line_regional_defaults_and_auto_mode() {
        let CommandLineAction::Run(disabled) = parse_args(&[]).unwrap() else { panic!() };
        assert!(!disabled.recolor_auto);
        let CommandLineAction::Run(options) = parse_args(&["--recolor-auto"]).unwrap() else { panic!() };
        assert!(options.recolor_auto);
        assert_eq!(options.recolor_region_size_m, 0.5);
        assert_eq!(options.recolor_every, 5);
    }

    #[test]
    fn command_line_regional_accepts_tuning_and_equals_syntax() {
        for args in [
            vec!["--recolor-auto", "--recolor-region-size", "0.75", "--recolor-every", "2"],
            vec!["--recolor-region-size=0.75", "--recolor-every=2", "--recolor-auto"],
        ] {
            let CommandLineAction::Run(options) = parse_args(&args).unwrap() else { panic!() };
            assert_eq!(options.recolor_region_size_m, 0.75);
            assert_eq!(options.recolor_every, 2);
        }
    }

    #[test]
    fn command_line_regional_can_run_alongside_single_image_comparison() {
        let CommandLineAction::Run(options) = parse_args(&[
            "--lidar", "avia", "--recolor-auto", "--recolor-frame", "300",
            "--recolor-roi", "0,0,960,600",
        ]).unwrap() else { panic!() };
        assert!(options.recolor_auto);
        assert_eq!(options.recolor_frame, Some(300));
    }

    #[test]
    fn command_line_regional_rejects_invalid_region_sizes() {
        for value in ["0", "-1", "NaN", "inf", "abc"] {
            assert!(parse_args(&["--recolor-auto", "--recolor-region-size", value]).is_err());
        }
    }

    #[test]
    fn command_line_regional_rejects_invalid_intervals_or_missing_auto_flag() {
        for value in ["0", "-1", "abc"] {
            assert!(parse_args(&["--recolor-auto", "--recolor-every", value]).is_err());
        }
        assert!(parse_args(&["--recolor-every", "5"]).is_err());
        assert!(parse_args(&["--recolor-region-size", "0.5"]).is_err());
        assert!(parse_args(&["--recolor-auto", "--recolor-every"]).is_err());
        assert!(parse_args(&["--recolor-auto", "--recolor-region-size"]).is_err());
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
