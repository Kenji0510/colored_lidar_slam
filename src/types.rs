use nalgebra::{Matrix4, Point3, Vector3};
use pcd_rs::{PcdDeserialize, PcdSerialize};
use serde::{Deserialize, Serialize};

use crate::voxel_map::LOCALMap;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoadIMU {
    pub timestamp: u64,
    pub angular_velocity: [f32; 3],
    pub linear_acceleration: [f32; 3],
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IMU {
    pub timestamp: f64,
    pub angular_velocity: [f32; 3],
    pub linear_acceleration: [f32; 3],
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZ {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZIT {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub timestamp: f64,
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZCov {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub cov_xx: f32,
    pub cov_xy: f32,
    pub cov_xz: f32,
    pub cov_yy: f32,
    pub cov_yz: f32,
    pub cov_zz: f32,
}

#[derive(Debug, Clone)]
pub struct FrameData {
    pub points: Vec<Point3<f32>>,
    pub covariances: Vec<[[f32; 3]; 3]>,
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZNormal {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub normal_x: f32,
    pub normal_y: f32,
    pub normal_z: f32,
}

pub struct CurrentFrameInfo {
    pub current_global_pose: Matrix4<f64>,
    pub current_velocity: Vector3<f64>,
}

pub struct SLAMMap {
    pub global_voxel_map: LOCALMap,
    pub local_voxel_map: LOCALMap,
}

pub struct ProcessTimes {
    pub total: f64,
    pub find_nearest_points: f64,
    pub icp: f64,
    pub update_map: f64,
}

/// Wall-clock timings for one point-cloud frame, in milliseconds.
///
/// The top-level stages do not overlap and can therefore be compared directly.
/// The `icp_*` detail fields are a breakdown of `icp_ms`, and the `coloring_*`
/// detail fields are a breakdown of `coloring_ms`; neither must be added to the
/// top-level stages a second time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FrameTiming {
    pub load_pcd_ms: f64,
    pub timestamps_ms: f64,
    pub imu_predict_ms: f64,
    pub rotation_trajectory_ms: f64,
    pub deskew_ms: f64,
    pub downsample_ms: f64,
    pub build_source_maps_ms: f64,
    pub icp_ms: f64,
    /// Grayscale conversion, direct image alignment and reference patch updates.
    pub visual_alignment_ms: f64,
    pub pose_update_ms: f64,
    pub global_filter_ms: f64,
    pub global_map_update_ms: f64,
    pub delayed_surface_ms: f64,
    pub local_map_update_ms: f64,
    /// Total time spent coloring the registered frame (sum of `coloring_*`).
    pub coloring_ms: f64,
    /// Per-frame processing time excluding PCD file I/O.
    pub processing_total_ms: f64,
    /// Per-frame wall time including PCD file I/O.
    pub total_with_file_io_ms: f64,
    /// Time spent finding ICP correspondences, including final validation.
    pub icp_correspondence_search_ms: f64,
    /// Time spent building ICP point-to-plane linear systems.
    pub icp_linear_system_ms: f64,
    /// Time spent solving ICP linear systems.
    pub icp_solver_ms: f64,
    /// Remaining ICP work, calculated from `icp_ms`.
    pub icp_other_ms: f64,
    /// Time spent finding the image nearest to the LiDAR mid-time.
    pub coloring_image_select_ms: f64,
    /// Time spent loading and decoding the selected RGB image.
    pub coloring_image_load_ms: f64,
    /// Time spent predicting the LiDAR/camera pose at the image timestamp.
    pub coloring_image_pose_ms: f64,
    /// Time spent projecting deskewed points into the image.
    pub coloring_projection_ms: f64,
    /// Time spent sampling image colors for the deskewed frame points.
    pub coloring_colorize_ms: f64,
    /// Time spent sampling image colors for the GlobalMap input points.
    pub coloring_global_rgb_ms: f64,
}

/// Image-only diagnostics. ICP quality fields retain their pre-visual meaning.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VisualAlignmentLog {
    pub status: String,
    pub update_applied: bool,
    pub matched_patches: usize,
    pub residual_pixels: usize,
    pub iterations: usize,
    /// Zero-mean patch RMSE on grayscale values in 0..255.
    pub initial_rmse: Option<f64>,
    pub final_rmse: Option<f64>,
    pub translation_correction_m: f64,
    pub rotation_correction_deg: f64,
    pub new_landmarks: usize,
    pub refreshed_landmarks: usize,
    pub landmarks: usize,
    /// Frame index of the keyframe K used for alignment (None before bootstrap).
    pub keyframe_frame_index: Option<u64>,
    /// Frames elapsed since K was created.
    pub keyframe_age_frames: u64,
    /// Number of patch points stored in K.
    pub keyframe_points: usize,
    /// Camera-center distance between K and the current image [m].
    pub keyframe_baseline_m: Option<f64>,
    /// ln(current mean intensity / K mean intensity).
    pub brightness_log_ratio: Option<f64>,
    /// K patches visible (in image and not occluded) at the prior pose.
    pub visible_patches: usize,
    /// K patches rejected by the depth-buffer occlusion check.
    pub occluded_patches: usize,
    /// Visible fraction of K patches at the aligned pose (keyframe switch criterion).
    pub visible_ratio: Option<f64>,
    /// Candidate / occluded counts when building a new K this frame.
    pub keyframe_candidates: usize,
    pub keyframe_candidates_occluded: usize,
    /// Points in the newly built K (0 when no K was built).
    pub new_keyframe_points: usize,
    /// "none", "created:<reason>" or "skipped:<reason>:<cause>".
    pub keyframe_event: String,
    pub consecutive_failures: u32,
}

impl Default for VisualAlignmentLog {
    fn default() -> Self {
        Self {
            status: "not_run".into(),
            update_applied: false,
            matched_patches: 0,
            residual_pixels: 0,
            iterations: 0,
            initial_rmse: None,
            final_rmse: None,
            translation_correction_m: 0.0,
            rotation_correction_deg: 0.0,
            new_landmarks: 0,
            refreshed_landmarks: 0,
            landmarks: 0,
            keyframe_frame_index: None,
            keyframe_age_frames: 0,
            keyframe_points: 0,
            keyframe_baseline_m: None,
            brightness_log_ratio: None,
            visible_patches: 0,
            occluded_patches: 0,
            visible_ratio: None,
            keyframe_candidates: 0,
            keyframe_candidates_occluded: 0,
            new_keyframe_points: 0,
            keyframe_event: "none".into(),
            consecutive_failures: 0,
        }
    }
}

/// GlobalMap への色付け（ベストビュー方式）の診断。`color_selection` を参照。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ColoringLog {
    /// "colored"（色付けした）、または色付けしなかった理由:
    /// "lidar_rejected" / "no_image" / "no_image_pose" / "too_dark" / "too_saturated" /
    /// "too_blurry" / "no_candidates" / "invalid_config" / "invalid_camera_pose" /
    /// "invalid_image_dimensions"。
    pub status: String,
    /// 画像の中央 75% に写り、深度が範囲内だった点の数。
    pub candidates: usize,
    /// 点ごとの判定を通り、色の観測として渡した点の数。
    pub observations: usize,
    pub rejected_saturated: usize,
    pub rejected_oblique: usize,
    pub rejected_blurry: usize,
    /// 露光比 g = このフレームの輝度 / 地図にある色の輝度（同じセルどうしの中央値）。
    pub exposure_ratio: Option<f64>,
    /// 露光比の推定に使えた点の数。
    pub exposure_samples: usize,
    /// 予測ブレ量の中央値 [px]（幅 960 相当に換算）。
    pub median_blur_px: Option<f64>,
    /// 白飛び画素の割合。
    pub saturated_ratio: Option<f64>,
    /// 画像時刻のカメラの角速度の大きさ [deg/s]。
    pub angular_speed_deg_s: Option<f64>,
    /// このフレームの観測で色が更新された GlobalMap のセル数。
    pub color_updates: usize,
    /// 最終の色付き点群（dense、`dense_color_map` 参照）の候補として貯めた点の数。
    pub dense_points_stored: usize,
}

impl Default for ColoringLog {
    fn default() -> Self {
        Self {
            status: "not_run".into(),
            candidates: 0,
            observations: 0,
            rejected_saturated: 0,
            rejected_oblique: 0,
            rejected_blurry: 0,
            exposure_ratio: None,
            exposure_samples: 0,
            median_blur_px: None,
            saturated_ratio: None,
            angular_speed_deg_s: None,
            color_updates: 0,
            dense_points_stored: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameLog {
    pub frame_index: usize,
    pub timestamp: f64,
    pub icp_ok: bool,
    /// Final Point-to-Plane RMSE [m]. May be present for a candidate that was
    /// rejected by another quality gate; None when ICP produced no candidate.
    pub rmse: Option<f32>,
    #[serde(default)]
    pub correspondence_count: usize,
    #[serde(default)]
    pub correspondence_ratio: f32,
    #[serde(default)]
    pub observable_rank: usize,
    #[serde(default)]
    pub min_observable_eigenvalue_ratio: Option<f32>,
    #[serde(default)]
    pub icp_translation_correction_m: f32,
    #[serde(default)]
    pub icp_rotation_correction_deg: f32,
    #[serde(default)]
    pub map_updated: bool,
    /// Translation distance from previous frame [m].
    pub translation_m: f64,
    /// Rotation angle from previous frame [deg].
    pub rotation_deg: f64,
    /// Speed (velocity magnitude) [m/s].
    pub velocity_m_s: f64,
    pub pose_x: f64,
    pub pose_y: f64,
    pub pose_z: f64,
    #[serde(default)]
    pub timing: FrameTiming,
    #[serde(default)]
    pub visual_alignment: VisualAlignmentLog,
    /// ESKF の LiDAR 更新を採用したか（false なら IMU 予測のみ）。
    #[serde(default)]
    pub eskf_update_applied: bool,
    /// 推定ジャイロバイアス（LiDAR 軸）[rad/s]。
    #[serde(default)]
    pub gyro_bias_radps: [f64; 3],
    /// 推定加速度バイアス（LiDAR 軸）[m/s²]。
    #[serde(default)]
    pub accel_bias_mps2: [f64; 3],
    /// ワールド座標の速度 [m/s]。
    #[serde(default)]
    pub velocity_mps: [f64; 3],
    /// 以下、更新後の共分散の対角から求めた標準偏差。
    #[serde(default)]
    pub attitude_std_deg: [f64; 3],
    #[serde(default)]
    pub position_std_m: [f64; 3],
    #[serde(default)]
    pub velocity_std_mps: [f64; 3],
    #[serde(default)]
    pub gyro_bias_std_radps: [f64; 3],
    #[serde(default)]
    pub accel_bias_std_mps2: [f64; 3],
    /// 収束点での事前項 (x ⊟ x_pred)ᵀ P⁻¹ (x ⊟ x_pred)。
    /// おおむね 15 以下が目安。常に大きく超えるなら IMU ノイズ設定が小さすぎ
    /// （予測を信用しすぎ）。
    #[serde(default)]
    pub eskf_prior_cost: Option<f64>,
    /// GlobalMap への色付けの診断（色付けしなかったフレームは理由を記録）。
    #[serde(default)]
    pub coloring: ColoringLog,
}

#[cfg(test)]
mod tests {
    use super::FrameLog;

    #[test]
    fn frame_log_reads_legacy_json_without_quality_fields() {
        let json = r#"{
            "frame_index": 1,
            "timestamp": 2.0,
            "icp_ok": true,
            "rmse": 0.03,
            "translation_m": 0.1,
            "rotation_deg": 0.2,
            "velocity_m_s": 1.0,
            "pose_x": 1.0,
            "pose_y": 2.0,
            "pose_z": 3.0
        }"#;

        let log: FrameLog = serde_json::from_str(json).unwrap();
        assert_eq!(log.correspondence_count, 0);
        assert_eq!(log.correspondence_ratio, 0.0);
        assert_eq!(log.observable_rank, 0);
        assert!(!log.map_updated);
        assert_eq!(log.timing.processing_total_ms, 0.0);
        assert_eq!(log.coloring.status, "not_run");
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RosTimestamp {
    pub sec: i32,
    pub nanosec: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageFrameMetadata {
    pub frame_index: u64,
    pub file_name: String,
    pub timestamp_ns: i64,
    pub timestamp_sec: f64,
    pub stamp: RosTimestamp,
    pub frame_id: String,
    pub width: u32,
    pub height: u32,
    pub source_encoding: String,
    pub source_step: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageRecordingMetadata {
    topic: String,
    image_format: String,
    frame_count: usize,
    pub frames: Vec<ImageFrameMetadata>,
}

#[derive(Debug, Clone)]
pub struct ColoredPoint {
    pub position: Point3<f32>, // 世界座標
    pub rgb: Option<[u8; 3]>,
}

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZRGB {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub rgb: f32,
}

/// 色付き点と、その点を撮ったフレーム番号（確認用の出力。ビューアでフレーム番号ごとに色分けできる）。
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZRGBFrame {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub rgb: f32,
    pub frame: f32,
}

impl PointXYZRGB {
    pub fn new(position: Point3<f32>, rgb: [u8; 3]) -> Self {
        let packed_rgb = ((rgb[0] as u32) << 16) | ((rgb[1] as u32) << 8) | (rgb[2] as u32);

        Self {
            x: position.x,
            y: position.y,
            z: position.z,
            rgb: f32::from_bits(packed_rgb),
        }
    }
}

/// 位置合わせで得た姿勢（ワールド座標系における LiDAR フレーム開始時刻の LiDAR 姿勢）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrationPose {
    /// 位置 [m] (x, y, z)。
    pub position_m: [f64; 3],
    /// 姿勢の四元数 (x, y, z, w)。
    pub orientation_xyzw: [f64; 4],
    /// 同次変換行列 world_from_lidar（行優先の 4x4）。
    pub world_from_lidar: [[f64; 4]; 4],
}

impl RegistrationPose {
    pub fn from_matrix(pose: &Matrix4<f64>) -> Self {
        let rotation: nalgebra::Matrix3<f64> = pose.fixed_view::<3, 3>(0, 0).into_owned();
        let quaternion = nalgebra::UnitQuaternion::from_matrix(&rotation);
        let mut world_from_lidar = [[0.0; 4]; 4];
        for (row, values) in world_from_lidar.iter_mut().enumerate() {
            for (col, value) in values.iter_mut().enumerate() {
                *value = pose[(row, col)];
            }
        }
        Self {
            position_m: [pose[(0, 3)], pose[(1, 3)], pose[(2, 3)]],
            orientation_xyzw: [quaternion.i, quaternion.j, quaternion.k, quaternion.w],
            world_from_lidar,
        }
    }
}

/// LiDAR 点群による位置合わせ (Point-to-Plane ICP) の結果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LidarRegistrationResult {
    /// "ok"（採用）/ "rejected"（品質不足で IMU 予測を使用）/ "skipped"（初回フレーム等で未実行）。
    pub status: String,
    /// 最終 Point-to-Plane RMSE [m]。ICP が候補を出さなかった場合は None。
    pub rmse_m: Option<f32>,
    pub correspondence_count: usize,
    /// ICP 後（画像位置合わせ前）の姿勢。rejected / skipped の場合は IMU 予測姿勢。
    pub pose: RegistrationPose,
}

/// 画像による位置合わせ（キーフレームパッチの直接法）の結果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRegistrationResult {
    /// `VisualAlignmentLog::status` と同じ。
    pub status: String,
    /// 補正を状態へ反映したか。false の場合 `pose` は LiDAR 位置合わせの姿勢と同じ。
    pub update_applied: bool,
    /// パッチ平均除去後の輝度 RMSE（0..255）。補正前 / 補正後。
    pub initial_rmse: Option<f64>,
    pub final_rmse: Option<f64>,
    pub matched_patches: usize,
    /// 使用した画像ファイル名（画像が選べなかった場合は None）。
    pub image_file_name: Option<String>,
    /// 画像タイムスタンプ（時刻オフセット適用後）[s]。
    pub image_timestamp_sec: Option<f64>,
    /// 画像位置合わせ後の姿勢（このフレームの最終姿勢）。
    pub pose: RegistrationPose,
}

/// 1 フレーム分の位置合わせ結果。`registration_results.json` に配列で保存する。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameRegistrationResult {
    pub frame_index: usize,
    /// 入力 PCD のファイル名（例: cloud_000123.pcd）。
    pub pcd_file_name: String,
    /// LiDAR フレーム開始時刻 [s]。姿勢はこの時刻のもの。
    pub timestamp: f64,
    pub lidar: LidarRegistrationResult,
    pub image: ImageRegistrationResult,
    /// 保存したボクセル化済み色付き点群の相対パス（色付き点が無く保存しなかった場合は None）。
    pub colored_cloud_file: Option<String>,
    pub colored_cloud_points: usize,
}
