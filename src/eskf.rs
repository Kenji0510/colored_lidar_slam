//! 15次元 誤差状態カルマンフィルタ（反復更新つき。FAST-LIO2 の ESIKF と同系統）
//!
//! 誤差状態 δx = [δθ, δp, δv, δb_g, δb_a]（各3次元、合計15次元）
//!
//! 規約（`icp.rs` の `apply_delta` とヤコビアンに合わせている）:
//! - 回転誤差は左（ワールド側）摂動:  R = Exp(δθ) · R̂
//! - それ以外は加算:                 p = p̂ + δp, v = v̂ + δv, ...
//! - ボディ座標系: 軸は LiDAR 座標軸、原点は IMU 原点
//!   （IMU 計測値は `ImuModel::imu_to_body` で LiDAR 軸へ回してから使う）
//! - バイアスはボディ座標系で定義（回転・単位変換の後に引く）
//! - 重力は (0, 0, -g) でワールド固定。初期姿勢を静止区間の加速度から
//!   重力方向に揃えることで成立させる（`estimate_static_init`）
//! - 状態の時刻はフレーム開始時刻
//!
//! LiDAR 更新は ICP の 6×6 正規方程式（[δθ; δt_lidar]）をそのまま受け取り、
//! レバーアームを考慮した写像 M で 15次元へ埋め込み、IMU 予測を事前分布として
//! 情報行列形式で解く:
//!
//!   (Mᵀ H M / σ² + P⁻¹) δx = Mᵀ b / σ² − P⁻¹ (x_i ⊟ x_pred)

use std::fmt;

use nalgebra::{Cholesky, Matrix3, Matrix4, SMatrix, SVector, UnitQuaternion, Vector3};

use crate::icp::IcpLinearSystem;
use crate::types::IMU;

pub const ROT: usize = 0;
pub const POS: usize = 3;
pub const VEL: usize = 6;
pub const BG: usize = 9;
pub const BA: usize = 12;
pub const ERROR_STATE_DIM: usize = 15;

pub type Matrix15 = SMatrix<f64, ERROR_STATE_DIM, ERROR_STATE_DIM>;
pub type Vector15 = SVector<f64, ERROR_STATE_DIM>;
type Matrix6x15 = SMatrix<f64, 6, ERROR_STATE_DIM>;

pub const STANDARD_GRAVITY_MPS2: f64 = 9.80665;

// ---------------------------------------------------------------------------
// IMU 計測モデル
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct ImuModel {
    /// IMU 座標 → ボディ（LiDAR 軸）座標の回転。
    pub imu_to_body: UnitQuaternion<f64>,
    /// 生の加速度値に掛けて m/s² にする倍率（g 単位の IMU なら 9.80665）。
    pub accel_scale: f64,
}

impl ImuModel {
    pub fn new(imu_to_body: UnitQuaternion<f64>, accel_scale: f64) -> Self {
        Self {
            imu_to_body,
            accel_scale,
        }
    }

    /// (角速度 [rad/s], 比力 [m/s²]) をボディ座標で返す。バイアスはまだ引かない。
    #[inline]
    pub fn measurement(&self, sample: &IMU) -> (Vector3<f64>, Vector3<f64>) {
        let omega = Vector3::new(
            sample.angular_velocity[0] as f64,
            sample.angular_velocity[1] as f64,
            sample.angular_velocity[2] as f64,
        );
        let accel = Vector3::new(
            sample.linear_acceleration[0] as f64,
            sample.linear_acceleration[1] as f64,
            sample.linear_acceleration[2] as f64,
        );
        (
            self.imu_to_body * omega,
            self.imu_to_body * accel * self.accel_scale,
        )
    }
}

// ---------------------------------------------------------------------------
// 名目状態
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavState {
    /// ワールド ← ボディ の回転。
    pub rot: UnitQuaternion<f64>,
    /// IMU 原点のワールド位置 [m]。
    pub pos: Vector3<f64>,
    /// IMU 原点のワールド速度 [m/s]。
    pub vel: Vector3<f64>,
    /// ジャイロバイアス（ボディ座標）[rad/s]。
    pub bg: Vector3<f64>,
    /// 加速度バイアス（ボディ座標）[m/s²]。
    pub ba: Vector3<f64>,
}

impl Default for NavState {
    fn default() -> Self {
        Self {
            rot: UnitQuaternion::identity(),
            pos: Vector3::zeros(),
            vel: Vector3::zeros(),
            bg: Vector3::zeros(),
            ba: Vector3::zeros(),
        }
    }
}

impl NavState {
    /// x ⊞ δx（回転は左から掛ける）。
    pub fn boxplus(&self, dx: &Vector15) -> Self {
        let d_rot = UnitQuaternion::from_scaled_axis(dx.fixed_rows::<3>(ROT).into_owned());
        let mut rot = d_rot * self.rot;
        rot.renormalize();
        Self {
            rot,
            pos: self.pos + dx.fixed_rows::<3>(POS),
            vel: self.vel + dx.fixed_rows::<3>(VEL),
            bg: self.bg + dx.fixed_rows::<3>(BG),
            ba: self.ba + dx.fixed_rows::<3>(BA),
        }
    }

    /// self ⊟ other（`other.boxplus(&d) == self` となる d）。
    pub fn boxminus(&self, other: &Self) -> Vector15 {
        let mut d = Vector15::zeros();
        d.fixed_rows_mut::<3>(ROT)
            .copy_from(&(self.rot * other.rot.inverse()).scaled_axis());
        d.fixed_rows_mut::<3>(POS)
            .copy_from(&(self.pos - other.pos));
        d.fixed_rows_mut::<3>(VEL)
            .copy_from(&(self.vel - other.vel));
        d.fixed_rows_mut::<3>(BG).copy_from(&(self.bg - other.bg));
        d.fixed_rows_mut::<3>(BA).copy_from(&(self.ba - other.ba));
        d
    }

    /// LiDAR 原点の姿勢 (R_world_lidar, t_world_lidar)。
    /// ボディ軸 = LiDAR 軸なので回転は共通、並進だけレバーアーム分ずらす。
    pub fn lidar_rotation_translation(
        &self,
        imu_origin_in_lidar: &Vector3<f64>,
    ) -> (Matrix3<f64>, Vector3<f64>) {
        let r = self.rot.to_rotation_matrix().into_inner();
        (r, self.pos - r * imu_origin_in_lidar)
    }

    pub fn lidar_pose(&self, imu_origin_in_lidar: &Vector3<f64>) -> Matrix4<f64> {
        let (r, t) = self.lidar_rotation_translation(imu_origin_in_lidar);
        let mut pose = Matrix4::identity();
        pose.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        pose.fixed_view_mut::<3, 1>(0, 3).copy_from(&t);
        pose
    }
}

#[inline]
fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

// ---------------------------------------------------------------------------
// 設定
// ---------------------------------------------------------------------------

/// IMU ノイズ（連続時間の密度）。データシート値に振動分の余裕を掛けて設定する。
#[derive(Debug, Clone, Copy)]
pub struct ImuNoise {
    /// ジャイロ白色ノイズ密度 [rad/s/√Hz]
    pub gyro_noise_density: f64,
    /// 加速度白色ノイズ密度 [m/s²/√Hz]
    pub accel_noise_density: f64,
    /// ジャイロバイアスのランダムウォーク [rad/s²/√Hz]
    pub gyro_bias_random_walk: f64,
    /// 加速度バイアスのランダムウォーク [m/s³/√Hz]
    pub accel_bias_random_walk: f64,
}

/// 初期共分散（各ブロック 3軸共通の標準偏差）。
#[derive(Debug, Clone, Copy)]
pub struct InitialStd {
    pub attitude_rad: f64,
    pub position_m: f64,
    pub velocity_mps: f64,
    pub gyro_bias_radps: f64,
    pub accel_bias_mps2: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct EskfConfig {
    pub imu_noise: ImuNoise,
    pub initial_std: InitialStd,
    pub gravity_mps2: f64,
    /// point-to-plane 残差 1点あたりの標準偏差 [m]。
    /// 小さいほど LiDAR を強く信用する。点同士の誤差相関を無視しているので、
    /// 実際の平面ノイズより大きめにしておくと自信過剰になりにくい。
    pub lidar_point_sigma_m: f64,
}

// ---------------------------------------------------------------------------
// 静止区間からの初期化
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct StaticInitConfig {
    pub window_s: f64,
    pub min_samples: usize,
    /// 角速度の RMS 偏差がこれ以下なら静止とみなす [rad/s]
    pub max_gyro_std_radps: f64,
    /// 加速度の RMS 偏差がこれ以下なら静止とみなす [m/s²]
    pub max_accel_std_mps2: f64,
    /// 静止時の加速度の大きさを重力加速度に合わせて正規化する（FAST-LIO と同じ処理）。
    pub normalize_accel_magnitude: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct StaticInitResult {
    /// 重力方向を +z に合わせる初期姿勢（ワールド ← ボディ）。
    pub rotation: UnitQuaternion<f64>,
    /// 静止時の角速度平均（ボディ座標）。
    pub gyro_bias: Vector3<f64>,
    /// `ImuModel::accel_scale` に掛ける補正係数（正規化しない場合は 1.0）。
    pub accel_scale_correction: f64,
    pub sample_count: usize,
    pub gyro_std_radps: f64,
    pub accel_std_mps2: f64,
    pub mean_accel_norm_mps2: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum StaticInitError {
    TooFewSamples {
        count: usize,
        required: usize,
    },
    NotStatic {
        gyro_std_radps: f64,
        accel_std_mps2: f64,
    },
    BadAccelMagnitude {
        norm_mps2: f64,
    },
}

impl fmt::Display for StaticInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewSamples { count, required } => {
                write!(f, "too few IMU samples in window ({count} < {required})")
            }
            Self::NotStatic {
                gyro_std_radps,
                accel_std_mps2,
            } => write!(
                f,
                "sensor is not static (gyro_std={gyro_std_radps:.4} rad/s, \
                 accel_std={accel_std_mps2:.4} m/s^2)"
            ),
            Self::BadAccelMagnitude { norm_mps2 } => write!(
                f,
                "mean accel magnitude {norm_mps2:.3} m/s^2 is far from gravity \
                 (check accel units)"
            ),
        }
    }
}

/// `[t_start, t_start + window_s]` の IMU を静止区間とみなして初期姿勢と
/// ジャイロバイアスを推定する。
///
/// オフライン処理なので最初のフレームより「後」のデータを使っている。
/// オンライン化する場合は、起動後に window_s 分バッファしてから開始する。
pub fn estimate_static_init(
    imu_data: &[IMU],
    model: &ImuModel,
    t_start: f64,
    config: &StaticInitConfig,
    gravity_mps2: f64,
) -> Result<StaticInitResult, StaticInitError> {
    let t_end = t_start + config.window_s;
    let begin = imu_data.partition_point(|s| s.timestamp < t_start);
    let end = imu_data.partition_point(|s| s.timestamp <= t_end);
    let samples = &imu_data[begin..end.max(begin)];

    let required = config.min_samples.max(2);
    if samples.len() < required {
        return Err(StaticInitError::TooFewSamples {
            count: samples.len(),
            required,
        });
    }

    let n = samples.len() as f64;
    let (gyro_sum, accel_sum) =
        samples
            .iter()
            .fold((Vector3::zeros(), Vector3::zeros()), |(gs, as_), s| {
                let (w, a) = model.measurement(s);
                (gs + w, as_ + a)
            });
    let gyro_mean: Vector3<f64> = gyro_sum / n;
    let accel_mean: Vector3<f64> = accel_sum / n;

    let (gyro_var, accel_var) = samples.iter().fold((0.0, 0.0), |(gv, av), s| {
        let (w, a) = model.measurement(s);
        (
            gv + (w - gyro_mean).norm_squared(),
            av + (a - accel_mean).norm_squared(),
        )
    });
    let gyro_std = (gyro_var / n).sqrt();
    let accel_std = (accel_var / n).sqrt();

    if gyro_std > config.max_gyro_std_radps || accel_std > config.max_accel_std_mps2 {
        return Err(StaticInitError::NotStatic {
            gyro_std_radps: gyro_std,
            accel_std_mps2: accel_std,
        });
    }

    let accel_norm = accel_mean.norm();
    if !(0.7 * gravity_mps2..=1.3 * gravity_mps2).contains(&accel_norm) {
        return Err(StaticInitError::BadAccelMagnitude {
            norm_mps2: accel_norm,
        });
    }

    // 静止時の加速度計は「上向き」の比力を測るので、それをワールド +z に回す。
    // 最小回転なのでヨーはほぼ 0（センサが水平なら従来どおりほぼ単位回転）。
    let rotation =
        UnitQuaternion::rotation_between(&accel_mean, &Vector3::z()).unwrap_or_else(|| {
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI)
        });

    let accel_scale_correction = if config.normalize_accel_magnitude {
        gravity_mps2 / accel_norm
    } else {
        1.0
    };

    Ok(StaticInitResult {
        rotation,
        gyro_bias: gyro_mean,
        accel_scale_correction,
        sample_count: samples.len(),
        gyro_std_radps: gyro_std,
        accel_std_mps2: accel_std,
        mean_accel_norm_mps2: accel_norm,
    })
}

// ---------------------------------------------------------------------------
// 伝播
// ---------------------------------------------------------------------------

/// `[t_from, t_to]` を IMU サンプルで区切り、各区間で `step(ω, a, dt)` を呼ぶ。
///
/// 区間の値には区間「終端」側のサンプルを使う（従来の `predict_pose_by_imu` と同じ）。
/// IMU が `t_to` まで届いていない場合は、最後の計測値を保持して残りを埋める。
fn for_each_imu_step<F>(
    imu_data: &[IMU],
    model: &ImuModel,
    t_from: f64,
    t_to: f64,
    mut step: F,
) -> usize
where
    F: FnMut(&Vector3<f64>, &Vector3<f64>, f64),
{
    // NaN も弾けるように「t_to > t_from でない」を明示的に判定する
    if t_to.is_nan() || t_from.is_nan() || t_to <= t_from || imu_data.is_empty() {
        return 0;
    }

    let start = imu_data.partition_point(|s| s.timestamp < t_from);
    let mut last_time = t_from;
    let mut last_measurement = start
        .checked_sub(1)
        .map(|index| model.measurement(&imu_data[index]));
    let mut steps = 0;

    for sample in &imu_data[start..] {
        if last_time >= t_to {
            break;
        }
        let step_end = sample.timestamp.min(t_to);
        let dt = step_end - last_time;
        let measurement = model.measurement(sample);
        last_measurement = Some(measurement);
        if dt <= 1e-9 {
            continue;
        }
        step(&measurement.0, &measurement.1, dt);
        steps += 1;
        last_time = step_end;
    }

    let remaining = t_to - last_time;
    if let Some((omega, accel)) = last_measurement.filter(|_| remaining > 1e-9) {
        step(&omega, &accel, remaining);
        steps += 1;
    }

    steps
}

/// 名目状態を 1ステップ進める（区間の始点の姿勢で加速度をワールドへ回す）。
#[inline]
fn integrate_nominal(
    state: &mut NavState,
    omega_meas: &Vector3<f64>,
    accel_meas: &Vector3<f64>,
    gravity: &Vector3<f64>,
    dt: f64,
) {
    let omega = omega_meas - state.bg;
    let accel_world = state.rot * (accel_meas - state.ba) + gravity;

    state.pos += state.vel * dt + 0.5 * accel_world * (dt * dt);
    state.vel += accel_world * dt;
    state.rot *= UnitQuaternion::from_scaled_axis(omega * dt);
    state.rot.renormalize();
}

/// 誤差状態の遷移行列 Φ ≈ I + F dt（左摂動）。
///
///   δθ' = δθ − R δb_g dt
///   δp' = δp + δv dt − ½[R a]× δθ dt² − ½ R δb_a dt²
///   δv' = δv − [R a]× δθ dt − R δb_a dt
///   δb_g' = δb_g,  δb_a' = δb_a
fn error_transition(state: &NavState, accel_meas: &Vector3<f64>, dt: f64) -> Matrix15 {
    let r = state.rot.to_rotation_matrix().into_inner();
    let accel_world_skew = skew(&(r * (accel_meas - state.ba)));
    let dt2_half = 0.5 * dt * dt;

    let mut phi = Matrix15::identity();
    phi.fixed_view_mut::<3, 3>(ROT, BG).copy_from(&(-r * dt));
    phi.fixed_view_mut::<3, 3>(POS, ROT)
        .copy_from(&(-accel_world_skew * dt2_half));
    phi.fixed_view_mut::<3, 3>(POS, VEL)
        .copy_from(&(Matrix3::identity() * dt));
    phi.fixed_view_mut::<3, 3>(POS, BA)
        .copy_from(&(-r * dt2_half));
    phi.fixed_view_mut::<3, 3>(VEL, ROT)
        .copy_from(&(-accel_world_skew * dt));
    phi.fixed_view_mut::<3, 3>(VEL, BA).copy_from(&(-r * dt));
    phi
}

fn add_process_noise(cov: &mut Matrix15, noise: &ImuNoise, dt: f64) {
    let q_rot = noise.gyro_noise_density.powi(2) * dt;
    let q_vel = noise.accel_noise_density.powi(2) * dt;
    let q_bg = noise.gyro_bias_random_walk.powi(2) * dt;
    let q_ba = noise.accel_bias_random_walk.powi(2) * dt;
    for i in 0..3 {
        cov[(ROT + i, ROT + i)] += q_rot;
        cov[(VEL + i, VEL + i)] += q_vel;
        cov[(BG + i, BG + i)] += q_bg;
        cov[(BA + i, BA + i)] += q_ba;
    }
}

#[inline]
fn symmetrize(m: &mut Matrix15) {
    *m = (*m + m.transpose()) * 0.5;
}

// ---------------------------------------------------------------------------
// フィルタ本体
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Eskf {
    pub state: NavState,
    pub cov: Matrix15,
    pub config: EskfConfig,
    pub imu_model: ImuModel,
    /// IMU 原点を LiDAR 座標で表した位置（レバーアーム）。
    pub imu_origin_in_lidar: Vector3<f64>,
}

/// ログ用の標準偏差（共分散の対角の平方根）。
#[derive(Debug, Clone, Copy, Default)]
pub struct CovarianceSummary {
    pub attitude_deg: [f64; 3],
    pub position_m: [f64; 3],
    pub velocity_mps: [f64; 3],
    pub gyro_bias_radps: [f64; 3],
    pub accel_bias_mps2: [f64; 3],
}

impl Eskf {
    pub fn new(
        initial_state: NavState,
        config: EskfConfig,
        imu_model: ImuModel,
        imu_origin_in_lidar: Vector3<f64>,
    ) -> Self {
        let s = &config.initial_std;
        let mut cov = Matrix15::zeros();
        for i in 0..3 {
            cov[(ROT + i, ROT + i)] = s.attitude_rad.powi(2);
            cov[(POS + i, POS + i)] = s.position_m.powi(2);
            cov[(VEL + i, VEL + i)] = s.velocity_mps.powi(2);
            cov[(BG + i, BG + i)] = s.gyro_bias_radps.powi(2);
            cov[(BA + i, BA + i)] = s.accel_bias_mps2.powi(2);
        }
        Self {
            state: initial_state,
            cov,
            config,
            imu_model,
            imu_origin_in_lidar,
        }
    }

    pub fn gravity(&self) -> Vector3<f64> {
        Vector3::new(0.0, 0.0, -self.config.gravity_mps2)
    }

    /// 名目状態と共分散を `t_from → t_to` へ伝播する。返り値は積分ステップ数。
    pub fn propagate(&mut self, imu_data: &[IMU], t_from: f64, t_to: f64) -> usize {
        let gravity = self.gravity();
        let noise = self.config.imu_noise;
        let model = self.imu_model;
        let state = &mut self.state;
        let cov = &mut self.cov;

        let steps = for_each_imu_step(imu_data, &model, t_from, t_to, |omega, accel, dt| {
            // Φ は区間始点の状態で評価する（名目状態の積分と同じ線形化点）。
            let phi = error_transition(state, accel, dt);
            integrate_nominal(state, omega, accel, &gravity, dt);
            *cov = phi * *cov * phi.transpose();
            add_process_noise(cov, &noise, dt);
        });
        symmetrize(&mut self.cov);
        steps
    }

    /// 共分散を触らずに名目状態だけを予測する（カメラ画像時刻の姿勢など）。
    pub fn predict_nominal(&self, imu_data: &[IMU], t_from: f64, t_to: f64) -> NavState {
        let gravity = self.gravity();
        let mut state = self.state;
        for_each_imu_step(
            imu_data,
            &self.imu_model,
            t_from,
            t_to,
            |omega, accel, dt| {
                integrate_nominal(&mut state, omega, accel, &gravity, dt);
            },
        );
        state
    }

    pub fn lidar_pose(&self) -> Matrix4<f64> {
        self.state.lidar_pose(&self.imu_origin_in_lidar)
    }

    pub fn lidar_pose_of(&self, state: &NavState) -> Matrix4<f64> {
        state.lidar_pose(&self.imu_origin_in_lidar)
    }

    /// 現在の（予測後の）状態と共分散を事前分布として、LiDAR 反復更新を開始する。
    /// 共分散が正定値でない場合は None。
    pub fn begin_lidar_update(&self) -> Option<LidarUpdate> {
        LidarUpdate::new(
            self.state,
            self.cov,
            self.config.lidar_point_sigma_m,
            self.imu_origin_in_lidar,
        )
    }

    /// 反復更新の結果を採用する。
    pub fn accept_update(&mut self, state: NavState, cov: Matrix15) {
        self.state = state;
        self.cov = cov;
        symmetrize(&mut self.cov);
    }

    pub fn covariance_summary(&self) -> CovarianceSummary {
        let std = |offset: usize, scale: f64| -> [f64; 3] {
            std::array::from_fn(|i| self.cov[(offset + i, offset + i)].max(0.0).sqrt() * scale)
        };
        CovarianceSummary {
            attitude_deg: std(ROT, 180.0 / std::f64::consts::PI),
            position_m: std(POS, 1.0),
            velocity_mps: std(VEL, 1.0),
            gyro_bias_radps: std(BG, 1.0),
            accel_bias_mps2: std(BA, 1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// LiDAR 反復更新
// ---------------------------------------------------------------------------

/// 1フレーム分の反復更新の文脈（事前分布 = IMU 予測）。
#[derive(Debug, Clone)]
pub struct LidarUpdate {
    prior_state: NavState,
    prior_cov: Matrix15,
    prior_info: Matrix15,
    point_info: f64,
    imu_origin_in_lidar: Vector3<f64>,
}

impl LidarUpdate {
    fn new(
        prior_state: NavState,
        mut prior_cov: Matrix15,
        point_sigma_m: f64,
        imu_origin_in_lidar: Vector3<f64>,
    ) -> Option<Self> {
        if !point_sigma_m.is_finite() || point_sigma_m <= 0.0 {
            return None;
        }
        symmetrize(&mut prior_cov);
        let prior_info = invert_spd(&prior_cov)?;
        Some(Self {
            prior_state,
            prior_cov,
            prior_info,
            point_info: 1.0 / (point_sigma_m * point_sigma_m),
            imu_origin_in_lidar,
        })
    }

    pub fn prior_state(&self) -> &NavState {
        &self.prior_state
    }

    /// ICP に渡す LiDAR 姿勢（f32）。
    pub fn lidar_rt_f32(&self, state: &NavState) -> (Matrix3<f32>, Vector3<f32>) {
        let (r, t) = state.lidar_rotation_translation(&self.imu_origin_in_lidar);
        (r.cast::<f32>(), t.cast::<f32>())
    }

    /// ICP の変数 [δθ; δt_lidar] と誤差状態 δx の関係 M（6×15）。
    ///
    /// t_lidar = p − R o より、R' = Exp(δθ) R のとき
    ///   δt_lidar = δp + [R o]× δθ
    /// レバーアーム o = 0 なら単なる埋め込みになる。
    fn lidar_pose_jacobian(&self, state: &NavState) -> Matrix6x15 {
        let mut m = Matrix6x15::zeros();
        m.fixed_view_mut::<3, 3>(0, ROT)
            .copy_from(&Matrix3::identity());
        let lever_world = state.rot * self.imu_origin_in_lidar;
        m.fixed_view_mut::<3, 3>(3, ROT)
            .copy_from(&skew(&lever_world));
        m.fixed_view_mut::<3, 3>(3, POS)
            .copy_from(&Matrix3::identity());
        m
    }

    /// ICP 系を σ² で重み付けして 15次元の情報行列とベクトルにする。
    fn measurement_information(
        &self,
        state: &NavState,
        system: &IcpLinearSystem,
    ) -> Option<(Matrix15, Vector15)> {
        let h6 = system.h.cast::<f64>();
        let b6 = system.b.cast::<f64>();
        if h6.iter().chain(b6.iter()).any(|v| !v.is_finite()) {
            return None;
        }
        let m = self.lidar_pose_jacobian(state);
        let mt = m.transpose();
        let mut h = mt * h6 * m * self.point_info;
        symmetrize(&mut h);
        let b = mt * b6 * self.point_info;
        Some((h, b))
    }

    /// 現在の反復点 `state` からの更新量 δx を求める。
    ///
    ///   (Mᵀ H M/σ² + P⁻¹) δx = Mᵀ b/σ² − P⁻¹ (state ⊟ x_pred)
    ///
    /// 事前項のヤコビアンは単位行列で近似している（反復中の補正が小さい前提）。
    pub fn solve_step(&self, state: &NavState, system: &IcpLinearSystem) -> Option<Vector15> {
        let (h, b) = self.measurement_information(state, system)?;
        let r = state.boxminus(&self.prior_state);
        let mut a = h + self.prior_info;
        symmetrize(&mut a);
        let rhs = b - self.prior_info * r;
        let dx = Cholesky::new(a)?.solve(&rhs);
        dx.iter().all(|v| v.is_finite()).then_some(dx)
    }

    /// 事前分布からの乖離 (x ⊟ x_pred)ᵀ P⁻¹ (x ⊟ x_pred)。
    pub fn prior_cost(&self, state: &NavState) -> f64 {
        let r = state.boxminus(&self.prior_state);
        (r.transpose() * self.prior_info * r)[(0, 0)]
    }

    /// 点のロバストコスト（ICP の Σρ(e)、単位 m²）と事前項を合わせた目的関数。
    pub fn total_cost(&self, state: &NavState, robust_point_cost: f32) -> f64 {
        robust_point_cost as f64 * self.point_info + self.prior_cost(state)
    }

    /// 収束点での事後共分散 (Mᵀ H M/σ² + P⁻¹)⁻¹。
    pub fn posterior_covariance(
        &self,
        state: &NavState,
        system: &IcpLinearSystem,
    ) -> Option<Matrix15> {
        let (h, _) = self.measurement_information(state, system)?;
        invert_spd(&(h + self.prior_info))
    }

    /// 予測のある軸の位置が信用できないと分かったとき（Z リカバリなど）、
    /// 事前分布のその軸の分散を足して広げる。
    pub fn inflate_position_variance(&mut self, axis: usize, std_m: f64) -> bool {
        if axis >= 3 || !std_m.is_finite() {
            return false;
        }
        let mut cov = self.prior_cov;
        cov[(POS + axis, POS + axis)] += std_m * std_m;
        match invert_spd(&cov) {
            Some(info) => {
                self.prior_cov = cov;
                self.prior_info = info;
                true
            }
            None => false,
        }
    }
}

fn invert_spd(m: &Matrix15) -> Option<Matrix15> {
    let mut sym = *m;
    symmetrize(&mut sym);
    let mut inv = Cholesky::new(sym)?.inverse();
    symmetrize(&mut inv);
    inv.iter().all(|v| v.is_finite()).then_some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::icp::{Matrix6f, Vector6f};

    const G: f64 = STANDARD_GRAVITY_MPS2;

    fn test_config() -> EskfConfig {
        EskfConfig {
            imu_noise: ImuNoise {
                gyro_noise_density: 1.0e-4,
                accel_noise_density: 1.0e-3,
                gyro_bias_random_walk: 1.0e-6,
                accel_bias_random_walk: 1.0e-5,
            },
            initial_std: InitialStd {
                attitude_rad: 1.0e-3,
                position_m: 1.0e-3,
                velocity_mps: 1.0e-2,
                gyro_bias_radps: 1.0e-3,
                accel_bias_mps2: 0.2,
            },
            gravity_mps2: G,
            lidar_point_sigma_m: 0.05,
        }
    }

    fn sample(t: f64, omega: [f64; 3], accel_g: [f64; 3]) -> IMU {
        IMU {
            timestamp: t,
            angular_velocity: omega.map(|v| v as f32),
            linear_acceleration: accel_g.map(|v| v as f32),
        }
    }

    fn constant_imu(rate_hz: f64, duration_s: f64, omega: [f64; 3], accel_g: [f64; 3]) -> Vec<IMU> {
        let n = (rate_hz * duration_s).round() as usize;
        (1..=n)
            .map(|k| sample(k as f64 / rate_hz, omega, accel_g))
            .collect()
    }

    fn nontrivial_state() -> NavState {
        NavState {
            rot: UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3),
            pos: Vector3::new(1.0, -2.0, 0.5),
            vel: Vector3::new(0.3, 0.1, -0.2),
            bg: Vector3::new(0.01, -0.02, 0.005),
            ba: Vector3::new(0.05, 0.02, -0.03),
        }
    }

    #[test]
    fn boxplus_boxminus_roundtrip() {
        let x = nontrivial_state();
        let dx =
            Vector15::from_fn(|i, _| 0.01 * (i as f64 + 1.0) * if i % 2 == 0 { 1.0 } else { -1.0 });
        let y = x.boxplus(&dx);
        assert!((y.boxminus(&x) - dx).norm() < 1e-12);
    }

    #[test]
    fn stationary_level_imu_stays_at_rest_and_covariance_grows() {
        let model = ImuModel::new(UnitQuaternion::identity(), G);
        let mut eskf = Eskf::new(NavState::default(), test_config(), model, Vector3::zeros());
        let p0_vel = eskf.cov[(VEL, VEL)];
        let imu = constant_imu(200.0, 1.0, [0.0; 3], [0.0, 0.0, 1.0]);

        let steps = eskf.propagate(&imu, 0.0, 1.0);

        assert_eq!(steps, 200);
        assert!(eskf.state.pos.norm() < 1e-9);
        assert!(eskf.state.vel.norm() < 1e-9);
        assert!(eskf.cov[(VEL, VEL)] > p0_vel);
        // 加速度バイアスの不確かさが位置・速度へ伝わり、相関が生まれている
        assert!(eskf.cov[(POS, BA)].abs() > 0.0);
    }

    #[test]
    fn constant_acceleration_matches_kinematics() {
        let model = ImuModel::new(UnitQuaternion::identity(), G);
        let mut eskf = Eskf::new(NavState::default(), test_config(), model, Vector3::zeros());
        let imu = constant_imu(200.0, 1.0, [0.0; 3], [1.0 / G, 0.0, 1.0]);

        eskf.propagate(&imu, 0.0, 1.0);

        assert!((eskf.state.pos.x - 0.5).abs() < 1e-6);
        assert!((eskf.state.vel.x - 1.0).abs() < 1e-6);
        assert!(eskf.state.pos.z.abs() < 1e-6);
    }

    #[test]
    fn propagation_holds_last_measurement_when_imu_ends_early() {
        let model = ImuModel::new(UnitQuaternion::identity(), G);
        let mut eskf = Eskf::new(NavState::default(), test_config(), model, Vector3::zeros());
        let imu = constant_imu(100.0, 0.5, [0.0; 3], [1.0 / G, 0.0, 1.0]);

        eskf.propagate(&imu, 0.0, 1.0);

        assert!((eskf.state.vel.x - 1.0).abs() < 1e-6);
    }

    /// Φ を数値微分（x ⊞ ε eᵢ を 1ステップ積分して ⊟）と比較する。
    #[test]
    fn error_transition_matches_numerical_differentiation() {
        let x0 = nontrivial_state();
        let omega = Vector3::new(0.4, -0.3, 0.8);
        let accel = Vector3::new(1.2, -0.7, 9.6);
        let gravity = Vector3::new(0.0, 0.0, -G);
        let dt = 0.005;
        let eps = 1e-6;

        let phi = error_transition(&x0, &accel, dt);
        let mut x1 = x0;
        integrate_nominal(&mut x1, &omega, &accel, &gravity, dt);

        for col in 0..ERROR_STATE_DIM {
            let mut dx = Vector15::zeros();
            dx[col] = eps;
            let mut x1p = x0.boxplus(&dx);
            integrate_nominal(&mut x1p, &omega, &accel, &gravity, dt);
            let numeric = x1p.boxminus(&x1) / eps;
            let analytic = phi.column(col);
            let err = (numeric - analytic).norm();
            // 1次近似 Φ = I + F dt なので O(dt²) の差は許容する
            assert!(
                err < 5e-5,
                "column {col}: err={err:.3e}\nnumeric={numeric}\nanalytic={analytic}"
            );
        }
    }

    /// M（[δθ; δt_lidar] = M δx）をレバーアームありで数値微分と比較する。
    #[test]
    fn lidar_pose_jacobian_matches_numerical_differentiation_with_lever_arm() {
        let lever = Vector3::new(-0.04165, -0.02326, 0.02840);
        let x = nontrivial_state();
        let update = LidarUpdate::new(x, Matrix15::identity(), 0.05, lever).unwrap();
        let m = update.lidar_pose_jacobian(&x);
        let (r0, t0) = x.lidar_rotation_translation(&lever);
        let eps = 1e-7;

        for col in 0..ERROR_STATE_DIM {
            let mut dx = Vector15::zeros();
            dx[col] = eps;
            let (r1, t1) = x.boxplus(&dx).lidar_rotation_translation(&lever);
            let d_rot = UnitQuaternion::from_matrix(&(r1 * r0.transpose())).scaled_axis();
            let numeric = SVector::<f64, 6>::new(
                d_rot.x,
                d_rot.y,
                d_rot.z,
                t1.x - t0.x,
                t1.y - t0.y,
                t1.z - t0.z,
            ) / eps;
            let err = (numeric - m.column(col)).norm();
            assert!(err < 1e-6, "column {col}: err={err:.3e}");
        }
    }

    /// 説明で使った 1次元の例: IMU が 1.0 m/s² と言うが実際は 0.9（バイアス 0.1）。
    /// LiDAR が「位置は 0.45 m」と言うと、速度と加速度バイアスも補正される。
    #[test]
    fn position_residual_corrects_velocity_and_accel_bias() {
        let model = ImuModel::new(UnitQuaternion::identity(), G);
        let mut eskf = Eskf::new(NavState::default(), test_config(), model, Vector3::zeros());
        let imu = constant_imu(200.0, 1.0, [0.0; 3], [1.0 / G, 0.0, 1.0]);
        eskf.propagate(&imu, 0.0, 1.0);
        assert!((eskf.state.pos.x - 0.5).abs() < 1e-6);

        // 原点の点が x = 0.45 の平面（法線 +x）に乗っている観測を N 点ぶん作る。
        // 残差 e = n·t + d = 0.5 − 0.45 = 0.05、J = [0,0,0, 1,0,0]
        let n_points = 1000.0f32;
        let residual = (eskf.state.pos.x - 0.45) as f32;
        let mut h = Matrix6f::zeros();
        h[(3, 3)] = n_points;
        let mut b = Vector6f::zeros();
        b[3] = -n_points * residual;
        let system = IcpLinearSystem {
            h,
            b,
            cost: n_points * residual * residual,
            used_count: n_points as usize,
        };

        let update = eskf.begin_lidar_update().unwrap();
        let x_pred = eskf.state;
        let dx = update.solve_step(&x_pred, &system).unwrap();
        let x_new = x_pred.boxplus(&dx);

        assert!((x_new.pos.x - 0.45).abs() < 2e-3, "pos.x={}", x_new.pos.x);
        assert!((x_new.vel.x - 0.9).abs() < 0.02, "vel.x={}", x_new.vel.x);
        assert!((x_new.ba.x - 0.1).abs() < 0.02, "ba.x={}", x_new.ba.x);
        // 観測のない y は予測のまま。z はピッチ誤差（重力との結合）経由でごくわずかに動く
        assert!(x_new.pos.y.abs() < 1e-9, "pos.y={}", x_new.pos.y);
        assert!(x_new.pos.z.abs() < 1e-4, "pos.z={}", x_new.pos.z);

        // 目的関数が下がり、x 位置の分散が縮む
        let cost_new = update.total_cost(&x_new, 0.0);
        let cost_pred = update.total_cost(&x_pred, system.cost);
        assert!(cost_new < cost_pred);
        let cov_post = update.posterior_covariance(&x_new, &system).unwrap();
        assert!(cov_post[(POS, POS)] < eskf.cov[(POS, POS)]);
        let prior_y = eskf.cov[(POS + 1, POS + 1)];
        assert!((cov_post[(POS + 1, POS + 1)] - prior_y).abs() < 1e-9 * prior_y);
    }

    #[test]
    fn iterated_solve_converges_to_fixed_point() {
        let model = ImuModel::new(UnitQuaternion::identity(), G);
        let mut eskf = Eskf::new(NavState::default(), test_config(), model, Vector3::zeros());
        let imu = constant_imu(200.0, 1.0, [0.0; 3], [1.0 / G, 0.0, 1.0]);
        eskf.propagate(&imu, 0.0, 1.0);
        let update = eskf.begin_lidar_update().unwrap();

        let build = |x: &NavState| {
            let residual = (x.pos.x - 0.45) as f32;
            let mut h = Matrix6f::zeros();
            h[(3, 3)] = 1000.0;
            let mut b = Vector6f::zeros();
            b[3] = -1000.0 * residual;
            IcpLinearSystem {
                h,
                b,
                cost: 1000.0 * residual * residual,
                used_count: 1000,
            }
        };

        let mut x = *update.prior_state();
        let first = update.solve_step(&x, &build(&x)).unwrap();
        x = x.boxplus(&first);
        // 線形な観測なので 2回目の反復の更新量はほぼ 0（不動点に収束済み）
        let second = update.solve_step(&x, &build(&x)).unwrap();
        // ICP 系は f32 で組むので、その丸め誤差程度は残る
        assert!(
            second.norm() < 1e-6 * first.norm(),
            "second={}",
            second.norm()
        );
    }

    /// 実際の ICP 系（`build_robust_point_to_plane_system`）を使った反復更新が、
    /// レバーアームありでも真の LiDAR 姿勢へ収束することを確認する（符号規約の結合テスト）。
    #[test]
    fn iterated_update_with_real_icp_system_converges_with_lever_arm() {
        use crate::find_nearest_points::PointCorrespondence;
        use crate::icp::build_robust_point_to_plane_system;
        use nalgebra::Point3;

        let lever = Vector3::new(-0.04165, -0.02326, 0.02840);
        let truth = nontrivial_state();
        let (r_true, t_true) = truth.lidar_rotation_translation(&lever);

        // ワールドの平面 n·x + d = 0 を 4枚（3方向 + 斜め）用意し、真の姿勢で LiDAR 座標へ
        let planes = [
            (Vector3::<f64>::new(1.0, 0.0, 0.0), -4.0f64),
            (Vector3::new(0.0, 1.0, 0.0), 5.0),
            (Vector3::new(0.0, 0.0, 1.0), 1.0),
            (Vector3::new(1.0, 1.0, 1.0).normalize(), -2.0),
        ];
        let mut correspondences = Vec::new();
        for (normal, d) in planes {
            let (u, v) = {
                let a = if normal.x.abs() < 0.9 {
                    Vector3::x()
                } else {
                    Vector3::y()
                };
                let u = normal.cross(&a).normalize();
                (u, normal.cross(&u))
            };
            let origin = -d * normal;
            for i in -6..=6 {
                for j in -6..=6 {
                    let world = origin + u * (i as f64 * 0.5) + v * (j as f64 * 0.5);
                    let local = r_true.transpose() * (world - t_true);
                    correspondences.push(PointCorrespondence {
                        src_point: Point3::from(local.cast::<f32>()),
                        plane_normal: normal.cast::<f32>(),
                        plane_d: d as f32,
                    });
                }
            }
        }

        let mut config = test_config();
        config.initial_std.attitude_rad = 0.1;
        config.initial_std.position_m = 0.5;
        let model = ImuModel::new(UnitQuaternion::identity(), G);
        let mut prior = truth;
        prior.rot = UnitQuaternion::from_scaled_axis(Vector3::new(0.02, -0.015, 0.03)) * truth.rot;
        prior.pos += Vector3::new(0.12, -0.08, 0.05);
        let eskf = Eskf::new(prior, config, model, lever);
        let update = eskf.begin_lidar_update().unwrap();

        let mut x = prior;
        for _ in 0..10 {
            let (r, t) = update.lidar_rt_f32(&x);
            let system =
                build_robust_point_to_plane_system(&correspondences, &r, &t, f32::INFINITY);
            let dx = update.solve_step(&x, &system).unwrap();
            x = x.boxplus(&dx);
            if dx.norm() < 1e-9 {
                break;
            }
        }

        let (r_est, t_est) = x.lidar_rotation_translation(&lever);
        let rot_err = UnitQuaternion::from_matrix(&(r_est * r_true.transpose())).angle();
        assert!(rot_err < 1e-4, "rotation error {rot_err:.3e} rad");
        assert!(
            (t_est - t_true).norm() < 1e-3,
            "translation error {:.3e} m",
            (t_est - t_true).norm()
        );
    }

    /// 閉ループのシミュレーション: バイアス入りの IMU と平面への点群観測で 30 秒走らせ、
    /// 初期値 0 のバイアスが真値へ収束し、姿勢も追従することを確認する。
    #[test]
    fn simulated_lio_loop_estimates_biases() {
        use crate::find_nearest_points::PointCorrespondence;
        use crate::icp::build_robust_point_to_plane_system;
        use nalgebra::Point3;

        let imu_rate = 200.0;
        let lidar_rate = 10.0;
        let duration_s = 30.0;
        let lever = Vector3::new(-0.04165, -0.02326, 0.02840);
        let bg_true = Vector3::new(0.01, -0.008, 0.005);
        let ba_true = Vector3::new(0.08, -0.05, 0.06);
        let gravity = Vector3::new(0.0, 0.0, -G);

        // 真の運動（ボディ角速度・ワールド加速度）から、バイアス入り IMU 列と真の状態列を作る。
        // 真値はフィルタと同じ離散化で積分しておく（離散化誤差を検証から除くため）。
        let n_imu = (imu_rate * duration_s) as usize;
        let mut truth = NavState {
            bg: bg_true,
            ba: ba_true,
            ..NavState::default()
        };
        let mut truth_at = vec![(0.0, truth)];
        let mut imu = Vec::with_capacity(n_imu);
        let model = ImuModel::new(UnitQuaternion::identity(), G);
        for k in 1..=n_imu {
            let t = k as f64 / imu_rate;
            let omega_true = Vector3::new(0.3 * (0.5 * t).sin(), -0.2 * (0.3 * t).cos(), 0.4);
            let accel_world =
                Vector3::new(0.5 * t.sin(), 0.3 * (0.7 * t).cos(), 0.1 * (1.3 * t).sin());
            let specific_force = truth.rot.inverse() * (accel_world - gravity);
            let omega_meas = omega_true + bg_true;
            let accel_meas = specific_force + ba_true;
            let s = sample(t, omega_meas.into(), (accel_meas / G).into());
            let (w, a) = model.measurement(&s);
            integrate_nominal(&mut truth, &w, &a, &gravity, 1.0 / imu_rate);
            imu.push(s);
            truth_at.push((t, truth));
        }

        let planes = [
            (Vector3::<f64>::new(1.0, 0.0, 0.0), -8.0f64),
            (Vector3::new(-1.0, 0.0, 0.0), -8.0),
            (Vector3::new(0.0, 1.0, 0.0), -8.0),
            (Vector3::new(0.0, -1.0, 0.0), -8.0),
            (Vector3::new(0.0, 0.0, 1.0), 2.0),
            (Vector3::new(0.0, 0.0, -1.0), 6.0),
        ];
        let observe = |state: &NavState| -> Vec<PointCorrespondence> {
            let (r, t) = state.lidar_rotation_translation(&lever);
            let mut out = Vec::new();
            for (normal, d) in planes {
                let a = if normal.x.abs() < 0.9 {
                    Vector3::x()
                } else {
                    Vector3::y()
                };
                let u = normal.cross(&a).normalize();
                let v = normal.cross(&u);
                // 平面上で LiDAR 位置の足元付近をサンプリング
                let foot = t - normal * (normal.dot(&t) + d);
                for i in -3..=3 {
                    for j in -3..=3 {
                        let world = foot + u * (i as f64 * 1.0) + v * (j as f64 * 1.0);
                        let local = r.transpose() * (world - t);
                        out.push(PointCorrespondence {
                            src_point: Point3::from(local.cast::<f32>()),
                            plane_normal: normal.cast::<f32>(),
                            plane_d: d as f32,
                        });
                    }
                }
            }
            out
        };

        let mut config = test_config();
        config.initial_std.gyro_bias_radps = 0.02;
        config.initial_std.accel_bias_mps2 = 0.1;
        config.lidar_point_sigma_m = 0.02;
        let initial = NavState {
            bg: Vector3::zeros(),
            ba: Vector3::zeros(),
            ..truth_at[0].1
        };
        let mut eskf = Eskf::new(initial, config, model, lever);

        let frame_period = 1.0 / lidar_rate;
        let n_frames = (duration_s * lidar_rate) as usize;
        let imu_per_frame = (imu_rate * frame_period).round() as usize;
        for f in 1..=n_frames {
            let (t_prev, t_now) = ((f - 1) as f64 * frame_period, f as f64 * frame_period);
            eskf.propagate(&imu, t_prev, t_now);
            let truth_now = truth_at[f * imu_per_frame].1;
            let correspondences = observe(&truth_now);
            let update = eskf.begin_lidar_update().unwrap();
            let mut x = eskf.state;
            for _ in 0..4 {
                let (r, t) = update.lidar_rt_f32(&x);
                let system = build_robust_point_to_plane_system(&correspondences, &r, &t, 0.08);
                let dx = update.solve_step(&x, &system).unwrap();
                x = x.boxplus(&dx);
            }
            let (r, t) = update.lidar_rt_f32(&x);
            let final_system = build_robust_point_to_plane_system(&correspondences, &r, &t, 0.08);
            let cov = update.posterior_covariance(&x, &final_system).unwrap();
            eskf.accept_update(x, cov);
        }

        let truth_end = truth_at[n_frames * imu_per_frame].1;
        let bg_err = (eskf.state.bg - bg_true).norm();
        let ba_err = (eskf.state.ba - ba_true).norm();
        let pos_err = (eskf.state.pos - truth_end.pos).norm();
        let vel_err = (eskf.state.vel - truth_end.vel).norm();
        let rot_err = (eskf.state.rot * truth_end.rot.inverse()).angle();
        assert!(bg_err < 1e-3, "bg={:?} err={bg_err:.2e}", eskf.state.bg);
        assert!(ba_err < 1e-2, "ba={:?} err={ba_err:.2e}", eskf.state.ba);
        assert!(pos_err < 5e-3, "pos err={pos_err:.2e}");
        assert!(vel_err < 2e-2, "vel err={vel_err:.2e}");
        assert!(rot_err < 1e-3, "rot err={rot_err:.2e}");
    }

    #[test]
    fn static_init_recovers_tilt_and_gyro_bias() {
        let tilt = UnitQuaternion::from_euler_angles(5f64.to_radians(), -3f64.to_radians(), 0.0);
        // ボディ座標での上向き比力 = R_trueᵀ (0,0,1)（g 単位）
        let up_body = tilt.inverse() * Vector3::z();
        let bias = [0.01, -0.02, 0.005];
        let imu: Vec<IMU> = (0..200)
            .map(|k| sample(k as f64 * 0.005, bias, [up_body.x, up_body.y, up_body.z]))
            .collect();
        let model = ImuModel::new(UnitQuaternion::identity(), G);
        let config = StaticInitConfig {
            window_s: 1.0,
            min_samples: 50,
            max_gyro_std_radps: 0.02,
            max_accel_std_mps2: 0.1,
            normalize_accel_magnitude: true,
        };

        let init = estimate_static_init(&imu, &model, 0.0, &config, G).unwrap();

        assert!((init.rotation * up_body - Vector3::z()).norm() < 1e-6);
        assert!((init.gyro_bias - Vector3::from(bias)).norm() < 1e-6);
        assert!((init.accel_scale_correction - 1.0).abs() < 1e-6);
    }

    #[test]
    fn static_init_rejects_moving_sensor() {
        let imu: Vec<IMU> = (0..200)
            .map(|k| {
                let w = if k % 2 == 0 { 0.5 } else { -0.5 };
                sample(k as f64 * 0.005, [0.0, 0.0, w], [0.0, 0.0, 1.0])
            })
            .collect();
        let model = ImuModel::new(UnitQuaternion::identity(), G);
        let config = StaticInitConfig {
            window_s: 1.0,
            min_samples: 50,
            max_gyro_std_radps: 0.02,
            max_accel_std_mps2: 0.1,
            normalize_accel_magnitude: true,
        };

        assert!(matches!(
            estimate_static_init(&imu, &model, 0.0, &config, G),
            Err(StaticInitError::NotStatic { .. })
        ));
    }

    #[test]
    fn inflating_position_variance_relaxes_prior() {
        let model = ImuModel::new(UnitQuaternion::identity(), G);
        let eskf = Eskf::new(NavState::default(), test_config(), model, Vector3::zeros());
        let mut update = eskf.begin_lidar_update().unwrap();
        let mut shifted = *update.prior_state();
        shifted.pos.z += 0.3;
        let before = update.prior_cost(&shifted);

        assert!(update.inflate_position_variance(2, 0.3));

        assert!(update.prior_cost(&shifted) < before * 1e-3);
    }
}
