//! Point-to-Plane ICP の成功後にだけ実行する、小さな混合誤差による微調整。
//! 座標系: source はセンサ座標、target と plane はワールド座標。
//! Point-to-Point の相手はローカルマップのセル平均。物理的な同一点ではない。

use nalgebra::{Matrix3, Point3, SMatrix, Vector3};
use rayon::prelude::*;

use crate::find_nearest_points::{PointCorrespondence, find_k_nearest_target_voxels};
use crate::icp::{
    IcpLinearSystem, apply_delta, build_robust_point_to_plane_system, compute_rmse,
    compute_robust_cost, solve_icp_delta_observable,
};
use crate::voxel_map::VoxelMap;

#[derive(Debug, Clone, Copy)]
pub struct HybridRefineConfig {
    pub iterations: usize,
    /// mean(plane Huber loss) + point_weight * mean(point Huber loss)
    pub point_weight: f32,
    pub point_huber_delta_m: f32,
    pub max_point_distance_m: f32,
    pub min_point_matches: usize,
    /// 前段の Point-to-Plane 結果から許す「後段全体」の追加補正量。
    pub max_translation_m: f32,
    pub max_rotation_deg: f32,
    /// 前段で固定した平面対応に対する RMSE の増加許容幅。
    pub max_plane_rmse_increase_m: f32,
    pub damping: f32,
    pub relative_eigenvalue_threshold: f32,
}

impl Default for HybridRefineConfig {
    fn default() -> Self {
        Self {
            iterations: 3,
            point_weight: 0.10,
            point_huber_delta_m: 0.10,
            max_point_distance_m: 0.30,
            min_point_matches: 100,
            max_translation_m: 0.03,
            max_rotation_deg: 0.30,
            max_plane_rmse_increase_m: 0.003,
            damping: 1e-6,
            relative_eigenvalue_threshold: 0.02,
        }
    }
}

#[derive(Debug)]
pub struct HybridRefineResult {
    pub rotation: Matrix3<f32>,
    pub translation: Vector3<f32>,
    pub accepted_steps: usize,
    pub point_matches: usize,
    pub translation_correction_m: f32,
    pub rotation_correction_deg: f32,
    /// どちらも前段終了時に固定した同一対応点で比較した値。
    pub point_rmse_before_m: f32,
    pub point_rmse_after_m: f32,
    pub plane_rmse_before_m: f32,
    pub plane_rmse_after_m: f32,
}

#[derive(Clone, Copy)]
struct PointMatch {
    source: Point3<f32>,
    target: Point3<f32>,
}

/// 平面判定を通った点だけでなく、source_map 内の全代表点を候補にする。
fn find_point_matches(
    source_map: &VoxelMap,
    target_map: &VoxelMap,
    voxel_size: f32,
    rotation: &Matrix3<f32>,
    translation: &Vector3<f32>,
    max_distance_m: f32,
) -> Vec<PointMatch> {
    let search_range = (max_distance_m / voxel_size).ceil() as i32;
    let max_distance_sq = max_distance_m * max_distance_m;

    source_map
        .par_iter()
        .filter_map(|(_, cell)| {
            let source = cell.point.0;
            let query = Point3::from(rotation * source.coords + translation);
            if !query.coords.iter().all(|v| v.is_finite()) {
                return None;
            }
            let closest = find_k_nearest_target_voxels::<1>(
                &query,
                target_map,
                voxel_size,
                search_range,
                max_distance_sq,
            )?;
            Some(PointMatch {
                source,
                target: closest[0].0.mean,
            })
        })
        .collect()
}

/// ||e|| に Huber を掛ける。各 x/y/z に別々の Huber は掛けない。
fn huber(norm: f32, delta: f32) -> (f32, f32) {
    if norm <= delta {
        (1.0, norm * norm)
    } else {
        (delta / norm, 2.0 * delta * norm - delta * delta)
    }
}

fn build_point_system(
    matches: &[PointMatch],
    rotation: &Matrix3<f32>,
    translation: &Vector3<f32>,
    huber_delta_m: f32,
) -> IcpLinearSystem {
    let mut system = IcpLinearSystem::default();
    for pair in matches {
        let rp = rotation * pair.source.coords;
        let residual = rp + translation - pair.target.coords;
        let (weight, loss) = huber(residual.norm(), huber_delta_m);

        // apply_delta(): R' = Exp(delta_theta) R, t' = t + delta_t に対応。
        // J = [-skew(Rp), I] (3行6列)。並進 t は回転の腕に入れない。
        let jacobian = SMatrix::<f32, 3, 6>::from_row_slice(&[
            0.0, rp.z, -rp.y, 1.0, 0.0, 0.0, -rp.z, 0.0, rp.x, 0.0, 1.0, 0.0, rp.y, -rp.x, 0.0,
            0.0, 0.0, 1.0,
        ]);
        system.h += weight * jacobian.transpose() * jacobian;
        system.b -= weight * jacobian.transpose() * residual;
        system.cost += loss;
        system.used_count += 1;
    }
    system
}

fn point_cost(
    matches: &[PointMatch],
    rotation: &Matrix3<f32>,
    translation: &Vector3<f32>,
    huber_delta_m: f32,
) -> f32 {
    matches
        .iter()
        .map(|pair| {
            let error = rotation * pair.source.coords + translation - pair.target.coords;
            huber(error.norm(), huber_delta_m).1
        })
        .sum::<f32>()
        / matches.len() as f32
}

fn point_rmse(matches: &[PointMatch], rotation: &Matrix3<f32>, translation: &Vector3<f32>) -> f32 {
    (matches
        .iter()
        .map(|pair| {
            (rotation * pair.source.coords + translation - pair.target.coords).norm_squared()
        })
        .sum::<f32>()
        / matches.len() as f32)
        .sqrt()
}

fn mixed_cost(
    planes: &[PointCorrespondence],
    points: &[PointMatch],
    rotation: &Matrix3<f32>,
    translation: &Vector3<f32>,
    plane_huber_delta_m: f32,
    config: &HybridRefineConfig,
) -> f32 {
    compute_robust_cost(planes, rotation, translation, plane_huber_delta_m) / planes.len() as f32
        + config.point_weight
            * point_cost(points, rotation, translation, config.point_huber_delta_m)
}

fn build_mixed_system(
    planes: &[PointCorrespondence],
    points: &[PointMatch],
    rotation: &Matrix3<f32>,
    translation: &Vector3<f32>,
    plane_huber_delta_m: f32,
    config: &HybridRefineConfig,
) -> IcpLinearSystem {
    let mut plane_system =
        build_robust_point_to_plane_system(planes, rotation, translation, plane_huber_delta_m);
    let point_system =
        build_point_system(points, rotation, translation, config.point_huber_delta_m);

    // 対応数が増えただけで Point-to-Point が支配しないよう、各々の平均誤差を使う。
    let plane_scale = 1.0 / planes.len() as f32;
    let point_scale = config.point_weight / points.len() as f32;
    plane_system.h = plane_system.h * plane_scale + point_system.h * point_scale;
    plane_system.b = plane_system.b * plane_scale + point_system.b * point_scale;
    plane_system.cost = plane_system.cost * plane_scale + point_system.cost * point_scale;
    plane_system.used_count += point_system.used_count;
    plane_system
}

fn correction(
    base_r: &Matrix3<f32>,
    base_t: &Vector3<f32>,
    rotation: &Matrix3<f32>,
    translation: &Vector3<f32>,
) -> (f32, f32) {
    let relative = rotation * base_r.transpose();
    let angle = ((relative.trace() - 1.0) * 0.5)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    ((translation - base_t).norm(), angle)
}

/// 成功してもまだ main 側で「新しい平面対応」と「IMUからの補正量」を検証する。
/// 引数の姿勢やマップを変更しないので、Err や main 側の不採用では前段結果が残る。
pub fn refine_hybrid_pose(
    source_map: &VoxelMap,
    target_map: &VoxelMap,
    target_voxel_size: f32,
    planes: &[PointCorrespondence],
    base_r: &Matrix3<f32>,
    base_t: &Vector3<f32>,
    plane_huber_delta_m: f32,
    config: &HybridRefineConfig,
) -> Result<HybridRefineResult, &'static str> {
    let positive_values = [
        target_voxel_size,
        plane_huber_delta_m,
        config.point_weight,
        config.point_huber_delta_m,
        config.max_point_distance_m,
        config.max_translation_m,
        config.max_rotation_deg,
    ];
    if positive_values.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || !config.max_plane_rmse_increase_m.is_finite()
        || config.max_plane_rmse_increase_m < 0.0
        || !config.damping.is_finite()
        || config.damping < 0.0
        || !config.relative_eigenvalue_threshold.is_finite()
        || !(0.0..1.0).contains(&config.relative_eigenvalue_threshold)
        || config.iterations == 0
        || config.min_point_matches < 6
        || planes.len() < 6
        || !base_r.iter().chain(base_t.iter()).all(|v| v.is_finite())
    {
        return Err("invalid hybrid configuration or baseline");
    }

    let anchor_points = find_point_matches(
        source_map,
        target_map,
        target_voxel_size,
        base_r,
        base_t,
        config.max_point_distance_m,
    );
    if anchor_points.len() < config.min_point_matches {
        return Err("not enough point-to-point matches");
    }
    let base_plane_rmse = compute_rmse(planes, base_r, base_t);
    let base_point_cost = point_cost(&anchor_points, base_r, base_t, config.point_huber_delta_m);
    let base_cost = mixed_cost(
        planes,
        &anchor_points,
        base_r,
        base_t,
        plane_huber_delta_m,
        config,
    );
    if !base_cost.is_finite() || !base_plane_rmse.is_finite() {
        return Err("non-finite baseline error");
    }

    let mut rotation = *base_r;
    let mut translation = *base_t;
    let mut accepted_steps = 0;
    for _ in 0..config.iterations {
        // 点対応は姿勢が動くたびに探し直す。平面は前段で検証済みの対応を保持する。
        let points = find_point_matches(
            source_map,
            target_map,
            target_voxel_size,
            &rotation,
            &translation,
            config.max_point_distance_m,
        );
        if points.len() < config.min_point_matches {
            break;
        }
        let system = build_mixed_system(
            planes,
            &points,
            &rotation,
            &translation,
            plane_huber_delta_m,
            config,
        );
        let Some(solved) = solve_icp_delta_observable(
            &system,
            config.damping,
            config.relative_eigenvalue_threshold,
        ) else {
            break;
        };
        let old_cost = mixed_cost(
            planes,
            &points,
            &rotation,
            &translation,
            plane_huber_delta_m,
            config,
        );

        let mut accepted = false;
        // 1回の更新が大きい場合は短くして試す。上限値で無理にクリップしない。
        for scale in [1.0_f32, 0.5, 0.25, 0.125, 0.0625] {
            let delta = solved.delta * scale;
            let (next_r, next_t) = apply_delta(&rotation, &translation, &delta);
            let (distance, angle) = correction(base_r, base_t, &next_r, &next_t);
            if !distance.is_finite()
                || !angle.is_finite()
                || distance > config.max_translation_m
                || angle > config.max_rotation_deg
            {
                continue;
            }

            // 同じ対応で比較する。対応点が消えただけの「見かけの改善」を防ぐ。
            let next_cost = mixed_cost(
                planes,
                &points,
                &next_r,
                &next_t,
                plane_huber_delta_m,
                config,
            );
            let anchor_cost = mixed_cost(
                planes,
                &anchor_points,
                &next_r,
                &next_t,
                plane_huber_delta_m,
                config,
            );
            let anchor_point_cost =
                point_cost(&anchor_points, &next_r, &next_t, config.point_huber_delta_m);
            let plane_rmse = compute_rmse(planes, &next_r, &next_t);
            if next_cost.is_finite()
                && next_cost < old_cost
                && anchor_cost.is_finite()
                && anchor_cost <= base_cost
                && anchor_point_cost.is_finite()
                && anchor_point_cost <= base_point_cost
                && plane_rmse.is_finite()
                && plane_rmse <= base_plane_rmse + config.max_plane_rmse_increase_m
            {
                rotation = next_r;
                translation = next_t;
                accepted_steps += 1;
                accepted = true;
                break;
            }
        }
        if !accepted {
            break;
        }
    }
    if accepted_steps == 0 {
        return Err("no acceptable improving step");
    }
    let (distance, angle) = correction(base_r, base_t, &rotation, &translation);
    if distance < 1e-6 && angle < 1e-4 {
        return Err("correction is negligible");
    }
    Ok(HybridRefineResult {
        rotation,
        translation,
        accepted_steps,
        point_matches: anchor_points.len(),
        translation_correction_m: distance,
        rotation_correction_deg: angle,
        point_rmse_before_m: point_rmse(&anchor_points, base_r, base_t),
        point_rmse_after_m: point_rmse(&anchor_points, &rotation, &translation),
        plane_rmse_before_m: base_plane_rmse,
        plane_rmse_after_m: compute_rmse(planes, &rotation, &translation),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::icp::Vector6f;

    fn floor_scene(shift_x: f32) -> (VoxelMap, VoxelMap, Vec<PointCorrespondence>) {
        use crate::voxel_map::{VoxelCell, voxel_key};
        let mut source = VoxelMap::default();
        let mut target = VoxelMap::default();
        let mut planes = Vec::new();
        for ix in 0..10 {
            for iy in 0..10 {
                let point = Point3::new((ix as f32 - 4.5) * 0.5, (iy as f32 - 4.5) * 0.5, 1.0);
                let key = voxel_key(&point, 0.25);
                source.insert(key, VoxelCell::from_key(&key, 0.25, point, 0));
                let shifted = point + Vector3::new(shift_x, 0.0, 0.0);
                let target_key = voxel_key(&shifted, 0.25);
                target.insert(
                    target_key,
                    VoxelCell::from_key(&target_key, 0.25, shifted, 0),
                );
                planes.push(PointCorrespondence {
                    src_point: point,
                    plane_normal: Vector3::z(),
                    plane_d: -1.0,
                });
            }
        }
        (source, target, planes)
    }

    #[test]
    fn refinement_can_improve_a_tangential_offset_without_losing_the_plane() {
        let (source, target, planes) = floor_scene(0.012);
        let refined = refine_hybrid_pose(
            &source,
            &target,
            0.25,
            &planes,
            &Matrix3::identity(),
            &Vector3::zeros(),
            0.08,
            &HybridRefineConfig::default(),
        )
        .unwrap();
        assert!(refined.point_rmse_after_m < refined.point_rmse_before_m);
        assert!(refined.translation.x > 0.0);
        assert!(refined.plane_rmse_after_m < 0.003);
    }

    #[test]
    fn correction_limit_applies_to_the_entire_refinement_not_each_iteration() {
        let (source, target, planes) = floor_scene(0.080);
        let refined = refine_hybrid_pose(
            &source,
            &target,
            0.25,
            &planes,
            &Matrix3::identity(),
            &Vector3::zeros(),
            0.08,
            &HybridRefineConfig::default(),
        )
        .unwrap();
        assert!(refined.translation_correction_m <= 0.030001);
        assert!(refined.rotation_correction_deg <= 0.30001);
    }

    #[test]
    fn point_jacobian_matches_finite_differences_for_existing_pose_update() {
        let source = Point3::new(1.2, -0.4, 2.0);
        let rotation = nalgebra::UnitQuaternion::from_euler_angles(0.2, -0.1, 0.3)
            .to_rotation_matrix()
            .into_inner();
        let translation = Vector3::new(4.0, 2.0, -1.0);
        let rp = rotation * source.coords;
        let analytic = SMatrix::<f32, 3, 6>::from_row_slice(&[
            0.0, rp.z, -rp.y, 1.0, 0.0, 0.0, -rp.z, 0.0, rp.x, 0.0, 1.0, 0.0, rp.y, -rp.x, 0.0,
            0.0, 0.0, 1.0,
        ]);
        for column in 0..6 {
            let mut delta = Vector6f::zeros();
            delta[column] = 0.001;
            let (plus_r, plus_t) = apply_delta(&rotation, &translation, &delta);
            let (minus_r, minus_t) = apply_delta(&rotation, &translation, &(-delta));
            let derivative =
                (plus_r * source.coords + plus_t - minus_r * source.coords - minus_t) / 0.002;
            assert!((derivative - analytic.column(column)).norm() < 0.002);
        }
    }

    #[test]
    fn point_system_recovers_small_translation_with_correct_sign() {
        let shift = Vector3::new(0.01, -0.02, 0.005);
        let mut points = Vec::new();
        for x in [-1.0_f32, 0.0, 1.0] {
            for y in [-1.0_f32, 1.0] {
                for z in [-1.0_f32, 1.0] {
                    let source = Point3::new(x, y, z);
                    points.push(PointMatch {
                        source,
                        target: source + shift,
                    });
                }
            }
        }
        let system = build_point_system(&points, &Matrix3::identity(), &Vector3::zeros(), 0.1);
        let solved = solve_icp_delta_observable(&system, 1e-6, 0.02).unwrap();
        assert!((solved.delta.fixed_rows::<3>(3) - shift).norm() < 1e-5);
        assert!(solved.delta.fixed_rows::<3>(0).norm() < 1e-5);
    }

    #[test]
    fn duplicating_point_matches_does_not_change_hybrid_weight() {
        let planes = vec![PointCorrespondence {
            src_point: Point3::new(0.0, 0.0, 0.02),
            plane_normal: Vector3::z(),
            plane_d: 0.0,
        }];
        let points = vec![PointMatch {
            source: Point3::new(1.0, 0.0, 0.02),
            target: Point3::new(1.01, 0.0, 0.0),
        }];
        let repeated = vec![points[0]; 20];
        let config = HybridRefineConfig::default();
        let one = build_mixed_system(
            &planes,
            &points,
            &Matrix3::identity(),
            &Vector3::zeros(),
            0.08,
            &config,
        );
        let many = build_mixed_system(
            &planes,
            &repeated,
            &Matrix3::identity(),
            &Vector3::zeros(),
            0.08,
            &config,
        );
        assert!((one.h - many.h).norm() < 1e-5);
        assert!((one.b - many.b).norm() < 1e-5);
        assert!((one.cost - many.cost).abs() < 1e-6);
    }
}
