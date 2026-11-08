//! Fixed-map camera-pose refinement for recoloring. One SE(3) correction per image.
//! Image matches are local plane-warped NCC matches; the pose fit uses distorted
//! reprojection, a LiDAR-pose prior, robust weights, and held-out correspondences.

use anyhow::{Result, ensure};
use image::{GrayImage, RgbImage};
use nalgebra::{Matrix4, Point3, SMatrix, SVector, UnitQuaternion, Vector3};
use serde::Serialize;

use crate::camera::{CameraIntrinsics, project_camera_point_continuous};
use crate::patch_alignment::WarpSample;

type Vector6 = SVector<f64, 6>;
type Matrix6 = SMatrix<f64, 6, 6>;

#[derive(Debug, Clone, Serialize)]
pub struct CameraPoseAlignmentConfig {
    pub enabled: bool,
    pub graph_iterations: usize,
    pub max_pairs_per_image: usize,
    pub max_matches_per_pair: usize,
    pub max_seed_attempts: usize,
    pub search_radius_px: i32,
    pub min_ncc: f64,
    pub min_peak_gap: f64,
    pub min_texture_stddev: f64,
    pub min_matches: usize,
    pub min_inlier_fraction: f64,
    pub inlier_threshold_px: f64,
    pub huber_px: f64,
    pub max_iterations: usize,
    /// Hard bounds relative to the original LiDAR-derived camera pose.
    pub max_rotation_deg: f64,
    pub max_translation_m: f64,
    pub prior_rotation_std_deg: f64,
    pub prior_translation_std_m: f64,
    pub min_observable_ratio: f64,
    pub min_validation_improvement_px: f64,
    pub min_image_span_px: f64,
}

impl Default for CameraPoseAlignmentConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            graph_iterations: 3,
            max_pairs_per_image: 4,
            max_matches_per_pair: 96,
            max_seed_attempts: 384,
            search_radius_px: 6,
            min_ncc: 0.85,
            min_peak_gap: 0.04,
            min_texture_stddev: 8.0,
            min_matches: 16,
            min_inlier_fraction: 0.65,
            inlier_threshold_px: 1.5,
            huber_px: 1.0,
            max_iterations: 20,
            max_rotation_deg: 1.0,
            max_translation_m: 0.10,
            prior_rotation_std_deg: 0.5,
            prior_translation_std_m: 0.05,
            min_observable_ratio: 1e-6,
            min_validation_improvement_px: 0.05,
            min_image_span_px: 40.0,
        }
    }
}

impl CameraPoseAlignmentConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=8).contains(&self.graph_iterations),
            "invalid pose graph_iterations"
        );
        ensure!(
            (1..=12).contains(&self.max_pairs_per_image),
            "invalid pose pair limit"
        );
        ensure!(
            self.min_matches >= 12
                && self.max_matches_per_pair >= self.min_matches
                && self.max_matches_per_pair <= 512
                && self.max_seed_attempts >= self.max_matches_per_pair
                && self.max_seed_attempts <= 4096,
            "invalid pose match budget"
        );
        ensure!(
            (2..=16).contains(&self.search_radius_px),
            "invalid pose search radius"
        );
        for (name, value) in [
            ("NCC", self.min_ncc),
            ("peak gap", self.min_peak_gap),
            ("inlier fraction", self.min_inlier_fraction),
            ("observable ratio", self.min_observable_ratio),
        ] {
            ensure!(
                value.is_finite() && value > 0.0 && value <= 1.0,
                "invalid pose {name}"
            );
        }
        ensure!(
            self.min_inlier_fraction > 0.5,
            "pose inlier fraction must exceed 0.5"
        );
        for (name, value) in [
            ("texture", self.min_texture_stddev),
            ("inlier threshold", self.inlier_threshold_px),
            ("Huber threshold", self.huber_px),
            ("rotation bound", self.max_rotation_deg),
            ("translation bound", self.max_translation_m),
            ("rotation prior", self.prior_rotation_std_deg),
            ("translation prior", self.prior_translation_std_m),
            ("validation improvement", self.min_validation_improvement_px),
            ("image span", self.min_image_span_px),
        ] {
            ensure!(value.is_finite() && value > 0.0, "invalid pose {name}");
        }
        ensure!(
            self.max_rotation_deg <= 5.0 && self.max_translation_m <= 0.5,
            "pose correction bounds are too large for local matching"
        );
        ensure!(
            (1..=64).contains(&self.max_iterations),
            "invalid pose iteration count"
        );
        Ok(())
    }

    fn prior_weights(&self) -> Vector6 {
        Vector6::from_fn(|i, _| {
            if i < 3 {
                self.prior_rotation_std_deg.to_radians().powi(-2)
            } else {
                self.prior_translation_std_m.powi(-2)
            }
        })
    }
}

/// Parameters are a rotation vector [rad] followed by translation [m], in the
/// camera frame. T_corrected = correction(parameters) * T_original.
pub fn correction(parameters: &[f64; 6]) -> Matrix4<f64> {
    let mut result =
        UnitQuaternion::from_scaled_axis(Vector3::new(parameters[0], parameters[1], parameters[2]))
            .to_homogeneous();
    for axis in 0..3 {
        result[(axis, 3)] = parameters[axis + 3];
    }
    result
}

pub fn pose_prior_cost(parameters: &[f64; 6], config: &CameraPoseAlignmentConfig) -> f64 {
    let p = Vector6::from_column_slice(parameters);
    0.5 * p.component_mul(&p).dot(&config.prior_weights())
}

#[derive(Debug, Clone)]
pub struct PoseMatch {
    pub world: Point3<f64>,
    pub pixel: [f64; 2],
}

#[derive(Debug, Clone, Serialize)]
pub struct PoseEstimate {
    pub accepted: bool,
    pub reason: &'static str,
    pub correction_rotation_rad_translation_m: [f64; 6],
    pub matches: usize,
    pub inliers: usize,
    pub rms_before_px: Option<f64>,
    pub rms_after_px: Option<f64>,
    pub validation_before_px: Option<f64>,
    pub validation_after_px: Option<f64>,
    pub observable_ratio: Option<f64>,
    pub iterations: usize,
}

impl PoseEstimate {
    pub fn rejected(reason: &'static str) -> Self {
        Self {
            accepted: false,
            reason,
            correction_rotation_rad_translation_m: [0.0; 6],
            matches: 0,
            inliers: 0,
            rms_before_px: None,
            rms_after_px: None,
            validation_before_px: None,
            validation_after_px: None,
            observable_ratio: None,
            iterations: 0,
        }
    }
}

pub fn huber_cost(error_px: f64, threshold: f64) -> f64 {
    if error_px <= threshold {
        0.5 * error_px * error_px
    } else {
        threshold * (error_px - 0.5 * threshold)
    }
}

fn residual(m: &PoseMatch, pose: &Matrix4<f64>, camera: &CameraIntrinsics) -> Option<[f64; 2]> {
    let (u, v) = project_camera_point_continuous(&pose.transform_point(&m.world), camera)?;
    if u < 0.0 || v < 0.0 || u >= camera.width as f64 - 1.0 || v >= camera.height as f64 - 1.0 {
        return None;
    }
    Some([u - m.pixel[0], v - m.pixel[1]])
}

fn errors(
    matches: &[PoseMatch],
    pose: &Matrix4<f64>,
    camera: &CameraIntrinsics,
) -> Option<Vec<f64>> {
    matches
        .iter()
        .map(|m| residual(m, pose, camera).map(|e| e[0].hypot(e[1])))
        .collect()
}

fn objective(
    matches: &[PoseMatch],
    initial: &Matrix4<f64>,
    p: &Vector6,
    camera: &CameraIntrinsics,
    config: &CameraPoseAlignmentConfig,
) -> f64 {
    let array: [f64; 6] = (*p).into();
    let pose = correction(&array) * initial;
    let Some(errors) = errors(matches, &pose, camera) else {
        return f64::INFINITY;
    };
    errors
        .into_iter()
        .map(|e| huber_cost(e, config.huber_px))
        .sum::<f64>()
        + pose_prior_cost(&array, config)
}

fn system(
    matches: &[PoseMatch],
    initial: &Matrix4<f64>,
    p: &Vector6,
    camera: &CameraIntrinsics,
    config: &CameraPoseAlignmentConfig,
) -> Option<(Matrix6, Vector6)> {
    let array: [f64; 6] = (*p).into();
    let pose = correction(&array) * initial;
    // Central differences include the actual plumb_bob distortion model.
    let perturbations: Vec<_> = (0..6)
        .map(|axis| {
            let epsilon = if axis < 3 { 1e-6 } else { 1e-5 };
            let mut plus = array;
            let mut minus = array;
            plus[axis] += epsilon;
            minus[axis] -= epsilon;
            (
                correction(&plus) * initial,
                correction(&minus) * initial,
                2.0 * epsilon,
            )
        })
        .collect();
    let mut h = Matrix6::zeros();
    let mut b = Vector6::zeros();
    for m in matches {
        let e = residual(m, &pose, camera)?;
        let length = e[0].hypot(e[1]);
        let weight = if length > config.huber_px {
            config.huber_px / length
        } else {
            1.0
        };
        let mut jacobian = SMatrix::<f64, 2, 6>::zeros();
        for (axis, (plus, minus, denominator)) in perturbations.iter().enumerate() {
            let a = residual(m, plus, camera)?;
            let c = residual(m, minus, camera)?;
            for row in 0..2 {
                jacobian[(row, axis)] = (a[row] - c[row]) / denominator;
            }
        }
        h += weight * jacobian.transpose() * jacobian;
        b += weight * jacobian.transpose() * SVector::<f64, 2>::from_column_slice(&e);
    }
    Some((h, b))
}

fn observable_ratio(h: &Matrix6) -> f64 {
    if (0..6).any(|i| !h[(i, i)].is_finite() || h[(i, i)] <= 1e-12) {
        return 0.0;
    }
    let normalized = Matrix6::from_fn(|i, j| h[(i, j)] / (h[(i, i)] * h[(j, j)]).sqrt());
    let eigen = normalized.symmetric_eigen().eigenvalues;
    (eigen.min().max(0.0) / eigen.max().max(1e-12)).max(0.0)
}

fn within_bounds(p: &Vector6, config: &CameraPoseAlignmentConfig) -> bool {
    p.iter().all(|x| x.is_finite())
        && p.fixed_rows::<3>(0).norm() <= config.max_rotation_deg.to_radians()
        && p.fixed_rows::<3>(3).norm() <= config.max_translation_m
}

/// Refine all six pose parameters using matches from the entire image, not a
/// separate pose per region. Every fifth correspondence is held out of the fit.
pub fn estimate_camera_pose(
    initial: &Matrix4<f64>,
    camera: &CameraIntrinsics,
    matches: &[PoseMatch],
    config: &CameraPoseAlignmentConfig,
) -> PoseEstimate {
    let mut result = PoseEstimate::rejected("insufficient_matches");
    result.matches = matches.len();
    if config.validate().is_err() || matches.len() < config.min_matches {
        return result;
    }
    if !initial.iter().all(|x| x.is_finite())
        || initial.try_inverse().is_none()
        || matches.iter().any(|m| {
            !m.world
                .coords
                .iter()
                .chain(m.pixel.iter())
                .all(|x| x.is_finite())
        })
    {
        result.reason = "invalid_geometry";
        return result;
    }
    let mut low = [f64::INFINITY; 2];
    let mut high = [f64::NEG_INFINITY; 2];
    for m in matches {
        for axis in 0..2 {
            low[axis] = low[axis].min(m.pixel[axis]);
            high[axis] = high[axis].max(m.pixel[axis]);
        }
    }
    if (0..2).any(|axis| high[axis] - low[axis] < config.min_image_span_px) {
        result.reason = "insufficient_image_spread";
        return result;
    }
    let train: Vec<_> = matches
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 5 != 0)
        .map(|(_, m)| m.clone())
        .collect();
    let validation: Vec<_> = matches.iter().step_by(5).cloned().collect();
    let mut p = Vector6::zeros();
    let mut cost = objective(&train, initial, &p, camera, config);
    if !cost.is_finite() {
        result.reason = "outside_image";
        return result;
    }
    let weights = config.prior_weights();
    for iteration in 0..config.max_iterations {
        result.iterations = iteration + 1;
        let Some((mut h, mut b)) = system(&train, initial, &p, camera, config) else {
            result.reason = "invalid_geometry";
            return result;
        };
        let ratio = observable_ratio(&h);
        result.observable_ratio = Some(ratio);
        if ratio < config.min_observable_ratio {
            result.reason = "unobservable_pose";
            return result;
        }
        for i in 0..6 {
            h[(i, i)] += weights[i] + 1e-6;
            b[i] += weights[i] * p[i];
        }
        let Some(step) = h.cholesky().map(|l| l.solve(&(-b))) else {
            result.reason = "singular_system";
            return result;
        };
        if step.norm() < 1e-8 {
            break;
        }
        let mut accepted = None;
        for scale in [1.0, 0.5, 0.25, 0.125, 0.0625] {
            let trial = p + step * scale;
            if !within_bounds(&trial, config) {
                continue;
            }
            let trial_cost = objective(&train, initial, &trial, camera, config);
            if trial_cost < cost {
                accepted = Some((trial, trial_cost));
                break;
            }
        }
        let Some((trial, trial_cost)) = accepted else {
            break;
        };
        let improvement = cost - trial_cost;
        p = trial;
        cost = trial_cost;
        if improvement < 1e-7 {
            break;
        }
    }
    let array: [f64; 6] = p.into();
    let final_pose = correction(&array) * initial;
    let Some(before) = errors(matches, initial, camera) else {
        result.reason = "outside_image";
        return result;
    };
    let Some(after) = errors(matches, &final_pose, camera) else {
        result.reason = "outside_image";
        return result;
    };
    let inliers: Vec<_> = after
        .iter()
        .enumerate()
        .filter(|(_, e)| **e <= config.inlier_threshold_px)
        .map(|(i, _)| i)
        .collect();
    result.inliers = inliers.len();
    if inliers.len() < config.min_matches
        || (inliers.len() as f64 / matches.len() as f64) < config.min_inlier_fraction
    {
        result.reason = "no_pose_consensus";
        return result;
    }
    let rms = |e: &[f64], ids: &[usize]| {
        (ids.iter().map(|&i| e[i] * e[i]).sum::<f64>() / ids.len() as f64).sqrt()
    };
    result.rms_before_px = Some(rms(&before, &inliers));
    result.rms_after_px = Some(rms(&after, &inliers));
    // Use exactly the same validation identities before/after. Clipping limits
    // isolated bad matches without allowing points to disappear from the score.
    let clipped_rms = |values: Vec<f64>| {
        (values
            .iter()
            .map(|x| x.min(3.0 * config.inlier_threshold_px).powi(2))
            .sum::<f64>()
            / values.len() as f64)
            .sqrt()
    };
    let vb = clipped_rms(errors(&validation, initial, camera).unwrap());
    let va = clipped_rms(errors(&validation, &final_pose, camera).unwrap());
    result.validation_before_px = Some(vb);
    result.validation_after_px = Some(va);
    if vb - va < config.min_validation_improvement_px {
        result.reason = if vb < config.min_validation_improvement_px
            && va < config.min_validation_improvement_px
        {
            "already_aligned"
        } else {
            "validation_not_improved"
        };
        return result;
    }
    if !within_bounds(&p, config) {
        result.reason = "correction_out_of_bounds";
        return result;
    }
    // Do not accept a solution pressed against a hard correction limit.
    if p.fixed_rows::<3>(0).norm() > 0.98 * config.max_rotation_deg.to_radians()
        || p.fixed_rows::<3>(3).norm() > 0.98 * config.max_translation_m
    {
        result.reason = "correction_at_limit";
        return result;
    }
    result.accepted = true;
    result.reason = "accepted";
    result.correction_rotation_rad_translation_m = array;
    result
}

pub fn sample_rgb_bilinear(image: &RgbImage, pixel: [f64; 2]) -> Option<[u8; 3]> {
    let [u, v] = pixel;
    if !u.is_finite()
        || !v.is_finite()
        || image.width() == 0
        || image.height() == 0
        || u < 0.0
        || v < 0.0
        || u > (image.width() - 1) as f64
        || v > (image.height() - 1) as f64
    {
        return None;
    }
    let (x, y) = (u.floor() as u32, v.floor() as u32);
    let (nx, ny) = (
        (x + 1).min(image.width() - 1),
        (y + 1).min(image.height() - 1),
    );
    let (a, b) = (u - x as f64, v - y as f64);
    Some(std::array::from_fn(|c| {
        let get = |x, y| image.get_pixel(x, y).0[c] as f64;
        ((1.0 - b) * ((1.0 - a) * get(x, y) + a * get(nx, y))
            + b * ((1.0 - a) * get(x, ny) + a * get(nx, ny)))
        .round()
        .clamp(0.0, 255.0) as u8
    }))
}

fn gray_at(image: &GrayImage, pixel: [f64; 2]) -> Option<f64> {
    let [u, v] = pixel;
    if !u.is_finite()
        || !v.is_finite()
        || u < 0.0
        || v < 0.0
        || u + 1.0 >= image.width() as f64
        || v + 1.0 >= image.height() as f64
    {
        return None;
    }
    let (x, y) = (u.floor() as u32, v.floor() as u32);
    let (a, b) = (u - x as f64, v - y as f64);
    let get = |x, y| image.get_pixel(x, y).0[0] as f64;
    Some(
        (1.0 - b) * ((1.0 - a) * get(x, y) + a * get(x + 1, y))
            + b * ((1.0 - a) * get(x, y + 1) + a * get(x + 1, y + 1)),
    )
}

/// Match one plane-warped patch. Different patches may have different pixel
/// shifts: they are subsequently explained by one shared six-DoF camera pose.
pub fn match_patch(
    reference: &GrayImage,
    target: &GrayImage,
    patch: &[WarpSample],
    config: &CameraPoseAlignmentConfig,
) -> Option<([f64; 2], f64)> {
    let radius = config.search_radius_px + 1;
    let samples: Vec<_> = patch
        .iter()
        .filter_map(|s| {
            let value = gray_at(reference, s.reference)?;
            for offset in [-radius, radius] {
                gray_at(
                    target,
                    [s.target[0] + offset as f64, s.target[1] + offset as f64],
                )?;
            }
            Some((value, s.target))
        })
        .collect();
    if samples.len() < 32 {
        return None;
    }
    let n = samples.len() as f64;
    let mean = samples.iter().map(|s| s.0).sum::<f64>() / n;
    let variance = samples.iter().map(|s| (s.0 - mean).powi(2)).sum::<f64>();
    if variance / n < config.min_texture_stddev.powi(2) {
        return None;
    }
    let score_at = |shift: [f64; 2]| -> Option<f64> {
        let values: Option<Vec<_>> = samples
            .iter()
            .map(|s| gray_at(target, [s.1[0] + shift[0], s.1[1] + shift[1]]))
            .collect();
        let values = values?;
        let mb = values.iter().sum::<f64>() / n;
        let vb = values.iter().map(|b| (b - mb).powi(2)).sum::<f64>();
        if vb / n < config.min_texture_stddev.powi(2) {
            return None;
        }
        Some(
            samples
                .iter()
                .zip(&values)
                .map(|(a, b)| (a.0 - mean) * (b - mb))
                .sum::<f64>()
                / (variance * vb).sqrt(),
        )
    };
    let mut scores = Vec::new();
    for y in -radius..=radius {
        for x in -radius..=radius {
            if let Some(score) = score_at([x as f64, y as f64]) {
                scores.push(([x, y], score));
            }
        }
    }
    scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let &(best, best_score) = scores.first()?;
    if best.iter().any(|x| x.abs() > config.search_radius_px) || best_score < config.min_ncc {
        return None;
    }
    let second = scores
        .iter()
        .filter(|(s, _)| (s[0] - best[0]).abs() > 1 || (s[1] - best[1]).abs() > 1)
        .map(|x| x.1)
        .fold(-1.0_f64, f64::max);
    if best_score - second < config.min_peak_gap {
        return None;
    }
    let base = [best[0] as f64, best[1] as f64];
    let mut shift = base;
    for axis in 0..2 {
        let mut minus = base;
        let mut plus = base;
        minus[axis] -= 1.0;
        plus[axis] += 1.0;
        let (a, b) = (score_at(minus)?, score_at(plus)?);
        let denominator = a - 2.0 * best_score + b;
        if denominator < -1e-8 {
            shift[axis] += (0.5 * (a - b) / denominator).clamp(-0.5, 0.5);
        }
    }
    if shift
        .iter()
        .any(|x| x.abs() > config.search_radius_px as f64)
    {
        return None;
    }
    Some((shift, best_score))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb};

    fn camera() -> CameraIntrinsics {
        CameraIntrinsics::for_image_size(960, 600).unwrap()
    }

    fn scene(parameters: [f64; 6]) -> Vec<PoseMatch> {
        let camera_value = camera();
        let camera = &camera_value;
        let transform = correction(&parameters);
        (-5..=5)
            .flat_map(|x| {
                (-4..=4).map(move |y| {
                    let world = Point3::new(
                        x as f64 * 0.28,
                        y as f64 * 0.22,
                        4.0 + ((x + 2 * y + 30) % 5) as f64 * 0.7,
                    );
                    let pixel = project_camera_point_continuous(
                        &transform.transform_point(&world),
                        &camera,
                    )
                    .unwrap();
                    PoseMatch {
                        world,
                        pixel: [pixel.0, pixel.1],
                    }
                })
            })
            .collect()
    }

    #[test]
    fn recovers_all_six_dofs_with_distortion_and_varied_depth() {
        let truth = [0.002, -0.0018, 0.0012, 0.014, -0.011, 0.009];
        let matches = scene(truth);
        let estimate = estimate_camera_pose(
            &Matrix4::identity(),
            &camera(),
            &matches,
            &Default::default(),
        );
        assert!(estimate.accepted, "{estimate:?}");
        assert!(estimate.rms_after_px.unwrap() < 0.03, "{estimate:?}");
        assert!(estimate.validation_after_px.unwrap() < 0.04);
        for (i, &value) in truth.iter().enumerate() {
            let tolerance = if i < 3 { 0.00015 } else { 0.001 };
            assert!(
                (estimate.correction_rotation_rad_translation_m[i] - value).abs() < tolerance,
                "{estimate:?}"
            );
        }
    }

    #[test]
    fn robust_pose_fit_tolerates_bad_correspondences() {
        let mut matches = scene([0.0015, -0.001, 0.0008, 0.012, -0.008, 0.005]);
        for m in matches.iter_mut().step_by(9) {
            m.pixel[0] += 12.0;
            m.pixel[1] -= 8.0;
        }
        let estimate = estimate_camera_pose(
            &Matrix4::identity(),
            &camera(),
            &matches,
            &Default::default(),
        );
        assert!(estimate.accepted, "{estimate:?}");
        assert!(estimate.rms_after_px.unwrap() < 0.3, "{estimate:?}");
        assert!(estimate.inliers >= 80);
    }

    #[test]
    fn correction_is_applied_in_camera_coordinates_for_a_nonidentity_world_pose() {
        let camera = camera();
        let initial = correction(&[0.5, -0.3, 0.2, -12.0, 8.0, 3.0]);
        let inverse = initial.try_inverse().unwrap();
        let truth = [0.0015, -0.0018, 0.001, 0.012, -0.009, 0.006];
        let matches: Vec<_> = scene(truth)
            .into_iter()
            .map(|m| PoseMatch {
                world: inverse.transform_point(&m.world),
                pixel: m.pixel,
            })
            .collect();
        let estimate = estimate_camera_pose(&initial, &camera, &matches, &Default::default());
        assert!(estimate.accepted, "{estimate:?}");
        assert!(estimate.rms_after_px.unwrap() < 0.03);
        let fitted = correction(&estimate.correction_rotation_rad_translation_m) * initial;
        let expected = correction(&truth) * initial;
        let test = inverse.transform_point(&Point3::new(0.5, -0.3, 5.0));
        let (u, v) =
            project_camera_point_continuous(&fitted.transform_point(&test), &camera).unwrap();
        let (tu, tv) =
            project_camera_point_continuous(&expected.transform_point(&test), &camera).unwrap();
        assert!((u - tu).hypot(v - tv) < 0.05);
    }

    #[test]
    fn identity_bad_support_and_large_corrections_are_not_applied() {
        let config = CameraPoseAlignmentConfig::default();
        let aligned =
            estimate_camera_pose(&Matrix4::identity(), &camera(), &scene([0.0; 6]), &config);
        assert!(!aligned.accepted);
        assert_eq!(aligned.reason, "already_aligned");
        let mut too_far = scene([0.0; 6]);
        for m in &mut too_far {
            m.pixel[0] += 80.0;
        }
        assert!(!estimate_camera_pose(&Matrix4::identity(), &camera(), &too_far, &config).accepted);
        assert!(
            !estimate_camera_pose(&Matrix4::identity(), &camera(), &too_far[..10], &config)
                .accepted
        );
        let mut bad = scene([0.0; 6]);
        bad[0].world.x = f64::NAN;
        assert!(!estimate_camera_pose(&Matrix4::identity(), &camera(), &bad, &config).accepted);
    }

    #[test]
    fn collinear_geometry_is_rejected_without_using_the_prior_to_fake_rank() {
        let camera = camera();
        let matches: Vec<_> = (-30..=30)
            .map(|i| {
                let world = Point3::new(i as f64 * 0.035, i as f64 * 0.025, 5.0);
                let (u, v) = project_camera_point_continuous(&world, &camera).unwrap();
                PoseMatch {
                    world,
                    pixel: [u + 2.0, v - 1.0],
                }
            })
            .collect();
        let estimate =
            estimate_camera_pose(&Matrix4::identity(), &camera, &matches, &Default::default());
        assert!(!estimate.accepted, "{estimate:?}");
        assert_eq!(estimate.reason, "unobservable_pose");
    }

    #[test]
    fn validation_failure_returns_no_correction() {
        let mut matches = scene([0.001, -0.001, 0.001, 0.015, -0.009, 0.0]);
        let aligned = scene([0.0; 6]);
        for i in (0..matches.len()).step_by(5) {
            matches[i].pixel = aligned[i].pixel;
        }
        let estimate = estimate_camera_pose(
            &Matrix4::identity(),
            &camera(),
            &matches,
            &Default::default(),
        );
        assert!(!estimate.accepted, "{estimate:?}");
        assert_eq!(estimate.reason, "validation_not_improved");
        assert_eq!(estimate.correction_rotation_rad_translation_m, [0.0; 6]);
    }

    #[test]
    fn subpixel_rgb_preserves_channels_and_handles_the_last_pixel() {
        let image = RgbImage::from_fn(3, 3, |x, y| Rgb([(10 * x) as u8, (20 * y) as u8, 100]));
        assert_eq!(sample_rgb_bilinear(&image, [0.5, 1.25]), Some([5, 25, 100]));
        assert_eq!(sample_rgb_bilinear(&image, [2.0, 2.0]), Some([20, 40, 100]));
        for p in [[-0.01, 1.0], [3.0, 1.0], [f64::NAN, 1.0]] {
            assert_eq!(sample_rgb_bilinear(&image, p), None);
        }
    }

    #[test]
    fn plane_patch_matching_recovers_local_motion_and_rejects_flat_or_ambiguous_texture() {
        let texture = |x: u32, y: u32| {
            let mut value = (x as u64).wrapping_mul(0x9e3779b97f4a7c15)
                ^ (y as u64).wrapping_mul(0xbf58476d1ce4e5b9);
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            (value ^ (value >> 27)) as u8
        };
        let a = GrayImage::from_fn(100, 100, |x, y| Luma([texture(x, y)]));
        let b = GrayImage::from_fn(100, 100, |x, y| {
            if x >= 2 && y + 1 < 100 {
                *a.get_pixel(x - 2, y + 1)
            } else {
                Luma([0])
            }
        });
        let patch: Vec<_> = (-3..=3)
            .flat_map(|y| {
                (-3..=3).map(move |x| {
                    let p = [50.0 + 2.0 * x as f64, 50.0 + 2.0 * y as f64];
                    WarpSample {
                        world: Point3::origin(),
                        reference: p,
                        target: p,
                    }
                })
            })
            .collect();
        let (shift, ncc) = match_patch(&a, &b, &patch, &Default::default()).unwrap();
        assert!(
            (shift[0] - 2.0).abs() < 0.1 && (shift[1] + 1.0).abs() < 0.1,
            "{shift:?}"
        );
        assert!(ncc > 0.99);
        let flat = GrayImage::from_pixel(100, 100, Luma([120]));
        assert!(match_patch(&flat, &flat, &patch, &Default::default()).is_none());
        let repeated = GrayImage::from_fn(100, 100, |x, y| {
            Luma([if (x / 2 + y / 2) % 2 == 0 { 20 } else { 230 }])
        });
        assert!(match_patch(&repeated, &repeated, &patch, &Default::default()).is_none());
    }
}
