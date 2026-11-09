//! Online, fixed-LiDAR-map image alignment, independent of the navigation filter.
//!
//! Sparse 3D anchors carry dense 8x8 grayscale patches. Each patch pixel is lifted
//! onto the anchor's local plane, then projected into the current image. Robust
//! coarse-to-fine Gauss-Newton adjusts ONE camera_from_world transform. No map
//! coordinates, IMU biases, navigation states or filter covariance are changed.

use anyhow::{Result, ensure};
use image::RgbImage;
use nalgebra::{Matrix3, Matrix4, Point3, SMatrix, SVector, UnitQuaternion, Vector3};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

use crate::camera::CameraIntrinsics;

type Vec6 = SVector<f64, 6>;
type Mat6 = SMatrix<f64, 6, 6>;
const SIDE: usize = 8;
const PIXELS: usize = SIDE * SIDE;

#[derive(Debug, Clone)]
pub struct VisualAlignmentConfig {
    pub max_map_points: usize,
    pub max_active_points: usize,
    pub max_new_points: usize,
    pub min_patches: usize,
    pub pyramid_levels: usize,
    pub iterations_per_level: usize,
    /// At 960-pixel image width. Scaled with image width at runtime.
    pub grid_size_px: f64,
    pub min_texture_std: f64,
    pub min_corner_eigenvalue: f64,
    pub min_view_cos: f64,
    pub max_reference_age_sec: f64,
    pub reset_gap_sec: f64,
    pub reset_after_failures: usize,
    /// Bounds are relative to THIS frame's original LiDAR camera pose.
    pub max_rotation_deg: f64,
    pub max_translation_m: f64,
    pub prior_rotation_std_deg: f64,
    pub prior_translation_std_m: f64,
    pub huber_gray: f64,
    pub min_ncc: f64,
    pub min_inlier_fraction: f64,
    pub max_patch_rmse_gray: f64,
    pub min_validation_improvement_gray: f64,
    pub min_observable_ratio: f64,
    pub min_image_span_ratio: f64,
    pub occlusion_margin_m: f64,
    pub depth_cell_size_px: u32,
}

impl Default for VisualAlignmentConfig {
    fn default() -> Self {
        Self {
            max_map_points: 800,
            max_active_points: 200,
            max_new_points: 160,
            min_patches: 20,
            pyramid_levels: 3,
            iterations_per_level: 12,
            grid_size_px: 24.0,
            min_texture_std: 8.0,
            min_corner_eigenvalue: 2.0,
            min_view_cos: 0.30,
            max_reference_age_sec: 4.0,
            reset_gap_sec: 1.0,
            reset_after_failures: 3,
            max_rotation_deg: 1.0,
            max_translation_m: 0.10,
            prior_rotation_std_deg: 0.5,
            prior_translation_std_m: 0.05,
            huber_gray: 10.0,
            min_ncc: 0.85,
            min_inlier_fraction: 0.70,
            max_patch_rmse_gray: 18.0,
            min_validation_improvement_gray: 0.10,
            min_observable_ratio: 1e-6,
            min_image_span_ratio: 0.20,
            occlusion_margin_m: 0.20,
            depth_cell_size_px: 4,
        }
    }
}

impl VisualAlignmentConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.min_patches >= 12
                && self.max_active_points >= self.min_patches
                && self.max_map_points >= self.max_active_points
                && self.max_map_points <= 4000
                && self.max_new_points > 0
                && self.max_new_points <= self.max_map_points,
            "invalid visual patch budget"
        );
        ensure!(
            (1..=4).contains(&self.pyramid_levels)
                && (1..=30).contains(&self.iterations_per_level)
                && self.reset_after_failures > 0
                && self.reset_after_failures <= 30,
            "invalid visual iteration/lifetime settings"
        );
        for v in [
            self.grid_size_px,
            self.min_texture_std,
            self.min_corner_eigenvalue,
            self.max_reference_age_sec,
            self.reset_gap_sec,
            self.max_rotation_deg,
            self.max_translation_m,
            self.prior_rotation_std_deg,
            self.prior_translation_std_m,
            self.huber_gray,
            self.max_patch_rmse_gray,
            self.min_validation_improvement_gray,
            self.occlusion_margin_m,
        ] {
            ensure!(
                v.is_finite() && v > 0.0,
                "invalid visual positive parameter"
            );
        }
        for v in [
            self.min_view_cos,
            self.min_ncc,
            self.min_inlier_fraction,
            self.min_observable_ratio,
            self.min_image_span_ratio,
        ] {
            ensure!(v.is_finite() && v > 0.0 && v < 1.0, "invalid visual ratio");
        }
        ensure!(
            self.min_inlier_fraction > 0.5
                && self.max_rotation_deg <= 5.0
                && self.max_translation_m <= 0.5
                && (1..=16).contains(&self.depth_cell_size_px),
            "invalid visual bounds/depth grid"
        );
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct VisualPoint {
    pub position_world: Point3<f64>,
    pub normal_world: Vector3<f64>,
    /// Plane support radius; reference pixels beyond measured support are rejected.
    pub support_radius_m: f64,
}

/// PCA on an accumulated local-map neighborhood, not a fabricated per-pixel depth.
/// Mixed surfaces, line-like neighborhoods and unsupported centers are rejected.
pub fn fit_visual_point(center: Point3<f64>, neighbors: &[Point3<f64>]) -> Option<VisualPoint> {
    if neighbors.len() < 6
        || !center.coords.iter().all(|v| v.is_finite())
        || neighbors
            .iter()
            .any(|p| !p.coords.iter().all(|v| v.is_finite()))
    {
        return None;
    }
    let mean =
        neighbors.iter().fold(Vector3::zeros(), |s, p| s + p.coords) / neighbors.len() as f64;
    let covariance = neighbors.iter().fold(Matrix3::zeros(), |s, p| {
        let d = p.coords - mean;
        s + d * d.transpose()
    }) / neighbors.len() as f64;
    let eig = covariance.symmetric_eigen();
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    let [a, b, c] = order.map(|i| eig.eigenvalues[i].max(0.0));
    if c <= 1e-8 || (b - a) / c < 0.15 || a / (a + b + c) > 0.03 || a.sqrt() > 0.025 {
        return None;
    }
    let normal: Vector3<f64> = eig.eigenvectors.column(order[0]).into_owned();
    let offset = normal.dot(&(center.coords - mean));
    if offset.abs() > 0.04
        || neighbors
            .iter()
            .any(|p| normal.dot(&(p.coords - mean)).abs() > 0.07)
    {
        return None;
    }
    let position_world = Point3::from(center.coords - normal * offset);
    // Conservative: the minor in-plane spread must support the patch too.
    let support_radius_m = (2.0 * b.sqrt()).min(0.65);
    if support_radius_m < 0.04 {
        return None;
    }
    Some(VisualPoint {
        position_world,
        normal_world: normal,
        support_radius_m,
    })
}

pub struct VisualFrame<'a> {
    pub image_key: &'a str,
    pub timestamp_sec: f64,
    pub image: &'a RgbImage,
    pub camera: &'a CameraIntrinsics,
    pub camera_from_world_prior: &'a Matrix4<f64>,
    /// Current deskewed raw cloud, transformed with the unmodified LiDAR pose.
    pub geometry_world: &'a [Point3<f64>],
    pub new_points: &'a [VisualPoint],
    pub geometry_trusted: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct VisualAlignmentReport {
    pub image_key: String,
    pub timestamp_sec: f64,
    pub status: String,
    pub applied: bool,
    pub reference_points: usize,
    pub visible_patches: usize,
    pub train_patches: usize,
    pub validation_patches: usize,
    pub inlier_patches: usize,
    pub added_points: usize,
    pub iterations: usize,
    pub initial_rmse_gray: Option<f64>,
    pub final_rmse_gray: Option<f64>,
    pub initial_validation_rmse_gray: Option<f64>,
    pub final_validation_rmse_gray: Option<f64>,
    pub observable_ratio: Option<f64>,
    pub translation_correction_m: f64,
    pub rotation_correction_deg: f64,
    /// Explicit row-major matrices, separate from the navigation trajectory.
    pub camera_from_world_prior: [[f64; 4]; 4],
    pub camera_from_world_used: [[f64; 4]; 4],
}

#[derive(Clone)]
pub struct VisualAlignmentResult {
    pub camera_from_world: Matrix4<f64>,
    pub report: VisualAlignmentReport,
}

fn rows(m: &Matrix4<f64>) -> [[f64; 4]; 4] {
    std::array::from_fn(|i| std::array::from_fn(|j| m[(i, j)]))
}

#[derive(Clone)]
struct GrayLevel {
    width: usize,
    height: usize,
    values: Vec<f64>,
}

impl GrayLevel {
    fn sample(&self, u: f64, v: f64) -> Option<(f64, f64, f64)> {
        if !u.is_finite()
            || !v.is_finite()
            || u < 0.0
            || v < 0.0
            || u >= (self.width - 1) as f64
            || v >= (self.height - 1) as f64
        {
            return None;
        }
        let x = u.floor() as usize;
        let y = v.floor() as usize;
        let dx = u - x as f64;
        let dy = v - y as f64;
        let a = self.values[y * self.width + x];
        let b = self.values[y * self.width + x + 1];
        let c = self.values[(y + 1) * self.width + x];
        let d = self.values[(y + 1) * self.width + x + 1];
        Some((
            (1.0 - dy) * ((1.0 - dx) * a + dx * b) + dy * ((1.0 - dx) * c + dx * d),
            (1.0 - dy) * (b - a) + dy * (d - c),
            (1.0 - dx) * (c - a) + dx * (d - b),
        ))
    }
}

fn pyramid(image: &RgbImage, levels: usize) -> Vec<GrayLevel> {
    let mut result = vec![GrayLevel {
        width: image.width() as usize,
        height: image.height() as usize,
        values: image
            .pixels()
            .map(|p| 0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64)
            .collect(),
    }];
    for _ in 1..levels {
        let prev = result.last().unwrap();
        let width = prev.width / 2;
        let height = prev.height / 2;
        let values = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    (prev.values[2 * y * prev.width + 2 * x]
                        + prev.values[2 * y * prev.width + 2 * x + 1]
                        + prev.values[(2 * y + 1) * prev.width + 2 * x]
                        + prev.values[(2 * y + 1) * prev.width + 2 * x + 1])
                        * 0.25
                })
            })
            .collect();
        result.push(GrayLevel {
            width,
            height,
            values,
        });
    }
    result
}

fn scaled_camera(c: &CameraIntrinsics, level: usize) -> CameraIntrinsics {
    let s = (1u32 << level) as f64;
    CameraIntrinsics {
        width: c.width >> level,
        height: c.height >> level,
        fx: c.fx / s,
        fy: c.fy / s,
        cx: (c.cx + 0.5) / s - 0.5,
        cy: (c.cy + 0.5) / s - 0.5,
        distortion: c.distortion,
    }
}

fn distort(x: f64, y: f64, c: &CameraIntrinsics) -> (f64, f64, SMatrix<f64, 2, 2>) {
    let [k1, k2, p1, p2, k3] = c.distortion;
    let r2 = x * x + y * y;
    let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    let q = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r2 * r2;
    let rx = 2.0 * x * q;
    let ry = 2.0 * y * q;
    let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    (
        xd,
        yd,
        SMatrix::<f64, 2, 2>::new(
            radial + x * rx + 2.0 * p1 * y + 6.0 * p2 * x,
            x * ry + 2.0 * p1 * x + 2.0 * p2 * y,
            y * rx + 2.0 * p1 * x + 2.0 * p2 * y,
            radial + y * ry + 6.0 * p1 * y + 2.0 * p2 * x,
        ),
    )
}

/// Continuous distorted projection + derivative w.r.t. camera XYZ.
fn project(p: &Point3<f64>, c: &CameraIntrinsics) -> Option<([f64; 2], SMatrix<f64, 2, 3>)> {
    if !p.coords.iter().all(|v| v.is_finite()) || p.z <= 0.05 {
        return None;
    }
    let x = p.x / p.z;
    let y = p.y / p.z;
    let (xd, yd, jd) = distort(x, y, c);
    let jn = SMatrix::<f64, 2, 3>::new(1.0 / p.z, 0.0, -x / p.z, 0.0, 1.0 / p.z, -y / p.z);
    let k = SMatrix::<f64, 2, 2>::new(c.fx, 0.0, 0.0, c.fy);
    let pixel = [c.fx * xd + c.cx, c.fy * yd + c.cy];
    pixel
        .iter()
        .all(|v| v.is_finite())
        .then_some((pixel, k * jd * jn))
}

fn ray(pixel: [f64; 2], c: &CameraIntrinsics) -> Option<Vector3<f64>> {
    let target = nalgebra::Vector2::new((pixel[0] - c.cx) / c.fx, (pixel[1] - c.cy) / c.fy);
    let mut x = target;
    for _ in 0..12 {
        let (xd, yd, j) = distort(x.x, x.y, c);
        let error = nalgebra::Vector2::new(xd, yd) - target;
        if error.norm() < 1e-11 {
            break;
        }
        x -= j.try_inverse()? * error;
        if !x.iter().all(|v| v.is_finite()) || x.norm() > 20.0 {
            return None;
        }
    }
    let r = Vector3::new(x.x, x.y, 1.0);
    let (check, _) = project(&Point3::from(r), c)?;
    ((check[0] - pixel[0]).hypot(check[1] - pixel[1]) < 1e-5).then_some(r)
}

fn in_color_region(pixel: [f64; 2], c: &CameraIntrinsics) -> bool {
    pixel[0] >= c.width as f64 * 0.125
        && pixel[0] < c.width as f64 * 0.875
        && pixel[1] >= c.height as f64 * 0.125
        && pixel[1] < c.height as f64 * 0.875
}

fn rigid(m: &Matrix4<f64>) -> bool {
    let r = m.fixed_view::<3, 3>(0, 0);
    m.iter().all(|v| v.is_finite())
        && (r.transpose() * r - Matrix3::identity()).norm() < 1e-5
        && (r.determinant() - 1.0).abs() < 1e-5
        && m[(3, 0)].abs() + m[(3, 1)].abs() + m[(3, 2)].abs() + (m[(3, 3)] - 1.0).abs() < 1e-8
}

fn delta_pose(delta: &Vec6) -> Matrix4<f64> {
    let mut m =
        UnitQuaternion::from_scaled_axis(delta.fixed_rows::<3>(0).into_owned()).to_homogeneous();
    m.fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&delta.fixed_rows::<3>(3));
    m
}

fn correction_vector(pose: &Matrix4<f64>, prior: &Matrix4<f64>) -> Vec6 {
    let d = pose * prior.try_inverse().unwrap();
    let r = UnitQuaternion::from_matrix(&d.fixed_view::<3, 3>(0, 0).into_owned()).scaled_axis();
    Vec6::new(r.x, r.y, r.z, d[(0, 3)], d[(1, 3)], d[(2, 3)])
}

/// Coarse nearest-depth grid. No-depth pixels are unknown, never a fake surface.
struct DepthGrid {
    width: usize,
    height: usize,
    cell: f64,
    z: Vec<f64>,
    margin: f64,
}

impl DepthGrid {
    fn new(
        points: &[Point3<f64>],
        pose: &Matrix4<f64>,
        c: &CameraIntrinsics,
        config: &VisualAlignmentConfig,
    ) -> Self {
        let cell = config.depth_cell_size_px as usize;
        let width = (c.width as usize).div_ceil(cell);
        let height = (c.height as usize).div_ceil(cell);
        let mut result = Self {
            width,
            height,
            cell: cell as f64,
            z: vec![f64::INFINITY; width * height],
            margin: config.occlusion_margin_m,
        };
        for p in points {
            let p = pose.transform_point(p);
            if let Some(([u, v], _)) = project(&p, c) {
                if u >= 0.0 && v >= 0.0 && u < c.width as f64 && v < c.height as f64 {
                    let idx = (v / result.cell) as usize * width + (u / result.cell) as usize;
                    result.z[idx] = result.z[idx].min(p.z);
                }
            }
        }
        result
    }
    fn occluded(&self, pixel: [f64; 2], z: f64) -> bool {
        let x = (pixel[0] / self.cell).floor() as isize;
        let y = (pixel[1] / self.cell).floor() as isize;
        // Nearest-cell + neighbors compensate for sparse scan sampling.
        for yy in y - 1..=y + 1 {
            for xx in x - 1..=x + 1 {
                if xx >= 0
                    && yy >= 0
                    && xx < self.width as isize
                    && yy < self.height as isize
                    && z > self.z[yy as usize * self.width + xx as usize] + self.margin
                {
                    return true;
                }
            }
        }
        false
    }
}

struct PatchLevel {
    world: Vec<Point3<f64>>,
    intensity: Vec<f64>,
}
struct ReferencePoint {
    point: VisualPoint,
    levels: Vec<PatchLevel>,
    timestamp_sec: f64,
    last_seen_sec: f64,
    camera_origin_world: Point3<f64>,
    quality: f64,
}

fn make_reference(
    point: &VisualPoint,
    pose: &Matrix4<f64>,
    gray: &[GrayLevel],
    c: &CameraIntrinsics,
    timestamp: f64,
    config: &VisualAlignmentConfig,
    depth: &DepthGrid,
) -> Option<ReferencePoint> {
    if !point.normal_world.iter().all(|v| v.is_finite())
        || !point.support_radius_m.is_finite()
        || point.support_radius_m <= 0.0
    {
        return None;
    }
    let camera_point = pose.transform_point(&point.position_world);
    let ([u, v], _) = project(&camera_point, c)?;
    if !in_color_region([u, v], c) || depth.occluded([u, v], camera_point.z) {
        return None;
    }
    let world_from_camera = pose.try_inverse()?;
    let origin = world_from_camera.transform_point(&Point3::origin());
    let view = (point.position_world - origin).normalize();
    let normal = point.normal_world.try_normalize(1e-9)?;
    if normal.dot(&view).abs() < config.min_view_cos {
        return None;
    }
    let r = world_from_camera.fixed_view::<3, 3>(0, 0);
    let plane_distance = normal.dot(&(point.position_world - origin));
    let mut levels = Vec::new();
    let mut quality = 0.0;
    for (level, image) in gray.iter().enumerate() {
        let camera = scaled_camera(c, level);
        let scale = (1usize << level) as f64;
        let center = [(u + 0.5) / scale - 0.5, (v + 0.5) / scale - 0.5];
        let mut world = Vec::with_capacity(PIXELS);
        let mut intensity = Vec::with_capacity(PIXELS);
        let mut tensor = SMatrix::<f64, 2, 2>::zeros();
        for y in 0..SIDE {
            for x in 0..SIDE {
                let uv = [center[0] + x as f64 - 3.5, center[1] + y as f64 - 3.5];
                let (value, gx, gy) = image.sample(uv[0], uv[1])?;
                let direction = r * ray(uv, &camera)?;
                let denominator = normal.dot(&direction);
                if denominator.abs() < 1e-6 {
                    return None;
                }
                let distance = plane_distance / denominator;
                if distance <= 0.0 {
                    return None;
                }
                let wp = origin + direction * distance;
                if (wp - point.position_world).norm() > point.support_radius_m {
                    return None;
                }
                let cp = pose.transform_point(&wp);
                let (full_uv, _) = project(&cp, c)?;
                if !in_color_region(full_uv, c) || depth.occluded(full_uv, cp.z) {
                    return None;
                }
                world.push(wp);
                intensity.push(value);
                tensor +=
                    nalgebra::Vector2::new(gx, gy) * nalgebra::Vector2::new(gx, gy).transpose();
            }
        }
        let mean = intensity.iter().sum::<f64>() / PIXELS as f64;
        let variance = intensity.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / PIXELS as f64;
        if variance.sqrt()
            < if level == 0 {
                config.min_texture_std
            } else {
                2.0
            }
        {
            return None;
        }
        if level == 0 {
            quality = tensor.symmetric_eigen().eigenvalues.min() / PIXELS as f64;
            if quality < config.min_corner_eigenvalue {
                return None;
            }
        }
        levels.push(PatchLevel { world, intensity });
    }
    Some(ReferencePoint {
        point: point.clone(),
        levels,
        timestamp_sec: timestamp,
        last_seen_sec: timestamp,
        camera_origin_world: origin,
        quality,
    })
}

struct Observation {
    residual: Vec<f64>,
    jacobian: Vec<Vec6>,
    ncc: f64,
    rmse: f64,
}

fn observe(
    reference: &ReferencePoint,
    level: usize,
    pose: &Matrix4<f64>,
    gray: &GrayLevel,
    c: &CameraIntrinsics,
    with_jacobian: bool,
) -> Option<Observation> {
    let patch = &reference.levels[level];
    let mut values = Vec::with_capacity(PIXELS);
    let mut jacobian = Vec::with_capacity(PIXELS);
    for wp in &patch.world {
        let p = pose.transform_point(wp);
        let ([u, v], jp) = project(&p, c)?;
        let (value, gx, gy) = gray.sample(u, v)?;
        values.push(value);
        if with_jacobian {
            let grad = jp.transpose() * nalgebra::Vector2::new(gx, gy);
            let rotation = p.coords.cross(&grad);
            jacobian.push(Vec6::new(
                rotation.x, rotation.y, rotation.z, grad.x, grad.y, grad.z,
            ));
        }
    }
    let mean = values.iter().sum::<f64>() / PIXELS as f64;
    let ref_mean = patch.intensity.iter().sum::<f64>() / PIXELS as f64;
    let ref_var = patch
        .intensity
        .iter()
        .map(|v| (v - ref_mean).powi(2))
        .sum::<f64>();
    let cur_var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    if ref_var < 1e-6 || cur_var < 1e-6 {
        return None;
    }
    let cross = values
        .iter()
        .zip(&patch.intensity)
        .map(|(a, b)| (a - mean) * (b - ref_mean))
        .sum::<f64>();
    // Eliminate per-patch brightness offset and fit positive exposure gain.
    // Derivatives include the eliminated mean and fitted gain (variable projection).
    let raw_gain = cross / ref_var;
    let gain = raw_gain.clamp(0.5, 2.0);
    let residual: Vec<_> = values
        .iter()
        .zip(&patch.intensity)
        .map(|(a, b)| a - mean - gain * (b - ref_mean))
        .collect();
    if with_jacobian {
        let jm = jacobian.iter().fold(Vec6::zeros(), |s, j| s + j) / PIXELS as f64;
        let jg = if raw_gain > 0.5 && raw_gain < 2.0 {
            jacobian
                .iter()
                .zip(&patch.intensity)
                .fold(Vec6::zeros(), |s, (j, r)| s + j * (r - ref_mean))
                / ref_var
        } else {
            Vec6::zeros()
        };
        for (j, r) in jacobian.iter_mut().zip(&patch.intensity) {
            *j -= jm + jg * (r - ref_mean);
        }
    }
    let rmse = (residual.iter().map(|r| r * r).sum::<f64>() / PIXELS as f64).sqrt();
    Some(Observation {
        residual,
        jacobian,
        ncc: cross / (ref_var * cur_var).sqrt(),
        rmse,
    })
}

fn huber(r: f64, d: f64) -> f64 {
    let a = r.abs();
    if a <= d {
        0.5 * r * r
    } else {
        d * (a - 0.5 * d)
    }
}

fn patch_visible(
    reference: &ReferencePoint,
    pose: &Matrix4<f64>,
    c: &CameraIntrinsics,
    depth: &DepthGrid,
    config: &VisualAlignmentConfig,
) -> bool {
    let p = pose.transform_point(&reference.point.position_world);
    let Some((pixel, _)) = project(&p, c) else {
        return false;
    };
    let Some(world_from_camera) = pose.try_inverse() else {
        return false;
    };
    let origin = world_from_camera.transform_point(&Point3::origin());
    let view = (reference.point.position_world - origin).normalize();
    let old_view = (reference.point.position_world - reference.camera_origin_world).normalize();
    if !in_color_region(pixel, c)
        || reference.point.normal_world.dot(&view).abs() < config.min_view_cos
        || old_view.dot(&view) < 0.5
    {
        return false;
    }
    // Full-resolution support also rejects patches straddling depth boundaries.
    reference.levels[0].world.iter().all(|wp| {
        let p = pose.transform_point(wp);
        project(&p, c).is_some_and(|(uv, _)| in_color_region(uv, c) && !depth.occluded(uv, p.z))
    })
}

fn prior_weights(config: &VisualAlignmentConfig) -> Vec6 {
    let r = 1.0 / config.prior_rotation_std_deg.to_radians().powi(2);
    let t = 1.0 / config.prior_translation_std_m.powi(2);
    Vec6::new(r, r, r, t, t, t)
}

fn prior_system(pose: &Matrix4<f64>, prior: &Matrix4<f64>, weights: &Vec6) -> (Mat6, Vec6, f64) {
    let r = correction_vector(pose, prior);
    let mut j = Mat6::zeros();
    for k in 0..6 {
        let mut d = Vec6::zeros();
        d[k] = 1e-6;
        let a = correction_vector(&(delta_pose(&d) * pose), prior);
        let b = correction_vector(&(delta_pose(&(-d)) * pose), prior);
        j.set_column(k, &((a - b) / 2e-6));
    }
    let w = Mat6::from_diagonal(weights);
    (
        j.transpose() * w * j,
        j.transpose() * w * r,
        0.5 * r.dot(&(weights.component_mul(&r))),
    )
}

struct Evaluation {
    cost: f64,
    rmse: f64,
    inliers: usize,
    h: Mat6,
    g: Vec6,
}

fn evaluate(
    map: &[ReferencePoint],
    active: &[usize],
    level: usize,
    pose: &Matrix4<f64>,
    gray: &GrayLevel,
    c: &CameraIntrinsics,
    config: &VisualAlignmentConfig,
    jac: bool,
) -> Evaluation {
    let mut e = Evaluation {
        cost: 0.0,
        rmse: 0.0,
        inliers: 0,
        h: Mat6::zeros(),
        g: Vec6::zeros(),
    };
    // With 0..255 inputs and gain <=2, an individual valid residual is bounded
    // by 765. Use a larger penalty so disappearing patches never reduce cost.
    let invalid_rmse = 1024.0;
    // Fixed patch membership: leaving the image cannot improve the objective.
    for &i in active {
        if let Some(o) = observe(&map[i], level, pose, gray, c, jac) {
            e.rmse += o.rmse * o.rmse;
            if o.ncc >= config.min_ncc && o.rmse <= config.max_patch_rmse_gray {
                e.inliers += 1;
            }
            for (k, &r) in o.residual.iter().enumerate() {
                e.cost += huber(r, config.huber_gray) / PIXELS as f64;
                if jac {
                    let j = o.jacobian[k];
                    let w = (config.huber_gray / r.abs().max(config.huber_gray)) / PIXELS as f64;
                    e.h += w * j * j.transpose();
                    e.g += w * j * r;
                }
            }
        } else {
            e.cost += huber(invalid_rmse, config.huber_gray);
            e.rmse += invalid_rmse * invalid_rmse;
        }
    }
    e.rmse = (e.rmse / active.len().max(1) as f64).sqrt();
    e
}

pub struct VisualAligner {
    config: VisualAlignmentConfig,
    references: Vec<ReferencePoint>,
    camera_signature: Option<(u32, u32, [f64; 9])>,
    last_timestamp: Option<f64>,
    last_result: Option<(String, f64, VisualAlignmentResult)>,
    failures: usize,
}

impl VisualAligner {
    pub fn new(config: VisualAlignmentConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            references: Vec::new(),
            camera_signature: None,
            last_timestamp: None,
            last_result: None,
            failures: 0,
        })
    }

    pub fn process(&mut self, frame: VisualFrame<'_>) -> Result<VisualAlignmentResult> {
        let VisualFrame {
            image_key,
            timestamp_sec,
            image,
            camera,
            camera_from_world_prior: prior,
            geometry_world,
            new_points,
            geometry_trusted,
        } = frame;
        ensure!(
            timestamp_sec.is_finite() && rigid(prior),
            "invalid visual timestamp/camera pose"
        );
        ensure!(
            camera.width == image.width()
                && camera.height == image.height()
                && camera.width >= (16u32 << self.config.pyramid_levels)
                && camera.height >= (16u32 << self.config.pyramid_levels)
                && camera.fx.is_finite()
                && camera.fy.is_finite()
                && camera.fx > 0.0
                && camera.fy > 0.0
                && camera.cx.is_finite()
                && camera.cy.is_finite()
                && camera.distortion.iter().all(|v| v.is_finite()),
            "invalid visual image/intrinsics"
        );
        let signature = (
            camera.width,
            camera.height,
            [
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                camera.distortion[0],
                camera.distortion[1],
                camera.distortion[2],
                camera.distortion[3],
                camera.distortion[4],
            ],
        );
        let mut report = VisualAlignmentReport {
            image_key: image_key.to_string(),
            timestamp_sec,
            status: "bootstrap".into(),
            applied: false,
            reference_points: self.references.len(),
            visible_patches: 0,
            train_patches: 0,
            validation_patches: 0,
            inlier_patches: 0,
            added_points: 0,
            iterations: 0,
            initial_rmse_gray: None,
            final_rmse_gray: None,
            initial_validation_rmse_gray: None,
            final_validation_rmse_gray: None,
            observable_ratio: None,
            translation_correction_m: 0.0,
            rotation_correction_deg: 0.0,
            camera_from_world_prior: rows(prior),
            camera_from_world_used: rows(prior),
        };
        if !geometry_trusted {
            report.status = "lidar_quality_rejected".into();
            return Ok(VisualAlignmentResult {
                camera_from_world: *prior,
                report,
            });
        }
        if self.camera_signature.as_ref() == Some(&signature) {
            if let Some((key, time, last)) = &self.last_result {
                if key == image_key && *time == timestamp_sec {
                    // A reused image must not match against itself or add duplicate references.
                    let mut result = last.clone();
                    result.report.status = "reused_image".into();
                    result.report.added_points = 0;
                    result.report.iterations = 0;
                    result.report.camera_from_world_prior = rows(prior);
                    let d = correction_vector(&result.camera_from_world, prior);
                    if d.fixed_rows::<3>(0).norm().to_degrees() > self.config.max_rotation_deg
                        || d.fixed_rows::<3>(3).norm() > self.config.max_translation_m
                    {
                        report.status = "reused_image_prior_disagreement".into();
                        return Ok(VisualAlignmentResult {
                            camera_from_world: *prior,
                            report,
                        });
                    }
                    result.report.rotation_correction_deg =
                        d.fixed_rows::<3>(0).norm().to_degrees();
                    result.report.translation_correction_m = d.fixed_rows::<3>(3).norm();
                    return Ok(result);
                }
            }
        }
        if self.last_timestamp.is_some_and(|t| timestamp_sec <= t) {
            report.status = "non_monotonic_image".into();
            return Ok(VisualAlignmentResult {
                camera_from_world: *prior,
                report,
            });
        }
        if self.camera_signature.as_ref() != Some(&signature)
            || self
                .last_timestamp
                .is_some_and(|t| timestamp_sec - t > self.config.reset_gap_sec)
        {
            self.references.clear();
            self.failures = 0;
        }
        self.camera_signature = Some(signature);
        self.last_timestamp = Some(timestamp_sec);
        self.references
            .retain(|r| timestamp_sec - r.last_seen_sec <= self.config.max_reference_age_sec);
        let gray = pyramid(image, self.config.pyramid_levels);
        let depth = DepthGrid::new(geometry_world, prior, camera, &self.config);
        let grid = self.config.grid_size_px * camera.width as f64 / 960.0;
        let mut occupied = HashSet::new();
        let mut active = Vec::new();
        let mut order: Vec<_> = (0..self.references.len()).collect();
        order.sort_by(|&a, &b| {
            self.references[b]
                .quality
                .total_cmp(&self.references[a].quality)
        });
        for i in order {
            let reference = &self.references[i];
            if !patch_visible(reference, prior, camera, &depth, &self.config) {
                continue;
            }
            if (0..self.config.pyramid_levels).any(|l| {
                observe(
                    reference,
                    l,
                    prior,
                    &gray[l],
                    &scaled_camera(camera, l),
                    false,
                )
                .is_none()
            }) {
                continue;
            }
            let (uv, _) = project(
                &prior.transform_point(&reference.point.position_world),
                camera,
            )
            .unwrap();
            if occupied.insert(((uv[0] / grid) as i32, (uv[1] / grid) as i32)) {
                active.push(i);
            }
            if active.len() >= self.config.max_active_points {
                break;
            }
        }
        // Stable spatial order makes held-out patches cover the image, not one corner.
        active.sort_by(|&a, &b| {
            let uv = |i: usize| {
                project(
                    &prior.transform_point(&self.references[i].point.position_world),
                    camera,
                )
                .unwrap()
                .0
            };
            let p = uv(a);
            let q = uv(b);
            p[1].total_cmp(&q[1]).then(p[0].total_cmp(&q[0]))
        });
        report.visible_patches = active.len();
        let mut used = *prior;
        if self.references.is_empty() {
            report.status = "bootstrap".into();
        } else if active.len() < self.config.min_patches {
            report.status = "insufficient_overlap".into();
        } else {
            let validation: Vec<_> = active
                .iter()
                .enumerate()
                .filter(|(k, _)| k % 5 == 0)
                .map(|(_, &i)| i)
                .collect();
            let train: Vec<_> = active
                .iter()
                .enumerate()
                .filter(|(k, _)| k % 5 != 0)
                .map(|(_, &i)| i)
                .collect();
            report.train_patches = train.len();
            report.validation_patches = validation.len();
            let result = self.refine(
                prior,
                &train,
                &validation,
                &active,
                &gray,
                camera,
                geometry_world,
                &mut report,
            );
            if let Some(pose) = result {
                used = pose;
                report.applied = true;
                self.failures = 0;
            }
        }
        let may_seed =
            self.references.is_empty() || report.applied || report.status == "already_aligned";
        if may_seed {
            self.failures = 0;
            // Keep a successfully tracked reference stable instead of replacing
            // every patch periodically with the uncorrected LiDAR-derived pose.
            for &i in &active {
                self.references[i].last_seen_sec = timestamp_sec;
            }
        }
        if !may_seed {
            self.failures += 1;
            if self.failures >= self.config.reset_after_failures {
                self.references.clear();
                self.failures = 0;
                report.status = format!("reseeded_after_{}", report.status);
            }
        }
        if may_seed || self.references.is_empty() {
            let final_depth = DepthGrid::new(geometry_world, &used, camera, &self.config);
            report.added_points = self.add_references(
                new_points,
                &used,
                &gray,
                camera,
                timestamp_sec,
                &final_depth,
            );
        }
        report.reference_points = self.references.len();
        report.camera_from_world_used = rows(&used);
        let result = VisualAlignmentResult {
            camera_from_world: used,
            report,
        };
        self.last_result = Some((image_key.to_string(), timestamp_sec, result.clone()));
        Ok(result)
    }

    fn add_references(
        &mut self,
        points: &[VisualPoint],
        pose: &Matrix4<f64>,
        gray: &[GrayLevel],
        c: &CameraIntrinsics,
        timestamp: f64,
        depth: &DepthGrid,
    ) -> usize {
        let grid = self.config.grid_size_px * c.width as f64 / 960.0;
        let cell = |p: [f64; 2]| ((p[0] / grid) as i32, (p[1] / grid) as i32);
        let mut occupied = HashSet::new();
        for r in &self.references {
            if patch_visible(r, pose, c, depth, &self.config) {
                if let Some((uv, _)) = project(&pose.transform_point(&r.point.position_world), c) {
                    occupied.insert(cell(uv));
                }
            }
        }
        // Test every supplied candidate, retain only the best textured patch per grid cell.
        let mut best: HashMap<(i32, i32), ReferencePoint> = HashMap::new();
        for point in points {
            let Some((uv, _)) = project(&pose.transform_point(&point.position_world), c) else {
                continue;
            };
            let key = cell(uv);
            if occupied.contains(&key) {
                continue;
            }
            let Some(r) = make_reference(point, pose, gray, c, timestamp, &self.config, depth)
            else {
                continue;
            };
            if best.get(&key).is_none_or(|old| r.quality > old.quality) {
                best.insert(key, r);
            }
        }
        let mut best: Vec<_> = best.into_iter().collect();
        best.sort_by(|(ka, a), (kb, b)| b.quality.total_cmp(&a.quality).then(ka.cmp(kb)));
        let count = best.len().min(self.config.max_new_points);
        if self.references.len() + count > self.config.max_map_points {
            self.references.sort_by(|a, b| {
                a.last_seen_sec
                    .total_cmp(&b.last_seen_sec)
                    .then(a.timestamp_sec.total_cmp(&b.timestamp_sec))
            });
            self.references
                .drain(0..self.references.len() + count - self.config.max_map_points);
        }
        self.references
            .extend(best.into_iter().take(count).map(|(_, r)| r));
        count
    }

    #[allow(clippy::too_many_arguments)]
    fn refine(
        &self,
        prior: &Matrix4<f64>,
        train: &[usize],
        validation: &[usize],
        active: &[usize],
        gray: &[GrayLevel],
        camera: &CameraIntrinsics,
        geometry: &[Point3<f64>],
        report: &mut VisualAlignmentReport,
    ) -> Option<Matrix4<f64>> {
        let mut min = [f64::INFINITY; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        for &i in active {
            let (uv, _) = project(
                &prior.transform_point(&self.references[i].point.position_world),
                camera,
            )?;
            for k in 0..2 {
                min[k] = min[k].min(uv[k]);
                max[k] = max[k].max(uv[k]);
            }
        }
        if max[0] - min[0] < camera.width as f64 * self.config.min_image_span_ratio
            || max[1] - min[1] < camera.height as f64 * self.config.min_image_span_ratio
        {
            report.status = "insufficient_image_coverage".into();
            return None;
        }
        let initial = evaluate(
            &self.references,
            active,
            0,
            prior,
            &gray[0],
            camera,
            &self.config,
            false,
        );
        let initial_validation = evaluate(
            &self.references,
            validation,
            0,
            prior,
            &gray[0],
            camera,
            &self.config,
            false,
        );
        report.initial_rmse_gray = Some(initial.rmse);
        report.initial_validation_rmse_gray = Some(initial_validation.rmse);
        // A local optimizer needs some initial correspondence. Reject unrelated
        // imagery before spending iterations on arbitrary photometric minima.
        let coarse = self.config.pyramid_levels - 1;
        let coarse_camera = scaled_camera(camera, coarse);
        let coarse_support = train
            .iter()
            .filter(|&&i| {
                observe(
                    &self.references[i],
                    coarse,
                    prior,
                    &gray[coarse],
                    &coarse_camera,
                    false,
                )
                .is_some_and(|o| o.ncc > 0.20)
            })
            .count();
        if coarse_support as f64 / (train.len() as f64) < 0.30 {
            report.status = "no_coarse_photometric_support".into();
            return None;
        }
        let weights = prior_weights(&self.config);
        let mut pose = *prior;
        for level in (0..self.config.pyramid_levels).rev() {
            let c = scaled_camera(camera, level);
            for _ in 0..self.config.iterations_per_level {
                report.iterations += 1;
                let e = evaluate(
                    &self.references,
                    train,
                    level,
                    &pose,
                    &gray[level],
                    &c,
                    &self.config,
                    true,
                );
                let (ph, pg, pc) = prior_system(&pose, prior, &weights);
                let h = e.h + ph;
                let g = e.g + pg;
                let mut accepted = None;
                for damping in [1e-4, 1e-2, 1.0, 100.0] {
                    let mut lhs = h;
                    for k in 0..6 {
                        lhs[(k, k)] += damping * h[(k, k)].max(1.0);
                    }
                    let Some(delta) = lhs.cholesky().map(|chol| chol.solve(&(-g))) else {
                        continue;
                    };
                    if !delta.iter().all(|v| v.is_finite()) {
                        continue;
                    }
                    for alpha in [1.0, 0.5, 0.25, 0.125] {
                        let candidate = delta_pose(&(alpha * delta)) * pose;
                        let corr = correction_vector(&candidate, prior);
                        if corr.fixed_rows::<3>(0).norm().to_degrees()
                            > self.config.max_rotation_deg
                            || corr.fixed_rows::<3>(3).norm() > self.config.max_translation_m
                        {
                            continue;
                        }
                        let trial = evaluate(
                            &self.references,
                            train,
                            level,
                            &candidate,
                            &gray[level],
                            &c,
                            &self.config,
                            false,
                        );
                        let p = 0.5 * corr.dot(&weights.component_mul(&corr));
                        if trial.cost + p < e.cost + pc - 1e-8 {
                            accepted = Some((candidate, alpha * delta));
                            break;
                        }
                    }
                    if accepted.is_some() {
                        break;
                    }
                }
                let Some((candidate, step)) = accepted else {
                    break;
                };
                pose = candidate;
                if step.fixed_rows::<3>(0).norm() < 1e-6 && step.fixed_rows::<3>(3).norm() < 1e-5 {
                    break;
                }
            }
        }
        let final_e = evaluate(
            &self.references,
            active,
            0,
            &pose,
            &gray[0],
            camera,
            &self.config,
            true,
        );
        let final_v = evaluate(
            &self.references,
            validation,
            0,
            &pose,
            &gray[0],
            camera,
            &self.config,
            false,
        );
        report.final_rmse_gray = Some(final_e.rmse);
        report.final_validation_rmse_gray = Some(final_v.rmse);
        report.inlier_patches = final_e.inliers;
        let corr = correction_vector(&pose, prior);
        let scale = Vec6::new(
            self.config.prior_rotation_std_deg.to_radians(),
            self.config.prior_rotation_std_deg.to_radians(),
            self.config.prior_rotation_std_deg.to_radians(),
            self.config.prior_translation_std_m,
            self.config.prior_translation_std_m,
            self.config.prior_translation_std_m,
        );
        let scaled = Mat6::from_diagonal(&scale) * final_e.h * Mat6::from_diagonal(&scale);
        let eig = scaled.symmetric_eigen().eigenvalues;
        let ratio = eig.min().max(0.0) / eig.max().max(1e-12);
        report.observable_ratio = Some(ratio);
        if ratio < self.config.min_observable_ratio {
            report.status = "degenerate_image_geometry".into();
            return None;
        }
        if final_e.inliers as f64 / (active.len() as f64) < self.config.min_inlier_fraction
            || final_v.inliers as f64 / (validation.len() as f64) < self.config.min_inlier_fraction
        {
            report.status = "photometric_consensus_rejected".into();
            return None;
        }
        let depth = DepthGrid::new(geometry, &pose, camera, &self.config);
        let visible = active
            .iter()
            .filter(|&&i| patch_visible(&self.references[i], &pose, camera, &depth, &self.config))
            .count();
        if visible as f64 / (active.len() as f64) < 0.95 {
            report.status = "final_visibility_rejected".into();
            return None;
        }
        // Held-out patches did not participate in optimization or its pose prior.
        if initial_validation.rmse - final_v.rmse < self.config.min_validation_improvement_gray
            || final_e.cost >= initial.cost
        {
            report.status = if initial.rmse < 2.0
                && initial.inliers as f64 / active.len() as f64 >= self.config.min_inlier_fraction
            {
                "already_aligned".into()
            } else {
                "validation_no_improvement".into()
            };
            return None;
        }
        report.translation_correction_m = corr.fixed_rows::<3>(3).norm();
        report.rotation_correction_deg = corr.fixed_rows::<3>(0).norm().to_degrees();
        report.status = "aligned".into();
        Some(pose)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> CameraIntrinsics {
        CameraIntrinsics {
            width: 640,
            height: 480,
            fx: 430.0,
            fy: 425.0,
            cx: 317.3,
            cy: 243.1,
            distortion: [-0.04, 0.005, 0.002, -0.001, 0.0],
        }
    }

    // Analytic static scene with two depths, slanted surfaces and rich texture.
    // Rendering ray-casts independently for each camera pixel; it does not call
    // observe(), patch creation or the solver's Jacobian to generate measurements.
    fn scene_hit(
        origin: Point3<f64>,
        direction: Vector3<f64>,
    ) -> Option<(Point3<f64>, Vector3<f64>)> {
        let mut best = None;
        let mut nearest = f64::INFINITY;
        for (left, z, n) in [
            (true, 4.0, Vector3::new(-0.10, 0.06, 1.0)),
            (false, 6.0, Vector3::new(0.12, -0.08, 1.0)),
        ] {
            let denom = n.dot(&direction);
            if denom.abs() < 1e-8 {
                continue;
            }
            let t = (z - n.dot(&origin.coords)) / denom;
            let p = origin + direction * t;
            if t > 0.0 && t < nearest && (p.x < 0.0) == left {
                nearest = t;
                best = Some((p, n.normalize()));
            }
        }
        best
    }

    fn render(pose: &Matrix4<f64>, c: &CameraIntrinsics, gain: f64, bias: f64) -> RgbImage {
        let inv = pose.try_inverse().unwrap();
        let origin = inv.transform_point(&Point3::origin());
        let r = inv.fixed_view::<3, 3>(0, 0);
        RgbImage::from_fn(c.width, c.height, |u, v| {
            let value = ray([u as f64, v as f64], c)
                .and_then(|d| scene_hit(origin, r * d))
                .map(|(p, _)| {
                    116.0
                        + 32.0 * (19.0 * p.x + 3.0 * p.y).sin()
                        + 29.0 * (23.0 * p.y - 4.0 * p.x).sin()
                        + 18.0 * (37.0 * p.x + 31.0 * p.y).cos()
                        + 12.0 * (11.0 * p.x - 47.0 * p.y).cos()
                })
                .unwrap_or(0.0);
            let value = (gain * value + bias).round().clamp(0.0, 255.0) as u8;
            image::Rgb([value; 3])
        })
    }

    fn scene_points(c: &CameraIntrinsics, spacing: usize, offset: f64) -> Vec<VisualPoint> {
        let mut result = Vec::new();
        for v in (60..c.height - 60).step_by(spacing) {
            for u in (70..c.width - 70).step_by(spacing) {
                let d = ray([u as f64 + offset, v as f64 + offset], c).unwrap();
                if let Some((p, n)) = scene_hit(Point3::origin(), d) {
                    result.push(VisualPoint {
                        position_world: p,
                        normal_world: n,
                        support_radius_m: 0.65,
                    });
                }
            }
        }
        result
    }

    fn bootstrap() -> (
        VisualAligner,
        CameraIntrinsics,
        Vec<VisualPoint>,
        Vec<Point3<f64>>,
    ) {
        let c = camera();
        let points = scene_points(&c, 24, 0.0);
        let geometry = scene_points(&c, 4, 0.0)
            .iter()
            .map(|p| p.position_world)
            .collect::<Vec<_>>();
        let mut aligner = VisualAligner::new(VisualAlignmentConfig::default()).unwrap();
        let image = render(&Matrix4::identity(), &c, 1.0, 0.0);
        let result = aligner
            .process(VisualFrame {
                image_key: "ref",
                timestamp_sec: 1.0,
                image: &image,
                camera: &c,
                camera_from_world_prior: &Matrix4::identity(),
                geometry_world: &geometry,
                new_points: &points,
                geometry_trusted: true,
            })
            .unwrap();
        assert!(!result.report.applied);
        assert!(result.report.added_points >= 40, "{:?}", result.report);
        (aligner, c, points, geometry)
    }

    fn projection_error(
        a: &Matrix4<f64>,
        b: &Matrix4<f64>,
        points: &[VisualPoint],
        c: &CameraIntrinsics,
    ) -> f64 {
        points
            .iter()
            .map(|p| {
                let (x, _) = project(&a.transform_point(&p.position_world), c).unwrap();
                let (y, _) = project(&b.transform_point(&p.position_world), c).unwrap();
                (x[0] - y[0]).hypot(x[1] - y[1])
            })
            .sum::<f64>()
            / points.len() as f64
    }

    #[test]
    fn direct_alignment_recovers_six_dof_with_different_scan_samples_and_exposure() {
        let (mut aligner, c, points, _) = bootstrap();
        let truth = delta_pose(&Vec6::new(0.006, -0.009, 0.004, 0.065, -0.023, 0.015));
        let prior = delta_pose(&Vec6::new(0.004, -0.006, 0.003, 0.022, -0.013, 0.018)) * truth;
        let image = render(&truth, &c, 1.12, 5.0);
        // Different sampling offsets: the second scan never needs to hit the
        // identical anchor positions stored from the first frame.
        let geometry: Vec<_> = scene_points(&c, 4, 1.3)
            .iter()
            .map(|p| p.position_world)
            .collect();
        let result = aligner
            .process(VisualFrame {
                image_key: "next",
                timestamp_sec: 1.1,
                image: &image,
                camera: &c,
                camera_from_world_prior: &prior,
                geometry_world: &geometry,
                new_points: &[],
                geometry_trusted: true,
            })
            .unwrap();
        let before = projection_error(&prior, &truth, &points, &c);
        let after = projection_error(&result.camera_from_world, &truth, &points, &c);
        eprintln!(
            "6DoF: before={before:.4}px after={after:.4}px report={:?}",
            result.report
        );
        assert!(result.report.applied, "{:?}", result.report);
        assert!(after < 0.20 && after < before * 0.1, "{before}->{after}");
        let difference = correction_vector(&result.camera_from_world, &truth);
        assert!(difference.fixed_rows::<3>(0).norm().to_degrees() < 0.10);
        assert!(difference.fixed_rows::<3>(3).norm() < 0.012);
        assert!(result.report.validation_patches >= 4);
    }

    #[test]
    fn projection_and_photometric_jacobians_match_central_differences() {
        let c = camera();
        let p = Point3::new(0.8, -0.6, 3.0);
        let (_, analytic) = project(&p, &c).unwrap();
        for k in 0..3 {
            let mut a = p;
            let mut b = p;
            a[k] += 1e-6;
            b[k] -= 1e-6;
            let x = project(&a, &c).unwrap().0;
            let y = project(&b, &c).unwrap().0;
            for row in 0..2 {
                assert!((analytic[(row, k)] - (x[row] - y[row]) / 2e-6).abs() < 1e-5);
            }
        }
        let (aligner, c, _, _) = bootstrap();
        let pose = delta_pose(&Vec6::new(0.001, -0.002, 0.001, 0.004, -0.003, 0.002));
        let image = render(&Matrix4::identity(), &c, 1.13, 6.0);
        let gray = pyramid(&image, 3);
        let reference = &aligner.references[4];
        let observation = observe(reference, 0, &pose, &gray[0], &c, true).unwrap();
        for k in 0..6 {
            let mut delta = Vec6::zeros();
            delta[k] = 1e-7;
            let a = observe(
                reference,
                0,
                &(delta_pose(&delta) * pose),
                &gray[0],
                &c,
                false,
            )
            .unwrap();
            let b = observe(
                reference,
                0,
                &(delta_pose(&(-delta)) * pose),
                &gray[0],
                &c,
                false,
            )
            .unwrap();
            for pixel in 0..PIXELS {
                let numerical = (a.residual[pixel] - b.residual[pixel]) / 2e-7;
                assert!(
                    (numerical - observation.jacobian[pixel][k]).abs() < 1e-3,
                    "dof={k} pixel={pixel} {numerical} vs {}",
                    observation.jacobian[pixel][k]
                );
            }
        }
    }

    #[test]
    fn pyramid_pixel_centers_and_distorted_rays_are_consistent() {
        let c = camera();
        for level in 0..3 {
            for p in [Point3::new(0.8, -0.6, 3.0), Point3::new(-1.2, 0.8, 4.0)] {
                let scaled = scaled_camera(&c, level);
                let full = project(&p, &c).unwrap().0;
                let uv = project(&p, &scaled).unwrap().0;
                let s = (1 << level) as f64;
                for k in 0..2 {
                    assert!((uv[k] - ((full[k] + 0.5) / s - 0.5)).abs() < 1e-10);
                }
                let direction = ray(uv, &scaled).unwrap();
                assert!((direction - p.coords / p.z).norm() < 1e-9);
            }
        }
    }

    #[test]
    fn low_texture_does_not_create_reference_points() {
        let c = camera();
        let points = scene_points(&c, 24, 0.0);
        let image = RgbImage::from_pixel(c.width, c.height, image::Rgb([100; 3]));
        let mut aligner = VisualAligner::new(VisualAlignmentConfig::default()).unwrap();
        let result = aligner
            .process(VisualFrame {
                image_key: "blank",
                timestamp_sec: 1.0,
                image: &image,
                camera: &c,
                camera_from_world_prior: &Matrix4::identity(),
                geometry_world: &[],
                new_points: &points,
                geometry_trusted: true,
            })
            .unwrap();
        assert!(!result.report.applied);
        assert_eq!(result.report.reference_points, 0);
    }

    #[test]
    fn unrelated_image_is_rejected_without_poisoning_reference_map() {
        let (mut aligner, c, points, geometry) = bootstrap();
        let count = aligner.references.len();
        let image = RgbImage::from_fn(c.width, c.height, |u, v| {
            let value =
                (127.0 + 60.0 * (0.43 * u as f64).sin() * (0.37 * v as f64).cos()).round() as u8;
            image::Rgb([value; 3])
        });
        let prior = Matrix4::identity();
        let result = aligner
            .process(VisualFrame {
                image_key: "unrelated",
                timestamp_sec: 1.1,
                image: &image,
                camera: &c,
                camera_from_world_prior: &prior,
                geometry_world: &geometry,
                new_points: &points,
                geometry_trusted: true,
            })
            .unwrap();
        assert!(!result.report.applied, "{:?}", result.report);
        assert_eq!(result.camera_from_world, prior);
        assert_eq!(aligner.references.len(), count);
        assert_eq!(result.report.added_points, 0);
    }

    #[test]
    fn duplicate_images_are_reused_and_bad_lidar_frames_do_not_enter_map() {
        let (mut aligner, c, points, geometry) = bootstrap();
        let count = aligner.references.len();
        let image = render(&Matrix4::identity(), &c, 1.0, 0.0);
        let prior = Matrix4::identity();
        let duplicate = aligner
            .process(VisualFrame {
                image_key: "ref",
                timestamp_sec: 1.0,
                image: &image,
                camera: &c,
                camera_from_world_prior: &prior,
                geometry_world: &geometry,
                new_points: &points,
                geometry_trusted: true,
            })
            .unwrap();
        assert_eq!(duplicate.report.status, "reused_image");
        assert_eq!(duplicate.report.added_points, 0);
        assert_eq!(aligner.references.len(), count);
        let bad = aligner
            .process(VisualFrame {
                image_key: "bad_lidar",
                timestamp_sec: 1.1,
                image: &image,
                camera: &c,
                camera_from_world_prior: &prior,
                geometry_world: &geometry,
                new_points: &points,
                geometry_trusted: false,
            })
            .unwrap();
        assert_eq!(bad.report.status, "lidar_quality_rejected");
        assert_eq!(aligner.last_timestamp, Some(1.0));
        assert_eq!(aligner.references.len(), count);
        let old = aligner
            .process(VisualFrame {
                image_key: "old",
                timestamp_sec: 0.9,
                image: &image,
                camera: &c,
                camera_from_world_prior: &prior,
                geometry_world: &geometry,
                new_points: &points,
                geometry_trusted: true,
            })
            .unwrap();
        assert_eq!(old.report.status, "non_monotonic_image");
    }

    #[test]
    fn occluded_background_is_removed_from_alignment_support() {
        let (mut aligner, c, _, _) = bootstrap();
        let image = render(&Matrix4::identity(), &c, 1.0, 0.0);
        let mut foreground = Vec::new();
        for v in (0..c.height).step_by(3) {
            for u in (0..c.width).step_by(3) {
                foreground.push(Point3::from(ray([u as f64, v as f64], &c).unwrap() * 2.0));
            }
        }
        let result = aligner
            .process(VisualFrame {
                image_key: "occluded",
                timestamp_sec: 1.1,
                image: &image,
                camera: &c,
                camera_from_world_prior: &Matrix4::identity(),
                geometry_world: &foreground,
                new_points: &[],
                geometry_trusted: true,
            })
            .unwrap();
        assert_eq!(result.report.visible_patches, 0);
        assert!(!result.report.applied);
    }

    #[test]
    fn excessive_pose_error_is_not_applied() {
        let (mut aligner, c, _, geometry) = bootstrap();
        aligner.config.max_translation_m = 0.0005;
        aligner.config.max_rotation_deg = 0.0005;
        let truth = delta_pose(&Vec6::new(0.003, -0.004, 0.002, 0.025, -0.015, 0.015));
        let image = render(&truth, &c, 1.0, 0.0);
        let prior = Matrix4::identity();
        let result = aligner
            .process(VisualFrame {
                image_key: "outside_bound",
                timestamp_sec: 1.1,
                image: &image,
                camera: &c,
                camera_from_world_prior: &prior,
                geometry_world: &geometry,
                new_points: &[],
                geometry_trusted: true,
            })
            .unwrap();
        assert!(!result.report.applied, "{:?}", result.report);
        assert_eq!(result.camera_from_world, prior);
    }

    #[test]
    fn plane_fit_accepts_plane_and_rejects_line_mixed_surface_and_bad_center() {
        let neighbors: Vec<_> = (-2..=2)
            .flat_map(|y| (-2..=2).map(move |x| Point3::new(x as f64 * 0.12, y as f64 * 0.12, 4.0)))
            .collect();
        let p = fit_visual_point(Point3::new(0.0, 0.0, 4.01), &neighbors).unwrap();
        assert!((p.position_world.z - 4.0).abs() < 1e-9);
        assert!(p.normal_world.z.abs() > 0.999);
        let line: Vec<_> = (0..10)
            .map(|x| Point3::new(x as f64 * 0.1, 0.0, 4.0))
            .collect();
        assert!(fit_visual_point(Point3::new(0.0, 0.0, 4.0), &line).is_none());
        assert!(fit_visual_point(Point3::new(0.0, 0.0, 4.2), &neighbors).is_none());
        let mut mixed = neighbors;
        mixed.push(Point3::new(0.0, 0.0, 4.3));
        assert!(fit_visual_point(Point3::new(0.0, 0.0, 4.0), &mixed).is_none());
    }

    #[test]
    fn map_is_bounded_and_long_image_gaps_bootstrap_again() {
        let (mut aligner, c, points, geometry) = bootstrap();
        let image = render(&Matrix4::identity(), &c, 1.0, 0.0);
        let prior = Matrix4::identity();
        let result = aligner
            .process(VisualFrame {
                image_key: "after_gap",
                timestamp_sec: 3.0,
                image: &image,
                camera: &c,
                camera_from_world_prior: &prior,
                geometry_world: &geometry,
                new_points: &points,
                geometry_trusted: true,
            })
            .unwrap();
        assert_eq!(result.report.status, "bootstrap");
        assert!(!result.report.applied);
        assert!(aligner.references.len() <= aligner.config.max_map_points);
        aligner.config.max_map_points = 240;
        let gray = pyramid(&image, 3);
        for i in 1..6 {
            // New, distant scene copies force actual insertions and eviction.
            let shift = Vector3::new(i as f64 * 20.0, 0.0, 0.0);
            let mut moved_pose = Matrix4::identity();
            moved_pose.fixed_view_mut::<3, 1>(0, 3).copy_from(&(-shift));
            let moved_points: Vec<_> = points
                .iter()
                .map(|p| VisualPoint {
                    position_world: p.position_world + shift,
                    normal_world: p.normal_world,
                    support_radius_m: p.support_radius_m,
                })
                .collect();
            let moved_geometry: Vec<_> = geometry.iter().map(|p| p + shift).collect();
            let added = aligner.add_references(
                &moved_points,
                &moved_pose,
                &gray,
                &c,
                3.0 + i as f64 * 0.1,
                &DepthGrid::new(&moved_geometry, &moved_pose, &c, &aligner.config),
            );
            assert!(added >= 40);
            assert!(aligner.references.len() <= aligner.config.max_map_points);
        }
        assert_eq!(aligner.references.len(), 240);
        assert!(aligner.references.iter().all(|r| r.timestamp_sec > 3.0));
    }

    #[test]
    fn invalid_config_and_invalid_pose_fail_explicitly() {
        let mut config = VisualAlignmentConfig::default();
        config.min_patches = 0;
        assert!(VisualAligner::new(config).is_err());
        let c = camera();
        let image = RgbImage::new(c.width, c.height);
        let mut aligner = VisualAligner::new(VisualAlignmentConfig::default()).unwrap();
        let mut pose = Matrix4::identity();
        pose[(0, 0)] = 2.0;
        assert!(
            aligner
                .process(VisualFrame {
                    image_key: "invalid",
                    timestamp_sec: 1.0,
                    image: &image,
                    camera: &c,
                    camera_from_world_prior: &pose,
                    geometry_world: &[],
                    new_points: &[],
                    geometry_trusted: true
                })
                .is_err()
        );
    }

    #[test]
    fn held_out_patches_can_reject_a_training_fit() {
        let (aligner, c, _, geometry) = bootstrap();
        let truth = delta_pose(&Vec6::new(0.003, -0.004, 0.002, 0.02, -0.01, 0.012));
        let mut image = render(&truth, &c, 1.0, 0.0);
        // Hide the whole right-hand scene, while retaining enough valid training
        // patches on the left. Validation patches never enter the linear system.
        for v in 0..c.height {
            for u in c.width / 2..c.width {
                let z = (120.0 + 65.0 * (0.38 * u as f64).sin() * (0.42 * v as f64).cos()).round()
                    as u8;
                image.put_pixel(u, v, image::Rgb([z; 3]));
            }
        }
        let train: Vec<_> = aligner
            .references
            .iter()
            .enumerate()
            .filter(|(_, r)| r.point.position_world.x < -0.25)
            .map(|(i, _)| i)
            .collect();
        let validation: Vec<_> = aligner
            .references
            .iter()
            .enumerate()
            .filter(|(_, r)| r.point.position_world.x > 0.25)
            .map(|(i, _)| i)
            .collect();
        let active: Vec<_> = train.iter().chain(&validation).copied().collect();
        let prior = Matrix4::identity();
        let mut report = aligner.last_result.as_ref().unwrap().2.report.clone();
        let result = aligner.refine(
            &prior,
            &train,
            &validation,
            &active,
            &pyramid(&image, 3),
            &c,
            &geometry,
            &mut report,
        );
        assert!(result.is_none(), "{:?}", report);
        assert_eq!(report.status, "photometric_consensus_rejected");
    }

    #[test]
    fn aligned_pose_improves_rgb_sampling_without_moving_lidar_geometry() {
        let (mut aligner, c, points, geometry) = bootstrap();
        let truth = delta_pose(&Vec6::new(0.003, -0.004, 0.002, 0.025, -0.015, 0.012));
        let prior = Matrix4::identity();
        let image = render(&truth, &c, 1.0, 0.0);
        let result = aligner
            .process(VisualFrame {
                image_key: "color",
                timestamp_sec: 1.1,
                image: &image,
                camera: &c,
                camera_from_world_prior: &prior,
                geometry_world: &geometry,
                new_points: &[],
                geometry_trusted: true,
            })
            .unwrap();
        assert!(result.report.applied, "{:?}", result.report);
        let lidar_points: Vec<_> = points
            .iter()
            .map(|p| p.position_world.cast::<f32>())
            .collect();
        let pixels = |pose: &Matrix4<f64>| {
            lidar_points
                .iter()
                .map(|p| {
                    crate::camera::project_camera_point(&pose.transform_point(&p.cast::<f64>()), &c)
                })
                .collect::<Vec<_>>()
        };
        let raw_pixels = pixels(&prior);
        let aligned_pixels = pixels(&result.camera_from_world);
        let truth_pixels = pixels(&truth);
        let raw = crate::camera::colorize_world_points(
            &lidar_points,
            &prior,
            Some(&image),
            Some(&raw_pixels),
        )
        .unwrap();
        let aligned = crate::camera::colorize_world_points(
            &lidar_points,
            &prior,
            Some(&image),
            Some(&aligned_pixels),
        )
        .unwrap();
        let expected = crate::camera::colorize_world_points(
            &lidar_points,
            &prior,
            Some(&image),
            Some(&truth_pixels),
        )
        .unwrap();
        let error = |cloud: &[crate::types::ColoredPoint]| {
            cloud
                .iter()
                .zip(&expected)
                .filter_map(|(a, b)| Some((a.rgb?[0] as f64 - b.rgb?[0] as f64).abs()))
                .sum::<f64>()
        };
        assert!(error(&aligned) < error(&raw) * 0.15);
        for (i, (a, b)) in raw.iter().zip(&aligned).enumerate() {
            assert_eq!(a.position, b.position);
            assert_eq!(a.position, lidar_points[i]);
        }
    }

    #[test]
    fn spatially_concentrated_patches_cannot_move_camera() {
        let (mut aligner, c, _, geometry) = bootstrap();
        aligner.references.retain(|r| {
            let uv = project(&r.point.position_world, &c).unwrap().0;
            uv[1] > 200.0 && uv[1] < 275.0
        });
        aligner.config.min_patches = 12;
        let truth = delta_pose(&Vec6::new(0.003, -0.004, 0.002, 0.025, -0.015, 0.012));
        let image = render(&truth, &c, 1.0, 0.0);
        let result = aligner
            .process(VisualFrame {
                image_key: "narrow",
                timestamp_sec: 1.1,
                image: &image,
                camera: &c,
                camera_from_world_prior: &Matrix4::identity(),
                geometry_world: &geometry,
                new_points: &[],
                geometry_trusted: true,
            })
            .unwrap();
        assert!(!result.report.applied);
        assert_eq!(result.report.status, "insufficient_image_coverage");
    }

    #[test]
    fn successfully_tracked_references_survive_the_idle_age_limit() {
        let (mut aligner, c, points, geometry) = bootstrap();
        let image = render(&Matrix4::identity(), &c, 1.0, 0.0);
        for i in 1..=6 {
            let key = format!("steady_{i}");
            let result = aligner
                .process(VisualFrame {
                    image_key: &key,
                    timestamp_sec: 1.0 + 0.8 * i as f64,
                    image: &image,
                    camera: &c,
                    camera_from_world_prior: &Matrix4::identity(),
                    geometry_world: &geometry,
                    new_points: &points,
                    geometry_trusted: true,
                })
                .unwrap();
            assert_eq!(
                result.report.status, "already_aligned",
                "{:?}",
                result.report
            );
        }
        assert!(aligner.references.iter().any(|r| r.timestamp_sec == 1.0));
        assert!(aligner.references.iter().any(|r| r.last_seen_sec > 5.0));
    }
}
