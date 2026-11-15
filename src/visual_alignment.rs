//! Sparse direct image alignment after the LiDAR update.
//!
//! Each selected landmark stores grayscale 9x9 patches at three pyramid levels.
//! The map is fixed during a solve; the frame-start ESKF state is optimized.
//! Camera motion to the image timestamp (including biases/velocity and the IMU
//! lever arm) is part of the measurement function. No normals, normal/view-angle
//! tests, depth-buffer tests or occlusion rejection are used here.
//!
//! Patch geometry uses constant optical-axis depth in the reference camera.
//! This is a local approximation, not FAST-LIVO2's plane-induced patch warp.

use image::RgbImage;
use nalgebra::{Cholesky, Matrix3, Matrix4, Point3, Vector2, Vector3};

use crate::camera::{CameraIntrinsics, project_camera_point_subpixel, unproject_pixel_at_depth};
use crate::eskf::{ERROR_STATE_DIM, Eskf, Matrix15, NavState, Vector15};
use crate::types::{IMU, VisualAlignmentLog};

pub const PATCH_SIZE: usize = 9;
pub const PATCH_RADIUS: i32 = 4;
pub const PATCH_PIXELS: usize = PATCH_SIZE * PATCH_SIZE;
pub const PYRAMID_LEVELS: usize = 3;

#[derive(Debug, Clone)]
pub struct VisualAlignmentConfig {
    pub grid_size_px: usize,
    pub max_landmarks: usize,
    pub max_patches_per_frame: usize,
    pub max_new_landmarks_per_frame: usize,
    pub max_reference_age_frames: u64,
    /// Refresh a successfully matched reference to limit viewpoint changes.
    pub reference_refresh_frames: u64,
    pub min_patches: usize,
    pub max_iterations_per_level: usize,
    /// RMS grayscale gradient on the 0..255 scale.
    pub min_gradient: f64,
    pub min_depth_m: f64,
    pub max_depth_m: f64,
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
            max_landmarks: 2000,
            max_patches_per_frame: 300,
            max_new_landmarks_per_frame: 150,
            max_reference_age_frames: 60,
            reference_refresh_frames: 5,
            min_patches: 20,
            max_iterations_per_level: 5,
            min_gradient: 3.0,
            min_depth_m: 0.5,
            max_depth_m: 40.0,
            patch_sigma: 10.0,
            huber_delta: 10.0,
            max_initial_patch_rmse: 45.0,
            max_final_rmse: 25.0,
            max_translation_correction_m: 0.05,
            max_rotation_correction_deg: 0.5,
        }
    }
}

impl VisualAlignmentConfig {
    fn valid(&self) -> bool {
        self.grid_size_px > 0
            && self.max_landmarks > 0
            && self.min_patches > 0
            && self.max_patches_per_frame >= self.min_patches
            && self.max_new_landmarks_per_frame > 0
            && self.max_iterations_per_level > 0
            && self.min_depth_m.is_finite()
            && self.min_depth_m > 0.0
            && self.max_depth_m.is_finite()
            && self.max_depth_m > self.min_depth_m
            && [
                self.min_gradient,
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

#[derive(Clone)]
struct ReferencePatch {
    /// Raw grayscale values, always 9x9, row-major around the projected point.
    gray: [f32; PATCH_PIXELS],
    world_samples: [Point3<f64>; PATCH_PIXELS],
}

struct VisualLandmark {
    world_point: Point3<f64>,
    patches: [ReferencePatch; PYRAMID_LEVELS],
    /// Last accepted reference capture (not the landmark birth time).
    created_frame: u64,
}

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

#[derive(Default)]
pub struct VisualAligner {
    pub config: VisualAlignmentConfig,
    landmarks: Vec<VisualLandmark>,
    last_image_time: Option<f64>,
}

impl VisualAligner {
    pub fn new(config: VisualAlignmentConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }

    /// Call BEFORE updating the current frame pose, RGB samples and either map.
    /// Call only for a frame whose LiDAR pose/map insertion has been accepted.
    pub fn process_frame(
        &mut self,
        filter: &mut Eskf,
        imu: &[IMU],
        frame_start_time: f64,
        image_time: f64,
        frame_index: u64,
        rgb: &RgbImage,
        camera: &CameraIntrinsics,
        camera_from_lidar: &Matrix4<f64>,
        lidar_points_at_frame_start: &[Point3<f32>],
    ) -> VisualAlignmentLog {
        let mut report = VisualAlignmentLog::default();
        if !self.config.valid() {
            report.status = "invalid_config".into();
            return report;
        }
        if self
            .last_image_time
            .is_some_and(|last| image_time <= last + 1e-9)
        {
            report.status = "reused_or_older_image".into();
            report.landmarks = self.landmarks.len();
            return report;
        }
        if !frame_start_time.is_finite()
            || !image_time.is_finite()
            || image_time < frame_start_time
            || (image_time > frame_start_time
                && !(imu.first().is_some_and(|s| s.timestamp <= frame_start_time)
                    && imu.last().is_some_and(|s| s.timestamp >= image_time)))
            || rgb.width() != camera.width
            || rgb.height() != camera.height
            || rgb.width() < 48
            || rgb.height() < 48
        {
            report.status = "invalid_image_time_or_dimensions".into();
            return report;
        }
        self.last_image_time = Some(image_time);
        self.landmarks.retain(|p| {
            frame_index.saturating_sub(p.created_frame) <= self.config.max_reference_age_frames
        });
        let images = pyramid(rgb, camera);
        let prior = filter.clone();
        let observation = ImageObservation {
            filter: &prior,
            imu,
            t_start: frame_start_time,
            t_image: image_time,
            camera_from_lidar,
        };
        let Some(initial_camera_pose) = observation.camera_pose(&prior.state) else {
            report.status = "invalid_camera_pose".into();
            return report;
        };
        let indices = self.select_landmarks(&images[0], &initial_camera_pose);
        report.matched_patches = indices.len();
        let mut seed_allowed = true;
        if indices.len() >= self.config.min_patches {
            seed_allowed = self.optimize(&observation, &images, &indices, filter, &mut report);
        } else {
            report.status = if self.landmarks.is_empty() {
                "bootstrap"
            } else {
                "insufficient_patches"
            }
            .into();
        }
        if seed_allowed {
            // New references are captured only AFTER accepting the image solve;
            // they can never be used as observations in that same solve.
            if let Some(camera_pose) = observation.camera_pose(&filter.state) {
                report.refreshed_landmarks =
                    self.refresh_landmarks(&images, &camera_pose, &indices, frame_index);
                let world_from_lidar = filter.lidar_pose();
                report.new_landmarks = self.add_landmarks(
                    &images,
                    &camera_pose,
                    &world_from_lidar,
                    lidar_points_at_frame_start,
                    frame_index,
                );
            }
        }
        report.landmarks = self.landmarks.len();
        report
    }

    fn select_landmarks(&self, image: &ImageLevel, pose: &Matrix4<f64>) -> Vec<usize> {
        let cols = image.image.width.div_ceil(self.config.grid_size_px);
        let rows = image.image.height.div_ceil(self.config.grid_size_px);
        let mut occupied = vec![false; cols * rows];
        let mut indices = Vec::new();
        // Prefer newer references, while retaining at most one per image cell.
        for (index, landmark) in self.landmarks.iter().enumerate().rev() {
            let point = pose.transform_point(&landmark.world_point);
            let Some((pixel, _)) = project_camera_point_subpixel(&point, &image.camera) else {
                continue;
            };
            if image.image.sample(&pixel).is_none() {
                continue;
            }
            let cell = pixel.y as usize / self.config.grid_size_px * cols
                + pixel.x as usize / self.config.grid_size_px;
            if occupied[cell] {
                continue;
            }
            occupied[cell] = true;
            indices.push(index);
            if indices.len() >= self.config.max_patches_per_frame {
                break;
            }
        }
        indices
    }

    fn build_system(
        &self,
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
        let information_per_pixel = 1.0 / (self.config.patch_sigma.powi(2) * PATCH_PIXELS as f64);
        // A fixed patch/pixel set is used across each level and its line search.
        // If a candidate loses a patch, reject it instead of rewarding lost data.
        for &index in indices {
            let evaluation = evaluate_patch(
                &self.landmarks[index].patches[level_index],
                level,
                pose,
                linearization,
            )?;
            for i in 0..PATCH_PIXELS {
                let residual = evaluation.residuals[i];
                let (weight, loss) = huber(residual, self.config.huber_delta);
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

    fn optimize(
        &self,
        observation: &ImageObservation<'_>,
        images: &[ImageLevel],
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
                    evaluate_patch(&self.landmarks[i].patches[level_index], level, &pose, None)
                        .is_some_and(|p| {
                            (p.residuals.iter().map(|r| r * r).sum::<f64>() / PATCH_PIXELS as f64)
                                .sqrt()
                                <= self.config.max_initial_patch_rmse
                        })
                })
                .collect();
            if active.len() < self.config.min_patches {
                continue;
            }
            if level_index == 0 {
                finest_indices = active.clone();
            }
            for _ in 0..self.config.max_iterations_per_level {
                let Some(linearization) = observation.linearize(&state) else {
                    break;
                };
                let Some(system) = self.build_system(
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
                    if translation > self.config.max_translation_correction_m
                        || rotation > self.config.max_rotation_correction_deg
                    {
                        continue;
                    }
                    let Some(candidate_pose) = observation.camera_pose(&candidate) else {
                        continue;
                    };
                    let Some(candidate_system) =
                        self.build_system(&active, level_index, level, &candidate_pose, None)
                    else {
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
        if finest_indices.len() < self.config.min_patches {
            report.status = "insufficient_fine_patches".into();
            return false;
        }
        let Some(initial_pose) = observation.camera_pose(&prior_state) else {
            return false;
        };
        let Some(final_linearization) = observation.linearize(&state) else {
            return false;
        };
        let Some(initial) = self.build_system(&finest_indices, 0, &images[0], &initial_pose, None)
        else {
            report.status = "reference_support_lost".into();
            return false;
        };
        let Some(final_system) = self.build_system(
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
        if !final_system.cost.is_finite()
            || final_system.rmse() > self.config.max_final_rmse
            || final_system.cost + prior_cost(&state) > initial.cost + 1e-9
            || final_system.squared_error > initial.squared_error + 1e-6
            || translation > self.config.max_translation_correction_m
            || rotation > self.config.max_rotation_correction_deg
        {
            report.status = "rejected".into();
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

    fn refresh_landmarks(
        &mut self,
        images: &[ImageLevel],
        camera_from_world: &Matrix4<f64>,
        indices: &[usize],
        frame_index: u64,
    ) -> usize {
        let Some(world_from_camera) = camera_from_world.try_inverse() else {
            return 0;
        };
        let mut refreshed = 0;
        for &index in indices {
            let landmark = &mut self.landmarks[index];
            if frame_index.saturating_sub(landmark.created_frame)
                < self.config.reference_refresh_frames
            {
                continue;
            }
            let Some(evaluation) =
                evaluate_patch(&landmark.patches[0], &images[0], camera_from_world, None)
            else {
                continue;
            };
            let rmse = (evaluation.residuals.iter().map(|r| r * r).sum::<f64>()
                / PATCH_PIXELS as f64)
                .sqrt();
            if rmse > self.config.max_final_rmse {
                continue;
            }
            let pc = camera_from_world.transform_point(&landmark.world_point);
            if let Some(patches) = capture_reference_patches(images, &pc, &world_from_camera) {
                landmark.patches = patches;
                landmark.created_frame = frame_index;
                refreshed += 1;
            }
        }
        refreshed
    }

    fn add_landmarks(
        &mut self,
        images: &[ImageLevel],
        camera_from_world: &Matrix4<f64>,
        world_from_lidar: &Matrix4<f64>,
        points: &[Point3<f32>],
        frame_index: u64,
    ) -> usize {
        let Some(world_from_camera) = camera_from_world.try_inverse() else {
            return 0;
        };
        let cols = images[0].image.width.div_ceil(self.config.grid_size_px);
        let rows = images[0].image.height.div_ceil(self.config.grid_size_px);
        let mut occupied = vec![false; cols * rows];
        for landmark in &self.landmarks {
            let pc = camera_from_world.transform_point(&landmark.world_point);
            if let Some((pixel, _)) = project_camera_point_subpixel(&pc, &images[0].camera) {
                if images[0].image.sample(&pixel).is_some() {
                    occupied[pixel.y as usize / self.config.grid_size_px * cols
                        + pixel.x as usize / self.config.grid_size_px] = true;
                }
            }
        }
        // Choose a textured candidate in each cell. Scores measure image
        // gradients only; there is no surface-normal or depth-neighbor test.
        let mut candidates: Vec<Option<(f64, Point3<f64>, Point3<f64>)>> = vec![None; cols * rows];
        for point in points {
            if !point.coords.iter().all(|v| v.is_finite()) {
                continue;
            }
            let world = world_from_lidar.transform_point(&point.cast::<f64>());
            let pc = camera_from_world.transform_point(&world);
            if pc.z < self.config.min_depth_m || pc.z > self.config.max_depth_m {
                continue;
            }
            let Some((pixel, _)) = project_camera_point_subpixel(&pc, &images[0].camera) else {
                continue;
            };
            if images[0].image.sample(&pixel).is_none() {
                continue;
            }
            let cell = pixel.y as usize / self.config.grid_size_px * cols
                + pixel.x as usize / self.config.grid_size_px;
            if occupied[cell] {
                continue;
            }
            let mut gradient_sum = 0.0;
            let mut valid = true;
            for y in -PATCH_RADIUS..=PATCH_RADIUS {
                for x in -PATCH_RADIUS..=PATCH_RADIUS {
                    match images[0]
                        .image
                        .sample(&(pixel + Vector2::new(x as f64, y as f64)))
                    {
                        Some((_, g)) => gradient_sum += g.norm_squared(),
                        None => valid = false,
                    }
                }
            }
            let score = (gradient_sum / PATCH_PIXELS as f64).sqrt();
            if valid
                && score >= self.config.min_gradient
                && candidates[cell]
                    .as_ref()
                    .is_none_or(|(old, _, _)| score > *old)
            {
                candidates[cell] = Some((score, world, pc));
            }
        }
        let mut candidates: Vec<_> = candidates.into_iter().flatten().collect();
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut added = 0;
        for (_, world_point, pc) in candidates {
            let Some(patches) = capture_reference_patches(images, &pc, &world_from_camera) else {
                continue;
            };
            self.landmarks.push(VisualLandmark {
                world_point,
                patches,
                created_frame: frame_index,
            });
            added += 1;
            if added >= self.config.max_new_landmarks_per_frame {
                break;
            }
        }
        if self.landmarks.len() > self.config.max_landmarks {
            self.landmarks
                .drain(..self.landmarks.len() - self.config.max_landmarks);
        }
        added
    }
}

fn capture_reference_patches(
    images: &[ImageLevel],
    pc: &Point3<f64>,
    world_from_camera: &Matrix4<f64>,
) -> Option<[ReferencePatch; PYRAMID_LEVELS]> {
    let mut patches = Vec::with_capacity(PYRAMID_LEVELS);
    for level in images {
        let (pixel, _) = project_camera_point_subpixel(pc, &level.camera)?;
        let mut gray = [0.0f32; PATCH_PIXELS];
        let mut world_samples = [Point3::origin(); PATCH_PIXELS];
        for y in -PATCH_RADIUS..=PATCH_RADIUS {
            for x in -PATCH_RADIUS..=PATCH_RADIUS {
                let i = ((y + PATCH_RADIUS) as usize) * PATCH_SIZE + (x + PATCH_RADIUS) as usize;
                let p = pixel + Vector2::new(x as f64, y as f64);
                let (value, _) = level.image.sample(&p)?;
                let camera_point = unproject_pixel_at_depth(&p, pc.z, &level.camera)?;
                gray[i] = value as f32;
                world_samples[i] = world_from_camera.transform_point(&camera_point);
            }
        }
        patches.push(ReferencePatch {
            gray,
            world_samples,
        });
    }
    patches.try_into().ok()
}

fn correction(prior: &NavState, state: &NavState, filter: &Eskf) -> (f64, f64) {
    let prior_pose = filter.lidar_pose_of(prior);
    let pose = filter.lidar_pose_of(state);
    let translation = (pose.fixed_view::<3, 1>(0, 3) - prior_pose.fixed_view::<3, 1>(0, 3)).norm();
    let rotation = (state.rot * prior.rot.inverse()).angle().to_degrees();
    (translation, rotation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eskf::{EskfConfig, ImuModel, ImuNoise, InitialStd};
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

    fn sources(camera: &CameraIntrinsics) -> Vec<Point3<f32>> {
        let mut result = Vec::new();
        for v in (32..camera.height - 32).step_by(8) {
            for u in (32..camera.width - 32).step_by(8) {
                result.push(
                    unproject_pixel_at_depth(
                        &Vector2::new(u as f64 + 0.2, v as f64 + 0.3),
                        5.0,
                        camera,
                    )
                    .unwrap()
                    .cast(),
                );
            }
        }
        result
    }

    fn config() -> VisualAlignmentConfig {
        VisualAlignmentConfig {
            grid_size_px: 24,
            max_new_landmarks_per_frame: 300,
            min_gradient: 1.0,
            max_translation_correction_m: 0.15,
            max_rotation_correction_deg: 2.0,
            max_iterations_per_level: 8,
            ..Default::default()
        }
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
    fn bootstrap_stores_exactly_centered_9x9_at_each_scale_and_no_self_update() {
        let camera = camera();
        let mut filter = filter();
        let image = render_plane(&camera, Matrix4::identity(), 0);
        let prior = filter.state;
        let cov = filter.cov;
        let mut aligner = VisualAligner::new(config());
        let report = aligner.process_frame(
            &mut filter,
            &[],
            0.0,
            0.0,
            0,
            &image,
            &camera,
            &Matrix4::identity(),
            &sources(&camera),
        );
        assert_eq!(report.status, "bootstrap");
        assert!(!report.update_applied);
        assert_eq!(filter.state, prior);
        assert_eq!(filter.cov, cov);
        assert!(report.new_landmarks >= 20, "{:?}", report);
        for landmark in &aligner.landmarks {
            for patch in &landmark.patches {
                assert_eq!(patch.gray.len(), 81);
                assert!((patch.world_samples[40] - landmark.world_point).norm() < 1e-8);
            }
        }
        let again = aligner.process_frame(
            &mut filter,
            &[],
            0.0,
            0.0,
            1,
            &image,
            &camera,
            &Matrix4::identity(),
            &sources(&camera),
        );
        assert_eq!(again.status, "reused_or_older_image");
        assert_eq!(filter.state, prior);
        assert_eq!(filter.cov, cov);
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
    fn direct_alignment_recovers_synthetic_translation_and_rotation() {
        let camera = camera();
        let mut filter = filter();
        let mut aligner = VisualAligner::new(config());
        let source = sources(&camera);
        let reference = render_plane(&camera, Matrix4::identity(), 0);
        aligner.process_frame(
            &mut filter,
            &[],
            0.0,
            0.0,
            0,
            &reference,
            &camera,
            &Matrix4::identity(),
            &source,
        );
        let mut truth = NavState::default();
        truth.pos = Vector3::new(0.027, -0.018, 0.009);
        truth.rot = UnitQuaternion::from_euler_angles(0.0015, -0.002, 0.003);
        let current = render_plane(&camera, truth.lidar_pose(&Vector3::zeros()), 0);
        // LiDAR estimate is close, but still has a measurable subpixel error.
        filter.state.pos = Vector3::new(0.01, -0.005, 0.002);
        filter.state.rot = UnitQuaternion::from_euler_angles(0.0005, -0.001, 0.001);
        let before = (filter.state.pos - truth.pos).norm();
        let original_state = filter.state;
        let original_cov = filter.cov;
        let imu = [
            IMU {
                timestamp: 0.0,
                angular_velocity: [0.0; 3],
                linear_acceleration: [0.0, 0.0, 9.80665],
            },
            IMU {
                timestamp: 0.2,
                angular_velocity: [0.0; 3],
                linear_acceleration: [0.0, 0.0, 9.80665],
            },
        ];
        let report = aligner.process_frame(
            &mut filter,
            &imu,
            0.1,
            0.1,
            1,
            &current,
            &camera,
            &Matrix4::identity(),
            &source,
        );
        assert!(report.update_applied, "{:?}", report);
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

    #[test]
    fn textureless_or_invalid_time_never_changes_filter() {
        let camera = camera();
        let mut filter = filter();
        let cov = filter.cov;
        let state = filter.state;
        let mut aligner = VisualAligner::new(config());
        let blank = RgbImage::from_pixel(320, 240, Rgb([120, 120, 120]));
        let report = aligner.process_frame(
            &mut filter,
            &[],
            0.0,
            0.0,
            0,
            &blank,
            &camera,
            &Matrix4::identity(),
            &sources(&camera),
        );
        assert_eq!(report.new_landmarks, 0);
        let invalid = aligner.process_frame(
            &mut filter,
            &[],
            1.0,
            0.9,
            1,
            &blank,
            &camera,
            &Matrix4::identity(),
            &sources(&camera),
        );
        assert_eq!(invalid.status, "invalid_image_time_or_dimensions");
        assert_eq!(filter.state, state);
        assert_eq!(filter.cov, cov);
    }

    #[test]
    fn candidate_cannot_reduce_cost_by_moving_patches_out_of_image() {
        let camera = camera();
        let mut filter = filter();
        let mut aligner = VisualAligner::new(config());
        let rgb = render_plane(&camera, Matrix4::identity(), 0);
        let images = pyramid(&rgb, &camera);
        aligner.process_frame(
            &mut filter,
            &[],
            0.0,
            0.0,
            0,
            &rgb,
            &camera,
            &Matrix4::identity(),
            &sources(&camera),
        );
        let indices = aligner.select_landmarks(&images[0], &Matrix4::identity());
        assert!(
            aligner
                .build_system(&indices, 0, &images[0], &Matrix4::identity(), None)
                .is_some()
        );
        let mut moved = Matrix4::identity();
        moved[(0, 3)] = 100.0;
        assert!(
            aligner
                .build_system(&indices, 0, &images[0], &moved, None)
                .is_none()
        );
    }

    #[test]
    fn rejected_visual_update_keeps_state_covariance_and_reference_patches() {
        let camera = camera();
        let mut filter = filter();
        let source = sources(&camera);
        let reference = render_plane(&camera, Matrix4::identity(), 0);
        let mut aligner = VisualAligner::new(config());
        aligner.process_frame(
            &mut filter,
            &[],
            0.0,
            0.0,
            0,
            &reference,
            &camera,
            &Matrix4::identity(),
            &source,
        );
        aligner.config.max_final_rmse = 0.001;
        let state = filter.state;
        let cov = filter.cov;
        let count = aligner.landmarks.len();
        let reference_patch = aligner.landmarks[0].patches[0].gray;
        let mut truth = NavState::default();
        truth.pos = Vector3::new(0.025, -0.015, 0.01);
        let current = render_plane(&camera, truth.lidar_pose(&Vector3::zeros()), 0);
        let report = aligner.process_frame(
            &mut filter,
            &[],
            0.1,
            0.1,
            6,
            &current,
            &camera,
            &Matrix4::identity(),
            &source,
        );
        assert_eq!(report.status, "rejected", "{report:?}");
        assert_eq!(filter.state, state);
        assert_eq!(filter.cov, cov);
        assert_eq!(aligner.landmarks.len(), count);
        assert_eq!(aligner.landmarks[0].patches[0].gray, reference_patch);
        assert_eq!(report.refreshed_landmarks, 0);
        assert_eq!(report.new_landmarks, 0);
    }

    #[test]
    fn old_references_refresh_after_alignment_and_landmark_storage_is_bounded() {
        let camera = camera();
        let mut filter = filter();
        let mut cfg = config();
        cfg.max_landmarks = 50;
        let mut aligner = VisualAligner::new(cfg);
        let rgb = render_plane(&camera, Matrix4::identity(), 0);
        let source = sources(&camera);
        aligner.process_frame(
            &mut filter,
            &[],
            0.0,
            0.0,
            0,
            &rgb,
            &camera,
            &Matrix4::identity(),
            &source,
        );
        assert!(aligner.landmarks.len() <= 50);
        let report = aligner.process_frame(
            &mut filter,
            &[],
            0.1,
            0.1,
            5,
            &rgb,
            &camera,
            &Matrix4::identity(),
            &source,
        );
        assert!(report.refreshed_landmarks > 0, "{report:?}");
        assert!(aligner.landmarks.len() <= 50);
        assert!(aligner.landmarks.iter().any(|p| p.created_frame == 5));
    }
}
