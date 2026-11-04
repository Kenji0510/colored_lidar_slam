//! 平面上の画像対応から、元画像上の小さな並進ずれだけを推定する。
//! RGBは混ぜない。XYZ・カメラ姿勢・ESIKFは更新しない。

use crate::camera::{CameraIntrinsics, camera_ray_for_pixel, project_camera_point_continuous};
use anyhow::{Result, ensure};
use image::GrayImage;
use nalgebra::{Matrix4, Point3, Vector3};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct PatchAlignmentConfig {
    pub enabled: bool,
    /// 元画像の画素単位。探索はさらに外側1画素を含め、範囲外の解を検出する。
    pub max_shift_px: i32,
    pub patch_half_width: i32,
    pub sample_step_px: f64,
    pub max_patches: usize,
    pub min_matches: usize,
    pub min_samples: usize,
    pub min_stddev: f64,
    pub min_ncc: f64,
    /// 最良解の周囲1画素を除いた次点との差。
    pub min_peak_gap: f64,
    pub min_improvement: f64,
    pub consensus_radius_px: f64,
    /// テクスチャのある評価パッチのうち、共通のずれに賛成する最低割合。
    pub min_consensus_fraction: f64,
}

impl Default for PatchAlignmentConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_shift_px: 4,
            patch_half_width: 4,
            sample_step_px: 2.0,
            max_patches: 24,
            min_matches: 3,
            min_samples: 40,
            min_stddev: 8.0,
            min_ncc: 0.85,
            min_peak_gap: 0.04,
            min_improvement: 0.02,
            consensus_radius_px: 1.0,
            min_consensus_fraction: 0.6,
        }
    }
}

impl PatchAlignmentConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=12).contains(&self.max_shift_px),
            "alignment max_shift_px must be 1..=12"
        );
        ensure!(
            (2..=12).contains(&self.patch_half_width),
            "invalid alignment patch size"
        );
        ensure!(
            self.sample_step_px.is_finite()
                && self.sample_step_px >= 1.0
                && self.sample_step_px <= 4.0,
            "alignment sample_step_px must be 1..=4"
        );
        ensure!(
            (3..=64).contains(&self.max_patches)
                && self.min_matches >= 3
                && self.min_matches <= self.max_patches,
            "invalid alignment patch count"
        );
        let area = (2 * self.patch_half_width + 1).pow(2) as usize;
        ensure!(
            self.min_samples >= 9 && self.min_samples <= area,
            "invalid alignment min_samples"
        );
        ensure!(
            self.min_stddev.is_finite() && self.min_stddev > 0.0,
            "invalid alignment texture threshold"
        );
        ensure!(
            self.min_ncc.is_finite() && (0.0..=1.0).contains(&self.min_ncc),
            "invalid alignment min_ncc"
        );
        ensure!(
            self.min_peak_gap.is_finite() && (0.0..=1.0).contains(&self.min_peak_gap),
            "invalid alignment peak gap"
        );
        ensure!(
            self.min_improvement.is_finite() && (0.0..=1.0).contains(&self.min_improvement),
            "invalid alignment gain"
        );
        ensure!(
            self.consensus_radius_px.is_finite()
                && self.consensus_radius_px > 0.0
                && self.consensus_radius_px <= 2.0,
            "invalid alignment consensus radius"
        );
        ensure!(
            self.min_consensus_fraction.is_finite()
                && self.min_consensus_fraction > 0.5
                && self.min_consensus_fraction <= 1.0,
            "alignment consensus fraction must be in (0.5,1]"
        );
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct WarpSample {
    pub world: Point3<f64>,
    pub reference: [f64; 2],
    pub target: [f64; 2],
}

/// 基準画像の各光線と局所平面を交差させ、対象画像へ投影する。
/// 歪みを含む投影・逆投影を使うため、元画像を事前に変形する必要はない。
pub fn plane_patch_samples(
    center: &Point3<f64>,
    normal: &Vector3<f64>,
    reference_from_world: &Matrix4<f64>,
    target_from_world: &Matrix4<f64>,
    reference_camera: &CameraIntrinsics,
    target_camera: &CameraIntrinsics,
    config: &PatchAlignmentConfig,
) -> Vec<WarpSample> {
    let Some(world_from_reference) = reference_from_world.try_inverse() else {
        return vec![];
    };
    let Some((u, v)) = project_camera_point_continuous(
        &reference_from_world.transform_point(center),
        reference_camera,
    ) else {
        return vec![];
    };
    let origin = world_from_reference.transform_point(&Point3::origin());
    let plane_distance = normal.dot(&(center - origin));
    let mut result = Vec::new();
    for dy in -config.patch_half_width..=config.patch_half_width {
        for dx in -config.patch_half_width..=config.patch_half_width {
            let pixel = [
                u + dx as f64 * config.sample_step_px,
                v + dy as f64 * config.sample_step_px,
            ];
            let Some(ray) = camera_ray_for_pixel(pixel[0], pixel[1], reference_camera) else {
                continue;
            };
            let direction = world_from_reference.fixed_view::<3, 3>(0, 0) * ray;
            let denominator = normal.dot(&direction);
            if !denominator.is_finite() || denominator.abs() < 1e-8 {
                continue;
            }
            let distance = plane_distance / denominator;
            if !distance.is_finite() || distance <= 0.0 {
                continue;
            }
            let world = origin + direction * distance;
            if let Some((tu, tv)) = project_camera_point_continuous(
                &target_from_world.transform_point(&world),
                target_camera,
            ) {
                result.push(WarpSample {
                    world,
                    reference: pixel,
                    target: [tu, tv],
                });
            }
        }
    }
    result
}

#[derive(Debug, Clone, Serialize)]
pub struct AlignmentEstimate {
    pub accepted: bool,
    /// 対象画像から色を取る座標へ加算する。基準画像側は0で固定。
    pub shift_px: [i32; 2],
    pub reason: &'static str,
    pub textured_patches: usize,
    pub matched_patches: usize,
    pub inlier_patches: usize,
    pub mean_ncc: Option<f64>,
}

impl AlignmentEstimate {
    pub fn rejected(reason: &'static str) -> Self {
        Self {
            accepted: false,
            shift_px: [0, 0],
            reason,
            textured_patches: 0,
            matched_patches: 0,
            inlier_patches: 0,
            mean_ncc: None,
        }
    }
}

fn gray_at(image: &GrayImage, pixel: [f64; 2]) -> Option<f64> {
    let (x, y) = (pixel[0], pixel[1]);
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x + 1.0 >= image.width() as f64
        || y + 1.0 >= image.height() as f64
    {
        return None;
    }
    let (ix, iy) = (x.floor() as u32, y.floor() as u32);
    let (a, b) = (x - ix as f64, y - iy as f64);
    let get = |u, v| image.get_pixel(u, v).0[0] as f64;
    Some(
        (1.0 - b) * ((1.0 - a) * get(ix, iy) + a * get(ix + 1, iy))
            + b * ((1.0 - a) * get(ix, iy + 1) + a * get(ix + 1, iy + 1)),
    )
}

pub fn reference_texture(image: &GrayImage, patch: &[WarpSample]) -> f64 {
    let values: Vec<_> = patch
        .iter()
        .filter_map(|s| gray_at(image, s.reference))
        .collect();
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
}

pub fn shifted_pixel(
    pixel: [u32; 2],
    shift: [i32; 2],
    width: u32,
    height: u32,
) -> Option<[u32; 2]> {
    let u = pixel[0] as i64 + shift[0] as i64;
    let v = pixel[1] as i64 + shift[1] as i64;
    if u < 0 || v < 0 || u >= width as i64 || v >= height as i64 {
        return None;
    }
    Some([u as u32, v as u32])
}

fn correlation(a: &[f64], b: &[f64], min_stddev: f64) -> Option<f64> {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut aa, mut bb, mut ab) = (0.0, 0.0, 0.0);
    for (&a, &b) in a.iter().zip(b) {
        aa += (a - ma).powi(2);
        bb += (b - mb).powi(2);
        ab += (a - ma) * (b - mb);
    }
    if aa / n < min_stddev.powi(2) || bb / n < min_stddev.powi(2) {
        return None;
    }
    Some((ab / (aa * bb).sqrt()).clamp(-1.0, 1.0))
}

/// 各パッチで一意な小並進を探索し、複数パッチの合意がある場合だけ採用する。
pub fn estimate_translation(
    reference: &GrayImage,
    target: &GrayImage,
    patches: &[Vec<WarpSample>],
    config: &PatchAlignmentConfig,
) -> AlignmentEstimate {
    let mut result = AlignmentEstimate::rejected("insufficient_matches");
    let radius = config.max_shift_px + 1;
    let mut matches: Vec<([i32; 2], f64)> = Vec::new();
    for patch in patches.iter().take(config.max_patches) {
        // 全探索位置で同じサンプルを比較し、画面端で評価対象が変わるのを防ぐ。
        let samples: Vec<_> = patch
            .iter()
            .filter_map(|s| {
                let value = gray_at(reference, s.reference)?;
                let p = s.target;
                gray_at(target, [p[0] - radius as f64, p[1] - radius as f64])?;
                gray_at(target, [p[0] + radius as f64, p[1] + radius as f64])?;
                Some((value, p))
            })
            .collect();
        if samples.len() < config.min_samples {
            continue;
        }
        let a: Vec<_> = samples.iter().map(|x| x.0).collect();
        if correlation(&a, &a, config.min_stddev).is_none() {
            continue;
        }
        result.textured_patches += 1;
        let mut scores = Vec::new();
        let mut b = vec![0.0; samples.len()];
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                for (index, (_, pixel)) in samples.iter().enumerate() {
                    b[index] =
                        gray_at(target, [pixel[0] + dx as f64, pixel[1] + dy as f64]).unwrap();
                }
                if let Some(score) = correlation(&a, &b, config.min_stddev) {
                    scores.push(([dx, dy], score));
                }
            }
        }
        scores.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| (a.0[0].abs() + a.0[1].abs()).cmp(&(b.0[0].abs() + b.0[1].abs())))
                .then_with(|| a.0.cmp(&b.0))
        });
        let Some(&(best, score)) = scores.first() else {
            continue;
        };
        let second = scores
            .iter()
            .filter(|(shift, _)| (shift[0] - best[0]).abs() > 1 || (shift[1] - best[1]).abs() > 1)
            .map(|x| x.1)
            .fold(-1.0_f64, f64::max);
        let baseline = scores.iter().find(|x| x.0 == [0, 0]).map_or(-1.0, |x| x.1);
        if best.iter().any(|x| x.abs() > config.max_shift_px)
            || score < config.min_ncc
            || score - second < config.min_peak_gap
            || (best != [0, 0] && score - baseline < config.min_improvement)
        {
            continue;
        }
        matches.push((best, score));
    }
    result.matched_patches = matches.len();
    if matches.len() < config.min_matches {
        return result;
    }
    let radius_sq = config.consensus_radius_px.powi(2);
    let distance_sq =
        |a: [i32; 2], b: [i32; 2]| ((a[0] - b[0]).pow(2) + (a[1] - b[1]).pow(2)) as f64;
    let mut best_inliers = Vec::new();
    let mut best_score = -1.0;
    for &(candidate, _) in &matches {
        let inliers: Vec<_> = matches
            .iter()
            .copied()
            .filter(|(shift, _)| distance_sq(*shift, candidate) <= radius_sq)
            .collect();
        let score = inliers.iter().map(|x| x.1).sum::<f64>();
        if inliers.len() > best_inliers.len()
            || (inliers.len() == best_inliers.len() && score > best_score)
        {
            best_inliers = inliers;
            best_score = score;
        }
    }
    let median = |axis: usize| {
        let mut values: Vec<_> = best_inliers.iter().map(|x| x.0[axis]).collect();
        values.sort_unstable();
        values[values.len() / 2]
    };
    let shift = [median(0), median(1)];
    let inliers: Vec<_> = matches
        .iter()
        .filter(|x| distance_sq(x.0, shift) <= radius_sq)
        .collect();
    result.inlier_patches = inliers.len();
    if inliers.len() < config.min_matches
        || (inliers.len() as f64 / result.textured_patches as f64) < config.min_consensus_fraction
    {
        result.reason = "no_consensus";
        return result;
    }
    result.accepted = true;
    result.shift_px = shift;
    result.reason = if shift == [0, 0] {
        "already_aligned"
    } else {
        "accepted"
    };
    result.mean_ncc = Some(inliers.iter().map(|x| x.1).sum::<f64>() / inliers.len() as f64);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;
    use nalgebra::Rotation3;

    fn texture(x: u32, y: u32) -> u8 {
        let mut v = (x as u64).wrapping_mul(0x9e3779b97f4a7c15)
            ^ (y as u64).wrapping_mul(0xbf58476d1ce4e5b9);
        v = (v ^ (v >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        v = (v ^ (v >> 27)).wrapping_mul(0x94d049bb133111eb);
        (v ^ (v >> 31)) as u8
    }
    fn image() -> GrayImage {
        GrayImage::from_fn(160, 160, |x, y| Luma([texture(x, y)]))
    }
    fn shifted(source: &GrayImage, shift: [i32; 2], gain: f64, bias: f64) -> GrayImage {
        GrayImage::from_fn(source.width(), source.height(), |x, y| {
            let p = shifted_pixel(
                [x, y],
                [-shift[0], -shift[1]],
                source.width(),
                source.height(),
            );
            Luma([p.map_or(0, |p| {
                (source.get_pixel(p[0], p[1]).0[0] as f64 * gain + bias)
                    .round()
                    .clamp(0.0, 255.0) as u8
            })])
        })
    }
    fn patches() -> Vec<Vec<WarpSample>> {
        [
            [35, 35],
            [75, 35],
            [115, 35],
            [35, 100],
            [75, 100],
            [115, 100],
        ]
        .into_iter()
        .map(|center| {
            (-4..=4)
                .flat_map(|y| {
                    (-4..=4).map(move |x| {
                        let p = [(center[0] + 2 * x) as f64, (center[1] + 2 * y) as f64];
                        WarpSample {
                            world: Point3::origin(),
                            reference: p,
                            target: p,
                        }
                    })
                })
                .collect()
        })
        .collect()
    }

    #[test]
    fn known_translation_and_exposure_change_recover_the_sampling_direction() {
        let reference = image();
        let target = shifted(&reference, [2, -1], 0.7, 30.0);
        let estimate = estimate_translation(
            &reference,
            &target,
            &patches(),
            &PatchAlignmentConfig::default(),
        );
        assert!(estimate.accepted, "{estimate:?}");
        assert_eq!(estimate.shift_px, [2, -1]);
        assert_eq!(estimate.inlier_patches, 6);
        assert!(estimate.mean_ncc.unwrap() > 0.99);
    }

    #[test]
    fn an_already_aligned_image_does_not_acquire_an_offset() {
        let image = image();
        let estimate =
            estimate_translation(&image, &image, &patches(), &PatchAlignmentConfig::default());
        assert!(estimate.accepted, "{estimate:?}");
        assert_eq!(estimate.shift_px, [0, 0]);
        assert_eq!(estimate.reason, "already_aligned");
    }

    #[test]
    fn flat_repeated_and_unrelated_images_are_not_corrected() {
        let flat = GrayImage::from_pixel(160, 160, Luma([100]));
        let repeated = GrayImage::from_fn(160, 160, |x, y| {
            Luma([if (x / 2 + y / 2) % 2 == 0 { 20 } else { 230 }])
        });
        let reference = image();
        let unrelated = GrayImage::from_fn(160, 160, |x, y| Luma([texture(x + 300, y + 300)]));
        for (a, b) in [
            (&flat, &flat),
            (&repeated, &repeated),
            (&reference, &unrelated),
        ] {
            let estimate = estimate_translation(a, b, &patches(), &PatchAlignmentConfig::default());
            assert!(!estimate.accepted, "{estimate:?}");
            assert_eq!(estimate.shift_px, [0, 0]);
        }
    }

    #[test]
    fn out_of_range_and_missing_overlap_are_rejected() {
        let reference = image();
        let target = shifted(&reference, [5, 0], 1.0, 0.0);
        let cfg = PatchAlignmentConfig::default();
        assert!(!estimate_translation(&reference, &target, &patches(), &cfg).accepted);
        let target = GrayImage::new(10, 10);
        assert!(!estimate_translation(&reference, &target, &patches(), &cfg).accepted);
    }

    #[test]
    fn multiple_patches_require_a_common_translation_and_tolerate_one_outlier() {
        let reference = image();
        let a = shifted(&reference, [2, -1], 1.0, 0.0);
        let b = shifted(&reference, [-2, 1], 1.0, 0.0);
        let split = GrayImage::from_fn(160, 160, |x, y| {
            *if y < 70 {
                a.get_pixel(x, y)
            } else {
                b.get_pixel(x, y)
            }
        });
        let cfg = PatchAlignmentConfig::default();
        let estimate = estimate_translation(&reference, &split, &patches(), &cfg);
        assert!(!estimate.accepted, "{estimate:?}");
        assert_eq!(estimate.reason, "no_consensus");
        let one_outlier = GrayImage::from_fn(160, 160, |x, y| {
            *if x > 95 && y > 70 {
                b.get_pixel(x, y)
            } else {
                a.get_pixel(x, y)
            }
        });
        let estimate = estimate_translation(&reference, &one_outlier, &patches(), &cfg);
        assert!(estimate.accepted, "{estimate:?}");
        assert_eq!(estimate.shift_px, [2, -1]);
        assert_eq!(estimate.inlier_patches, 5);
    }

    #[test]
    fn distortion_inverse_roundtrips_on_both_supported_image_sizes() {
        for (width, height) in [(960, 600), (1920, 1200)] {
            let camera = CameraIntrinsics::for_image_size(width, height).unwrap();
            for p in [
                [0.0, 0.0],
                [width as f64 - 1.0, height as f64 - 1.0],
                [camera.cx, camera.cy],
                [50.25, 200.75],
            ] {
                let ray = camera_ray_for_pixel(p[0], p[1], &camera).unwrap();
                let pixel = project_camera_point_continuous(&Point3::from(ray), &camera).unwrap();
                assert!((pixel.0 - p[0]).abs() < 1e-3 && (pixel.1 - p[1]).abs() < 1e-3);
            }
        }
    }

    fn camera() -> CameraIntrinsics {
        CameraIntrinsics {
            width: 160,
            height: 160,
            fx: 120.0,
            fy: 120.0,
            cx: 80.0,
            cy: 80.0,
            distortion: [0.0; 5],
        }
    }

    #[test]
    fn plane_warp_accounts_for_known_viewpoint_displacement() {
        let camera = camera();
        let mut target = Matrix4::identity();
        target[(0, 3)] = -0.25;
        let samples = plane_patch_samples(
            &Point3::new(0.0, 0.0, 5.0),
            &Vector3::z(),
            &Matrix4::identity(),
            &target,
            &camera,
            &camera,
            &PatchAlignmentConfig::default(),
        );
        assert_eq!(samples.len(), 81);
        for sample in samples {
            assert!((sample.world.z - 5.0).abs() < 1e-10);
            assert!((sample.target[0] - sample.reference[0] + 6.0).abs() < 1e-10);
            assert!((sample.target[1] - sample.reference[1]).abs() < 1e-10);
        }
    }

    #[test]
    fn oblique_plane_different_camera_pose_and_pixel_error_are_separated() {
        let camera = camera();
        let cfg = PatchAlignmentConfig::default();
        let center = Point3::new(0.0, 0.0, 5.0);
        let normal = Vector3::new(0.3, -0.2, 1.0).normalize();
        let reference_pose = Matrix4::identity();
        let mut target_pose = Rotation3::from_euler_angles(0.02, -0.03, 0.015).to_homogeneous();
        target_pose[(0, 3)] = -0.15;
        target_pose[(1, 3)] = 0.1;
        target_pose[(2, 3)] = 0.2;
        let render = |pose: &Matrix4<f64>, shift: [f64; 2]| {
            let inverse = pose.try_inverse().unwrap();
            let origin = inverse.transform_point(&Point3::origin());
            GrayImage::from_fn(camera.width, camera.height, |u, v| {
                let ray = camera_ray_for_pixel(u as f64 - shift[0], v as f64 - shift[1], &camera)
                    .unwrap();
                let direction = inverse.fixed_view::<3, 3>(0, 0) * ray;
                let p =
                    origin + direction * (normal.dot(&(center - origin)) / normal.dot(&direction));
                let value = 125.0
                    + 35.0 * (8.0 * p.x + 3.0 * p.y).sin()
                    + 35.0 * (5.0 * p.x - 10.0 * p.y).cos()
                    + 20.0 * (15.0 * p.x + 13.0 * p.y).sin();
                Luma([value.round().clamp(0.0, 255.0) as u8])
            })
        };
        let reference = render(&reference_pose, [0.0, 0.0]);
        let target = render(&target_pose, [2.0, -1.0]);
        let mut patches = Vec::new();
        for y in [-0.8, 0.0, 0.8] {
            for x in [-0.8, 0.0, 0.8] {
                let p = Point3::new(x, y, 5.0 - (normal.x * x + normal.y * y) / normal.z);
                patches.push(plane_patch_samples(
                    &p,
                    &normal,
                    &reference_pose,
                    &target_pose,
                    &camera,
                    &camera,
                    &cfg,
                ));
            }
        }
        let estimate = estimate_translation(&reference, &target, &patches, &cfg);
        assert!(estimate.accepted, "{estimate:?}");
        assert_eq!(estimate.shift_px, [2, -1]);
    }

    #[test]
    fn sampling_at_image_edges_never_wraps_coordinates() {
        assert_eq!(shifted_pixel([0, 0], [-1, 0], 960, 600), None);
        assert_eq!(shifted_pixel([959, 599], [1, 0], 960, 600), None);
        assert_eq!(
            shifted_pixel([100, 100], [2, -1], 960, 600),
            Some([102, 99])
        );
    }
}
