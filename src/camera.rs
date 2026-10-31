use nalgebra::{Matrix3, Matrix4, Point3, Rotation3, Vector3};

use crate::types::ColoredPoint;

// 画像の縦横それぞれ、中央75%だけを着色に使用する
const COLOR_IMAGE_KEEP_RATIO: f64 = 0.75;

/// LiDAR座標の点をカメラ座標へ変換する行列。
/// camera_position_in_lidar_m は、LiDARから見たカメラ原点位置。
pub fn build_camera_from_lidar(
    camera_position_in_lidar_m: Vector3<f64>,
    roll_deg: f64,
    pitch_deg: f64,
    yaw_deg: f64,
) -> Matrix4<f64> {
    // 以前の着色コードと同じ軸変換。
    let r_nominal = Matrix3::<f64>::new(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0);

    let camera_mount = Rotation3::from_euler_angles(
        roll_deg.to_radians(),
        pitch_deg.to_radians(),
        yaw_deg.to_radians(),
    );

    let rotation = r_nominal * camera_mount.inverse().matrix();

    // p_camera = rotation * (p_lidar - camera_position)
    let translation = -(rotation * camera_position_in_lidar_m);

    let mut transform = Matrix4::<f64>::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);

    transform
}

#[derive(Debug, Clone)]
pub struct CameraIntrinsics {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    // [k1, k2, p1, p2, k3]
    pub distortion: [f64; 5],
}

impl CameraIntrinsics {
    pub fn for_image_size(width: u32, height: u32) -> anyhow::Result<Self> {
        let scale = match (width, height) {
            (1920, 1200) => 1.0,
            (960, 600) => 0.5,
            _ => anyhow::bail!("Unsupported image size: {width}x{height}"),
        };

        Ok(Self {
            width,
            height,
            fx: 1282.49623 * scale,
            fy: 1287.95761 * scale,
            cx: 928.79093 * scale,
            cy: 671.31092 * scale,
            distortion: [-0.040856, -0.004680, 0.010953, -0.002704, 0.0],
        })
    }
}

pub fn project_camera_point(point: &Point3<f64>, camera: &CameraIntrinsics) -> Option<(u32, u32)> {
    project_camera_point_with_keep_ratio(point, camera, COLOR_IMAGE_KEEP_RATIO)
}

/// 単一画像からの最終再着色では、看板の端も使えるよう画像全体を投影範囲にする。
pub fn project_camera_point_full_image(
    point: &Point3<f64>,
    camera: &CameraIntrinsics,
) -> Option<(u32, u32)> {
    project_camera_point_with_keep_ratio(point, camera, 1.0)
}

fn project_camera_point_with_keep_ratio(
    point: &Point3<f64>,
    camera: &CameraIntrinsics,
    keep_ratio: f64,
) -> Option<(u32, u32)> {
    // 非有限の座標と、カメラの後方・カメラ平面上の点を除外。
    if !point.coords.iter().all(|v| v.is_finite()) || point.z <= 0.0 {
        return None;
    }

    let x = point.x / point.z;
    let y = point.y / point.z;

    let [k1, k2, p1, p2, k3] = camera.distortion;
    let r2 = x * x + y * y;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

    let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);

    let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

    let u = (camera.fx * xd + camera.cx).round();
    let v = (camera.fy * yd + camera.cy).round();

    if !u.is_finite() || !v.is_finite() {
        return None;
    }

    let width = camera.width as f64;
    let height = camera.height as f64;

    let margin_ratio = (1.0 - keep_ratio) * 0.5;
    let margin_x = width * margin_ratio;
    let margin_y = height * margin_ratio;

    if u < margin_x || u >= width - margin_x || v < margin_y || v >= height - margin_y {
        return None;
    }

    Some((u as u32, v as u32))
}

pub fn colorize_world_points(
    lidar_points: &[Point3<f32>],
    world_from_lidar_start: &Matrix4<f64>,
    rgb_image: Option<&image::RgbImage>,
    projected_pixels: Option<&[Option<(u32, u32)>]>,
) -> anyhow::Result<Vec<ColoredPoint>> {
    if let Some(pixels) = projected_pixels {
        anyhow::ensure!(
            lidar_points.len() == pixels.len(),
            "Point/pixel count mismatch: {} vs {}",
            lidar_points.len(),
            pixels.len(),
        );
    }

    let points = lidar_points
        .iter()
        .enumerate()
        .filter_map(|(index, point)| {
            let position = world_from_lidar_start
                .transform_point(&point.cast::<f64>())
                .cast::<f32>();

            // XYZが非有限の点だけ除外する。
            if !position.coords.iter().all(|v| v.is_finite()) {
                return None;
            }

            let pixel = projected_pixels.and_then(|pixels| pixels[index]);

            let rgb = match (rgb_image, pixel) {
                (Some(image), Some((u, v))) => image.get_pixel_checked(u, v).map(|pixel| pixel.0),
                _ => None,
            };

            Some(ColoredPoint { position, rgb })
        })
        .collect();

    Ok(points)
}

/// 画像撮影時のカメラ座標で、画素ごとの最小Zを保持する。
/// COLOR_IMAGE_KEEP_RATIOと同じ投影領域を使用する。
pub struct ColorDepthMap {
    width: u32,
    height: u32,
    nearest_z: Vec<f64>,
}

impl ColorDepthMap {
    pub fn build(
        lidar_points: &[Point3<f32>],
        camera_from_lidar: &Matrix4<f64>,
        camera: &CameraIntrinsics,
    ) -> Self {
        let mut map = Self {
            width: camera.width,
            height: camera.height,
            nearest_z: vec![f64::INFINITY; camera.width as usize * camera.height as usize],
        };

        for point in lidar_points {
            let camera_point = camera_from_lidar.transform_point(&point.cast::<f64>());
            let Some((u, v)) = project_camera_point(&camera_point, camera) else {
                continue;
            };

            let index = v as usize * map.width as usize + u as usize;
            map.nearest_z[index] = map.nearest_z[index].min(camera_point.z);
        }

        map
    }

    /// 単一画像での再着色用。通常の中央75%制限を適用せず深度を作る。
    pub fn build_full_image(
        lidar_points: &[Point3<f32>],
        camera_from_lidar: &Matrix4<f64>,
        camera: &CameraIntrinsics,
    ) -> Self {
        let mut map = Self {
            width: camera.width,
            height: camera.height,
            nearest_z: vec![f64::INFINITY; camera.width as usize * camera.height as usize],
        };

        for point in lidar_points {
            let camera_point = camera_from_lidar.transform_point(&point.cast::<f64>());
            let Some((u, v)) = project_camera_point_full_image(&camera_point, camera) else {
                continue;
            };

            let index = v as usize * map.width as usize + u as usize;
            map.nearest_z[index] = map.nearest_z[index].min(camera_point.z);
        }

        map
    }

    /// 周辺に十分手前の点があればtrue。深度がない画素は判定に使わない。
    /// zは直線距離ではなく、画像時刻のカメラ座標のZ。
    pub fn is_occluded(&self, u: u32, v: u32, z: f64, margin_m: f64, radius_px: u32) -> bool {
        assert!(margin_m.is_finite() && margin_m >= 0.0);

        if u >= self.width || v >= self.height || !z.is_finite() || z <= 0.0 {
            return false;
        }

        let min_u = u.saturating_sub(radius_px);
        let max_u = u.saturating_add(radius_px).min(self.width - 1);
        let min_v = v.saturating_sub(radius_px);
        let max_v = v.saturating_add(radius_px).min(self.height - 1);

        for y in min_v..=max_v {
            for x in min_u..=max_u {
                let nearest_z = self.nearest_z[y as usize * self.width as usize + x as usize];
                if nearest_z.is_finite() && z > nearest_z + margin_m {
                    return true;
                }
            }
        }

        false
    }
}

/// 着色に使用する中央75%の範囲で、Laplacianの分散を計算する。
/// 値が大きいほど、輪郭が鮮明な傾向がある。
pub fn color_image_sharpness(rgb_image: &image::RgbImage) -> f64 {
    if rgb_image.width() == 0 || rgb_image.height() == 0 {
        return 0.0;
    }

    // 1920×1200と960×600で比較条件を揃える。
    // 判定用画像だけ縮小し、着色用の元画像はそのまま使う。
    let resized =
        image::imageops::resize(rgb_image, 960, 600, image::imageops::FilterType::Triangle);

    let gray = image::DynamicImage::ImageRgb8(resized).to_luma8();
    let (width, height) = gray.dimensions();

    // 既存の着色範囲と同じ、縦横それぞれ中央75%。
    let margin_ratio = (1.0 - COLOR_IMAGE_KEEP_RATIO) * 0.5;
    let margin_x = (width as f64 * margin_ratio).ceil() as u32;
    let margin_y = (height as f64 * margin_ratio).ceil() as u32;

    // Laplacianが上下左右を参照するので、さらに1画素内側を使う。
    let start_x = margin_x + 1;
    let end_x = width - margin_x - 1;
    let start_y = margin_y + 1;
    let end_y = height - margin_y - 1;

    let mut count = 0.0_f64;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;

    for y in start_y..end_y {
        for x in start_x..end_x {
            let center = gray.get_pixel(x, y).0[0] as f64;
            let left = gray.get_pixel(x - 1, y).0[0] as f64;
            let right = gray.get_pixel(x + 1, y).0[0] as f64;
            let up = gray.get_pixel(x, y - 1).0[0] as f64;
            let down = gray.get_pixel(x, y + 1).0[0] as f64;

            // 上下左右と中心画素からLaplacianを計算する。
            let laplacian = left + right + up + down - 4.0 * center;

            // Welford法で分散を集計する。
            count += 1.0;
            let delta = laplacian - mean;
            mean += delta / count;
            m2 += delta * (laplacian - mean);
        }
    }

    if count > 0.0 { m2 / count } else { 0.0 }
}

/// 鮮明な1枚の画像と、その撮影時刻の姿勢・深度を固定して保持する。
/// 最後に完成したGlobalMapのcell.meanを、この画像へ再投影する。
pub struct SingleImageColorReference {
    rgb_image: image::RgbImage,
    camera_from_world: Matrix4<f64>,
    intrinsics: CameraIntrinsics,
    depth_map: ColorDepthMap,
    /// [x_min, y_min, x_max, y_max]。右端・下端は含まない。
    roi: Option<[u32; 4]>,
}

impl SingleImageColorReference {
    pub fn capture(
        rgb_image: &image::RgbImage,
        camera_from_world: &Matrix4<f64>,
        world_from_lidar_start: &Matrix4<f64>,
        deskewed_points: &[Point3<f32>],
        roi: Option<[u32; 4]>,
    ) -> anyhow::Result<Self> {
        let intrinsics =
            CameraIntrinsics::for_image_size(rgb_image.width(), rgb_image.height())?;

        anyhow::ensure!(
            camera_from_world.iter().all(|value| value.is_finite())
                && world_from_lidar_start.iter().all(|value| value.is_finite()),
            "Single-image recoloring requires finite camera/LiDAR poses"
        );

        if let Some([x_min, y_min, x_max, y_max]) = roi {
            anyhow::ensure!(
                x_min < x_max && y_min < y_max
                    && x_max <= intrinsics.width && y_max <= intrinsics.height,
                "Recolor ROI {:?} must be inside the original {}x{} image",
                roi,
                intrinsics.width,
                intrinsics.height,
            );
        }

        // 参照フレームの全点を使う。GlobalMap用の平面フィルター前の点群であること。
        let camera_from_lidar_start = camera_from_world * world_from_lidar_start;
        let depth_map = ColorDepthMap::build_full_image(
            deskewed_points,
            &camera_from_lidar_start,
            &intrinsics,
        );

        Ok(Self {
            rgb_image: rgb_image.clone(),
            camera_from_world: camera_from_world.clone(),
            intrinsics,
            depth_map,
            roi,
        })
    }

    /// 呼び出し元は、最終座標cell.meanを渡す。
    /// 範囲外・遮蔽候補はNone。過去フレームのRGBへフォールバックしない。
    pub fn color_for_world_point(
        &self,
        world_point: &Point3<f32>,
        occlusion_margin_m: f64,
        occlusion_radius_px: u32,
    ) -> Option<[u8; 3]> {
        let camera_point = self
            .camera_from_world
            .transform_point(&world_point.cast::<f64>());
        let (u, v) = project_camera_point_full_image(&camera_point, &self.intrinsics)?;

        if let Some([x_min, y_min, x_max, y_max]) = self.roi {
            if u < x_min || u >= x_max || v < y_min || v >= y_max {
                return None;
            }
        }

        if self.depth_map.is_occluded(
            u, v, camera_point.z, occlusion_margin_m, occlusion_radius_px,
        ) {
            return None;
        }

        self.rgb_image.get_pixel_checked(u, v).map(|pixel| pixel.0)
    }
}

#[cfg(test)]
mod single_image_recolor_tests {
    use super::*;

    fn image() -> image::RgbImage {
        image::RgbImage::from_pixel(960, 600, image::Rgb([10, 20, 30]))
    }

    fn capture(points: &[Point3<f32>], roi: Option<[u32; 4]>) -> SingleImageColorReference {
        SingleImageColorReference::capture(
            &image(), &Matrix4::identity(), &Matrix4::identity(), points, roi,
        ).unwrap()
    }

    #[test]
    fn samples_the_final_world_coordinate_instead_of_the_original_observation() {
        let mut image = image();
        image.put_pixel(464, 336, image::Rgb([255, 0, 0]));
        image.put_pixel(468, 336, image::Rgb([0, 255, 0]));
        let original = Point3::new(0.0_f32, 0.0, 5.0);
        let reference = SingleImageColorReference::capture(
            &image, &Matrix4::identity(), &Matrix4::identity(), &[original], None,
        ).unwrap();
        assert_eq!(reference.color_for_world_point(&original, 0.2, 0), Some([255, 0, 0]));
        let final_mean = Point3::new(0.03_f32, 0.0, 5.0);
        assert_eq!(reference.color_for_world_point(&final_mean, 0.2, 0), Some([0, 255, 0]));
    }

    #[test]
    fn captures_the_image_pixels_and_pose_by_value() {
        let mut image = image();
        let mut camera_from_world = Matrix4::identity();
        camera_from_world[(0, 3)] = -1.0;
        let mut world_from_lidar = Matrix4::identity();
        world_from_lidar[(0, 3)] = 1.0;
        image.put_pixel(464, 336, image::Rgb([1, 2, 3]));
        let reference = SingleImageColorReference::capture(
            &image, &camera_from_world, &world_from_lidar,
            &[Point3::new(0.0_f32, 0.0, 5.0)], None,
        ).unwrap();
        image.put_pixel(464, 336, image::Rgb([9, 9, 9]));
        camera_from_world[(0, 3)] = 100.0;
        assert_eq!(
            reference.color_for_world_point(&Point3::new(1.0, 0.0, 5.0), 0.2, 0),
            Some([1, 2, 3]),
        );
    }

    #[test]
    fn rejects_background_using_the_closest_depth_in_either_input_order() {
        let front = Point3::new(0.0_f32, 0.0, 3.0);
        let back = Point3::new(0.0_f32, 0.0, 8.0);
        for points in [[front, back], [back, front]] {
            let reference = capture(&points, None);
            assert_eq!(reference.color_for_world_point(&front, 0.2, 0), Some([10, 20, 30]));
            assert_eq!(reference.color_for_world_point(&back, 0.2, 0), None);
        }
    }

    #[test]
    fn roi_uses_inclusive_left_top_and_exclusive_right_bottom() {
        let point = Point3::new(0.0_f32, 0.0, 5.0);
        assert_eq!(
            capture(&[point], Some([464, 336, 465, 337]))
                .color_for_world_point(&point, 0.2, 0),
            Some([10, 20, 30]),
        );
        assert_eq!(
            capture(&[point], Some([0, 0, 464, 600]))
                .color_for_world_point(&point, 0.2, 0), None,
        );
        assert_eq!(
            capture(&[point], Some([0, 0, 960, 336]))
                .color_for_world_point(&point, 0.2, 0), None,
        );
    }

    #[test]
    fn reference_projection_and_depth_cover_the_full_image() {
        let intrinsics = CameraIntrinsics::for_image_size(960, 600).unwrap();
        let front = Point3::new(-3.2_f32, 0.0, 5.0);
        assert_eq!(project_camera_point(&front.cast::<f64>(), &intrinsics), None);
        assert!(project_camera_point_full_image(&front.cast::<f64>(), &intrinsics).is_some());
        let reference = capture(&[front], None);
        assert_eq!(reference.color_for_world_point(&front, 0.2, 0), Some([10, 20, 30]));
        assert_eq!(
            reference.color_for_world_point(&Point3::new(-5.12, 0.0, 8.0), 0.2, 0), None,
        );
    }

    #[test]
    fn rejects_behind_camera_non_finite_and_out_of_frame_points() {
        let reference = capture(&[], None);
        for point in [
            Point3::new(0.0_f32, 0.0, -1.0),
            Point3::new(f32::NAN, 0.0, 5.0),
            Point3::new(1000.0, 0.0, 1.0),
        ] {
            assert_eq!(reference.color_for_world_point(&point, 0.2, 0), None);
        }
        let camera = CameraIntrinsics::for_image_size(960, 600).unwrap();
        assert_eq!(
            project_camera_point_full_image(&Point3::new(f64::MAX / 2.0, 0.0, 1.0), &camera), None,
        );
    }

    #[test]
    fn unknown_depth_retains_the_existing_allow_policy() {
        let reference = capture(&[], None);
        assert_eq!(
            reference.color_for_world_point(&Point3::new(0.0, 0.0, 5.0), 0.2, 2),
            Some([10, 20, 30]),
        );
    }

    #[test]
    fn neighborhood_radius_controls_conservative_occlusion_rejection() {
        let foreground = Point3::new(0.009_f32, 0.0, 3.0);
        let candidate = Point3::new(0.0_f32, 0.0, 8.0);
        let reference = capture(&[foreground], None);
        assert_eq!(reference.color_for_world_point(&candidate, 0.2, 0), Some([10, 20, 30]));
        assert_eq!(reference.color_for_world_point(&candidate, 0.2, 2), None);
    }

    #[test]
    fn rejects_invalid_roi_and_non_finite_pose() {
        for roi in [[0, 0, 961, 600], [10, 10, 10, 20], [20, 20, 10, 30]] {
            assert!(SingleImageColorReference::capture(
                &image(), &Matrix4::identity(), &Matrix4::identity(), &[], Some(roi),
            ).is_err());
        }
        let mut bad_pose = Matrix4::identity();
        bad_pose[(0, 3)] = f64::NAN;
        assert!(SingleImageColorReference::capture(
            &image(), &bad_pose, &Matrix4::identity(), &[], None,
        ).is_err());
    }
}
