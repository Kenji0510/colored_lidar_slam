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

    let width = camera.width as f64;
    let height = camera.height as f64;

    let margin_ratio = (1.0 - COLOR_IMAGE_KEEP_RATIO) * 0.5;
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

    /// 周辺に十分手前の点があればtrue。深度がない画素は判定に使わない。
    /// zは直線距離ではなく、画像時刻のカメラ座標のZ。
    pub fn is_occluded(
        &self,
        u: u32,
        v: u32,
        z: f64,
        margin_m: f64,
        radius_px: u32,
    ) -> bool {
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
