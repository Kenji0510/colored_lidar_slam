use nalgebra::{Matrix3, Matrix4, Point3, Rotation3, Vector3};

use crate::types::ColoredPoint;

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

    if !u.is_finite()
        || !v.is_finite()
        || u < 0.0
        || v < 0.0
        || u >= camera.width as f64
        || v >= camera.height as f64
    {
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
