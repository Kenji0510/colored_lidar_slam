use nalgebra::{Matrix2, Matrix3, Matrix4, Point3, Rotation3, SMatrix, Vector2, Vector3};

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
    // LiDAR(x前・y左・z上) → カメラ(x右・y下・z前) の標準的な軸変換。
    // x_cam = -y_lidar, y_cam = -z_lidar, z_cam = x_lidar (画像の反転なし)
    let r_nominal = Matrix3::<f64>::new(0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0);

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

/// Brown-Conrady distortion and its analytic derivative in normalized coordinates.
fn distort_with_jacobian(x: f64, y: f64, d: [f64; 5]) -> (Vector2<f64>, Matrix2<f64>) {
    let [k1, k2, p1, p2, k3] = d;
    let r2 = x * x + y * y;
    let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    let slope = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r2 * r2;
    let rx = 2.0 * x * slope;
    let ry = 2.0 * y * slope;
    let distorted = Vector2::new(
        x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
        y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
    );
    let jacobian = Matrix2::new(
        radial + x * rx + 2.0 * p1 * y + 6.0 * p2 * x,
        x * ry + 2.0 * p1 * x + 2.0 * p2 * y,
        y * rx + 2.0 * p1 * x + 2.0 * p2 * y,
        radial + y * ry + 6.0 * p1 * y + 2.0 * p2 * x,
    );
    (distorted, jacobian)
}

/// Subpixel projection for direct alignment. This does not apply the RGB
/// central-75% crop: callers validate the entire patch against image bounds.
pub fn project_camera_point_subpixel(
    point: &Point3<f64>,
    camera: &CameraIntrinsics,
) -> Option<(Vector2<f64>, SMatrix<f64, 2, 3>)> {
    if !point.coords.iter().all(|v| v.is_finite()) || point.z <= 1e-6 {
        return None;
    }
    let z_inv = 1.0 / point.z;
    let x = point.x * z_inv;
    let y = point.y * z_inv;
    let (d, jd) = distort_with_jacobian(x, y, camera.distortion);
    let pixel = Vector2::new(camera.fx * d.x + camera.cx, camera.fy * d.y + camera.cy);
    let normalized_jacobian =
        SMatrix::<f64, 2, 3>::new(z_inv, 0.0, -x * z_inv, 0.0, z_inv, -y * z_inv);
    let projection_jacobian =
        Matrix2::new(camera.fx, 0.0, 0.0, camera.fy) * jd * normalized_jacobian;
    (pixel
        .iter()
        .chain(projection_jacobian.iter())
        .all(|v| v.is_finite()))
    .then_some((pixel, projection_jacobian))
}

/// Invert Brown-Conrady distortion, then place the pixel at optical-axis depth Z.
/// Used for the constant-depth approximation of the 9x9 reference patch.
pub fn unproject_pixel_at_depth(
    pixel: &Vector2<f64>,
    depth: f64,
    camera: &CameraIntrinsics,
) -> Option<Point3<f64>> {
    if !depth.is_finite() || depth <= 1e-6 || camera.fx <= 0.0 || camera.fy <= 0.0 {
        return None;
    }
    let target = Vector2::new(
        (pixel.x - camera.cx) / camera.fx,
        (pixel.y - camera.cy) / camera.fy,
    );
    let mut xy = target;
    for _ in 0..15 {
        let (distorted, jacobian) = distort_with_jacobian(xy.x, xy.y, camera.distortion);
        let residual = distorted - target;
        if residual.norm_squared() < 1e-24 {
            break;
        }
        xy -= jacobian.try_inverse()? * residual;
        if !xy.iter().all(|v| v.is_finite()) {
            return None;
        }
    }
    let (distorted, _) = distort_with_jacobian(xy.x, xy.y, camera.distortion);
    if (distorted - target).norm() > 1e-8 {
        return None;
    }
    Some(Point3::new(xy.x * depth, xy.y * depth, depth))
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
            fx: 1175.01946 * scale,
            fy: 1177.56737 * scale,
            cx: 939.43935 * scale,
            cy: 627.06837 * scale,
            distortion: [-0.136349, 0.028462, -0.002188, 0.001479, 0.0],
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
