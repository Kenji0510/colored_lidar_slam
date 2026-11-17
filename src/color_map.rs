//! 色専用のボクセルマップ（GlobalMap とは別に持つ）。
//!
//! GlobalMap は ICP と平面判定のための幾何マップで、2.5 cm セル・平面フィルタあり・
//! 色は「一番近い観測 1 回分」しか持たない。色の入れ物としては粗すぎるので、ここでは
//!
//! - deskew 済みの全点（間引き前）を、位置合わせ後の姿勢で画像へ投影し、
//! - バイリニア補間で色を読み、
//! - 細かいボクセル（既定 1 cm）ごとに「位置の平均」と「距離で重み付けした色の平均」を
//!   蓄積する。
//!
//! 平面フィルタは通さないので、凹凸のある細部も残る。データ量はセル数で決まり、
//! 点数が増えても膨らまない。GlobalMap（幾何）には一切触れない。

use image::RgbImage;
use nalgebra::{Matrix4, Point3, Vector2};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::camera::{CameraIntrinsics, project_camera_point_subpixel};
use crate::types::PointXYZRGB;

#[derive(Debug, Clone)]
pub struct ColorMapConfig {
    /// 色を平均するボクセルの大きさ [m]。
    pub voxel_size_m: f32,
    /// この距離 [m] より近い観測は重み 1。遠い観測は (reference / 距離)² に下げる
    /// （遠いほど 1 画素が広い範囲を写すので、色がぼやけているため）。
    pub reference_distance_m: f64,
    pub min_depth_m: f64,
    pub max_depth_m: f64,
    /// 画像の縦横それぞれ、中央のこの割合だけを使う（周辺は歪み補正の誤差が大きい）。
    pub image_keep_ratio: f64,
    /// カメラから見て、同じフレームの手前の点に隠れている点には色を付けない。
    /// カメラと LiDAR の位置がずれているため、輪郭付近で奥の点に手前の色が付くのを防ぐ。
    pub occlusion_check: bool,
    /// 遮蔽チェックの深度バッファのマス [px]。
    pub occlusion_cell_px: usize,
    /// マスの最小深度より `margin_m + ratio * 最小深度` 以上奥なら遮蔽とみなす。
    pub occlusion_margin_m: f64,
    pub occlusion_margin_ratio: f64,
    /// 出力に含める最小観測回数。
    pub min_observations: u32,
}

impl Default for ColorMapConfig {
    fn default() -> Self {
        Self {
            voxel_size_m: 0.01,
            reference_distance_m: 5.0,
            min_depth_m: 0.5,
            max_depth_m: 40.0,
            image_keep_ratio: 0.75,
            occlusion_check: true,
            occlusion_cell_px: 6,
            occlusion_margin_m: 0.15,
            occlusion_margin_ratio: 0.1,
            min_observations: 1,
        }
    }
}

/// 1 点分の色の観測（ワールド座標）。
#[derive(Debug, Clone, Copy)]
pub struct ColorSample {
    pub world: Point3<f64>,
    /// バイリニア補間した RGB（0..255）。
    pub rgb: [f32; 3],
    pub weight: f32,
}

/// 1 フレーム分の観測と、その内訳。
#[derive(Debug, Clone, Default)]
pub struct ColorFrameSamples {
    pub samples: Vec<ColorSample>,
    pub input_points: usize,
    /// 深度の範囲内で、画像の使用範囲に投影された点。
    pub in_image: usize,
    /// そのうち、同じフレームの手前の点に隠れていると判定した点。
    pub occluded: usize,
}

/// 1 ボクセル分の累積値。位置はボクセルの最小角からの相対値で足し合わせ、
/// f32 でもワールド座標の大きさに関係なく桁落ちしないようにする。
#[derive(Debug, Clone, Copy, Default)]
struct ColorVoxel {
    offset_sum: [f32; 3],
    color_sum: [f32; 3],
    weight_sum: f32,
    count: u32,
}

// キーは各軸 21 bit（±2^20 セル。1 cm なら ±10 km）を 1 つの u64 に詰める。
const KEY_BITS: u32 = 21;
const KEY_MASK: u64 = (1 << KEY_BITS) - 1;
const KEY_OFFSET: i64 = 1 << (KEY_BITS - 1);

fn pack_key(index: [i64; 3]) -> Option<u64> {
    let mut key = 0u64;
    for (axis, value) in index.into_iter().enumerate() {
        let shifted = value + KEY_OFFSET;
        if !(0..(1i64 << KEY_BITS)).contains(&shifted) {
            return None;
        }
        key |= (shifted as u64) << (KEY_BITS * axis as u32);
    }
    Some(key)
}

fn unpack_key(key: u64) -> [i64; 3] {
    std::array::from_fn(|axis| ((key >> (KEY_BITS * axis as u32)) & KEY_MASK) as i64 - KEY_OFFSET)
}

/// RGB 画像をサブピクセル位置でバイリニア補間する（画素中心が整数座標）。
pub fn sample_rgb_bilinear(image: &RgbImage, pixel: &Vector2<f64>) -> Option<[f32; 3]> {
    let (width, height) = (image.width() as f64, image.height() as f64);
    if !pixel.iter().all(|v| v.is_finite())
        || pixel.x < 0.0
        || pixel.y < 0.0
        || pixel.x >= width - 1.0
        || pixel.y >= height - 1.0
    {
        return None;
    }
    let x0 = pixel.x.floor();
    let y0 = pixel.y.floor();
    let (dx, dy) = ((pixel.x - x0) as f32, (pixel.y - y0) as f32);
    let (x0, y0) = (x0 as u32, y0 as u32);
    let a = image.get_pixel(x0, y0).0;
    let b = image.get_pixel(x0 + 1, y0).0;
    let c = image.get_pixel(x0, y0 + 1).0;
    let d = image.get_pixel(x0 + 1, y0 + 1).0;
    Some(std::array::from_fn(|k| {
        (1.0 - dy) * ((1.0 - dx) * a[k] as f32 + dx * b[k] as f32)
            + dy * ((1.0 - dx) * c[k] as f32 + dx * d[k] as f32)
    }))
}

pub struct ColorMap {
    pub config: ColorMapConfig,
    voxels: FxHashMap<u64, ColorVoxel>,
    /// キーの範囲（±2^20 セル）を超えて捨てた観測の数。
    out_of_range: u64,
}

impl ColorMap {
    pub fn new(config: ColorMapConfig) -> Self {
        assert!(
            config.voxel_size_m.is_finite() && config.voxel_size_m > 0.0,
            "voxel_size_m must be positive"
        );
        assert!(
            config.occlusion_cell_px > 0,
            "occlusion_cell_px must be positive"
        );
        Self {
            config,
            voxels: FxHashMap::default(),
            out_of_range: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.voxels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    pub fn out_of_range_samples(&self) -> u64 {
        self.out_of_range
    }

    /// 1 フレームの点群を画像に投影し、色の観測を作る（マップはまだ変更しない）。
    ///
    /// - `points_lidar`: deskew 済みの点（フレーム開始時刻の LiDAR 座標）
    /// - `world_from_lidar`: フレーム開始時刻の LiDAR 姿勢（位置合わせ後）
    /// - `camera_from_world`: 画像を撮った時刻のカメラ姿勢
    pub fn sample_frame(
        &self,
        points_lidar: &[Point3<f32>],
        world_from_lidar: &Matrix4<f64>,
        camera_from_world: &Matrix4<f64>,
        image: &RgbImage,
        camera: &CameraIntrinsics,
    ) -> ColorFrameSamples {
        let config = &self.config;
        let camera_from_lidar = camera_from_world * world_from_lidar;
        let (width, height) = (camera.width as f64, camera.height as f64);
        let margin_ratio = ((1.0 - config.image_keep_ratio) * 0.5).clamp(0.0, 0.5);
        let (margin_x, margin_y) = (width * margin_ratio, height * margin_ratio);

        // 1. 投影（深度の範囲内で画像に写る点）。
        let projected: Vec<Option<(Point3<f64>, Vector2<f64>)>> = points_lidar
            .par_iter()
            .map(|point| {
                let pc = camera_from_lidar.transform_point(&point.cast::<f64>());
                if !(pc.z >= config.min_depth_m && pc.z <= config.max_depth_m) {
                    return None;
                }
                let (pixel, _) = project_camera_point_subpixel(&pc, camera)?;
                (pixel.x >= 0.0 && pixel.y >= 0.0 && pixel.x < width && pixel.y < height)
                    .then_some((pc, pixel))
            })
            .collect();

        // 2. 遮蔽チェック用の深度バッファ（同じフレームの点のマスごとの最小深度）。
        let cell = config.occlusion_cell_px as f64;
        let cols = (camera.width as usize)
            .div_ceil(config.occlusion_cell_px)
            .max(1);
        let rows = (camera.height as usize)
            .div_ceil(config.occlusion_cell_px)
            .max(1);
        let cell_index = |pixel: &Vector2<f64>| {
            ((pixel.y / cell) as usize).min(rows - 1) * cols
                + ((pixel.x / cell) as usize).min(cols - 1)
        };
        let depth_buffer = config.occlusion_check.then(|| {
            let mut buffer = vec![f64::INFINITY; cols * rows];
            for (pc, pixel) in projected.iter().flatten() {
                let index = cell_index(pixel);
                buffer[index] = buffer[index].min(pc.z);
            }
            buffer
        });

        // 3. 使用範囲内・遮蔽なしの点だけ色を読む。
        let results: Vec<(bool, Option<ColorSample>)> = projected
            .par_iter()
            .zip(points_lidar.par_iter())
            .filter_map(|(projection, point)| {
                let (pc, pixel) = projection.as_ref()?;
                if pixel.x < margin_x
                    || pixel.y < margin_y
                    || pixel.x > width - margin_x - 1.0
                    || pixel.y > height - margin_y - 1.0
                {
                    return None;
                }
                if let Some(buffer) = depth_buffer.as_ref() {
                    let nearest = buffer[cell_index(pixel)];
                    if pc.z
                        > nearest
                            + config.occlusion_margin_m
                            + config.occlusion_margin_ratio * nearest
                    {
                        return Some((true, None));
                    }
                }
                let rgb = sample_rgb_bilinear(image, pixel)?;
                let distance = pc.coords.norm();
                let ratio = config.reference_distance_m / distance.max(1e-6);
                let weight = (ratio * ratio).min(1.0) as f32;
                let world = world_from_lidar.transform_point(&point.cast::<f64>());
                world
                    .coords
                    .iter()
                    .all(|v| v.is_finite())
                    .then_some((false, Some(ColorSample { world, rgb, weight })))
            })
            .collect();

        let occluded = results.iter().filter(|(occluded, _)| *occluded).count();
        let samples: Vec<ColorSample> = results.into_iter().filter_map(|(_, s)| s).collect();
        ColorFrameSamples {
            in_image: samples.len() + occluded,
            occluded,
            samples,
            input_points: points_lidar.len(),
        }
    }

    /// 観測をボクセルに足し込む。返り値は新しくできたボクセルの数。
    pub fn accumulate(&mut self, samples: &[ColorSample]) -> usize {
        let voxel = self.config.voxel_size_m as f64;
        let inv_voxel = 1.0 / voxel;
        let before = self.voxels.len();
        for sample in samples {
            if !(sample.weight.is_finite() && sample.weight > 0.0) {
                continue;
            }
            let index: [i64; 3] =
                std::array::from_fn(|axis| (sample.world[axis] * inv_voxel).floor() as i64);
            let Some(key) = pack_key(index) else {
                self.out_of_range += 1;
                continue;
            };
            let offset: [f32; 3] = std::array::from_fn(|axis| {
                (sample.world[axis] - index[axis] as f64 * voxel) as f32
            });
            let entry = self.voxels.entry(key).or_default();
            for (sum, value) in entry.offset_sum.iter_mut().zip(offset) {
                *sum += value;
            }
            for (sum, value) in entry.color_sum.iter_mut().zip(sample.rgb) {
                *sum += value * sample.weight;
            }
            entry.weight_sum += sample.weight;
            entry.count += 1;
        }
        self.voxels.len() - before
    }

    /// 平均した位置と色を、観測回数が `min_observations` 以上のボクセルについて出力する。
    pub fn export_points(&self) -> Vec<PointXYZRGB> {
        let voxel = self.config.voxel_size_m as f64;
        let min_observations = self.config.min_observations.max(1);
        self.voxels
            .par_iter()
            .filter(|(_, v)| v.count >= min_observations && v.weight_sum > 0.0)
            .map(|(&key, v)| {
                let index = unpack_key(key);
                let position = Point3::from(std::array::from_fn::<f32, 3, _>(|axis| {
                    (index[axis] as f64 * voxel + (v.offset_sum[axis] / v.count as f32) as f64)
                        as f32
                }));
                let rgb = std::array::from_fn(|k| {
                    (v.color_sum[k] / v.weight_sum).round().clamp(0.0, 255.0) as u8
                });
                PointXYZRGB::new(position, rgb)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    use nalgebra::Vector3;

    fn camera() -> CameraIntrinsics {
        CameraIntrinsics {
            width: 200,
            height: 100,
            fx: 100.0,
            fy: 100.0,
            cx: 99.5,
            cy: 49.5,
            distortion: [0.0; 5],
        }
    }

    fn unpack_rgb(point: &PointXYZRGB) -> [u8; 3] {
        let bits = point.rgb.to_bits();
        [(bits >> 16) as u8, (bits >> 8) as u8, bits as u8]
    }

    /// カメラ（= LiDAR）座標で深度 `depth`、画素 (u, v) に写る点。
    fn point_at(camera: &CameraIntrinsics, u: f64, v: f64, depth: f64) -> Point3<f32> {
        Point3::new(
            ((u - camera.cx) / camera.fx * depth) as f32,
            ((v - camera.cy) / camera.fy * depth) as f32,
            depth as f32,
        )
    }

    fn config_full_image() -> ColorMapConfig {
        ColorMapConfig {
            image_keep_ratio: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn bilinear_sampling_interpolates_between_pixel_centres() {
        let mut image = RgbImage::new(2, 2);
        image.put_pixel(0, 0, Rgb([0, 100, 200]));
        image.put_pixel(1, 0, Rgb([100, 100, 200]));
        image.put_pixel(0, 1, Rgb([0, 200, 0]));
        image.put_pixel(1, 1, Rgb([100, 200, 0]));
        let rgb = sample_rgb_bilinear(&image, &Vector2::new(0.25, 0.5)).unwrap();
        assert!((rgb[0] - 25.0).abs() < 1e-4);
        assert!((rgb[1] - 150.0).abs() < 1e-4);
        assert!((rgb[2] - 100.0).abs() < 1e-4);
        assert!(sample_rgb_bilinear(&image, &Vector2::new(1.0, 0.5)).is_none());
        assert!(sample_rgb_bilinear(&image, &Vector2::new(-0.1, 0.5)).is_none());
    }

    #[test]
    fn keys_roundtrip_including_negative_coordinates_and_reject_out_of_range() {
        for index in [[0, 0, 0], [-1, 2, -3], [1_000_000, -1_000_000, 5]] {
            assert_eq!(unpack_key(pack_key(index).unwrap()), index);
        }
        assert!(pack_key([KEY_OFFSET, 0, 0]).is_none());
        assert!(pack_key([0, -KEY_OFFSET - 1, 0]).is_none());
    }

    #[test]
    fn accumulate_averages_positions_and_weights_colors_per_voxel() {
        let mut map = ColorMap::new(ColorMapConfig::default());
        let samples = [
            ColorSample {
                world: Point3::new(-3.0012, 2.0034, 10.0051),
                rgb: [200.0, 0.0, 0.0],
                weight: 1.0,
            },
            ColorSample {
                world: Point3::new(-3.0018, 2.0036, 10.0059),
                rgb: [0.0, 0.0, 100.0],
                weight: 0.25,
            },
            // 1 cm 離れた別のセル
            ColorSample {
                world: Point3::new(-3.0115, 2.0034, 10.0051),
                rgb: [10.0, 20.0, 30.0],
                weight: 1.0,
            },
        ];
        assert_eq!(map.accumulate(&samples), 2);
        assert_eq!(map.len(), 2);
        let points = map.export_points();
        let merged = points
            .iter()
            .find(|p| (p.x - -3.0015).abs() < 1e-4)
            .expect("merged voxel");
        assert!((merged.y - 2.0035).abs() < 1e-5 && (merged.z - 10.0055).abs() < 1e-5);
        // 重み 1 : 0.25 の加重平均 → R = 200/1.25 = 160, B = 25/1.25 = 20
        assert_eq!(unpack_rgb(merged), [160, 0, 20]);

        let mut strict = ColorMap::new(ColorMapConfig {
            min_observations: 2,
            ..Default::default()
        });
        strict.accumulate(&samples);
        assert_eq!(strict.export_points().len(), 1);
    }

    #[test]
    fn sample_frame_reads_bilinear_colour_respects_crop_depth_and_distance_weight() {
        let camera = camera();
        // 左半分が赤、右半分が青
        let image = RgbImage::from_fn(200, 100, |u, _| {
            if u < 100 {
                Rgb([255, 0, 0])
            } else {
                Rgb([0, 0, 255])
            }
        });
        let points = vec![
            point_at(&camera, 40.0, 50.0, 4.0),   // 赤、近い（重み 1）
            point_at(&camera, 160.0, 50.0, 10.0), // 青、遠い（重み 0.25）
            point_at(&camera, 5.0, 50.0, 4.0),    // 中央 75% の外
            point_at(&camera, 40.0, 50.0, 60.0),  // 遠すぎる
            Point3::new(0.0, 0.0, -2.0),          // カメラの後ろ
        ];
        let map = ColorMap::new(ColorMapConfig::default());
        let frame = map.sample_frame(
            &points,
            &Matrix4::identity(),
            &Matrix4::identity(),
            &image,
            &camera,
        );
        assert_eq!(frame.input_points, 5);
        assert_eq!(frame.samples.len(), 2, "{frame:?}");
        let red = frame.samples.iter().find(|s| s.rgb[0] > 200.0).unwrap();
        let blue = frame.samples.iter().find(|s| s.rgb[2] > 200.0).unwrap();
        assert_eq!(red.rgb, [255.0, 0.0, 0.0]);
        assert_eq!(blue.rgb, [0.0, 0.0, 255.0]);
        // 近い点は重み 1、遠い点は (5 m / 距離)²
        assert!((red.weight - 1.0).abs() < 1e-6);
        let blue_distance = points[1].cast::<f64>().coords.norm();
        assert!((blue.weight as f64 - (5.0 / blue_distance).powi(2)).abs() < 1e-5);
        assert!(blue.weight < 0.2);
        assert!((red.world - Point3::new(-2.38, 0.02, 4.0)).norm() < 1e-5);
    }

    #[test]
    fn world_pose_and_image_time_camera_pose_are_both_applied() {
        let camera = camera();
        let image = RgbImage::from_pixel(200, 100, Rgb([0, 255, 0]));
        // LiDAR はワールドで x=+2 m の位置。カメラは画像時刻にさらに +0.1 m 動いている。
        let mut world_from_lidar = Matrix4::identity();
        world_from_lidar[(0, 3)] = 2.0;
        let mut world_from_camera = world_from_lidar;
        world_from_camera[(0, 3)] = 2.1;
        let camera_from_world = world_from_camera.try_inverse().unwrap();
        let point = point_at(&camera, 120.0, 30.0, 5.0);
        let frame = ColorMap::new(config_full_image()).sample_frame(
            &[point],
            &world_from_lidar,
            &camera_from_world,
            &image,
            &camera,
        );
        let sample = frame.samples[0];
        let expected_world = point.cast::<f64>() + Vector3::new(2.0, 0.0, 0.0);
        assert!((sample.world - expected_world).norm() < 1e-6);
    }

    #[test]
    fn points_hidden_behind_nearer_points_of_the_same_frame_are_not_coloured() {
        let camera = camera();
        let image = RgbImage::from_pixel(200, 100, Rgb([90, 90, 90]));
        let mut points = Vec::new();
        // 手前（2 m）の物体と、その真後ろ（6 m）の壁が同じ画素範囲に写る
        for v in (30..70).step_by(2) {
            for u in (60..100).step_by(2) {
                points.push(point_at(&camera, u as f64, v as f64, 2.0));
                points.push(point_at(&camera, u as f64 + 0.5, v as f64 + 0.5, 6.0));
            }
        }
        // 物体のない場所の壁
        for v in (30..70).step_by(2) {
            for u in (120..160).step_by(2) {
                points.push(point_at(&camera, u as f64, v as f64, 6.0));
            }
        }
        let per_region = 20 * 20;
        let map = ColorMap::new(config_full_image());
        let frame = map.sample_frame(
            &points,
            &Matrix4::identity(),
            &Matrix4::identity(),
            &image,
            &camera,
        );
        assert_eq!(frame.in_image, 3 * per_region);
        assert_eq!(frame.occluded, per_region, "{frame:?}");
        assert!(
            frame
                .samples
                .iter()
                .all(|s| s.world.z < 3.0 || s.world.x > 0.0)
        );

        let unchecked = ColorMap::new(ColorMapConfig {
            occlusion_check: false,
            ..config_full_image()
        })
        .sample_frame(
            &points,
            &Matrix4::identity(),
            &Matrix4::identity(),
            &image,
            &camera,
        );
        assert_eq!(unchecked.occluded, 0);
        assert_eq!(unchecked.samples.len(), 3 * per_region);
    }

    #[test]
    fn repeated_frames_reduce_colour_noise_by_averaging() {
        let camera = camera();
        let points: Vec<Point3<f32>> = (30..70)
            .flat_map(|v| (60..140).map(move |u| (u, v)))
            .map(|(u, v)| point_at(&camera, u as f64, v as f64, 3.0))
            .collect();
        let mut map = ColorMap::new(config_full_image());
        // 露出のばらつきを模して、フレームごとに明るさが ±20 変わる
        for (frame, level) in [100u8, 140, 120, 80, 160, 100, 140, 120, 80, 160]
            .iter()
            .enumerate()
        {
            let image = RgbImage::from_pixel(200, 100, Rgb([*level, *level, *level]));
            let samples = map.sample_frame(
                &points,
                &Matrix4::identity(),
                &Matrix4::identity(),
                &image,
                &camera,
            );
            let new_voxels = map.accumulate(&samples.samples);
            if frame > 0 {
                assert_eq!(new_voxels, 0, "same points must land in the same voxels");
            }
        }
        for point in map.export_points() {
            assert_eq!(unpack_rgb(&point), [120, 120, 120]);
        }
    }
}
