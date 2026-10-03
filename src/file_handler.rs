use std::{fmt::format, fs, path::PathBuf};

use anyhow::{Context, Result};
use pcd_rs::Reader;

use crate::types::{
    ImageFrameMetadata, ImageRecordingMetadata, LoadIMU, PointXYZ, PointXYZCov, PointXYZIT,
    PointXYZNormal,
};

pub fn load_pcd_files(dir_path: &str) -> Result<Vec<PathBuf>> {
    // let re = regex::Regex::new(r"voxelized-005_frame_(\d+)\.pcd$")
    let re = regex::Regex::new(r"^cloud_(\d+)\.pcd$").context("Invalid regex pattern")?;

    let entries =
        fs::read_dir(dir_path).context(format!("Failed to read directory: {}", dir_path))?;

    let mut files_with_numbers: Vec<(PathBuf, u32)> = Vec::new();

    for entry in entries {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        let filename = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name,
            None => continue,
        };

        if let Some(captures) = re.captures(filename) {
            if let Some(num_str) = captures.get(1) {
                if let Ok(num) = num_str.as_str().parse::<u32>() {
                    files_with_numbers.push((path.clone(), num));
                }
            }
        }
    }

    files_with_numbers.sort_by_key(|(_path, num)| *num);

    let sorted_paths: Vec<PathBuf> = files_with_numbers
        .into_iter()
        .map(|(path, _num)| path)
        .collect();

    Ok(sorted_paths)
}

pub fn load_imu_data(file_path: &str) -> Result<Vec<LoadIMU>> {
    let data = fs::read_to_string(file_path)
        .context(format!("Failed to read IMU data file: {}", file_path))?;

    let imu_data: Vec<LoadIMU> =
        serde_json::from_str(&data).context("Failed to parse IMU data from JSON")?;

    Ok(imu_data)
}

pub fn load_image_meta(file_path: &str) -> Result<ImageRecordingMetadata> {
    let data = fs::read_to_string(file_path)
        .context(format!("Failed to read image metadata: {}", file_path))?;

    let image_meta: ImageRecordingMetadata =
        serde_json::from_str(&data).context("Failed to parse image meta from JSON")?;

    Ok(image_meta)
}

pub fn save_pcd_xyz(points: &[PointXYZ], file_path: &str) -> Result<()> {
    let mut writer = pcd_rs::WriterInit {
        width: 1,
        height: points.len() as u64,
        viewpoint: Default::default(),
        data_kind: pcd_rs::DataKind::Ascii,
        schema: None,
    }
    .create(file_path)?;

    for point in points {
        writer.push(point)?;
    }
    writer.finish()?;

    Ok(())
}

pub fn save_pcd_xyzit(points: &[PointXYZIT], file_path: &str) -> Result<()> {
    let mut writer = pcd_rs::WriterInit {
        width: 1,
        height: points.len() as u64,
        viewpoint: Default::default(),
        data_kind: pcd_rs::DataKind::Ascii,
        schema: None,
    }
    .create(file_path)?;

    for point in points {
        writer.push(point)?;
    }
    writer.finish()?;

    Ok(())
}

pub fn save_pcd_xyzcov(points: &[PointXYZCov], file_path: &str) -> Result<()> {
    let mut writer = pcd_rs::WriterInit {
        width: 1,
        height: points.len() as u64,
        viewpoint: Default::default(),
        data_kind: pcd_rs::DataKind::Ascii,
        schema: None,
    }
    .create(file_path)?;

    for point in points {
        writer.push(point)?;
    }
    writer.finish()?;

    Ok(())
}

pub fn load_pcd_xyzit(file_path: &str) -> Result<Vec<PointXYZIT>> {
    let reader = match Reader::open(file_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to open PCD file: {}", e);
            return Err(anyhow::anyhow!("Failed to open PCD file: {}", e));
        }
    };

    let points: Vec<PointXYZIT> = match reader.collect() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to read PCD data: {}", e);
            return Err(anyhow::anyhow!("Failed to read PCD data: {}", e));
        }
    };

    Ok(points)
}

pub fn save_pcd_xyznormal(points: &[PointXYZNormal], file_path: &str) -> Result<()> {
    let mut writer = pcd_rs::WriterInit {
        width: 1,
        height: points.len() as u64,
        viewpoint: Default::default(),
        data_kind: pcd_rs::DataKind::Ascii,
        schema: None,
    }
    .create(file_path)?;

    for point in points {
        writer.push(point)?;
    }
    writer.finish()?;

    Ok(())
}

pub fn find_nearest_image_frame(
    frames: &[ImageFrameMetadata],
    target_time: f64,
    max_time_diff_sec: f64,
) -> Option<&ImageFrameMetadata> {
    if !target_time.is_finite() || !max_time_diff_sec.is_finite() || max_time_diff_sec < 0.0 {
        return None;
    }

    frames
        .iter()
        .filter(|frame| frame.timestamp_sec.is_finite())
        .min_by(|a, b| {
            let diff_a = (a.timestamp_sec - target_time).abs();
            let diff_b = (b.timestamp_sec - target_time).abs();
            diff_a.total_cmp(&diff_b)
        })
        .filter(|frame| (frame.timestamp_sec - target_time).abs() <= max_time_diff_sec)
}

pub fn load_rgb_image(path: &std::path::Path) -> Result<image::RgbImage> {
    let decoded = image::ImageReader::open(path)
        .with_context(|| format!("Failed to open image: {}", path.display()))?
        .decode()
        .with_context(|| format!("Failed to decode image: {}", path.display()))?;

    Ok(decoded.into_rgb8())
}
