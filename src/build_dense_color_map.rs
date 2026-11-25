//! SLAM の出力（frames/cloud_N.pcd・frame_logs.json・registration_results.json）から、
//! 最終の色付き点群（区画ごとに 1 回の訪問の点だけ）を作り直す。SLAM を再実行せずに、
//! 区画の大きさや窓の長さを試すためのもの。
//!
//! 使い方:
//!   cargo run --release --example build_dense_color_map -- \
//!       --run-dir data/output/11222026/avia [--tile 0.5] [--window 20] [--visits 3] \
//!       [--keep-blurry] [--out <file.pcd>]
//!
//! SLAM 本体と同じく、LiDAR の位置合わせが採用され、画像の質の判定（暗すぎ・白飛び・ブレすぎ）を
//! 通ったフレームだけを使う。ただし frames/cloud_N.pcd は白飛びした画素も平均に含んでいる
//! （SLAM 本体では白飛びの点を除いてから貯める）。

use anyhow::{Context, Result, bail};
use pcd_rs::Reader;
use re_lidar_slam::{
    dense_color_map::{DenseColorMap, DenseColorMapConfig},
    file_handler::{save_pcd_xyzrgb, save_pcd_xyzrgb_frame},
    types::{FrameLog, FrameRegistrationResult, PointXYZRGB},
};
use std::collections::HashMap;

struct Options {
    run_dir: String,
    tile: f32,
    window: u32,
    visits: usize,
    keep_blurry: bool,
    out: Option<String>,
}

fn parse() -> Result<Options> {
    let mut options = Options {
        run_dir: String::new(),
        tile: 0.5,
        window: 20,
        visits: 3,
        keep_blurry: false,
        out: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .with_context(|| format!("{name} requires a value"))
        };
        match arg.as_str() {
            "--run-dir" => options.run_dir = value("--run-dir")?,
            "--tile" => options.tile = value("--tile")?.parse()?,
            "--window" => options.window = value("--window")?.parse()?,
            "--visits" => options.visits = value("--visits")?.parse()?,
            "--keep-blurry" => options.keep_blurry = true,
            "--out" => options.out = Some(value("--out")?),
            _ => bail!("unknown argument '{arg}'"),
        }
    }
    if options.run_dir.is_empty() {
        bail!("--run-dir <SLAM output directory> is required");
    }
    Ok(options)
}

fn main() -> Result<()> {
    let options = parse()?;
    let run_dir = options.run_dir.trim_end_matches('/');
    let registration: Vec<FrameRegistrationResult> = serde_json::from_str(
        &std::fs::read_to_string(format!("{run_dir}/registration_results.json"))
            .context("registration_results.json")?,
    )?;
    let logs: Vec<FrameLog> = serde_json::from_str(
        &std::fs::read_to_string(format!("{run_dir}/frame_logs.json"))
            .context("frame_logs.json")?,
    )?;
    let logs: HashMap<usize, FrameLog> =
        logs.into_iter().map(|log| (log.frame_index, log)).collect();

    let mut map = DenseColorMap::new(DenseColorMapConfig {
        cell_size: 0.025,
        tile_size: options.tile,
        visit_window_frames: options.window,
        max_visits_per_tile: options.visits,
    });
    let (mut used, mut skipped) = (0usize, 0usize);
    let mut results: Vec<&FrameRegistrationResult> = registration.iter().collect();
    results.sort_by_key(|result| result.frame_index);
    for result in results {
        let Some(file) = &result.colored_cloud_file else {
            continue;
        };
        let ok = logs.get(&result.frame_index).is_some_and(|log| {
            log.map_updated
                && match log.coloring.status.as_str() {
                    "colored" | "no_candidates" => true,
                    "too_blurry" => options.keep_blurry,
                    _ => false,
                }
        });
        if !ok {
            skipped += 1;
            continue;
        }
        let points: Vec<PointXYZRGB> = Reader::open(format!("{run_dir}/{file}"))
            .with_context(|| format!("open {file}"))?
            .collect::<Result<_, _>>()
            .with_context(|| format!("read {file}"))?;
        map.insert_frame(result.frame_index as u64, &points);
        used += 1;
    }

    let result = map.build();
    let summary = result.summary;
    println!(
        "frames used {used} (skipped {skipped}) | tile {:.2} m, window {} frames, visits {}",
        options.tile, options.window, options.visits
    );
    println!(
        "points {} (best visit {}, fill {}), cells {}, tiles {}, buffered {}",
        summary.points,
        summary.primary_points,
        summary.fill_points,
        summary.cells,
        summary.tiles,
        summary.buffered_points
    );
    let out = options.out.unwrap_or_else(|| {
        format!(
            "{run_dir}/dense_tile{:.2}_win{}_vis{}.pcd",
            options.tile, options.window, options.visits
        )
    });
    let rgb: Vec<PointXYZRGB> = result
        .points
        .iter()
        .map(|p| PointXYZRGB {
            x: p.x,
            y: p.y,
            z: p.z,
            rgb: p.rgb,
        })
        .collect();
    save_pcd_xyzrgb(&rgb, &out)?;
    let frames_out = format!("{}_frames.pcd", out.trim_end_matches(".pcd"));
    save_pcd_xyzrgb_frame(&result.points, &frames_out)?;
    println!("saved {out}\nsaved {frames_out} (with frame indices)");
    Ok(())
}
