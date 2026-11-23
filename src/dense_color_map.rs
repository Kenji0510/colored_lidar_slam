//! 最終の色付き点群を、リアルタイム表示と同じ細かさで作るための蓄積。
//!
//! GlobalMap は 1 セル（2.5 cm）に 1 点・1 色しか持たないので、点が 2.5 cm 間隔の格子に並び、
//! リアルタイム表示（フレームごとの色付き点群を何フレームか重ねたもの）より粗く見える。
//! ここでは、フレームごとの色付き点群（2.5 cm ごとに位置と色を平均した点。`frames/cloud_N.pcd`
//! と同じもの）を、1 セルに何点でもそのまま貯める。
//!
//! 同じ場所を何度も通ると、周回ごとの小さな姿勢のずれで点と色が重なってぼやけるので、
//! セル（一辺 `voxel_size` の立方体）ごとに「最初の訪問」の点だけを残す。
//!
//! - セルに初めて点が入ったフレームから `visit_window_frames` フレームのあいだを最初の訪問とし、
//!   その間の点を `max_points_per_cell` 点まで貯める。
//! - 訪問が終わったセルは固定し、以後のフレームの点は捨てる。
//! - ただし、最初の訪問で `min_points_to_lock` 点に満たなかったセルは固定せず、後のフレームの点で
//!   埋める（端をかすめただけで点がほとんどないまま固定されないように）。
//! - 後のフレームが、そのセルを最初の訪問より `closer_override_ratio` 倍以上近くから見ている
//!   場合だけ、セルの点をすべて入れ替え、そのフレームから新しい訪問を始める（遠くから見ただけの
//!   粗い色で固定されないように）。

use nalgebra::Point3;
use rustc_hash::FxHashMap;

use crate::types::PointXYZRGB;
use crate::voxel_map::{VoxelKey, voxel_key};

#[derive(Debug, Clone)]
pub struct DenseColorMapConfig {
    /// 「同じ場所」とみなすセルの一辺 [m]。
    pub voxel_size: f32,
    /// 最初の訪問とみなすフレーム数（10 Hz なら 20 フレーム = 2 秒）。
    pub visit_window_frames: u64,
    /// 1 セルに貯める点の上限。
    pub max_points_per_cell: u16,
    /// 最初の訪問でこの点数に満たないセルは固定しない。
    pub min_points_to_lock: u16,
    /// 後のフレームの距離 × この倍率 ≤ 最初の訪問で最も近かった距離 なら、セルの点を入れ替える。
    /// `f32::INFINITY` で入れ替えなし。
    pub closer_override_ratio: f32,
}

impl Default for DenseColorMapConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.025,
            visit_window_frames: 20,
            max_points_per_cell: 20,
            min_points_to_lock: 4,
            closer_override_ratio: 2.0,
        }
    }
}

/// セルの状態（点そのものは `DenseColorMap::points` にまとめて持つ）。
#[derive(Debug, Clone, Copy)]
struct DenseColorCell {
    /// 今の訪問が始まったフレーム。
    visit_start_frame: u64,
    /// 今の訪問で最も近かったカメラからの距離 [m]。
    visit_min_distance_m: f32,
    /// 今の訪問で貯めた点の数。
    count: u16,
    /// 入れ替えのたびに 1 増える。これと違う世代の点は捨てられた点。
    generation: u16,
}

#[derive(Debug, Clone, Copy)]
struct DenseColorPoint {
    position: Point3<f32>,
    /// `PointXYZRGB::rgb` と同じ詰め方の色。
    rgb: f32,
    generation: u16,
}

/// 1 フレーム分の追加の結果。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DenseColorFrameStats {
    pub input_points: usize,
    /// 貯めた点の数（下の 3 つを含む）。
    pub stored_points: usize,
    /// 新しくできたセルの数。
    pub new_cells: usize,
    /// 固定前の点が少ないセルを埋めた点の数。
    pub filled_sparse_points: usize,
    /// より近くから見たので点を入れ替えたセルの数。
    pub replaced_cells: usize,
    /// 固定済みのセルなので捨てた点の数。
    pub dropped_locked_points: usize,
    /// セルが上限に達していて捨てた点の数。
    pub dropped_full_points: usize,
    /// 座標が有限でない点の数。
    pub rejected_non_finite: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DenseColorMapSummary {
    pub cells: usize,
    pub points: usize,
    /// 入れ替えで捨てた（今は使われていない）点の数。
    pub discarded_points: usize,
}

pub struct DenseColorMap {
    pub config: DenseColorMapConfig,
    cells: FxHashMap<VoxelKey, DenseColorCell>,
    points: Vec<DenseColorPoint>,
}

impl DenseColorMap {
    pub fn new(config: DenseColorMapConfig) -> Self {
        assert!(
            config.voxel_size.is_finite() && config.voxel_size > 0.0,
            "voxel_size must be positive"
        );
        assert!(
            config.visit_window_frames > 0,
            "visit_window_frames must be positive"
        );
        assert!(
            config.max_points_per_cell > 0,
            "max_points_per_cell must be positive"
        );
        assert!(
            config.closer_override_ratio > 1.0,
            "closer_override_ratio must be greater than 1 (use INFINITY to disable)"
        );
        Self {
            config,
            cells: FxHashMap::default(),
            points: Vec::new(),
        }
    }

    /// 1 フレーム分の色付き点（ワールド座標）を追加する。
    ///
    /// `camera_center` はそのフレームの画像を撮ったときのカメラ中心（ワールド座標）。
    /// フレーム番号は増える順に渡すこと。
    pub fn insert_frame(
        &mut self,
        frame_index: u64,
        camera_center: &Point3<f32>,
        points: &[PointXYZRGB],
    ) -> DenseColorFrameStats {
        let config = &self.config;
        let mut stats = DenseColorFrameStats {
            input_points: points.len(),
            ..DenseColorFrameStats::default()
        };
        for point in points {
            let position = Point3::new(point.x, point.y, point.z);
            if !position.coords.iter().all(|value| value.is_finite()) {
                stats.rejected_non_finite += 1;
                continue;
            }
            let distance = (position - camera_center).norm();
            let key = voxel_key(&position, config.voxel_size);

            let generation = match self.cells.get_mut(&key) {
                None => {
                    self.cells.insert(
                        key,
                        DenseColorCell {
                            visit_start_frame: frame_index,
                            visit_min_distance_m: distance,
                            count: 1,
                            generation: 0,
                        },
                    );
                    stats.new_cells += 1;
                    0
                }
                Some(cell) => {
                    let in_first_visit = frame_index.saturating_sub(cell.visit_start_frame)
                        < config.visit_window_frames;
                    if in_first_visit {
                        if cell.count >= config.max_points_per_cell {
                            stats.dropped_full_points += 1;
                            continue;
                        }
                    } else if distance * config.closer_override_ratio <= cell.visit_min_distance_m {
                        // ずっと近くから見ている: 点を入れ替え、ここから新しい訪問を始める。
                        cell.generation = cell.generation.wrapping_add(1);
                        cell.visit_start_frame = frame_index;
                        cell.visit_min_distance_m = distance;
                        cell.count = 0;
                        stats.replaced_cells += 1;
                    } else if cell.count < config.min_points_to_lock {
                        stats.filled_sparse_points += 1;
                    } else {
                        stats.dropped_locked_points += 1;
                        continue;
                    }
                    cell.count += 1;
                    cell.visit_min_distance_m = cell.visit_min_distance_m.min(distance);
                    cell.generation
                }
            };
            self.points.push(DenseColorPoint {
                position,
                rgb: point.rgb,
                generation,
            });
            stats.stored_points += 1;
        }
        stats
    }

    /// 今残っている点（入れ替えで捨てた点を除く）。
    pub fn colored_points(&self) -> Vec<PointXYZRGB> {
        self.points
            .iter()
            .filter(|point| self.is_current(point))
            .map(|point| PointXYZRGB {
                x: point.position.x,
                y: point.position.y,
                z: point.position.z,
                rgb: point.rgb,
            })
            .collect()
    }

    pub fn summary(&self) -> DenseColorMapSummary {
        let points = self
            .points
            .iter()
            .filter(|point| self.is_current(point))
            .count();
        DenseColorMapSummary {
            cells: self.cells.len(),
            points,
            discarded_points: self.points.len() - points,
        }
    }

    fn is_current(&self, point: &DenseColorPoint) -> bool {
        self.cells
            .get(&voxel_key(&point.position, self.config.voxel_size))
            .is_some_and(|cell| cell.generation == point.generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOXEL: f32 = 0.025;

    fn map() -> DenseColorMap {
        DenseColorMap::new(DenseColorMapConfig {
            voxel_size: VOXEL,
            visit_window_frames: 3,
            max_points_per_cell: 5,
            min_points_to_lock: 2,
            closer_override_ratio: 2.0,
        })
    }

    /// セル (0,0,0) の中の点。`offset` で少しずつ位置をずらす。
    fn point(offset: f32, value: u8) -> PointXYZRGB {
        PointXYZRGB::new(
            Point3::new(0.005 + offset, 0.010, 0.012),
            [value, value, value],
        )
    }

    fn camera_at(distance: f32) -> Point3<f32> {
        Point3::new(0.005, 0.010, 0.012 + distance)
    }

    fn values(map: &DenseColorMap) -> Vec<u8> {
        let mut values: Vec<u8> = map
            .colored_points()
            .iter()
            .map(|p| (p.rgb.to_bits() & 0xff) as u8)
            .collect();
        values.sort_unstable();
        values
    }

    #[test]
    fn keeps_every_point_of_the_first_visit_and_ignores_later_visits() {
        let mut map = map();
        // 最初の訪問（フレーム 10〜12）: 1 フレーム 1 点ずつ、位置も色も少しずつ違う
        for (frame, value) in [(10, 100), (11, 110), (12, 120)] {
            let stats =
                map.insert_frame(frame, &camera_at(5.0), &[point(frame as f32 * 1e-3, value)]);
            assert_eq!(stats.stored_points, 1);
        }
        assert_eq!(
            values(&map),
            vec![100, 110, 120],
            "1 セルに 3 点そのまま残る"
        );

        // 訪問が終わった後（フレーム 13 以降、別の周回）は、同じくらいの距離なら捨てる
        let stats = map.insert_frame(13, &camera_at(4.0), &[point(0.0, 200)]);
        assert_eq!(stats.dropped_locked_points, 1);
        let stats = map.insert_frame(500, &camera_at(5.0), &[point(0.0, 200)]);
        assert_eq!(stats.dropped_locked_points, 1);
        assert_eq!(values(&map), vec![100, 110, 120]);
        assert_eq!(
            map.summary(),
            DenseColorMapSummary {
                cells: 1,
                points: 3,
                discarded_points: 0
            }
        );
    }

    #[test]
    fn cells_with_too_few_points_stay_open_until_they_have_enough() {
        let mut map = map();
        map.insert_frame(10, &camera_at(5.0), &[point(0.0, 100)]);
        // 訪問が終わっても 1 点（< 2）しかないので、後の周回の点で埋める
        let stats = map.insert_frame(400, &camera_at(5.0), &[point(0.001, 150)]);
        assert_eq!(stats.filled_sparse_points, 1);
        // 2 点になったので固定
        let stats = map.insert_frame(401, &camera_at(5.0), &[point(0.002, 200)]);
        assert_eq!(stats.dropped_locked_points, 1);
        assert_eq!(values(&map), vec![100, 150]);
    }

    #[test]
    fn a_much_closer_later_view_replaces_the_cell() {
        let mut map = map();
        for frame in 10..13 {
            map.insert_frame(frame, &camera_at(12.0), &[point(0.0, 50)]);
        }
        // 1.5 倍近いだけでは入れ替えない
        let stats = map.insert_frame(100, &camera_at(8.0), &[point(0.0, 200)]);
        assert_eq!(stats.dropped_locked_points, 1);
        // 2 倍以上近い: 古い 3 点を捨てて、ここから新しい訪問
        let stats = map.insert_frame(200, &camera_at(5.0), &[point(0.0, 200)]);
        assert_eq!(stats.replaced_cells, 1);
        let stats = map.insert_frame(201, &camera_at(5.0), &[point(0.001, 210)]);
        assert_eq!(stats.stored_points, 1);
        assert_eq!(values(&map), vec![200, 210]);
        assert_eq!(
            map.summary(),
            DenseColorMapSummary {
                cells: 1,
                points: 2,
                discarded_points: 3
            }
        );
        // 新しい訪問も 3 フレームで終わる
        let stats = map.insert_frame(203, &camera_at(5.0), &[point(0.002, 220)]);
        assert_eq!(stats.dropped_locked_points, 1);
    }

    #[test]
    fn the_number_of_points_per_cell_is_capped_and_cells_are_independent() {
        let mut map = map();
        let many: Vec<PointXYZRGB> = (0..8).map(|k| point(k as f32 * 1e-3, k as u8)).collect();
        let stats = map.insert_frame(10, &camera_at(5.0), &many);
        assert_eq!(stats.stored_points, 5);
        assert_eq!(stats.dropped_full_points, 3);

        // 隣のセル（x が 1 セル先）は別に数える
        let neighbor = PointXYZRGB::new(Point3::new(0.030, 0.010, 0.012), [9, 9, 9]);
        let stats = map.insert_frame(10, &camera_at(5.0), &[neighbor]);
        assert_eq!(stats.new_cells, 1);
        assert_eq!(map.summary().cells, 2);
        assert_eq!(map.summary().points, 6);
    }

    #[test]
    fn non_finite_points_are_ignored() {
        let mut map = map();
        let bad = PointXYZRGB::new(Point3::new(f32::NAN, 0.0, 0.0), [1, 2, 3]);
        let stats = map.insert_frame(0, &camera_at(5.0), &[bad]);
        assert_eq!(stats.rejected_non_finite, 1);
        assert_eq!(map.summary(), DenseColorMapSummary::default());
    }
}
