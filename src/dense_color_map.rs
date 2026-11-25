//! 最終の色付き点群を、リアルタイム表示と同じ細かさ・くっきり度で作るための蓄積と選択。
//!
//! GlobalMap は 1 セル（2.5 cm）に 1 点・1 色しか持たないので、点が 2.5 cm 間隔の格子に並び、
//! リアルタイム表示（フレームごとの色付き点群を連続した何フレームか重ねたもの）より粗く見える。
//! ここでは、フレームごとの色付き点群（2.5 cm ごとに位置と色を平均した点。`frames/cloud_N.pcd`
//! と同じもの）を貯めておき、最後に場所ごとに使う点を選ぶ。
//!
//! リアルタイム表示がくっきりしているのは、連続した数十フレーム（1 回の訪問）だけを重ねているから。
//! 周回が違うと、小さな姿勢のずれで点と色が重なってぼやける。そこで次のように選ぶ。
//!
//! - 場所を区画（一辺 `tile_size` の立方体）に分け、区画ごとに連続 `visit_window_frames`
//!   フレームの窓（＝1 回の訪問）を 1 つ選び、その窓の点だけを使う。区画の中では周回を混ぜない。
//! - 選ぶのは、その区画に入る点が最も多い窓。1 フレームの点は 2.5 cm ごとに 1 点なので、点の数は
//!   区画をどれだけ細かく覆ったか（近く・正面から見ているほど多い）を表す。遠くから見えただけの
//!   最初の訪問ではなく、一番細かく写った訪問が選ばれる。
//! - 選んだ窓で点のなかったセル（一辺 `cell_size`）だけ、2 番目、3 番目…に点の多い窓
//!   （`max_visits_per_tile` 個まで）の点で埋める。セルの中で窓は混ぜない。

use nalgebra::Point3;
use rustc_hash::FxHashMap;

use crate::types::{PointXYZRGB, PointXYZRGBFrame};
use crate::voxel_map::{VoxelKey, voxel_key};

#[derive(Debug, Clone)]
pub struct DenseColorMapConfig {
    /// 穴埋めの単位のセルの一辺 [m]（フレームごとの色付き点群のボクセルと同じにする）。
    pub cell_size: f32,
    /// 1 回の訪問を選ぶ区画の一辺 [m]。
    pub tile_size: f32,
    /// 1 回の訪問とみなす連続フレーム数（10 Hz なら 20 フレーム = 2 秒）。
    pub visit_window_frames: u32,
    /// 1 区画で使う訪問の数の上限。1 なら穴埋めしない。
    pub max_visits_per_tile: usize,
}

impl Default for DenseColorMapConfig {
    fn default() -> Self {
        Self {
            cell_size: 0.025,
            tile_size: 0.5,
            visit_window_frames: 20,
            max_visits_per_tile: 3,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DensePoint {
    position: Point3<f32>,
    /// `PointXYZRGB::rgb` と同じ詰め方の色。
    rgb: f32,
}

struct FrameRecord {
    frame_index: u32,
    points: Vec<DensePoint>,
}

/// 区画ごとの、フレームごとの点の数（フレーム番号の昇順）。
type TileHistory = Vec<(u32, u32)>;

/// 選んだ訪問（フレーム番号 [start, end)）。並びは良い順。
type TileVisits = Vec<(u32, u32)>;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DenseColorMapSummary {
    /// 貯めたフレーム数と点の数。
    pub frames: usize,
    pub buffered_points: usize,
    pub tiles: usize,
    /// 出力した点の数（下の 2 つの和）。
    pub points: usize,
    /// 区画ごとの一番良い訪問の点。
    pub primary_points: usize,
    /// 穴埋めに使った 2 番目以降の訪問の点。
    pub fill_points: usize,
    /// 点のあるセルの数。
    pub cells: usize,
}

pub struct DenseColorResult {
    /// 出力する点と、その点を撮ったフレーム番号。
    pub points: Vec<PointXYZRGBFrame>,
    pub summary: DenseColorMapSummary,
}

pub struct DenseColorMap {
    pub config: DenseColorMapConfig,
    frames: Vec<FrameRecord>,
    tile_history: FxHashMap<VoxelKey, TileHistory>,
}

impl DenseColorMap {
    pub fn new(config: DenseColorMapConfig) -> Self {
        assert!(
            config.cell_size.is_finite() && config.cell_size > 0.0,
            "cell_size must be positive"
        );
        assert!(
            config.tile_size.is_finite() && config.tile_size >= config.cell_size,
            "tile_size must be at least cell_size"
        );
        assert!(
            config.visit_window_frames > 0,
            "visit_window_frames must be positive"
        );
        assert!(
            config.max_visits_per_tile > 0,
            "max_visits_per_tile must be positive"
        );
        Self {
            config,
            frames: Vec::new(),
            tile_history: FxHashMap::default(),
        }
    }

    /// 1 フレーム分の色付き点（ワールド座標）を貯める。戻り値は貯めた点の数。
    /// フレーム番号は増える順に渡すこと。
    pub fn insert_frame(&mut self, frame_index: u64, points: &[PointXYZRGB]) -> usize {
        let frame_index = u32::try_from(frame_index).expect("frame index exceeds u32");
        if let Some(last) = self.frames.last() {
            assert!(
                frame_index > last.frame_index,
                "frames must be inserted in order"
            );
        }
        let mut stored = Vec::with_capacity(points.len());
        let mut tile_counts: FxHashMap<VoxelKey, u32> = FxHashMap::default();
        for point in points {
            let position = Point3::new(point.x, point.y, point.z);
            if !position.coords.iter().all(|value| value.is_finite()) {
                continue;
            }
            *tile_counts
                .entry(voxel_key(&position, self.config.tile_size))
                .or_insert(0) += 1;
            stored.push(DensePoint {
                position,
                rgb: point.rgb,
            });
        }
        for (tile, count) in tile_counts {
            self.tile_history
                .entry(tile)
                .or_default()
                .push((frame_index, count));
        }
        let count = stored.len();
        if count > 0 {
            stored.shrink_to_fit();
            self.frames.push(FrameRecord {
                frame_index,
                points: stored,
            });
        }
        count
    }

    /// 区画ごとに訪問を選び、出力する点を作る。
    pub fn build(&self) -> DenseColorResult {
        let window = self.config.visit_window_frames;
        let visits: FxHashMap<VoxelKey, TileVisits> = self
            .tile_history
            .iter()
            .map(|(tile, history)| {
                (
                    *tile,
                    choose_visits(history, window, self.config.max_visits_per_tile),
                )
            })
            .collect();

        let rank_of = |frame_index: u32, position: &Point3<f32>| -> Option<u8> {
            visits
                .get(&voxel_key(position, self.config.tile_size))?
                .iter()
                .position(|&(start, end)| (start..end).contains(&frame_index))
                .map(|rank| rank as u8)
        };

        // 1. セルごとに、点のある一番良い訪問を求める
        let mut best_rank: FxHashMap<VoxelKey, u8> = FxHashMap::default();
        for frame in &self.frames {
            for point in &frame.points {
                if let Some(rank) = rank_of(frame.frame_index, &point.position) {
                    best_rank
                        .entry(voxel_key(&point.position, self.config.cell_size))
                        .and_modify(|best| *best = (*best).min(rank))
                        .or_insert(rank);
                }
            }
        }

        // 2. その訪問の点だけを出力する
        let mut summary = DenseColorMapSummary {
            frames: self.frames.len(),
            buffered_points: self.frames.iter().map(|frame| frame.points.len()).sum(),
            tiles: self.tile_history.len(),
            cells: best_rank.len(),
            ..DenseColorMapSummary::default()
        };
        let mut points = Vec::new();
        for frame in &self.frames {
            for point in &frame.points {
                let Some(rank) = rank_of(frame.frame_index, &point.position) else {
                    continue;
                };
                let cell = voxel_key(&point.position, self.config.cell_size);
                if best_rank.get(&cell) != Some(&rank) {
                    continue;
                }
                if rank == 0 {
                    summary.primary_points += 1;
                } else {
                    summary.fill_points += 1;
                }
                points.push(PointXYZRGBFrame {
                    x: point.position.x,
                    y: point.position.y,
                    z: point.position.z,
                    rgb: point.rgb,
                    frame: frame.frame_index as f32,
                });
            }
        }
        summary.points = points.len();
        DenseColorResult { points, summary }
    }
}

/// 点の数が多い順に、重ならない窓（連続 `window` フレーム）を `max_visits` 個まで選ぶ。
fn choose_visits(history: &[(u32, u32)], window: u32, max_visits: usize) -> TileVisits {
    let mut remaining: Vec<(u32, u32)> = history.to_vec();
    let mut visits: TileVisits = Vec::new();
    while visits.len() < max_visits {
        // 尺取り法: 窓の始まりを各フレームに置いたときの、窓内の点の合計の最大
        let mut best: Option<(u64, u32)> = None;
        let mut sum: u64 = 0;
        let mut end = 0usize;
        for begin in 0..remaining.len() {
            let start = remaining[begin].0;
            while end < remaining.len() && remaining[end].0 < start.saturating_add(window) {
                sum += u64::from(remaining[end].1);
                end += 1;
            }
            if sum > 0 && best.is_none_or(|(best_sum, _)| sum > best_sum) {
                best = Some((sum, start));
            }
            sum -= u64::from(remaining[begin].1);
        }
        let Some((_, start)) = best else {
            break;
        };
        let end_frame = start.saturating_add(window);
        visits.push((start, end_frame));
        // 選んだ窓のフレームは次の候補から外す
        remaining.retain(|&(frame, _)| !(start..end_frame).contains(&frame));
        if remaining.is_empty() {
            break;
        }
    }
    visits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(max_visits: usize) -> DenseColorMap {
        DenseColorMap::new(DenseColorMapConfig {
            cell_size: 0.025,
            tile_size: 0.5,
            visit_window_frames: 3,
            max_visits_per_tile: max_visits,
        })
    }

    /// 区画 (0,0,0) の中の、セル番号 `cell` の中央の点（x 方向に 2.5 cm ずつ並ぶ）。
    fn point(cell: u32, value: u8) -> PointXYZRGB {
        PointXYZRGB::new(
            Point3::new(0.0125 + 0.025 * cell as f32, 0.1, 0.1),
            [value, value, value],
        )
    }

    fn frames_of(result: &DenseColorResult) -> Vec<u32> {
        let mut frames: Vec<u32> = result.points.iter().map(|p| p.frame as u32).collect();
        frames.sort_unstable();
        frames
    }

    #[test]
    fn the_densest_visit_is_used_instead_of_the_first_one() {
        let mut map = map(1);
        // 1 周目（フレーム 0〜2）: 遠いので 1 フレームに 1 点しか入らない
        for frame in 0..3 {
            map.insert_frame(frame, &[point(frame as u32, 10)]);
        }
        // 2 周目（フレーム 100〜102）: 近いので 1 フレームに 4 点
        for frame in 100..103 {
            let points: Vec<_> = (0..4).map(|cell| point(cell, 200)).collect();
            map.insert_frame(frame, &points);
        }
        let result = map.build();
        // 2 周目の窓だけを使い、1 周目の点は混ざらない
        assert_eq!(
            frames_of(&result),
            vec![100, 100, 100, 100, 101, 101, 101, 101, 102, 102, 102, 102]
        );
        assert_eq!(result.summary.primary_points, 12);
        assert_eq!(result.summary.fill_points, 0);
        assert_eq!(result.summary.cells, 4);
    }

    #[test]
    fn only_one_contiguous_window_is_used_per_visit() {
        let mut map = map(1);
        // 同じ周回でも、窓（3 フレーム）の外のフレームは使わない
        for frame in 10..16 {
            let count = if (11..14).contains(&frame) { 3 } else { 1 };
            let points: Vec<_> = (0..count).map(|cell| point(cell, 100)).collect();
            map.insert_frame(frame, &points);
        }
        let result = map.build();
        let mut used: Vec<u32> = frames_of(&result);
        used.dedup();
        assert_eq!(used, vec![11, 12, 13]);
    }

    #[test]
    fn cells_missing_from_the_best_visit_are_filled_by_the_next_visit_without_mixing() {
        let mut map = map(3);
        // 1 周目: セル 0〜3 を少しずつ
        for frame in 0..3 {
            map.insert_frame(frame, &[point(0, 10), point(3, 10)]);
        }
        // 2 周目（一番良い）: セル 0〜2 をたくさん。セル 3 は写っていない
        for frame in 50..53 {
            map.insert_frame(frame, &[point(0, 200), point(1, 200), point(2, 200)]);
        }
        let result = map.build();
        let mut by_cell: FxHashMap<i32, Vec<u32>> = FxHashMap::default();
        for p in &result.points {
            let cell = (p.x / 0.025).floor() as i32;
            by_cell.entry(cell).or_default().push(p.frame as u32);
        }
        // セル 0〜2 は 2 周目だけ（1 周目のセル 0 の点は捨てる）、セル 3 だけ 1 周目で埋める
        assert_eq!(by_cell[&0], vec![50, 51, 52]);
        assert_eq!(by_cell[&1], vec![50, 51, 52]);
        assert_eq!(by_cell[&3], vec![0, 1, 2]);
        assert_eq!(result.summary.primary_points, 9);
        assert_eq!(result.summary.fill_points, 3);

        // 穴埋めしない設定ではセル 3 は空のまま
        let mut single = super::DenseColorMap::new(DenseColorMapConfig {
            max_visits_per_tile: 1,
            ..map.config.clone()
        });
        for frame in 0..3 {
            single.insert_frame(frame, &[point(0, 10), point(3, 10)]);
        }
        for frame in 50..53 {
            single.insert_frame(frame, &[point(0, 200), point(1, 200), point(2, 200)]);
        }
        assert_eq!(single.build().summary.cells, 3);
    }

    #[test]
    fn each_tile_chooses_its_own_visit() {
        let mut map = map(1);
        let far_tile = |value| PointXYZRGB::new(Point3::new(2.1, 0.1, 0.1), [value; 3]);
        // 区画 A はフレーム 0〜2 が良く、区画 B はフレーム 40〜42 が良い
        for frame in 0..3 {
            map.insert_frame(frame, &[point(0, 1), point(1, 1), far_tile(1)]);
        }
        for frame in 40..43 {
            let b: Vec<_> = (0..3)
                .map(|k| PointXYZRGB::new(Point3::new(2.1 + 0.025 * k as f32, 0.1, 0.1), [2; 3]))
                .collect();
            let mut points = b;
            points.push(point(0, 2));
            map.insert_frame(frame, &points);
        }
        let result = map.build();
        for p in &result.points {
            if p.x < 0.5 {
                assert!(p.frame < 3.0, "tile A must use its own visit");
            } else {
                assert!(p.frame >= 40.0, "tile B must use its own visit");
            }
        }
        assert_eq!(result.summary.tiles, 2);
    }

    #[test]
    fn choose_visits_picks_non_overlapping_windows_in_order_of_density() {
        let history = vec![(0, 1), (1, 1), (2, 1), (10, 5), (11, 5), (30, 2)];
        let visits = choose_visits(&history, 3, 3);
        assert_eq!(visits, vec![(10, 13), (0, 3), (30, 33)]);
        assert_eq!(choose_visits(&history, 3, 1), vec![(10, 13)]);
        assert!(choose_visits(&[], 3, 3).is_empty());
    }

    #[test]
    fn non_finite_points_are_ignored() {
        let mut map = map(1);
        let bad = PointXYZRGB::new(Point3::new(f32::NAN, 0.0, 0.0), [1, 2, 3]);
        assert_eq!(map.insert_frame(0, &[bad]), 0);
        assert_eq!(map.build().summary, DenseColorMapSummary::default());
    }
}
