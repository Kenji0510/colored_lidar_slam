//! Restrict each coplanar surface to one verified image component. A singleton
//! component is a single-image fallback, not evidence that its pose is refined.
use super::*;
use std::collections::BTreeSet;

pub(super) struct ImageGroups(Vec<usize>);
impl ImageGroups {
    pub(super) fn new(size: usize) -> Self {
        Self((0..size).collect())
    }
    pub(super) fn root(&mut self, id: usize) -> usize {
        let mut root = id;
        while self.0[root] != root {
            root = self.0[root];
        }
        let mut cursor = id;
        while self.0[cursor] != cursor {
            let next = self.0[cursor];
            self.0[cursor] = root;
            cursor = next;
        }
        root
    }
    pub(super) fn join(&mut self, a: usize, b: usize) {
        let (a, b) = (self.root(a), self.root(b));
        self.0[a.max(b)] = a.min(b);
    }
}

#[derive(Debug, Default, Serialize)]
pub struct ViewConsistencyStats {
    pub surfaces: usize,
    pub restricted_surfaces: usize,
    pub single_image_surfaces: usize,
    pub removed_candidate_views: usize,
    pub patches_without_compatible_view: usize,
    pub points_without_compatible_view: usize,
    pub shared_candidates_added: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SurfaceViewRecord {
    pub surface_id: usize,
    pub patches: Vec<SurfacePatchKey>,
    pub selected_image_group: Option<usize>,
    pub candidate_image_groups: usize,
    pub eligible_frame_ids: Vec<usize>,
    pub estimated_colored_points_before: usize,
    pub estimated_colored_points_after: usize,
    pub patches_without_compatible_view: usize,
}

/// Check all members against a fixed seed plane as well as local adjacency.
/// This prevents a chain of locally similar normals from merging a curved wall.
pub(super) fn coplanar_surfaces(
    patches: &[(SurfacePatchKey, Vec<usize>)],
    points: &[RegionalRecolorPoint],
    adjacency: &[Vec<usize>],
    config: &RegionalRecolorConfig,
) -> Vec<Vec<usize>> {
    let geometry: Vec<_> = patches
        .iter()
        .map(|(_, indices)| {
            let mut sum = Vector3::<f64>::zeros();
            let mut nsum = Vector3::<f64>::zeros();
            for &index in indices {
                sum += points[index].position.cast::<f64>().coords;
                if let Some(mut normal) = unit_normal(points[index].normal.as_ref()) {
                    if nsum.dot(&normal) < 0.0 {
                        normal = -normal;
                    }
                    nsum += normal;
                }
            }
            if indices.is_empty() || nsum.norm() < 1e-6 {
                None
            } else {
                Some((Point3::from(sum / indices.len() as f64), nsum.normalize()))
            }
        })
        .collect();
    let mut seen = vec![false; patches.len()];
    let mut result = Vec::new();
    for seed in 0..patches.len() {
        if seen[seed] {
            continue;
        }
        seen[seed] = true;
        let mut nodes = vec![seed];
        if let Some((center, normal)) = &geometry[seed] {
            let mut cursor = 0;
            while cursor < nodes.len() {
                let node = nodes[cursor];
                cursor += 1;
                for &other in &adjacency[node] {
                    if seen[other] {
                        continue;
                    }
                    let Some((position, n)) = &geometry[other] else {
                        continue;
                    };
                    let delta = position - center;
                    if n.dot(normal).abs() < config.coherence_normal_cos
                        || normal.dot(&delta).abs() > config.coherence_plane_distance_m
                        || n.dot(&delta).abs() > config.coherence_plane_distance_m
                    {
                        continue;
                    }
                    seen[other] = true;
                    nodes.push(other);
                }
            }
        }
        nodes.sort_unstable();
        result.push(nodes);
    }
    result
}

impl RegionalRecolor {
    /// Add shared whole-surface references before loading/scoring images. This
    /// rescues a wider image that the original per-region top-N pool discarded.
    pub(super) fn add_shared_surface_candidates(
        &self,
        surfaces: &[Vec<usize>],
        patches: &[(SurfacePatchKey, Vec<usize>)],
        points: &[RegionalRecolorPoint],
        by_frame: &mut BTreeMap<usize, Vec<usize>>,
    ) -> usize {
        if !self.config.pose_alignment.enabled {
            return 0;
        }
        let mut added = 0;
        for surface in surfaces {
            if surface.len() < 2 {
                continue;
            }
            let mut ranking = Vec::new();
            // Evaluate all cached images using geometry only, including images
            // no longer retained in this surface's local candidate pools.
            for (id, frame) in self.frames.iter().enumerate() {
                let mut covered = Vec::new();
                let mut support = 0;
                let mut distance = 0.0;
                for &node in surface {
                    let indices = &patches[node].1;
                    let budget = indices.len().min(16);
                    if budget == 0 {
                        continue;
                    }
                    let mut visible = 0;
                    for step in 0..budget {
                        let p = &points[indices[step * indices.len() / budget]];
                        let world = p.position.cast::<f64>();
                        let view = frame.camera_center - world;
                        let length = view.norm();
                        if length <= 0.0 || !length.is_finite() {
                            continue;
                        }
                        if unit_normal(p.normal.as_ref()).is_some_and(|n| {
                            n.dot(&(view / length)).abs() < self.config.min_front_cos
                        }) {
                            continue;
                        }
                        if project_camera_point_full_image(
                            &frame.camera_from_world.transform_point(&world),
                            &frame.intrinsics,
                        )
                        .is_some()
                        {
                            visible += 1;
                            distance += length;
                        }
                    }
                    if visible as f64 / budget as f64 >= self.config.min_visible_fraction {
                        covered.push(node);
                        support += indices.len();
                    }
                }
                if !covered.is_empty() {
                    ranking.push((support, distance, id, covered));
                }
            }
            ranking.sort_by(|a, b| {
                b.0.cmp(&a.0)
                    .then_with(|| a.1.total_cmp(&b.1))
                    .then_with(|| a.2.cmp(&b.2))
            });
            for (_, _, id, nodes) in ranking
                .into_iter()
                .take(self.config.pose_alignment.shared_surface_candidates)
            {
                let existing = by_frame.entry(id).or_default();
                for node in nodes {
                    if !existing.contains(&node) {
                        existing.push(node);
                        added += 1;
                    }
                }
            }
        }
        for nodes in by_frame.values_mut() {
            nodes.sort_unstable();
            nodes.dedup();
        }
        added
    }
}

/// Select the image group with the greatest predicted colored-point coverage,
/// then normalized quality. Only images within that group remain eligible.
/// Missing views stay explicitly uncolored rather than silently mixing groups.
pub(super) fn restrict_surface_views(
    options: &mut [Vec<PatchSelection>],
    surfaces: &[Vec<usize>],
    patches: &[(SurfacePatchKey, Vec<usize>)],
    image_groups: &[usize],
    image_identity: &[usize],
    stats: &mut ViewConsistencyStats,
) -> Vec<SurfaceViewRecord> {
    let mut records = Vec::new();
    for (surface_id, nodes) in surfaces.iter().enumerate() {
        stats.surfaces += 1;
        let groups: BTreeSet<_> = nodes
            .iter()
            .flat_map(|&n| options[n].iter().map(|v| image_groups[v.frame_id]))
            .collect();
        let predicted = |views: &[PatchSelection]| {
            views
                .iter()
                .map(|v| (v.coverage * v.point_count as f64).round() as usize)
                .max()
                .unwrap_or(0)
        };
        let before = nodes.iter().map(|&n| predicted(&options[n])).sum();
        let mut ranking = Vec::new();
        for &group in &groups {
            let mut support = 0;
            let mut quality = 0.0;
            for &node in nodes {
                let best = options[node]
                    .iter()
                    .map(|v| v.score)
                    .fold(0.0_f64, f64::max)
                    .max(f64::MIN_POSITIVE);
                let available: Vec<_> = options[node]
                    .iter()
                    .filter(|v| image_groups[v.frame_id] == group)
                    .collect();
                support += available
                    .iter()
                    .map(|v| (v.coverage * v.point_count as f64).round() as usize)
                    .max()
                    .unwrap_or(0);
                quality += available
                    .iter()
                    .map(|v| v.score / best * v.point_count as f64)
                    .fold(0.0_f64, f64::max);
            }
            ranking.push((support, quality, group));
        }
        ranking.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.total_cmp(&a.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        let selected = ranking.first().map(|r| r.2);
        if groups.len() > 1 {
            stats.restricted_surfaces += 1;
        }
        let mut missing = 0;
        for &node in nodes {
            let count = options[node].len();
            options[node].retain(|v| Some(image_groups[v.frame_id]) == selected);
            stats.removed_candidate_views += count - options[node].len();
            if count > 0 && options[node].is_empty() {
                missing += 1;
                stats.patches_without_compatible_view += 1;
                stats.points_without_compatible_view += patches[node].1.len();
            }
        }
        let frame_ids: BTreeSet<_> = nodes
            .iter()
            .flat_map(|&n| options[n].iter().map(|v| v.frame_id))
            .collect();
        if frame_ids
            .iter()
            .map(|&id| image_identity[id])
            .collect::<BTreeSet<_>>()
            .len()
            == 1
        {
            stats.single_image_surfaces += 1;
        }
        records.push(SurfaceViewRecord {
            surface_id,
            patches: nodes.iter().map(|&n| patches[n].0).collect(),
            selected_image_group: selected,
            candidate_image_groups: groups.len(),
            eligible_frame_ids: frame_ids.into_iter().collect(),
            estimated_colored_points_before: before,
            estimated_colored_points_after: nodes.iter().map(|&n| predicted(&options[n])).sum(),
            patches_without_compatible_view: missing,
        });
    }
    records
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patches(size: usize) -> Vec<(SurfacePatchKey, Vec<usize>)> {
        (0..size)
            .map(|i| {
                (
                    SurfacePatchKey {
                        region: [i as i64, 0, 0],
                        normal_axis: 2,
                    },
                    (i * 10..i * 10 + 10).collect(),
                )
            })
            .collect()
    }
    fn view(node: usize, image: usize, score: f64) -> PatchSelection {
        PatchSelection {
            key: SurfacePatchKey {
                region: [node as i64, 0, 0],
                normal_axis: 2,
            },
            frame_id: image,
            frame_index: image,
            score,
            coverage: 1.0,
            sharpness: 50.0,
            mean_distance_m: 5.0,
            mean_front_cos: 1.0,
            point_count: 10,
            colored_count: 0,
        }
    }

    #[test]
    fn an_unverified_image_cannot_win_a_tile_inside_a_verified_surface() {
        let mut options = vec![
            vec![view(0, 0, 1.0)],
            vec![view(1, 2, 10.0), view(1, 1, 1.0)],
            vec![view(2, 1, 1.0)],
        ];
        let mut stats = ViewConsistencyStats::default();
        let records = restrict_surface_views(
            &mut options,
            &[vec![0, 1, 2]],
            &patches(3),
            &[0, 0, 2],
            &[0, 1, 2],
            &mut stats,
        );
        assert_eq!(records[0].selected_image_group, Some(0));
        assert_eq!(options[1][0].frame_id, 1);
        assert_eq!(stats.removed_candidate_views, 1);
        assert_eq!(stats.points_without_compatible_view, 0);
        assert_eq!(records[0].eligible_frame_ids, vec![0, 1]);
    }

    #[test]
    fn a_lower_quality_common_image_is_preferred_to_disconnected_sharp_tiles() {
        let mut options = vec![
            vec![view(0, 0, 1.0), view(0, 1, 100.0)],
            vec![view(1, 0, 0.05), view(1, 2, 100.0)],
            vec![view(2, 0, 0.05), view(2, 2, 100.0)],
        ];
        let mut stats = ViewConsistencyStats::default();
        let records = restrict_surface_views(
            &mut options,
            &[vec![0, 1, 2]],
            &patches(3),
            &[0, 1, 2],
            &[0, 1, 2],
            &mut stats,
        );
        assert!(options.iter().all(|o| o.len() == 1 && o[0].frame_id == 0));
        assert_eq!(records[0].estimated_colored_points_after, 30);
        assert_eq!(stats.single_image_surfaces, 1);
    }

    #[test]
    fn incompatible_coverage_is_reported_without_silently_reintroducing_a_seam() {
        let mut options = vec![vec![view(0, 0, 1.0)], vec![view(1, 1, 1.0)]];
        let mut stats = ViewConsistencyStats::default();
        let records = restrict_surface_views(
            &mut options,
            &[vec![0, 1]],
            &patches(2),
            &[0, 1],
            &[0, 1],
            &mut stats,
        );
        assert_eq!(options[0].len(), 1);
        assert!(options[1].is_empty());
        assert_eq!(stats.patches_without_compatible_view, 1);
        assert_eq!(stats.points_without_compatible_view, 10);
        assert_eq!(records[0].estimated_colored_points_before, 20);
        assert_eq!(records[0].estimated_colored_points_after, 10);
    }

    #[test]
    fn separate_surfaces_keep_their_own_images() {
        let mut options = vec![vec![view(0, 0, 1.0)], vec![view(1, 1, 1.0)]];
        let mut stats = ViewConsistencyStats::default();
        restrict_surface_views(
            &mut options,
            &[vec![0], vec![1]],
            &patches(2),
            &[0, 1],
            &[0, 1],
            &mut stats,
        );
        assert!(options.iter().all(|o| o.len() == 1));
        assert_eq!(stats.points_without_compatible_view, 0);
    }

    #[test]
    fn chains_of_normals_do_not_merge_different_planes() {
        let keys = vec![
            (
                SurfacePatchKey {
                    region: [0, 0, 5],
                    normal_axis: 2,
                },
                vec![0],
            ),
            (
                SurfacePatchKey {
                    region: [1, 0, 5],
                    normal_axis: 2,
                },
                vec![1],
            ),
            (
                SurfacePatchKey {
                    region: [2, 0, 5],
                    normal_axis: 2,
                },
                vec![2],
            ),
        ];
        let points: Vec<_> = [0.0_f32, 10.0, 20.0]
            .into_iter()
            .enumerate()
            .map(|(i, angle)| {
                let t = angle.to_radians();
                RegionalRecolorPoint {
                    position: Point3::new(i as f32 * 0.1, 0.0, 5.0),
                    normal: Some(Vector3::new(t.sin(), 0.0, t.cos())),
                }
            })
            .collect();
        let groups = coplanar_surfaces(
            &keys,
            &points,
            &[vec![1], vec![0, 2], vec![1]],
            &Default::default(),
        );
        assert_eq!(groups, vec![vec![0, 1], vec![2]]);
    }
}
