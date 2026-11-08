//! Image-wide pose correction before regional image selection and coloring.
//! Each physical image has one pose, even when several LiDAR scans use it.
use super::*;
use crate::camera::{camera_ray_for_pixel, project_camera_point_continuous};
use crate::camera_pose_alignment::{
    PoseEstimate, PoseMatch, correction, estimate_camera_pose, huber_cost, match_patch,
    pose_prior_cost,
};
use image::GrayImage;
use std::collections::{BTreeSet, VecDeque};
use std::rc::Rc;

#[derive(Debug, Clone, Default, Serialize)]
pub struct CameraPoseAlignmentStats {
    pub images: usize,
    pub duplicate_observations: usize,
    pub candidate_pairs: usize,
    pub matched_pairs: usize,
    pub patch_matches: usize,
    pub anchor_images: usize,
    pub refined_images: usize,
    pub unchanged_images: usize,
    pub graph_iterations: usize,
    pub graph_cost_before: f64,
    pub graph_cost_after: f64,
    pub images_loaded: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CameraPoseAlignmentRecord {
    pub frame_id: usize,
    pub frame_index: usize,
    pub image_path: PathBuf,
    pub corrected_depth_path: PathBuf,
    pub shared_by_frame_indices: Vec<usize>,
    pub anchor: bool,
    pub estimate: PoseEstimate,
    /// All matrices use nalgebra's column-major order, T_camera_from_world.
    pub original_camera_from_world_column_major: [f64; 16],
    pub corrected_camera_from_world_column_major: [f64; 16],
    pub rotation_correction_deg: f64,
    pub camera_center_displacement_m: f64,
}

struct PairMatch {
    plane_center: Point3<f64>,
    normal: Vector3<f64>,
    a_pixel: [f64; 2],
    b_pixel: [f64; 2],
}

struct Pair {
    a: usize,
    b: usize,
    matches: Vec<PairMatch>,
}

struct View {
    gray: GrayImage,
    depth: DepthRaster,
}

/// Bound working image memory. Images can be revisited by neighboring pairs.
#[derive(Default)]
struct ViewCache(VecDeque<(usize, Rc<View>)>);
impl ViewCache {
    fn load<F>(
        &mut self,
        id: usize,
        frames: &[CachedFrame],
        load_image: &mut F,
        stats: &mut CameraPoseAlignmentStats,
    ) -> Result<Rc<View>>
    where
        F: FnMut(&Path) -> Result<RgbImage>,
    {
        if let Some((_, view)) = self.0.iter().find(|(key, _)| *key == id) {
            return Ok(Rc::clone(view));
        }
        let frame = &frames[id];
        let image = load_image(&frame.image_path)?;
        ensure!(
            image.dimensions() == (frame.intrinsics.width, frame.intrinsics.height),
            "camera pose alignment image dimensions changed: {}",
            frame.image_path.display()
        );
        let view = Rc::new(View {
            gray: image::DynamicImage::ImageRgb8(image).into_luma8(),
            depth: DepthRaster::read_sparse(&frame.depth_path, &frame.intrinsics)?,
        });
        stats.images_loaded += 1;
        if self.0.len() == 4 {
            self.0.pop_front();
        }
        self.0.push_back((id, Rc::clone(&view)));
        Ok(view)
    }
}

fn matrix_array(matrix: &Matrix4<f64>) -> [f64; 16] {
    let mut result = [0.0; 16];
    result.copy_from_slice(matrix.as_slice());
    result
}

fn lift(
    pixel: [f64; 2],
    center: &Point3<f64>,
    normal: &Vector3<f64>,
    world_from_camera: &Matrix4<f64>,
    camera: &CameraIntrinsics,
) -> Option<Point3<f64>> {
    let ray = camera_ray_for_pixel(pixel[0], pixel[1], camera)?;
    let origin = world_from_camera.transform_point(&Point3::origin());
    let direction = world_from_camera.fixed_view::<3, 3>(0, 0) * ray;
    let denominator = normal.dot(&direction);
    if !denominator.is_finite() || denominator.abs() < 1e-8 {
        return None;
    }
    let distance = normal.dot(&(center - origin)) / denominator;
    if !distance.is_finite() || distance <= 0.0 {
        return None;
    }
    Some(origin + direction * distance)
}

fn matches_for(
    id: usize,
    pairs: &[Pair],
    incident: &[usize],
    poses: &[Matrix4<f64>],
    frames: &[CachedFrame],
) -> Vec<PoseMatch> {
    let mut result = Vec::new();
    for &edge in incident {
        let pair = &pairs[edge];
        let other = if pair.a == id { pair.b } else { pair.a };
        let Some(world_from_other) = poses[other].try_inverse() else {
            continue;
        };
        for m in &pair.matches {
            let (source, target) = if pair.b == id {
                (m.a_pixel, m.b_pixel)
            } else {
                (m.b_pixel, m.a_pixel)
            };
            if let Some(world) = lift(
                source,
                &m.plane_center,
                &m.normal,
                &world_from_other,
                &frames[other].intrinsics,
            ) {
                result.push(PoseMatch {
                    world,
                    pixel: target,
                });
            }
        }
    }
    result
}

fn pair_cost(
    pair: &Pair,
    poses: &[Matrix4<f64>],
    frames: &[CachedFrame],
    config: &CameraPoseAlignmentConfig,
) -> f64 {
    let Some(wa) = poses[pair.a].try_inverse() else {
        return f64::INFINITY;
    };
    let Some(wb) = poses[pair.b].try_inverse() else {
        return f64::INFINITY;
    };
    let mut cost = 0.0;
    for m in &pair.matches {
        for (pixel, target, world_from_source, source, destination) in [
            (m.a_pixel, m.b_pixel, &wa, pair.a, pair.b),
            (m.b_pixel, m.a_pixel, &wb, pair.b, pair.a),
        ] {
            let Some(world) = lift(
                pixel,
                &m.plane_center,
                &m.normal,
                world_from_source,
                &frames[source].intrinsics,
            ) else {
                return f64::INFINITY;
            };
            let Some((u, v)) = project_camera_point_continuous(
                &poses[destination].transform_point(&world),
                &frames[destination].intrinsics,
            ) else {
                return f64::INFINITY;
            };
            if u < 0.0
                || v < 0.0
                || u >= frames[destination].intrinsics.width as f64 - 1.0
                || v >= frames[destination].intrinsics.height as f64 - 1.0
            {
                return f64::INFINITY;
            }
            cost += huber_cost((u - target[0]).hypot(v - target[1]), config.huber_px);
        }
    }
    cost
}

impl RegionalRecolor {
    pub(super) fn pose_aligned_frames<F>(
        &self,
        patches: &[(SurfacePatchKey, Vec<usize>)],
        points: &[RegionalRecolorPoint],
        adjacency: &[Vec<usize>],
        by_frame: &BTreeMap<usize, Vec<usize>>,
        load_image: &mut F,
        stats: &mut RegionalRecolorStats,
    ) -> Result<(Vec<CachedFrame>, Vec<CameraPoseAlignmentRecord>)>
    where
        F: FnMut(&Path) -> Result<RgbImage>,
    {
        let mut frames = self.frames.clone();
        let config = &self.config.pose_alignment;
        if !config.enabled {
            return Ok((frames, Vec::new()));
        }
        let pose_stats = &mut stats.camera_pose_alignment;

        // Repeated selection of the same camera file must not create independent
        // pose variables. The first observation defines its LiDAR-pose prior.
        let mut canonical = vec![0; frames.len()];
        let mut by_path = BTreeMap::<PathBuf, usize>::new();
        let mut aliases = BTreeMap::<usize, Vec<usize>>::new();
        for (id, frame) in frames.iter().enumerate() {
            let root = *by_path.entry(frame.image_path.clone()).or_insert(id);
            canonical[id] = root;
            aliases.entry(root).or_default().push(id);
        }
        let active: BTreeSet<_> = by_frame.keys().map(|&id| canonical[id]).collect();
        pose_stats.images = active.len();
        pose_stats.duplicate_observations = by_frame.len().saturating_sub(active.len());

        // Candidate pools and neighboring surface regions provide overlap edges.
        // This runs even when coherence_weight=0, so pose alignment is independent
        // of the image-label smoothing option.
        let mut patch_frames = vec![Vec::new(); patches.len()];
        for (&id, nodes) in by_frame {
            for &node in nodes {
                patch_frames[node].push(canonical[id]);
            }
        }
        let mut sources = BTreeMap::<(usize, usize), BTreeSet<usize>>::new();
        for node in 0..patches.len() {
            let mut ids = patch_frames[node].clone();
            for &other in &adjacency[node] {
                ids.extend(&patch_frames[other]);
            }
            ids.sort_unstable();
            ids.dedup();
            // Favor cameras near this region, rather than earliest frame IDs.
            let center = region_center(patches[node].0.region, self.config.region_size_m);
            ids.sort_by(|&a, &b| {
                (frames[a].camera_center - center)
                    .norm_squared()
                    .total_cmp(&(frames[b].camera_center - center).norm_squared())
                    .then_with(|| a.cmp(&b))
            });
            ids.truncate(8);
            ids.sort_unstable();
            for (position, &a) in ids.iter().enumerate() {
                for &b in &ids[position + 1..] {
                    let entry = sources.entry((a, b)).or_default();
                    entry.insert(node);
                }
            }
        }
        pose_stats.candidate_pairs = sources.len();
        let mut sources: Vec<_> = sources.into_iter().collect();
        sources.sort_by(|a, b| {
            let count =
                |nodes: &BTreeSet<usize>| nodes.iter().map(|&n| patches[n].1.len()).sum::<usize>();
            count(&b.1).cmp(&count(&a.1)).then_with(|| a.0.cmp(&b.0))
        });
        let mut degree = vec![0; frames.len()];
        let mut pairs = Vec::new();
        let mut cache = ViewCache::default();
        let total = sources.len();
        for (step, ((a, b), nodes)) in sources.into_iter().enumerate() {
            if degree[a] >= config.max_pairs_per_image || degree[b] >= config.max_pairs_per_image {
                continue;
            }
            if step % 25 == 0 {
                log::info!(
                    "Camera pose alignment: matching pair {}/{}",
                    step + 1,
                    total
                );
            }
            let va = cache.load(a, &frames, load_image, pose_stats)?;
            let vb = cache.load(b, &frames, load_image, pose_stats)?;
            let matches =
                self.pose_pair_matches(&nodes, patches, points, &frames[a], &frames[b], &va, &vb);
            if matches.len() < config.min_matches {
                continue;
            }
            degree[a] += 1;
            degree[b] += 1;
            pose_stats.patch_matches += matches.len();
            pairs.push(Pair { a, b, matches });
        }
        drop(cache);
        pose_stats.matched_pairs = pairs.len();
        let mut incident = vec![Vec::new(); frames.len()];
        for (edge, pair) in pairs.iter().enumerate() {
            incident[pair.a].push(edge);
            incident[pair.b].push(edge);
        }

        // A connected image component has one fixed anchor. Every incident plane
        // contributes to the same image pose, including across separate surfaces.
        let mut anchors = BTreeSet::new();
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        for &start in &active {
            if !visited.insert(start) {
                continue;
            }
            let mut component = vec![start];
            let mut cursor = 0;
            while cursor < component.len() {
                let id = component[cursor];
                cursor += 1;
                for &edge in &incident[id] {
                    let pair = &pairs[edge];
                    let other = if pair.a == id { pair.b } else { pair.a };
                    if visited.insert(other) {
                        component.push(other);
                    }
                }
            }
            let support = |id: usize| {
                incident[id]
                    .iter()
                    .map(|&edge| pairs[edge].matches.len())
                    .sum::<usize>()
            };
            let anchor = *component
                .iter()
                .max_by(|&&a, &&b| support(a).cmp(&support(b)).then_with(|| b.cmp(&a)))
                .unwrap();
            anchors.insert(anchor);
            let mut queue = VecDeque::from([anchor]);
            let mut seen = HashSet::from([anchor]);
            while let Some(id) = queue.pop_front() {
                if id != anchor {
                    order.push(id);
                }
                for &edge in &incident[id] {
                    let pair = &pairs[edge];
                    let other = if pair.a == id { pair.b } else { pair.a };
                    if seen.insert(other) {
                        queue.push_back(other);
                    }
                }
            }
        }
        pose_stats.anchor_images = anchors.len();
        let original: Vec<_> = frames.iter().map(|f| f.camera_from_world).collect();
        let mut poses = original.clone();
        let mut parameters = vec![[0.0; 6]; frames.len()];
        let mut estimates = HashMap::<usize, PoseEstimate>::new();
        let mut attempts = HashMap::<usize, PoseEstimate>::new();
        pose_stats.graph_cost_before = pairs
            .iter()
            .map(|p| pair_cost(p, &poses, &frames, config))
            .sum();
        for sweep in 0..config.graph_iterations {
            let mut changed = 0;
            for &id in &order {
                let matches = matches_for(id, &pairs, &incident[id], &poses, &frames);
                let mut estimate =
                    estimate_camera_pose(&original[id], &frames[id].intrinsics, &matches, config);
                if estimate.accepted {
                    let before = incident[id]
                        .iter()
                        .map(|&edge| pair_cost(&pairs[edge], &poses, &frames, config))
                        .sum::<f64>()
                        + pose_prior_cost(&parameters[id], config);
                    let old = poses[id];
                    poses[id] =
                        correction(&estimate.correction_rotation_rad_translation_m) * original[id];
                    let after = incident[id]
                        .iter()
                        .map(|&edge| pair_cost(&pairs[edge], &poses, &frames, config))
                        .sum::<f64>()
                        + pose_prior_cost(&estimate.correction_rotation_rad_translation_m, config);
                    if after.is_finite() && after < before - 1e-7 {
                        parameters[id] = estimate.correction_rotation_rad_translation_m;
                        estimates.insert(id, estimate.clone());
                        changed += 1;
                    } else {
                        poses[id] = old;
                        estimate.accepted = false;
                        estimate.reason = "graph_cost_not_improved";
                        estimate.correction_rotation_rad_translation_m = [0.0; 6];
                    }
                }
                attempts.insert(id, estimate);
            }
            pose_stats.graph_iterations = sweep + 1;
            log::info!(
                "Camera pose alignment: graph sweep={} updated_images={}",
                sweep + 1,
                changed
            );
            if changed == 0 {
                break;
            }
        }
        pose_stats.graph_cost_after = pairs
            .iter()
            .map(|p| pair_cost(p, &poses, &frames, config))
            .sum::<f64>()
            + parameters
                .iter()
                .map(|p| pose_prior_cost(p, config))
                .sum::<f64>();
        let mut records = Vec::new();
        let depth_dir = self.cache_dir.join(format!(
            "pose-depth-{}",
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        if !active.is_empty() {
            fs::create_dir(&depth_dir)?;
        }
        for (step, &id) in active.iter().enumerate() {
            if step % 25 == 0 || step + 1 == active.len() {
                log::info!(
                    "Camera pose alignment: updating depth {}/{}",
                    step + 1,
                    active.len()
                );
            }
            let anchor = anchors.contains(&id);
            let estimate = estimates.remove(&id).unwrap_or_else(|| {
                if anchor {
                    PoseEstimate::rejected(if incident[id].is_empty() {
                        "no_overlap_matches"
                    } else {
                        "fixed_component_anchor"
                    })
                } else {
                    attempts
                        .remove(&id)
                        .unwrap_or_else(|| PoseEstimate::rejected("no_overlap_matches"))
                }
            });
            if estimate.accepted {
                pose_stats.refined_images += 1;
            }
            let original_center = original[id]
                .try_inverse()
                .context("singular original camera pose")?
                .transform_point(&Point3::origin());
            let center = poses[id]
                .try_inverse()
                .context("singular corrected camera pose")?
                .transform_point(&Point3::origin());
            let rotation = Vector3::new(parameters[id][0], parameters[id][1], parameters[id][2])
                .norm()
                .to_degrees();
            let depth_path = depth_dir.join(format!("frame-{id:08}.depth"));
            records.push(CameraPoseAlignmentRecord {
                frame_id: id,
                frame_index: frames[id].frame_index,
                image_path: frames[id].image_path.clone(),
                corrected_depth_path: depth_path.clone(),
                shared_by_frame_indices: aliases[&id]
                    .iter()
                    .map(|&alias| frames[alias].frame_index)
                    .collect(),
                anchor,
                estimate,
                original_camera_from_world_column_major: matrix_array(&original[id]),
                corrected_camera_from_world_column_major: matrix_array(&poses[id]),
                rotation_correction_deg: rotation,
                camera_center_displacement_m: (center - original_center).norm(),
            });
            // All original observed depths, including aliases, are reprojected
            // into the one corrected image pose. Preserve raw occluders that may
            // not be present in the final planar point map.
            let mut depth = DepthRaster::build(&[], &Matrix4::identity(), &frames[id].intrinsics);
            for &alias in &aliases[&id] {
                let raw = DepthRaster::read_sparse(
                    &self.frames[alias].depth_path,
                    &frames[id].intrinsics,
                )?;
                let old_inverse = self.frames[alias]
                    .camera_from_world
                    .try_inverse()
                    .context("singular depth camera pose")?;
                depth.merge(&raw.reproject(&(poses[id] * old_inverse), &frames[id].intrinsics));
            }
            depth.write_sparse(&depth_path)?;
            for &alias in &aliases[&id] {
                frames[alias].camera_from_world = poses[id];
                frames[alias].camera_center = center;
                frames[alias].depth_path = depth_path.clone();
            }
        }
        pose_stats.unchanged_images = pose_stats.images - pose_stats.refined_images;
        log::info!(
            "Camera pose alignment: images={} refined={} anchors={} matched_pairs={} patch_matches={} cost={:.3}->{:.3}",
            pose_stats.images,
            pose_stats.refined_images,
            pose_stats.anchor_images,
            pose_stats.matched_pairs,
            pose_stats.patch_matches,
            pose_stats.graph_cost_before,
            pose_stats.graph_cost_after
        );
        Ok((frames, records))
    }

    #[allow(clippy::too_many_arguments)]
    fn pose_pair_matches(
        &self,
        nodes: &BTreeSet<usize>,
        patches: &[(SurfacePatchKey, Vec<usize>)],
        points: &[RegionalRecolorPoint],
        a: &CachedFrame,
        b: &CachedFrame,
        va: &View,
        vb: &View,
    ) -> Vec<PairMatch> {
        let config = &self.config.pose_alignment;
        let regions: HashSet<_> = nodes.iter().map(|&node| patches[node].0.region).collect();
        let mut bins = BTreeMap::<(u32, u32), (f64, usize)>::new();
        for &node in nodes {
            let indices = &patches[node].1;
            for &index in indices.iter().step_by((indices.len() / 256).max(1)) {
                let point = &points[index];
                if unit_normal(point.normal.as_ref()).is_none() {
                    continue;
                }
                if self.visible_sample(point, a, &va.depth).is_none()
                    || self.visible_sample(point, b, &vb.depth).is_none()
                {
                    continue;
                }
                let world = point.position.cast::<f64>();
                let Some((u, v)) = project_camera_point_continuous(
                    &a.camera_from_world.transform_point(&world),
                    &a.intrinsics,
                ) else {
                    continue;
                };
                let bin = ((v / 24.0) as u32, (u / 24.0) as u32);
                let rank = (u - (bin.1 as f64 + 0.5) * 24.0).powi(2)
                    + (v - (bin.0 as f64 + 0.5) * 24.0).powi(2);
                let entry = bins.entry(bin).or_insert((rank, index));
                if rank < entry.0 || (rank == entry.0 && index < entry.1) {
                    *entry = (rank, index);
                }
            }
        }
        let seeds: Vec<_> = bins.into_values().map(|x| x.1).collect();
        let budget = seeds.len().min(config.max_seed_attempts);
        let patch_config = PatchAlignmentConfig {
            patch_half_width: 3,
            sample_step_px: 1.5,
            ..Default::default()
        };
        let mut matches = Vec::new();
        for step in 0..budget {
            let point = &points[seeds[step * seeds.len() / budget]];
            let world = point.position.cast::<f64>();
            let normal = unit_normal(point.normal.as_ref()).unwrap();
            let mut patch = plane_patch_samples(
                &world,
                &normal,
                &a.camera_from_world,
                &b.camera_from_world,
                &a.intrinsics,
                &b.intrinsics,
                &patch_config,
            );
            patch.retain(|s| {
                if !spatial_key(&s.world, self.config.region_size_m)
                    .is_some_and(|key| regions.contains(&key))
                {
                    return false;
                }
                let p = RegionalRecolorPoint {
                    position: s.world.cast::<f32>(),
                    normal: Some(normal.cast::<f32>()),
                };
                let (Some(ar), Some(br)) = (
                    self.visible_sample(&p, a, &va.depth),
                    self.visible_sample(&p, b, &vb.depth),
                ) else {
                    return false;
                };
                !va.depth.has_depth_discontinuity(
                    ar.u,
                    ar.v,
                    a.camera_from_world.transform_point(&s.world).z,
                    self.config.occlusion_margin_m,
                    self.config.occlusion_radius_px + 1,
                ) && !vb.depth.has_depth_discontinuity(
                    br.u,
                    br.v,
                    b.camera_from_world.transform_point(&s.world).z,
                    self.config.occlusion_margin_m,
                    config.search_radius_px as u32 + self.config.occlusion_radius_px + 1,
                )
            });
            let Some((shift, _ncc)) = match_patch(&va.gray, &vb.gray, &patch, config) else {
                continue;
            };
            let Some((au, av)) = project_camera_point_continuous(
                &a.camera_from_world.transform_point(&world),
                &a.intrinsics,
            ) else {
                continue;
            };
            let Some((bu, bv)) = project_camera_point_continuous(
                &b.camera_from_world.transform_point(&world),
                &b.intrinsics,
            ) else {
                continue;
            };
            let pixel = [bu + shift[0], bv + shift[1]];
            if pixel[0] < 0.0
                || pixel[1] < 0.0
                || pixel[0] >= b.intrinsics.width as f64 - 1.0
                || pixel[1] >= b.intrinsics.height as f64 - 1.0
            {
                continue;
            }
            matches.push(PairMatch {
                plane_center: world,
                normal,
                a_pixel: [au, av],
                b_pixel: pixel,
            });
        }
        // Uniform subsampling retains the image-wide spread of the correspondences.
        if matches.len() > config.max_matches_per_pair {
            let step = matches.len() as f64 / config.max_matches_per_pair as f64;
            let keep: HashSet<_> = (0..config.max_matches_per_pair)
                .map(|i| (i as f64 * step).floor() as usize)
                .collect();
            matches = matches
                .into_iter()
                .enumerate()
                .filter_map(|(i, m)| keep.contains(&i).then_some(m))
                .collect();
        }
        matches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_pose_alignment::sample_rgb_bilinear;
    use image::Rgb;

    struct Scratch(PathBuf);
    impl Scratch {
        fn new(label: &str) -> Self {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "pose-recolor-{label}-{}-{stamp}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn config() -> RegionalRecolorConfig {
        RegionalRecolorConfig {
            region_size_m: 1.0,
            evaluate_every: 1,
            min_points_per_frame_region: 1,
            coherence_weight: 0.0,
            occlusion_radius_px: 1,
            ..Default::default()
        }
    }
    fn texture() -> RgbImage {
        let raw = RgbImage::from_fn(960, 600, |x, y| {
            let mut v = (x as u64).wrapping_mul(0x9e3779b97f4a7c15)
                ^ (y as u64).wrapping_mul(0xbf58476d1ce4e5b9);
            v = (v ^ (v >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            v = (v ^ (v >> 27)).wrapping_mul(0x94d049bb133111eb);
            let g = (v ^ (v >> 31)) as u8;
            Rgb([g, g, g])
        });
        image::imageops::blur(&raw, 0.6)
    }
    fn rendered(reference: &RgbImage, camera_from_world: &Matrix4<f64>, plane_z: f64) -> RgbImage {
        let camera = CameraIntrinsics::for_image_size(960, 600).unwrap();
        let inverse = camera_from_world.try_inverse().unwrap();
        RgbImage::from_fn(960, 600, |x, y| {
            let Some(world) = lift(
                [x as f64, y as f64],
                &Point3::new(0.0, 0.0, plane_z),
                &Vector3::z(),
                &inverse,
                &camera,
            ) else {
                return Rgb([0; 3]);
            };
            let Some((u, v)) = project_camera_point_continuous(&world, &camera) else {
                return Rgb([0; 3]);
            };
            Rgb(sample_rgb_bilinear(reference, [u, v]).unwrap_or([0; 3]))
        })
    }

    #[test]
    fn image_wide_pose_reduces_color_seams_and_is_shared_by_duplicate_observations() {
        let scratch = Scratch::new("rendered");
        let mut state = RegionalRecolor::new(config(), &scratch.0).unwrap();
        let reference = texture();
        let nominal = Matrix4::identity();
        let truth = correction(&[0.0015, -0.0025, 0.002, 0.009, -0.012, 0.008]);
        let target = rendered(&reference, &truth, 5.1);
        let raw: Vec<_> = (-55..=55)
            .filter(|&x| x != 0)
            .flat_map(|x| {
                (-35..=35).map(move |y| Point3::new(x as f32 * 0.03, y as f32 * 0.03, 5.1))
            })
            .collect();
        let left: Vec<_> = raw.iter().copied().filter(|p| p.x < 0.0).collect();
        let right: Vec<_> = raw.iter().copied().filter(|p| p.x > 0.0).collect();
        for (id, path, image, points) in [
            (0, "reference", &reference, &left),
            (1, "target", &target, &right),
            (2, "target", &target, &right),
        ] {
            state
                .observe_frame(
                    id,
                    if id == 2 { 1.0 } else { id as f64 },
                    Path::new(path),
                    image,
                    &nominal,
                    &Matrix4::identity(),
                    points,
                )
                .unwrap();
        }
        let points: Vec<_> = raw
            .iter()
            .map(|p| RegionalRecolorPoint {
                position: *p,
                normal: Some(Vector3::z()),
            })
            .collect();
        let load = |path: &Path| {
            Ok(if path == Path::new("reference") {
                reference.clone()
            } else {
                target.clone()
            })
        };
        let before = state.recolor_points(&points, load).unwrap();
        assert!(before.camera_poses.is_empty());
        state.config.pose_alignment.enabled = true;
        let after = state.recolor_points(&points, load).unwrap();
        assert_eq!(after.camera_poses.len(), 2);
        let anchor = &after.camera_poses[0];
        let corrected = &after.camera_poses[1];
        assert!(anchor.anchor && !anchor.estimate.accepted);
        assert_eq!(
            anchor.original_camera_from_world_column_major,
            anchor.corrected_camera_from_world_column_major
        );
        assert!(
            corrected.estimate.accepted,
            "{:?}; stats={:?}",
            corrected, after.stats.camera_pose_alignment
        );
        assert_eq!(corrected.shared_by_frame_indices, vec![1, 2]);
        assert!(corrected.camera_center_displacement_m <= 0.1);
        assert!(
            after.stats.camera_pose_alignment.graph_cost_after
                < after.stats.camera_pose_alignment.graph_cost_before
        );
        assert_eq!(after.stats.colored_points, before.stats.colored_points);
        assert!(after.selections.iter().filter(|s| s.frame_id != 0).count() > 1);
        let camera = CameraIntrinsics::for_image_size(960, 600).unwrap();
        let mut error_before = 0.0;
        let mut error_after = 0.0;
        let mut count = 0;
        for (i, point) in points.iter().enumerate() {
            assert_eq!(point.position, raw[i]);
            if point.position.x <= 0.0 {
                continue;
            }
            let (u, v) =
                project_camera_point_continuous(&point.position.cast::<f64>(), &camera).unwrap();
            let expected = sample_rgb_bilinear(&reference, [u, v]).unwrap();
            let (Some(a), Some(b)) = (before.colors[i], after.colors[i]) else {
                continue;
            };
            error_before += (a[0] as f64 - expected[0] as f64).abs();
            error_after += (b[0] as f64 - expected[0] as f64).abs();
            count += 1;
        }
        assert!(count > 1000);
        eprintln!(
            "synthetic wall: color MAE {:.3} -> {:.3}, matched_patches={}, reprojection_RMS={:.3} -> {:.3}px",
            error_before / count as f64,
            error_after / count as f64,
            corrected.estimate.matches,
            corrected.estimate.rms_before_px.unwrap(),
            corrected.estimate.rms_after_px.unwrap()
        );
        assert!(
            error_after < 0.7 * error_before,
            "color MAE {} -> {}; {:?}",
            error_before / count as f64,
            error_after / count as f64,
            corrected.estimate
        );
        let report_path = scratch.0.join("report.json");
        state.write_report(&report_path, &after).unwrap();
        let report: serde_json::Value =
            serde_json::from_reader(File::open(report_path).unwrap()).unwrap();
        assert_eq!(report["camera_poses"][1]["estimate"]["accepted"], true);
        assert_eq!(
            report["camera_poses"][1]["shared_by_frame_indices"],
            serde_json::json!([1, 2])
        );
    }

    #[test]
    fn depth_occlusion_moves_with_the_corrected_pose() {
        let camera = CameraIntrinsics::for_image_size(960, 600).unwrap();
        let mut raw = DepthRaster::build(&[], &Matrix4::identity(), &camera);
        let (u, v) = (470_u32, 300_u32);
        raw.nearest_z[v as usize * camera.width as usize + u as usize] = 3.0;
        let delta = correction(&[0.0, 0.0, 0.0, 0.12, 0.0, 0.0]);
        let changed = raw.reproject(&delta, &camera);
        let original_point =
            Point3::from(camera_ray_for_pixel(u as f64, v as f64, &camera).unwrap() * 3.0);
        let new_point = delta.transform_point(&original_point);
        let (nu, nv) = project_camera_point_full_image(&new_point, &camera).unwrap();
        assert!(nu > u + 10);
        assert_eq!(changed.classify(nu, nv, 5.0, 0.2, 0), DepthState::Occluded);
        assert_eq!(changed.classify(u, v, 5.0, 0.2, 0), DepthState::Unknown);
        assert_eq!(raw.classify(u, v, 5.0, 0.2, 0), DepthState::Occluded);
    }

    #[test]
    fn textureless_images_keep_original_poses_and_color_counts() {
        let scratch = Scratch::new("flat");
        let mut state = RegionalRecolor::new(config(), &scratch.0).unwrap();
        let image = RgbImage::from_pixel(960, 600, Rgb([90; 3]));
        let raw: Vec<_> = (-25..=25)
            .flat_map(|x| {
                (-20..=20).map(move |y| Point3::new(x as f32 * 0.04, y as f32 * 0.04, 5.1))
            })
            .collect();
        for id in 0..2 {
            state
                .observe_frame(
                    id,
                    id as f64,
                    Path::new(if id == 0 { "a" } else { "b" }),
                    &image,
                    &Matrix4::identity(),
                    &Matrix4::identity(),
                    &raw,
                )
                .unwrap();
        }
        let points: Vec<_> = raw
            .iter()
            .map(|p| RegionalRecolorPoint {
                position: *p,
                normal: Some(Vector3::z()),
            })
            .collect();
        state.config.pose_alignment.enabled = true;
        let result = state
            .recolor_points(&points, |_| Ok(image.clone()))
            .unwrap();
        assert_eq!(result.stats.camera_pose_alignment.refined_images, 0);
        assert_eq!(result.stats.colored_points, points.len());
        for record in &result.camera_poses {
            assert!(!record.estimate.accepted);
            assert_eq!(record.estimate.reason, "no_overlap_matches");
            assert_eq!(
                record.original_camera_from_world_column_major,
                record.corrected_camera_from_world_column_major
            );
        }
        assert!(result.colors.iter().all(|c| *c == Some([90; 3])));
    }
}
