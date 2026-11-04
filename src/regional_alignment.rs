//! regional_recolorの内部処理。同じ平面・対象画像で補正量を共有する。
use super::*;
use crate::camera::project_camera_point_continuous;
use image::GrayImage;

impl RegionalRecolor {
    pub(super) fn alignment_shifts<F>(
        &self,
        patches: &[(SurfacePatchKey, Vec<usize>)],
        points: &[RegionalRecolorPoint],
        adjacency: &[Vec<usize>],
        winners: &[Option<PatchSelection>],
        load_image: &mut F,
        stats: &mut RegionalRecolorStats,
    ) -> Result<(Vec<[i32; 2]>, Vec<AlignmentRecord>)>
    where
        F: FnMut(&Path) -> Result<RgbImage>,
    {
        let mut shifts = vec![[0, 0]; patches.len()];
        let mut records = Vec::new();
        if !self.config.alignment.enabled {
            return Ok((shifts, records));
        }
        let mut visited = vec![false; patches.len()];
        let mut groups = Vec::new();
        for start in 0..patches.len() {
            if visited[start] || winners[start].is_none() {
                continue;
            }
            visited[start] = true;
            let mut stack = vec![start];
            let mut group = Vec::new();
            while let Some(node) = stack.pop() {
                group.push(node);
                for &other in &adjacency[node] {
                    if !visited[other] && winners[other].is_some() {
                        visited[other] = true;
                        stack.push(other);
                    }
                }
            }
            group.sort_unstable();
            let frames: HashSet<_> = group
                .iter()
                .map(|&node| winners[node].as_ref().unwrap().frame_id)
                .collect();
            if frames.len() > 1 {
                groups.push(group);
            }
        }
        stats.alignment_groups = groups.len();
        for (group_id, group) in groups.iter().enumerate() {
            if group_id % 25 == 0 || group_id + 1 == groups.len() {
                log::info!(
                    "Regional alignment: plane group {}/{}",
                    group_id + 1,
                    groups.len()
                );
            }
            let indices: Vec<_> = group
                .iter()
                .flat_map(|&node| patches[node].1.iter().copied())
                .collect();
            let Some((center, normal)) = group_plane(&indices, points, &self.config) else {
                stats.alignment_nonplanar_groups += 1;
                continue;
            };
            let regions: HashSet<_> = group.iter().map(|&node| patches[node].0.region).collect();
            let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
            for &node in group {
                *counts
                    .entry(winners[node].as_ref().unwrap().frame_id)
                    .or_default() += patches[node].1.len();
            }
            // 最も多くの最終点で使われる画像を固定基準とする。補正を連鎖させない。
            let reference_id = *counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
                .unwrap()
                .0;
            let reference = &self.frames[reference_id];
            let reference_gray = load_gray(reference, load_image)?;
            let reference_depth =
                DepthRaster::read_sparse(&reference.depth_path, &reference.intrinsics)?;
            stats.images_loaded_for_alignment += 1;
            for &target_id in counts.keys().filter(|&&id| id != reference_id) {
                stats.alignment_attempted_pairs += 1;
                let target = &self.frames[target_id];
                let target_gray = load_gray(target, load_image)?;
                let target_depth =
                    DepthRaster::read_sparse(&target.depth_path, &target.intrinsics)?;
                stats.images_loaded_for_alignment += 1;
                let samples = self.alignment_patches(
                    &indices,
                    points,
                    &regions,
                    &normal,
                    reference,
                    target,
                    &reference_gray,
                    &reference_depth,
                    &target_depth,
                );
                let estimate = estimate_translation(
                    &reference_gray,
                    &target_gray,
                    &samples,
                    &self.config.alignment,
                );
                if estimate.accepted {
                    stats.alignment_accepted_pairs += 1;
                }
                let mut target_patches = Vec::new();
                for &node in group {
                    if winners[node].as_ref().unwrap().frame_id == target_id {
                        target_patches.push(patches[node].0);
                        if estimate.accepted {
                            shifts[node] = estimate.shift_px;
                        }
                    }
                }
                log::debug!(
                    "Regional alignment: group={} reference={} target={} shift={:?} reason={} inliers={}/{}",
                    group_id,
                    reference.frame_index,
                    target.frame_index,
                    estimate.shift_px,
                    estimate.reason,
                    estimate.inlier_patches,
                    estimate.textured_patches
                );
                records.push(AlignmentRecord {
                    group_id,
                    reference_frame_index: reference.frame_index,
                    target_frame_index: target.frame_index,
                    target_patches,
                    plane_point_world: center.coords.into(),
                    plane_normal_world: normal.into(),
                    estimate,
                });
            }
        }
        log::info!(
            "Regional alignment: groups={} nonplanar={} accepted={}/{} image_loads={}",
            stats.alignment_groups,
            stats.alignment_nonplanar_groups,
            stats.alignment_accepted_pairs,
            stats.alignment_attempted_pairs,
            stats.images_loaded_for_alignment
        );
        Ok((shifts, records))
    }

    #[allow(clippy::too_many_arguments)]
    fn alignment_patches(
        &self,
        indices: &[usize],
        points: &[RegionalRecolorPoint],
        regions: &HashSet<[i64; 3]>,
        normal: &Vector3<f64>,
        reference: &CachedFrame,
        target: &CachedFrame,
        reference_gray: &GrayImage,
        reference_depth: &DepthRaster,
        target_depth: &DepthRaster,
    ) -> Vec<Vec<crate::patch_alignment::WarpSample>> {
        let config = &self.config.alignment;
        // 32画素のセルごとに中心に近い実在点を1つ取り、偏りを減らす。
        let mut bins: BTreeMap<(u32, u32), (f64, Point3<f64>)> = BTreeMap::new();
        for &index in indices {
            let point = &points[index];
            if self
                .visible_sample(point, reference, reference_depth)
                .is_none()
                || self.visible_sample(point, target, target_depth).is_none()
            {
                continue;
            }
            let world = point.position.cast::<f64>();
            let Some((u, v)) = project_camera_point_continuous(
                &reference.camera_from_world.transform_point(&world),
                &reference.intrinsics,
            ) else {
                continue;
            };
            if u < 0.0 || v < 0.0 {
                continue;
            }
            let key = ((v / 32.0) as u32, (u / 32.0) as u32);
            let rank = (u - (key.1 as f64 * 32.0 + 16.0)).powi(2)
                + (v - (key.0 as f64 * 32.0 + 16.0)).powi(2);
            let entry = bins.entry(key).or_insert((rank, world));
            if rank < entry.0
                || (rank == entry.0 && world.coords.as_slice() < entry.1.coords.as_slice())
            {
                *entry = (rank, world);
            }
        }
        let seeds: Vec<_> = bins.into_values().map(|x| x.1).collect();
        let budget = seeds.len().min(config.max_patches * 4);
        let mut textured = Vec::new();
        for step in 0..budget {
            let seed = &seeds[step * seeds.len() / budget];
            let mut samples = plane_patch_samples(
                seed,
                normal,
                &reference.camera_from_world,
                &target.camera_from_world,
                &reference.intrinsics,
                &target.intrinsics,
                config,
            );
            samples.retain(|sample| {
                if !spatial_key(&sample.world, self.config.region_size_m)
                    .is_some_and(|key| regions.contains(&key))
                {
                    return false;
                }
                let point = RegionalRecolorPoint {
                    position: sample.world.cast::<f32>(),
                    normal: Some(normal.cast::<f32>()),
                };
                let Some(r) = self.visible_sample(&point, reference, reference_depth) else {
                    return false;
                };
                let Some(t) = self.visible_sample(&point, target, target_depth) else {
                    return false;
                };
                let rz = reference.camera_from_world.transform_point(&sample.world).z;
                let tz = target.camera_from_world.transform_point(&sample.world).z;
                !reference_depth.has_depth_discontinuity(
                    r.u,
                    r.v,
                    rz,
                    self.config.occlusion_margin_m,
                    self.config.occlusion_radius_px,
                ) && !target_depth.has_depth_discontinuity(
                    t.u,
                    t.v,
                    tz,
                    self.config.occlusion_margin_m,
                    config.max_shift_px as u32 + 1 + self.config.occlusion_radius_px,
                )
            });
            if samples.len() >= config.min_samples {
                let texture = reference_texture(reference_gray, &samples);
                if texture >= config.min_stddev {
                    textured.push((texture, step, samples));
                }
            }
        }
        textured.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        textured
            .into_iter()
            .take(config.max_patches)
            .map(|x| x.2)
            .collect()
    }
}

fn load_gray<F>(frame: &CachedFrame, load_image: &mut F) -> Result<GrayImage>
where
    F: FnMut(&Path) -> Result<RgbImage>,
{
    let image = load_image(&frame.image_path)?;
    ensure!(
        image.dimensions() == (frame.intrinsics.width, frame.intrinsics.height),
        "alignment reference image dimensions changed: {}",
        frame.image_path.display()
    );
    Ok(image::DynamicImage::ImageRgb8(image).into_luma8())
}

fn group_plane(
    indices: &[usize],
    points: &[RegionalRecolorPoint],
    config: &RegionalRecolorConfig,
) -> Option<(Point3<f64>, Vector3<f64>)> {
    if indices.len() < 3 {
        return None;
    }
    let mut center = Vector3::<f64>::zeros();
    let mut normal = Vector3::<f64>::zeros();
    for &index in indices {
        center += points[index].position.cast::<f64>().coords;
        let mut n = unit_normal(points[index].normal.as_ref())?;
        if n.dot(&normal) < 0.0 {
            n = -n;
        }
        normal += n;
    }
    center /= indices.len() as f64;
    if normal.norm() < 1e-6 {
        return None;
    }
    normal.normalize_mut();
    // 隣接判定を連鎖して曲面全体が1枚の平面になることを防ぐ。
    for &index in indices {
        let point = &points[index];
        if normal.dot(&unit_normal(point.normal.as_ref())?).abs() < config.coherence_normal_cos
            || normal
                .dot(&(point.position.cast::<f64>().coords - center))
                .abs()
                > config.coherence_plane_distance_m
        {
            return None;
        }
    }
    Some((Point3::from(center), normal))
}
