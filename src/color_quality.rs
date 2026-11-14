//! Prefer colors from image poses verified by photometric alignment.
//! Distance breaks ties within the same quality; it never overrides quality.

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum ColorQuality {
    #[default]
    Unverified,
    Verified,
}

pub fn should_replace_color(
    has_stored_color: bool,
    stored_quality: ColorQuality,
    stored_distance_m: Option<f64>,
    candidate_quality: ColorQuality,
    candidate_distance_m: f64,
) -> bool {
    if !candidate_distance_m.is_finite() || candidate_distance_m <= 0.0 {
        return false;
    }
    if !has_stored_color {
        return true;
    }
    if candidate_quality != stored_quality {
        return candidate_quality > stored_quality;
    }
    stored_distance_m.is_none_or(|old| !old.is_finite() || old <= 0.0 || candidate_distance_m < old)
}

#[cfg(test)]
mod tests {
    use super::{ColorQuality::*, should_replace_color};

    #[test]
    fn nearer_failed_alignment_cannot_overwrite_verified_color() {
        assert!(!should_replace_color(
            true,
            Verified,
            Some(10.0),
            Unverified,
            5.0
        ));
    }

    #[test]
    fn verified_color_replaces_nearer_unverified_color() {
        assert!(should_replace_color(
            true,
            Unverified,
            Some(5.0),
            Verified,
            10.0
        ));
    }

    #[test]
    fn same_quality_keeps_the_nearer_observation() {
        for quality in [Unverified, Verified] {
            assert!(should_replace_color(
                true,
                quality,
                Some(10.0),
                quality,
                9.0
            ));
            assert!(!should_replace_color(
                true,
                quality,
                Some(10.0),
                quality,
                11.0
            ));
            assert!(!should_replace_color(
                true,
                quality,
                Some(10.0),
                quality,
                10.0
            ));
        }
    }

    #[test]
    fn unverified_observation_can_fill_a_colorless_point() {
        assert!(should_replace_color(
            false, Unverified, None, Unverified, 8.0
        ));
    }

    #[test]
    fn invalid_distances_never_promote_an_observation() {
        for invalid in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            assert!(!should_replace_color(
                true,
                Unverified,
                Some(8.0),
                Verified,
                invalid
            ));
            assert!(!should_replace_color(
                false, Unverified, None, Unverified, invalid
            ));
        }
    }
}
