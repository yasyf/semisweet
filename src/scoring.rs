use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use crate::error::Result;
use crate::newtype::{Context, Embedding, Entity};
use crate::vector::ScoredHit;

// Thresholds and bonuses live on the *fused* score scale. A pure-dense hit (sparse 0) fuses to
// `dense_weight * cosine`, so scaling the original pure-dense cosine thresholds by the default
// `dense_weight` of 0.7 (0.90 -> 0.63, 0.86 -> 0.602, 0.04 -> 0.028) makes hybrid an exact superset
// of pure-dense matching: with sparse 0 the accept decision is identical to the old cosine gate
// (`cosine >= 0.90 - 0.04*entity_ratio`, floored at 0.86), and any lexical or context signal only
// ever lowers the bar further. Setting `dense_weight` away from 0.7 rescales this relationship.
const DEFAULT_BASE_THRESHOLD: f32 = 0.63;
const DEFAULT_FLOOR_THRESHOLD: f32 = 0.602;
const DEFAULT_ENTITY_BONUS_WEIGHT: f32 = 0.028;
const DEFAULT_DENSE_WEIGHT: f32 = 0.7;
const DEFAULT_SPARSE_WEIGHT: f32 = 0.3;
const DEFAULT_CONTEXT_BONUS_WEIGHT: f32 = 0.028;
const DEFAULT_TOP_K: NonZeroUsize = match NonZeroUsize::new(10) {
    Some(top_k) => top_k,
    None => unreachable!(),
};
const TIE_EPSILON: f32 = 1e-6;

fn token_set(text: &str) -> BTreeSet<&str> {
    text.split_whitespace().collect()
}

/// Lexical overlap of two token sets in `[0, 1]` (Jaccard). Disjoint or empty -> 0.
fn jaccard(a: &BTreeSet<&str>, b: &BTreeSet<&str>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    intersection as f32 / union as f32
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextMode {
    Ignore,
    Tiebreak,
}

#[derive(Debug, Clone)]
pub struct ScoringConfig {
    pub base_threshold: f32,
    pub floor_threshold: f32,
    pub entity_bonus_weight: f32,
    pub dense_weight: f32,
    pub sparse_weight: f32,
    pub context_bonus_weight: f32,
    pub top_k: NonZeroUsize,
    pub entity_filter: bool,
    pub context: ContextMode,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            base_threshold: DEFAULT_BASE_THRESHOLD,
            floor_threshold: DEFAULT_FLOOR_THRESHOLD,
            entity_bonus_weight: DEFAULT_ENTITY_BONUS_WEIGHT,
            dense_weight: DEFAULT_DENSE_WEIGHT,
            sparse_weight: DEFAULT_SPARSE_WEIGHT,
            context_bonus_weight: DEFAULT_CONTEXT_BONUS_WEIGHT,
            top_k: DEFAULT_TOP_K,
            entity_filter: true,
            context: ContextMode::Tiebreak,
        }
    }
}

impl ScoringConfig {
    /// Whether the read path should embed the query context for the dense half of the context
    /// boost. False when context is ignored or either weight zeroes the dense context term out,
    /// so the extra embedding call on a context-bearing GET is skipped.
    pub fn uses_context_dense(&self) -> bool {
        self.context != ContextMode::Ignore
            && self.context_bonus_weight > 0.0
            && self.dense_weight > 0.0
    }

    /// Convex combination of a dense cosine (already in `[0, 1]`) and a backend-normalized
    /// sparse score (in `[0, 1]`). Normalizing by the weight sum keeps the result in `[0, 1]`
    /// and makes `sparse_weight == 0` reduce exactly to the dense score. The caller guarantees
    /// `dense_weight + sparse_weight > 0` (validated in `ScoringDto::to_config`).
    fn fuse(&self, dense: f32, sparse: f32) -> f32 {
        (self.dense_weight * dense.clamp(0.0, 1.0) + self.sparse_weight * sparse.clamp(0.0, 1.0))
            / (self.dense_weight + self.sparse_weight)
    }

    /// The bioqa-style context boost: a present, matching query context lowers the effective
    /// threshold by up to `context_bonus_weight`. Absent context (or `ContextMode::Ignore`)
    /// yields 0 — it never penalizes. The match itself is hybrid: dense cosine of the context
    /// embeddings fused with the lexical overlap of the context text.
    fn context_relaxation(
        &self,
        query_context: &Option<Context>,
        query_context_vector: &Option<Embedding>,
        hit: &ScoredHit,
    ) -> Result<f32> {
        let Some(query_context) = query_context else {
            return Ok(0.0);
        };
        if self.context == ContextMode::Ignore {
            return Ok(0.0);
        }
        let sparse_c = match &hit.context {
            Some(hit_context) => jaccard(
                &token_set(query_context.as_str()),
                &token_set(hit_context.as_str()),
            ),
            None => 0.0,
        };
        let dense_c = match (query_context_vector, &hit.context_vector) {
            (Some(query_vector), Some(hit_vector)) => query_vector.dot(hit_vector)?.clamp(0.0, 1.0),
            _ => 0.0,
        };
        Ok(self.context_bonus_weight * self.fuse(dense_c, sparse_c))
    }

    fn accept(
        &self,
        query_entities: &BTreeSet<Entity>,
        query_context: &Option<Context>,
        query_context_vector: &Option<Embedding>,
        hit: &ScoredHit,
    ) -> Result<bool> {
        let n = query_entities.len();
        let overlap = query_entities.intersection(&hit.entities).count();
        if n > 0 {
            let required = (n / 3).max(1);
            if overlap < required {
                return Ok(false);
            }
        }
        let ratio = if n == 0 {
            0.0
        } else {
            overlap as f32 / n as f32
        };
        let tau_eff = (self.base_threshold
            - self.entity_bonus_weight * ratio
            - self.context_relaxation(query_context, query_context_vector, hit)?)
        .clamp(self.floor_threshold, self.base_threshold);
        Ok(self.fuse(hit.dense_score, hit.sparse_score) >= tau_eff)
    }

    pub fn select(
        &self,
        query_entities: &BTreeSet<Entity>,
        query_context: &Option<Context>,
        query_context_vector: &Option<Embedding>,
        hits: Vec<ScoredHit>,
    ) -> Result<Option<ScoredHit>> {
        // (hit, fused query score, context relaxation) — the relaxation doubles as the
        // continuous tiebreaker among near-equal fused scores.
        let mut scored: Vec<(ScoredHit, f32, f32)> = Vec::with_capacity(hits.len());
        for hit in hits {
            if !self.accept(query_entities, query_context, query_context_vector, &hit)? {
                continue;
            }
            let fused = self.fuse(hit.dense_score, hit.sparse_score);
            let relaxation = self.context_relaxation(query_context, query_context_vector, &hit)?;
            scored.push((hit, fused, relaxation));
        }
        let Some(best_fused) = scored
            .iter()
            .map(|(_, fused, _)| *fused)
            .max_by(f32::total_cmp)
        else {
            return Ok(None);
        };
        let winner = scored
            .into_iter()
            .filter(|(_, fused, _)| (best_fused - fused).abs() <= TIE_EPSILON)
            .max_by(|(_, fused_a, relax_a), (_, fused_b, relax_b)| {
                relax_a.total_cmp(relax_b).then(fused_a.total_cmp(fused_b))
            })
            .map(|(hit, _, _)| hit);
        Ok(winner)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::error::Error;
    use crate::newtype::{Context, Embedding, Entity, EntryId, Key, QueryText};

    use super::*;

    fn ents(names: &[&str]) -> BTreeSet<Entity> {
        names
            .iter()
            .map(|n| Entity::new((*n).to_owned()).unwrap())
            .collect()
    }

    fn hit(
        label: &str,
        dense: f32,
        sparse: f32,
        entities: BTreeSet<Entity>,
        context: Option<&str>,
        context_vector: Option<Embedding>,
    ) -> ScoredHit {
        let id = EntryId::derive(
            &QueryText::new(label.to_owned()).unwrap(),
            &BTreeSet::<Key>::new(),
        );
        ScoredHit {
            id,
            dense_score: dense,
            sparse_score: sparse,
            entities,
            context: context.map(|c| Context::new(c.to_owned()).unwrap()),
            context_vector,
        }
    }

    /// A hit whose fused query score equals `score` (dense == sparse) and carries no context
    /// vector — the common case for the entity-gate and threshold tests.
    fn make_hit(
        label: &str,
        score: f32,
        entities: BTreeSet<Entity>,
        context: Option<&str>,
    ) -> ScoredHit {
        hit(label, score, score, entities, context, None)
    }

    fn accepts(config: &ScoringConfig, query: &BTreeSet<Entity>, hit: &ScoredHit) -> bool {
        config.accept(query, &None, &None, hit).unwrap()
    }

    #[test]
    fn entity_gate_boundary_n2_requires_one() {
        let config = ScoringConfig::default();
        let query = ents(&["a", "b"]);
        let below = make_hit("h", 0.99, ents(&["c"]), None);
        let at = make_hit("h", 0.99, ents(&["a"]), None);
        assert!(!accepts(&config, &query, &below));
        assert!(accepts(&config, &query, &at));
    }

    #[test]
    fn entity_gate_boundary_n3_requires_one() {
        let config = ScoringConfig::default();
        let query = ents(&["a", "b", "c"]);
        let below = make_hit("h", 0.99, ents(&["x"]), None);
        let at = make_hit("h", 0.99, ents(&["b"]), None);
        assert!(!accepts(&config, &query, &below));
        assert!(accepts(&config, &query, &at));
    }

    #[test]
    fn entity_gate_boundary_n5_requires_one() {
        let config = ScoringConfig::default();
        let query = ents(&["a", "b", "c", "d", "e"]);
        let below = make_hit("h", 0.99, ents(&["x", "y"]), None);
        let at = make_hit("h", 0.99, ents(&["c"]), None);
        assert!(!accepts(&config, &query, &below));
        assert!(accepts(&config, &query, &at));
    }

    #[test]
    fn entity_gate_boundary_n6_requires_two() {
        let config = ScoringConfig::default();
        let query = ents(&["a", "b", "c", "d", "e", "f"]);
        let below = make_hit("h", 0.99, ents(&["a"]), None);
        let at = make_hit("h", 0.99, ents(&["a", "b"]), None);
        assert!(!accepts(&config, &query, &below));
        assert!(accepts(&config, &query, &at));
    }

    #[test]
    fn bonus_relaxes_threshold_with_full_overlap() {
        // Full entity overlap drops tau_eff from base 0.63 to the floor 0.602; a fused 0.61 hit is
        // below base but above the relaxed bar, so it only accepts because of the relaxation.
        let config = ScoringConfig::default();
        let query = ents(&["a"]);
        let full = make_hit("h", 0.61, ents(&["a"]), None);
        let none = make_hit("h", 0.61, BTreeSet::new(), None);
        assert!(accepts(&config, &query, &full));
        assert!(!accepts(&config, &BTreeSet::new(), &none));
    }

    #[test]
    fn bonus_does_not_save_no_overlap() {
        let config = ScoringConfig::default();
        let query = ents(&["a"]);
        let none = make_hit("h", 0.61, ents(&["b"]), None);
        assert!(!accepts(&config, &query, &none));
    }

    #[test]
    fn full_overlap_still_rejects_below_floor() {
        let config = ScoringConfig::default();
        let query = ents(&["a"]);
        let low = make_hit("h", 0.59, ents(&["a"]), None);
        assert!(!accepts(&config, &query, &low));
    }

    #[test]
    fn partial_overlap_relaxes_only_partially() {
        // 1/3 entity overlap relaxes tau_eff only to ~0.621; a fused 0.61 hit stays below it,
        // though full overlap (tau_eff 0.602) would have accepted it.
        let config = ScoringConfig::default();
        let query = ents(&["a", "b", "c"]);
        let partial = make_hit("h", 0.61, ents(&["a"]), None);
        assert!(!accepts(&config, &query, &partial));
    }

    #[test]
    fn weight_zero_reduces_to_bioqa() {
        // Zeroing every relaxation/sparse weight and restoring the old band reproduces the
        // pure-dense threshold gate: fuse() collapses to the dense score.
        let config = ScoringConfig {
            entity_bonus_weight: 0.0,
            sparse_weight: 0.0,
            context_bonus_weight: 0.0,
            base_threshold: 0.90,
            floor_threshold: 0.86,
            ..ScoringConfig::default()
        };
        let query = ents(&["a", "b"]);
        let above_gate_above_thresh = make_hit("h", 0.91, ents(&["a", "b"]), None);
        let above_gate_below_thresh = make_hit("h", 0.89, ents(&["a", "b"]), None);
        let below_gate = make_hit("h", 0.99, ents(&["x"]), None);
        assert!(accepts(&config, &query, &above_gate_above_thresh));
        assert!(!accepts(&config, &query, &above_gate_below_thresh));
        assert!(!accepts(&config, &query, &below_gate));
    }

    #[test]
    fn base_threshold_with_no_entities() {
        let config = ScoringConfig::default();
        let query = BTreeSet::<Entity>::new();
        let above = make_hit("h", 0.66, BTreeSet::new(), None);
        let below = make_hit("h", 0.60, BTreeSet::new(), None);
        assert!(accepts(&config, &query, &above));
        assert!(!accepts(&config, &query, &below));
    }

    #[test]
    fn fuse_is_convex_combination_in_unit_range() {
        let config = ScoringConfig::default();
        assert!(config.fuse(0.0, 0.0).abs() < 1e-6);
        assert!((config.fuse(1.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((config.fuse(1.0, 0.0) - 0.7).abs() < 1e-6);
        assert!((config.fuse(0.0, 1.0) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn sparse_weight_zero_reduces_fuse_to_dense() {
        let config = ScoringConfig {
            sparse_weight: 0.0,
            ..ScoringConfig::default()
        };
        assert!((config.fuse(0.9, 0.1) - 0.9).abs() < 1e-6);
        assert!((config.fuse(0.3, 1.0) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn context_relaxation_rescues_subthreshold_hit() {
        // A fused 0.61 hit is below base 0.63, so it misses on its own. A perfectly matching
        // present context (dense_c == sparse_c == 1.0) relaxes tau_eff to the floor 0.602 and
        // rescues it.
        let config = ScoringConfig::default();
        let vector = Embedding::new(vec![1.0, 0.0, 0.0]).unwrap();
        let candidate = hit(
            "h",
            0.61,
            0.61,
            BTreeSet::new(),
            Some("alpha beta"),
            Some(vector.clone()),
        );
        let query_context = Some(Context::new("alpha beta".to_owned()).unwrap());

        assert!(
            config
                .accept(&BTreeSet::new(), &query_context, &Some(vector), &candidate)
                .unwrap()
        );
        assert!(!accepts(&config, &BTreeSet::new(), &candidate));
    }

    #[test]
    fn present_query_context_with_absent_hit_context_is_no_penalty() {
        // The query carries a context, but the candidate stored none: relaxation is 0, so the
        // decision matches the no-context case exactly — never a penalty.
        let config = ScoringConfig::default();
        let vector = Embedding::new(vec![1.0, 0.0, 0.0]).unwrap();
        let query_context = Some(Context::new("alpha beta".to_owned()).unwrap());

        let on_its_own = make_hit("h", 0.66, BTreeSet::new(), None);
        let too_low = make_hit("h", 0.60, BTreeSet::new(), None);
        assert!(
            config
                .accept(
                    &BTreeSet::new(),
                    &query_context,
                    &Some(vector.clone()),
                    &on_its_own,
                )
                .unwrap()
        );
        assert!(
            !config
                .accept(&BTreeSet::new(), &query_context, &Some(vector), &too_low)
                .unwrap()
        );
    }

    #[test]
    fn dim_mismatch_context_vector_errors() {
        // The context cosine is fail-fast: a stored context vector whose dimension differs from
        // the query context embedding surfaces DimMismatch rather than coercing to 0.
        let config = ScoringConfig::default();
        let query_vector = Embedding::new(vec![1.0, 0.0, 0.0]).unwrap();
        let hit_vector = Embedding::new(vec![1.0, 0.0]).unwrap();
        let candidate = hit(
            "h",
            0.95,
            0.95,
            BTreeSet::new(),
            Some("alpha"),
            Some(hit_vector),
        );
        let query_context = Some(Context::new("alpha".to_owned()).unwrap());

        let result = config.accept(
            &BTreeSet::new(),
            &query_context,
            &Some(query_vector),
            &candidate,
        );
        assert!(matches!(
            result,
            Err(Error::DimMismatch { got: 2, want: 3 })
        ));
    }

    #[test]
    fn select_returns_none_when_all_rejected() {
        let config = ScoringConfig::default();
        let query = ents(&["a"]);
        let hits = vec![make_hit("h", 0.50, ents(&["a"]), None)];
        assert!(config.select(&query, &None, &None, hits).unwrap().is_none());
    }

    #[test]
    fn select_picks_highest_score_without_context() {
        let config = ScoringConfig::default();
        let query = BTreeSet::<Entity>::new();
        let low = make_hit("low", 0.90, BTreeSet::new(), None);
        let high = make_hit("high", 0.95, BTreeSet::new(), None);
        let winner = config
            .select(&query, &None, &None, vec![low, high.clone()])
            .unwrap()
            .unwrap();
        assert_eq!(winner.id, high.id);
    }

    #[test]
    fn tiebreak_breaks_near_ties_by_context_overlap() {
        let config = ScoringConfig::default();
        let query = BTreeSet::<Entity>::new();
        let context = Some(Context::new("alpha beta gamma".to_owned()).unwrap());
        let overlapping = make_hit("a", 0.95, BTreeSet::new(), Some("alpha beta"));
        let disjoint = make_hit("b", 0.95, BTreeSet::new(), Some("delta epsilon"));
        let winner = config
            .select(&query, &context, &None, vec![disjoint, overlapping.clone()])
            .unwrap()
            .unwrap();
        assert_eq!(winner.id, overlapping.id);
    }

    #[test]
    fn tiebreak_does_not_override_clear_score_lead() {
        let config = ScoringConfig::default();
        let query = BTreeSet::<Entity>::new();
        let context = Some(Context::new("alpha beta gamma".to_owned()).unwrap());
        let overlapping = make_hit("a", 0.95, BTreeSet::new(), Some("alpha beta"));
        let higher = make_hit("b", 0.97, BTreeSet::new(), None);
        let winner = config
            .select(&query, &context, &None, vec![overlapping, higher.clone()])
            .unwrap()
            .unwrap();
        assert_eq!(winner.id, higher.id);
    }

    #[test]
    fn ignore_mode_picks_highest_score_despite_context() {
        let config = ScoringConfig {
            context: ContextMode::Ignore,
            ..ScoringConfig::default()
        };
        let query = BTreeSet::<Entity>::new();
        let context = Some(Context::new("alpha beta".to_owned()).unwrap());
        let overlapping = make_hit("a", 0.95, BTreeSet::new(), Some("alpha beta"));
        let higher = make_hit("b", 0.96, BTreeSet::new(), None);
        let winner = config
            .select(&query, &context, &None, vec![overlapping, higher.clone()])
            .unwrap()
            .unwrap();
        assert_eq!(winner.id, higher.id);
    }
}
