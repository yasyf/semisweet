//! Scoring: deciding when a candidate entry is a semantic hit.
//!
//! A cache miss is cheap (recompute), but a *false hit* returns a wrong cached answer — so the
//! model is precision-first and built entirely from HARD gates, never score fusion or a soft
//! threshold relaxation. A candidate is accepted only if it clears, in order:
//!
//! 1. **Entity hard-gate** — when the query carries `n` entities, at least `max(1, n/3)` must
//!    overlap the candidate's. Rejects same-template / different-subject neighbors.
//! 2. **Context hard-gate** (`Gate` mode, context-bearing query) — the candidate's backend
//!    context-BM25 match (`hit.sparse_score`, the query context scored against the stored
//!    context) must reach `context_gate`, else reject. This is the sole disambiguator between
//!    entries that share a query but differ in context.
//! 3. **Dense floor** — the query/candidate cosine must clear a threshold. A context-bearing
//!    query that passed the context gate is already precision-backstopped, so it need only
//!    reach the lower `context_threshold`; everything else must reach the full `threshold`.
//!
//! Entry identity (`EntryId`, see [`crate::newtype`]) includes `context`, so the same query and
//! keys with a different context are *distinct* entries the context gate disambiguates, rather
//! than one overwriting the other. Defaults are calibrated precision-first on representative
//! bench data (see `bench/`).

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use crate::error::Result;
use crate::newtype::{Context, Entity};
use crate::vector::ScoredHit;

// Defaults calibrated on representative v3 bench data (Phase C), precision-first. Tests
// derive expected behavior from the config's own fields rather than these literals.
const DEFAULT_THRESHOLD: f32 = 0.92;
// Locked on v3 via the daemon: wrong-context BM25 tops out ~0.05 and correct contexts start
// ~0.13, so 0.10 sits in the gap — 0 wrong-entry, 29/36 disambiguation (matching Jaccard).
const DEFAULT_CONTEXT_GATE: f32 = 0.10;
const DEFAULT_CONTEXT_THRESHOLD: f32 = 0.88;
const DEFAULT_TOP_K: NonZeroUsize = match NonZeroUsize::new(10) {
    Some(top_k) => top_k,
    None => unreachable!(),
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextMode {
    Ignore,
    Gate,
}

#[derive(Debug, Clone)]
pub struct ScoringConfig {
    pub threshold: f32,
    pub entity_filter: bool,
    pub context: ContextMode,
    pub context_gate: f32,
    pub context_threshold: f32,
    pub top_k: NonZeroUsize,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            entity_filter: true,
            context: ContextMode::Gate,
            context_gate: DEFAULT_CONTEXT_GATE,
            context_threshold: DEFAULT_CONTEXT_THRESHOLD,
            top_k: DEFAULT_TOP_K,
        }
    }
}

impl ScoringConfig {
    /// Accept a candidate through the entity hard-gate and a context-selected dense floor.
    /// First, when the query carries entities, at least `max(1, n/3)` of its `n` entities
    /// must overlap the hit's. Then the dense floor: in `Gate` mode a context-bearing query
    /// whose candidate context-BM25 match (`hit.sparse_score`) clears `context_gate` is
    /// backstopped on precision, so its dense cosine need only reach the lower
    /// `context_threshold`; a candidate that misses the gate is rejected outright. A
    /// context-less query, or `Ignore` mode, must reach the full `threshold`.
    fn accept(
        &self,
        query_entities: &BTreeSet<Entity>,
        query_context: &Option<Context>,
        hit: &ScoredHit,
    ) -> Result<bool> {
        let n = query_entities.len();
        if n > 0 {
            let required = (n / 3).max(1);
            if query_entities.intersection(&hit.entities).count() < required {
                return Ok(false);
            }
        }
        let dense_floor = match (self.context, query_context) {
            (ContextMode::Gate, Some(_)) => {
                if hit.sparse_score < self.context_gate {
                    return Ok(false);
                }
                self.context_threshold
            }
            _ => self.threshold,
        };
        Ok(hit.dense_score >= dense_floor)
    }

    /// The accepted candidate with the highest dense cosine, or `None` when every hit is
    /// filtered out. An exact dense-score tie keeps the first hit in `hits` order.
    pub fn select(
        &self,
        query_entities: &BTreeSet<Entity>,
        query_context: &Option<Context>,
        hits: Vec<ScoredHit>,
    ) -> Result<Option<ScoredHit>> {
        let mut best: Option<ScoredHit> = None;
        for hit in hits {
            if !self.accept(query_entities, query_context, &hit)? {
                continue;
            }
            let wins = match &best {
                Some(current) => hit.dense_score.total_cmp(&current.dense_score).is_gt(),
                None => true,
            };
            if wins {
                best = Some(hit);
            }
        }
        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::newtype::{Context, Entity, EntryId, Key, QueryText};

    use super::*;

    const EPS: f32 = 1e-3;

    fn ents(names: &[&str]) -> BTreeSet<Entity> {
        names
            .iter()
            .map(|n| Entity::new((*n).to_owned()).unwrap())
            .collect()
    }

    fn ctx(text: &str) -> Option<Context> {
        Some(Context::new(text.to_owned()).unwrap())
    }

    fn hit(label: &str, dense: f32, entities: BTreeSet<Entity>, sparse: f32) -> ScoredHit {
        let id = EntryId::derive(
            &QueryText::new(label.to_owned()).unwrap(),
            &BTreeSet::<Key>::new(),
            &None,
        );
        ScoredHit {
            id,
            dense_score: dense,
            sparse_score: sparse,
            entities,
            context: None,
        }
    }

    /// A dense cosine comfortably above `threshold`, so a test isolates the entity or
    /// context gate rather than the dense gate.
    fn clears_dense(config: &ScoringConfig) -> f32 {
        (config.threshold + 1.0) / 2.0
    }

    /// A context-BM25 match comfortably above `context_gate`, so a test isolates the dense
    /// floor rather than the context gate.
    fn clears_gate(config: &ScoringConfig) -> f32 {
        (config.context_gate + 1.0) / 2.0
    }

    fn accepts(
        config: &ScoringConfig,
        query_entities: &BTreeSet<Entity>,
        query_context: &Option<Context>,
        hit: &ScoredHit,
    ) -> bool {
        config.accept(query_entities, query_context, hit).unwrap()
    }

    // --- entity hard-gate ---

    #[test]
    fn entity_gate_boundary_n2_requires_one() {
        let config = ScoringConfig::default();
        let d = clears_dense(&config);
        let query = ents(&["a", "b"]);
        let below = hit("h", d, ents(&["c"]), 0.0);
        let at = hit("h", d, ents(&["a"]), 0.0);
        assert!(!accepts(&config, &query, &None, &below));
        assert!(accepts(&config, &query, &None, &at));
    }

    #[test]
    fn entity_gate_boundary_n3_requires_one() {
        let config = ScoringConfig::default();
        let d = clears_dense(&config);
        let query = ents(&["a", "b", "c"]);
        let below = hit("h", d, ents(&["x"]), 0.0);
        let at = hit("h", d, ents(&["b"]), 0.0);
        assert!(!accepts(&config, &query, &None, &below));
        assert!(accepts(&config, &query, &None, &at));
    }

    #[test]
    fn entity_gate_boundary_n5_requires_one() {
        let config = ScoringConfig::default();
        let d = clears_dense(&config);
        let query = ents(&["a", "b", "c", "d", "e"]);
        let below = hit("h", d, ents(&["x", "y"]), 0.0);
        let at = hit("h", d, ents(&["c"]), 0.0);
        assert!(!accepts(&config, &query, &None, &below));
        assert!(accepts(&config, &query, &None, &at));
    }

    #[test]
    fn entity_gate_boundary_n6_requires_two() {
        let config = ScoringConfig::default();
        let d = clears_dense(&config);
        let query = ents(&["a", "b", "c", "d", "e", "f"]);
        let below = hit("h", d, ents(&["a"]), 0.0);
        let at = hit("h", d, ents(&["a", "b"]), 0.0);
        assert!(!accepts(&config, &query, &None, &below));
        assert!(accepts(&config, &query, &None, &at));
    }

    #[test]
    fn entity_overlap_does_not_rescue_subthreshold_dense() {
        // The entity gate only filters — the removed entity_bonus no longer relaxes the
        // dense bar, so a sub-threshold hit with full overlap is still rejected.
        let config = ScoringConfig::default();
        let query = ents(&["a"]);
        let candidate = hit("h", config.threshold - EPS, ents(&["a"]), 0.0);
        assert!(!accepts(&config, &query, &None, &candidate));
    }

    // --- dense gate ---

    #[test]
    fn dense_gate_accepts_at_threshold_rejects_below() {
        let config = ScoringConfig::default();
        let no_entities = BTreeSet::<Entity>::new();
        let at = hit("h", config.threshold, BTreeSet::new(), 0.0);
        let below = hit("h", config.threshold - EPS, BTreeSet::new(), 0.0);
        assert!(accepts(&config, &no_entities, &None, &at));
        assert!(!accepts(&config, &no_entities, &None, &below));
    }

    // --- context hard-gate ---

    #[test]
    fn context_gate_accepts_matching_context() {
        // The backend reports a context match clearing the gate; with the dense floor cleared
        // too, the candidate is accepted.
        let config = ScoringConfig::default();
        let d = clears_dense(&config);
        let candidate = hit("h", d, BTreeSet::new(), clears_gate(&config));
        assert!(accepts(&config, &BTreeSet::new(), &ctx("q"), &candidate));
    }

    #[test]
    fn context_gate_rejects_below_gate_match() {
        let config = ScoringConfig::default();
        assert!(config.context_gate > 0.0);
        let d = clears_dense(&config);
        let candidate = hit("h", d, BTreeSet::new(), config.context_gate - EPS);
        assert!(!accepts(&config, &BTreeSet::new(), &ctx("q"), &candidate));
    }

    #[test]
    fn context_gate_rejects_candidate_without_context() {
        // The query carries a context; the candidate stored none -> backend match 0 -> reject.
        let config = ScoringConfig::default();
        assert!(config.context_gate > 0.0);
        let d = clears_dense(&config);
        let candidate = hit("h", d, BTreeSet::new(), 0.0);
        assert!(!accepts(&config, &BTreeSet::new(), &ctx("q"), &candidate));
    }

    #[test]
    fn context_gate_skipped_without_query_context() {
        // A context-less query is never rejected by the context gate, even in Gate mode and
        // even when the candidate's context match is 0.
        let config = ScoringConfig::default();
        let d = clears_dense(&config);
        let candidate = hit("h", d, BTreeSet::new(), 0.0);
        assert!(accepts(&config, &BTreeSet::new(), &None, &candidate));
    }

    #[test]
    fn context_gate_skipped_in_ignore_mode() {
        let config = ScoringConfig {
            context: ContextMode::Ignore,
            ..ScoringConfig::default()
        };
        let d = clears_dense(&config);
        let candidate = hit("h", d, BTreeSet::new(), 0.0);
        assert!(accepts(&config, &BTreeSet::new(), &ctx("q"), &candidate));
    }

    #[test]
    fn context_gate_is_inclusive_at_the_bar() {
        // The reject is strict `<`, so a context match exactly equal to the gate is accepted
        // and a strictly smaller one is rejected.
        let config = ScoringConfig::default();
        let d = clears_dense(&config);
        let at = hit("h", d, BTreeSet::new(), config.context_gate);
        assert!(accepts(&config, &BTreeSet::new(), &ctx("q"), &at));
        let below = hit("h", d, BTreeSet::new(), config.context_gate - EPS);
        assert!(!accepts(&config, &BTreeSet::new(), &ctx("q"), &below));
    }

    #[test]
    fn context_match_does_not_rescue_below_context_threshold() {
        // A matching context lowers the dense floor to `context_threshold`, but never below
        // it: a hit under that floor with a clearing context match is still rejected.
        let config = ScoringConfig::default();
        let candidate = hit(
            "h",
            config.context_threshold - EPS,
            BTreeSet::new(),
            clears_gate(&config),
        );
        assert!(!accepts(&config, &BTreeSet::new(), &ctx("q"), &candidate));
    }

    // --- context-present dense floor ---

    #[test]
    fn context_present_uses_lower_dense_floor() {
        // A context match clearing the gate drops the dense floor from `threshold` to the
        // lower `context_threshold`, so a dense score between the two clears the floor only
        // when the query actually carries a context.
        let config = ScoringConfig::default();
        assert!(config.context_threshold < config.threshold);
        let between = (config.context_threshold + config.threshold) / 2.0;
        let candidate = hit("h", between, BTreeSet::new(), clears_gate(&config));

        // Context-bearing query + a context match clearing the gate: the lower floor applies,
        // so a sub-`threshold` dense is accepted.
        assert!(accepts(&config, &BTreeSet::new(), &ctx("q"), &candidate));

        // Same hit and dense, but a context-less query: the full `threshold` applies and
        // rejects.
        assert!(!accepts(&config, &BTreeSet::new(), &None, &candidate));

        // Context-bearing but the candidate's match misses the gate: the hard-gate rejects
        // regardless of dense — even a perfect 1.0.
        let mismatched = hit("h", 1.0, BTreeSet::new(), config.context_gate - EPS);
        assert!(!accepts(&config, &BTreeSet::new(), &ctx("q"), &mismatched));
    }

    // --- select ---

    #[test]
    fn select_returns_none_when_all_rejected() {
        let config = ScoringConfig::default();
        let query = ents(&["a"]);
        let hits = vec![hit("h", config.threshold - EPS, ents(&["a"]), 0.0)];
        assert!(config.select(&query, &None, hits).unwrap().is_none());
    }

    #[test]
    fn select_picks_highest_dense_score() {
        let config = ScoringConfig::default();
        let query = BTreeSet::<Entity>::new();
        let low = hit("low", clears_dense(&config), BTreeSet::new(), 0.0);
        let high = hit("high", 1.0, BTreeSet::new(), 0.0);
        let winner = config
            .select(&query, &None, vec![low, high.clone()])
            .unwrap()
            .unwrap();
        assert_eq!(winner.id, high.id);
    }

    #[test]
    fn select_breaks_exact_tie_by_first_seen() {
        // Two accepted hits with identical dense scores: select keeps the first in input order.
        let config = ScoringConfig::default();
        let query = BTreeSet::<Entity>::new();
        let first = hit("first", 1.0, BTreeSet::new(), 0.0);
        let second = hit("second", 1.0, BTreeSet::new(), 0.0);
        assert_ne!(first.id, second.id);
        let winner = config
            .select(&query, &None, vec![first.clone(), second.clone()])
            .unwrap()
            .unwrap();
        assert_eq!(winner.id, first.id);
    }

    #[test]
    fn select_context_gate_filters_failing_context() {
        // The higher-dense candidate misses the context gate (match below the bar) and is
        // dropped; the lower-dense candidate whose context match clears the gate wins.
        let config = ScoringConfig::default();
        let query = BTreeSet::<Entity>::new();
        let context = ctx("q");
        let matching = hit(
            "a",
            clears_dense(&config),
            BTreeSet::new(),
            clears_gate(&config),
        );
        let failing = hit("b", 1.0, BTreeSet::new(), config.context_gate - EPS);
        let winner = config
            .select(&query, &context, vec![failing, matching.clone()])
            .unwrap()
            .unwrap();
        assert_eq!(winner.id, matching.id);
    }

    #[test]
    fn ignore_mode_select_ignores_context() {
        let config = ScoringConfig {
            context: ContextMode::Ignore,
            ..ScoringConfig::default()
        };
        let query = BTreeSet::<Entity>::new();
        let context = ctx("q");
        let matching = hit(
            "a",
            clears_dense(&config),
            BTreeSet::new(),
            clears_gate(&config),
        );
        let higher = hit("b", 1.0, BTreeSet::new(), 0.0);
        let winner = config
            .select(&query, &context, vec![matching, higher.clone()])
            .unwrap()
            .unwrap();
        assert_eq!(winner.id, higher.id);
    }
}
