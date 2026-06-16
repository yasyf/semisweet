use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use crate::newtype::{Context, Entity};
use crate::vector::ScoredHit;

const DEFAULT_BASE_THRESHOLD: f32 = 0.90;
const DEFAULT_FLOOR_THRESHOLD: f32 = 0.86;
const DEFAULT_ENTITY_BONUS_WEIGHT: f32 = 0.04;
const DEFAULT_TOP_K: NonZeroUsize = match NonZeroUsize::new(10) {
    Some(top_k) => top_k,
    None => unreachable!(),
};
const TIE_EPSILON: f32 = 1e-6;

fn token_set(text: &str) -> BTreeSet<&str> {
    text.split_whitespace().collect()
}

fn context_overlap(query_tokens: &BTreeSet<&str>, hit_context: Option<&Context>) -> usize {
    match hit_context {
        Some(context) => {
            let hit_tokens = token_set(context.as_str());
            query_tokens.intersection(&hit_tokens).count()
        }
        None => 0,
    }
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
            top_k: DEFAULT_TOP_K,
            entity_filter: true,
            context: ContextMode::Tiebreak,
        }
    }
}

impl ScoringConfig {
    fn accept(&self, query_entities: &BTreeSet<Entity>, hit: &ScoredHit) -> bool {
        let n = query_entities.len();
        let overlap = query_entities.intersection(&hit.entities).count();
        if n > 0 {
            let required = (n / 3).max(1);
            if overlap < required {
                return false;
            }
        }
        let ratio = if n == 0 {
            0.0
        } else {
            overlap as f32 / n as f32
        };
        let tau_eff = (self.base_threshold - self.entity_bonus_weight * ratio)
            .clamp(self.floor_threshold, self.base_threshold);
        hit.score >= tau_eff
    }

    pub fn select(
        &self,
        query_entities: &BTreeSet<Entity>,
        query_context: &Option<Context>,
        hits: Vec<ScoredHit>,
    ) -> Option<ScoredHit> {
        let accepted: Vec<ScoredHit> = hits
            .into_iter()
            .filter(|hit| self.accept(query_entities, hit))
            .collect();

        match (self.context, query_context) {
            (ContextMode::Tiebreak, Some(context)) => {
                let best_score = accepted
                    .iter()
                    .map(|hit| hit.score)
                    .max_by(f32::total_cmp)?;
                let query_tokens = token_set(context.as_str());
                accepted
                    .into_iter()
                    .filter(|hit| (best_score - hit.score).abs() <= TIE_EPSILON)
                    .max_by(|a, b| {
                        let oa = context_overlap(&query_tokens, a.context.as_ref());
                        let ob = context_overlap(&query_tokens, b.context.as_ref());
                        oa.cmp(&ob).then(a.score.total_cmp(&b.score))
                    })
            }
            (ContextMode::Ignore, _) | (ContextMode::Tiebreak, None) => accepted
                .into_iter()
                .max_by(|a, b| a.score.total_cmp(&b.score)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::newtype::{Context, Entity, EntryId, Key, QueryText};

    use super::*;

    fn ents(names: &[&str]) -> BTreeSet<Entity> {
        names
            .iter()
            .map(|n| Entity::new((*n).to_owned()).unwrap())
            .collect()
    }

    fn make_hit(
        label: &str,
        score: f32,
        entities: BTreeSet<Entity>,
        context: Option<&str>,
    ) -> ScoredHit {
        let id = EntryId::derive(
            &QueryText::new(label.to_owned()).unwrap(),
            &BTreeSet::<Key>::new(),
        );
        ScoredHit {
            id,
            score,
            entities,
            context: context.map(|c| Context::new(c.to_owned()).unwrap()),
        }
    }

    #[test]
    fn entity_gate_boundary_n2_requires_one() {
        let config = ScoringConfig::default();
        let query = ents(&["a", "b"]);
        let below = make_hit("h", 0.99, ents(&["c"]), None);
        let at = make_hit("h", 0.99, ents(&["a"]), None);
        assert!(!config.accept(&query, &below));
        assert!(config.accept(&query, &at));
    }

    #[test]
    fn entity_gate_boundary_n3_requires_one() {
        let config = ScoringConfig::default();
        let query = ents(&["a", "b", "c"]);
        let below = make_hit("h", 0.99, ents(&["x"]), None);
        let at = make_hit("h", 0.99, ents(&["b"]), None);
        assert!(!config.accept(&query, &below));
        assert!(config.accept(&query, &at));
    }

    #[test]
    fn entity_gate_boundary_n5_requires_one() {
        let config = ScoringConfig::default();
        let query = ents(&["a", "b", "c", "d", "e"]);
        let below = make_hit("h", 0.99, ents(&["x", "y"]), None);
        let at = make_hit("h", 0.99, ents(&["c"]), None);
        assert!(!config.accept(&query, &below));
        assert!(config.accept(&query, &at));
    }

    #[test]
    fn entity_gate_boundary_n6_requires_two() {
        let config = ScoringConfig::default();
        let query = ents(&["a", "b", "c", "d", "e", "f"]);
        let below = make_hit("h", 0.99, ents(&["a"]), None);
        let at = make_hit("h", 0.99, ents(&["a", "b"]), None);
        assert!(!config.accept(&query, &below));
        assert!(config.accept(&query, &at));
    }

    #[test]
    fn bonus_relaxes_threshold_to_floor_with_full_overlap() {
        let config = ScoringConfig::default();
        let query = ents(&["a"]);
        let full = make_hit("h", 0.87, ents(&["a"]), None);
        assert!(config.accept(&query, &full));
    }

    #[test]
    fn bonus_does_not_save_no_overlap() {
        let config = ScoringConfig::default();
        let query = ents(&["a"]);
        let none = make_hit("h", 0.87, ents(&["b"]), None);
        assert!(!config.accept(&query, &none));
    }

    #[test]
    fn full_overlap_still_rejects_below_floor() {
        let config = ScoringConfig::default();
        let query = ents(&["a"]);
        let low = make_hit("h", 0.85, ents(&["a"]), None);
        assert!(!config.accept(&query, &low));
    }

    #[test]
    fn partial_overlap_relaxes_only_partially() {
        let config = ScoringConfig::default();
        let query = ents(&["a", "b", "c"]);
        let partial = make_hit("h", 0.87, ents(&["a"]), None);
        assert!(!config.accept(&query, &partial));
    }

    #[test]
    fn weight_zero_reduces_to_bioqa() {
        let config = ScoringConfig {
            entity_bonus_weight: 0.0,
            ..ScoringConfig::default()
        };
        let query = ents(&["a", "b"]);
        let above_gate_above_thresh = make_hit("h", 0.91, ents(&["a", "b"]), None);
        let above_gate_below_thresh = make_hit("h", 0.89, ents(&["a", "b"]), None);
        let below_gate = make_hit("h", 0.99, ents(&["x"]), None);
        assert!(config.accept(&query, &above_gate_above_thresh));
        assert!(!config.accept(&query, &above_gate_below_thresh));
        assert!(!config.accept(&query, &below_gate));
    }

    #[test]
    fn weight_zero_no_entities_uses_base_threshold() {
        let config = ScoringConfig {
            entity_bonus_weight: 0.0,
            ..ScoringConfig::default()
        };
        let query = BTreeSet::<Entity>::new();
        let above = make_hit("h", 0.90, BTreeSet::new(), None);
        let below = make_hit("h", 0.899, BTreeSet::new(), None);
        assert!(config.accept(&query, &above));
        assert!(!config.accept(&query, &below));
    }

    #[test]
    fn select_returns_none_when_all_rejected() {
        let config = ScoringConfig::default();
        let query = ents(&["a"]);
        let hits = vec![make_hit("h", 0.50, ents(&["a"]), None)];
        assert!(config.select(&query, &None, hits).is_none());
    }

    #[test]
    fn select_picks_highest_score_without_context() {
        let config = ScoringConfig::default();
        let query = BTreeSet::<Entity>::new();
        let low = make_hit("low", 0.93, BTreeSet::new(), None);
        let high = make_hit("high", 0.97, BTreeSet::new(), None);
        let winner = config
            .select(&query, &None, vec![low, high.clone()])
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
            .select(&query, &context, vec![disjoint, overlapping.clone()])
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
            .select(&query, &context, vec![overlapping, higher.clone()])
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
            .select(&query, &context, vec![overlapping, higher.clone()])
            .unwrap();
        assert_eq!(winner.id, higher.id);
    }
}
