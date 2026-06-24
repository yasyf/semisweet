"""Labeled synthetic semantic-cache dataset generator for the empirical-sweep harness.

Run as ``python -m bench.gen_dataset`` to emit a versioned, fully-labeled dataset under
``bench/data/v<ver>/``: canonical entries to ``set()``, labeled lookup queries, the
float32 vectors each record's ``vector_ref``/``context_vector_ref`` indexes into, nested
distractor corpora, a small real-ish holdout, and (on the LLM path) a draft cache.

Two authoring paths feed one shared pipeline that embeds with
:class:`bench.common.ParityEmbedder`, computes every axis label from the *real* vectors
(never an authored self-label), and writes the files:

* LLM path (default) drives the authenticated Claude CLI through ``spawnllm`` structured
  output -- one clinical pair-cluster per call, cached so re-runs are API-free.
* ``--deterministic`` expands hand-authored seed clusters across four domains (clinical,
  software, personal-finance, how-to) with seeded text transforms; it needs no network and
  is the reproducibility gate.

The deterministic path authors two cluster shapes:

* Pair clusters -- two canonicals sharing one ``keys`` set but naming different entities,
  with paraphrase positives, hard negatives, and a realistic holdout. They exercise recall
  and same-keys wrong-entry separation; their canonicals carry no context.
* Context-disambiguation clusters -- one shared query ``Q`` and one ``keys`` set with two
  or three context variants, each its own canonical (same ``Q``, same keys, distinct
  context, distinct payload). ``EntryId`` now hashes ``(query, context, keys)`` together
  (``src/newtype.rs`` ``EntryId::derive``), so these no longer collide -- they are
  genuinely distinct entries instead of overwriting one another. ``Q`` names only the
  shared topic entity, never the disambiguator, so the entity hard-gate passes every
  variant and the dense score is identical across variants; the lexical context gate
  (``src/scoring.rs``) is the sole disambiguator. Each eval query is a high-similarity
  paraphrase of ``Q`` (still naming no disambiguator) paired with a context that lexically
  matches exactly one variant, so returning another variant is a measurable
  ``wrong_entry_hit``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from spawnllm import (
    ClaudeCliBackend,
    extract_sync,
)
from spawnllm.proc import map_concurrent

from bench.common import (
    BGE_DIM,
    BGE_MODEL_ID,
    BGE_QUERY_PREFIX,
    CanonicalEntry,
    Distractor,
    EntityOverlap,
    LabeledQuery,
    LexicalBand,
    Manifest,
    ParityEmbedder,
    QueryKind,
)

# Constants

SCHEMA_VERSION = 1
DATA_ROOT = Path(__file__).resolve().parent / "data"

LLM_TIMEOUT_SECS = 240
LLM_CONCURRENCY = 4

SEMANTIC_BANDS: tuple[tuple[float, str], ...] = (
    (0.95, "0.95"),
    (0.90, "0.90"),
    (0.85, "0.85"),
    (0.80, "0.80"),
)
SEMANTIC_BELOW = "below"

LEXICAL_BANDS: tuple[tuple[float, LexicalBand], ...] = (
    (0.55, "high"),
    (0.25, "mid"),
    (0.0, "low"),
)

STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "for", "how", "in", "is", "of", "on",
        "or", "should", "than", "that", "the", "this", "to", "what", "when", "which",
        "with", "i", "my", "should", "appropriate", "patient", "patients", "currently",
        "recommended", "given", "much", "does", "do", "can", "will",
    }
)

SYNONYMS: dict[str, str] = {
    "dosing": "dose",
    "dose": "dosage",
    "for": "in",
    "selection": "choice",
    "titration": "adjustment",
    "regimen": "schedule",
    "initiation": "start",
    "management": "treatment",
    "control": "stabilization",
}

FILLERS: tuple[str, ...] = (
    "in general",
    "as a rule",
    "in practice",
    "for reference",
)

CLINICAL_THEMES: tuple[str, ...] = (
    "oral anticoagulation dosing in atrial fibrillation",
    "empiric antibiotic selection for community acquired pneumonia",
    "analgesic dosing for osteoarthritis pain",
    "antihypertensive dosing for essential hypertension",
    "oral hypoglycemic initiation for type two diabetes",
    "statin dosing for hyperlipidemia",
    "inhaled corticosteroid dosing for persistent asthma",
    "proton pump inhibitor dosing for gastroesophageal reflux disease",
    "antidepressant selection for major depressive disorder",
    "thyroid hormone dosing for hypothyroidism",
    "antiepileptic dosing for focal epilepsy",
    "guideline-directed therapy for chronic heart failure",
    "dmard dosing for rheumatoid arthritis",
    "antihistamine dosing for allergic rhinitis",
    "alpha blocker dosing for benign prostatic hyperplasia",
    "anxiolytic selection for generalized anxiety disorder",
    "bisphosphonate dosing for osteoporosis",
    "bronchodilator dosing for chronic obstructive pulmonary disease",
    "antibiotic selection for uncomplicated urinary tract infection",
    "triptan dosing for acute migraine",
    "dopaminergic dosing for parkinson disease",
    "mood stabilizer dosing for bipolar disorder",
    "biologic selection for plaque psoriasis",
    "antiemetic dosing for chemotherapy induced nausea",
)

# Off-topic-but-plausible vocabulary for the distractor corpora; disjoint from the cluster
# conditions above so the entity filter blocks them on cluster queries while they still fill
# top_k slots and shift BM25-IDF.
DISTRACTOR_DRUGS: tuple[str, ...] = (
    "sumatriptan", "levothyroxine", "nitrofurantoin", "allopurinol", "sertraline",
    "gabapentin", "spironolactone", "tamsulosin", "finasteride", "montelukast",
    "cetirizine", "ranitidine", "ondansetron", "prochlorperazine", "loperamide",
    "ferrous sulfate", "cyanocobalamin", "alendronate", "raloxifene", "calcitonin",
    "colchicine", "methotrexate", "hydroxychloroquine", "sulfasalazine", "azathioprine",
    "tacrolimus", "cyclosporine", "mycophenolate", "prednisone", "hydrocortisone",
    "fludrocortisone", "desmopressin", "oxybutynin", "solifenacin", "mirabegron",
    "donepezil", "memantine", "rivastigmine", "pramipexole", "ropinirole",
    "carbidopa", "valproate", "lamotrigine", "topiramate", "levetiracetam",
    "clonazepam", "buspirone", "trazodone",
)
DISTRACTOR_CONDITIONS: tuple[str, ...] = (
    "migraine", "hypothyroidism", "urinary tract infection", "chronic gout",
    "generalized anxiety", "diabetic neuropathy", "primary hyperaldosteronism",
    "benign prostatic hyperplasia", "androgenetic alopecia", "allergic rhinitis",
    "chronic urticaria", "peptic ulcer disease", "chemotherapy nausea",
    "vertigo", "acute diarrhea", "iron deficiency anemia", "vitamin b12 deficiency",
    "postmenopausal osteoporosis", "paget disease of bone", "acute gout flare",
    "rheumatoid arthritis", "systemic lupus erythematosus", "psoriatic arthritis",
    "ulcerative colitis", "kidney transplant rejection", "polymyalgia rheumatica",
    "adrenal insufficiency", "central diabetes insipidus", "overactive bladder",
    "alzheimer dementia", "parkinson disease", "focal epilepsy",
    "absence seizures", "trigeminal neuralgia", "insomnia", "social phobia",
    "fibromyalgia", "restless legs syndrome", "tension headache", "panic disorder",
)
DISTRACTOR_TEMPLATES: tuple[str, ...] = (
    "{drug} dosing for {cond}",
    "{drug} side effects in {cond}",
    "monitoring parameters for {drug} in {cond}",
    "duration of {drug} therapy for {cond}",
    "{drug} contraindications in {cond}",
    "switching off {drug} in {cond}",
)

# Helpers


def _token_set(text: str) -> set[str]:
    return set(text.lower().split())


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _salient(text: str) -> set[str]:
    return {tok for tok in text.lower().split() if len(tok) >= 4 and tok not in STOPWORDS}


def _semantic_band(cosine: float) -> str:
    for threshold, label in SEMANTIC_BANDS:
        if cosine >= threshold:
            return label
    return SEMANTIC_BELOW


def _lexical_band(jaccard: float) -> LexicalBand:
    if jaccard <= 0.0:
        return "zero"
    for threshold, label in LEXICAL_BANDS:
        if jaccard >= threshold:
            return label
    return "low"


def _entity_overlap(query: str, ref_query: str, ref_entity: str) -> EntityOverlap:
    ref_salient = _salient(ref_query)
    shared = _salient(query) & ref_salient
    entity_present = ref_entity.lower() in query.lower().split()
    if entity_present and ref_salient and len(shared) >= len(ref_salient) / 2:
        return "full"
    if shared:
        return "partial"
    return "none"


def _slugify(text: str) -> str:
    return "-".join(tok for tok in text.lower().split() if tok.isalnum())[:40]


def _query_length_band(text: str) -> str:
    count = len(text.split())
    if count <= 6:
        return "short"
    if count <= 15:
        return "medium"
    return "long"


def _context_length_band(context: str | None) -> str:
    if context is None:
        return "none"
    return "short" if len(context.split()) <= 12 else "long"


def _reorder(text: str, rng: random.Random) -> str:
    tokens = text.split()
    shuffled = tokens[:]
    for _ in range(8):
        rng.shuffle(shuffled)
        if shuffled != tokens:
            break
    return " ".join(shuffled)


def _filler(text: str, rng: random.Random) -> str:
    tokens = text.split()
    filler = rng.choice(FILLERS)
    return " ".join([tokens[0], filler, *tokens[1:]])


def _synonym_swap(text: str) -> str:
    tokens = text.split()
    swapped = [SYNONYMS.get(tok.lower(), tok) for tok in tokens]
    if swapped == tokens:
        swapped.append("overall")
    return " ".join(swapped)


def _near_duplicate(text: str) -> str:
    return f"recommended {text}"


def _parse_corpus_sizes(spec: str) -> list[int]:
    sizes: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip().lower()
        multiplier = 1
        if token.endswith("k"):
            multiplier = 1000
            token = token[:-1]
        sizes.add(int(token) * multiplier)
    return sorted(sizes)


def _harness_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


# LLM draft schema


class CanonicalDraft(BaseModel):
    query: str = Field(
        description="Concise clinical lookup query, e.g. 'warfarin dosing for atrial fibrillation'."
    )
    entity: str = Field(
        description="The single primary drug named in the query, lowercased single word, e.g. 'warfarin'."
    )


class HardNegativeDraft(BaseModel):
    text: str = Field(description="A query that looks related but should NOT match canonical_a.")
    subtype: str = Field(
        description="One of: same_keyword_diff_meaning, same_template_diff_entity, topically_adjacent."
    )


class ClusterDraft(BaseModel):
    scenario: str = Field(description="One-line description of the clinical scenario.")
    condition: str = Field(
        description="Shared clinical condition both canonicals scope to, e.g. 'atrial fibrillation'."
    )
    canonical_a: CanonicalDraft
    canonical_b: CanonicalDraft = Field(
        description=(
            "Partner canonical: SAME condition, a DIFFERENT drug, and phrased with a different "
            "descriptive verb/noun than canonical_a (e.g. 'titration' vs 'dosing') so the two "
            "queries differ by more than the drug name."
        )
    )
    paraphrases: list[str] = Field(
        description=(
            "2-3 same-meaning rewordings of canonical_a's question; at least one must share very "
            "few words with it (different vocabulary, same clinical intent)."
        )
    )
    hard_negatives: list[HardNegativeDraft] = Field(
        description="Exactly three, one of each subtype; the diff_entity one must name a drug not in this cluster."
    )
    holdout_query: str = Field(
        description="A natural, realistic phrasing a clinician would actually type that should match canonical_a."
    )


# Cluster intermediate representation


@dataclass(frozen=True)
class CanonSpec:
    query: str
    entity: str
    payload: str


@dataclass(frozen=True)
class HardNegSpec:
    text: str
    subtype: str


@dataclass(frozen=True)
class ClusterSpec:
    cluster_id: str
    domain: str
    keys: list[str]
    canon_a: CanonSpec
    canon_b: CanonSpec
    seed_positives: list[str]
    hard_negatives: list[HardNegSpec]
    holdout_query: str


@dataclass(frozen=True)
class DisambigVariant:
    """One context arm of a same-query disambiguation cluster.

    ``context`` is stored on the variant's canonical; ``eval_context`` is the matching
    context phrasing attached to every eval query for this arm. They share the arm's
    distinctive tokens and are token-disjoint from sibling arms, so the lexical context
    gate accepts this arm and rejects the others.
    """

    payload_suffix: str
    context: str
    eval_context: str


@dataclass(frozen=True)
class DisambigSpec:
    cluster_id: str
    domain: str
    keys: list[str]
    entity: str
    query: str
    paraphrases: list[str]
    variants: list[DisambigVariant]


@dataclass(frozen=True)
class DistractorSpec:
    payload: str
    query: str
    keys: list[str]


# Seed clusters (deterministic path)


@dataclass(frozen=True)
class SeedCluster:
    domain: str
    condition: str
    entity_a: str
    query_a: str
    entity_b: str
    query_b: str
    paraphrase: str
    low_overlap: str
    neg_same_keyword: str
    neg_diff_entity: str
    neg_adjacent: str
    holdout: str


SEED_CLUSTERS: tuple[SeedCluster, ...] = (
    SeedCluster(
        domain="clinical",
        condition="community acquired pneumonia",
        entity_a="amoxicillin",
        query_a="amoxicillin selection for community acquired pneumonia",
        entity_b="azithromycin",
        query_b="azithromycin choice for community acquired pneumonia",
        paraphrase="which amoxicillin regimen treats community acquired pneumonia",
        low_overlap="first line antibiotic to prescribe for a lung infection caught outside hospital",
        neg_same_keyword="amoxicillin allergy management in community acquired pneumonia",
        neg_diff_entity="doxycycline selection for community acquired pneumonia",
        neg_adjacent="chest radiograph findings in community acquired pneumonia",
        holdout="what amoxicillin should i start for outpatient pneumonia",
    ),
    SeedCluster(
        domain="clinical",
        condition="hypertension",
        entity_a="lisinopril",
        query_a="lisinopril dosing for hypertension",
        entity_b="amlodipine",
        query_b="amlodipine titration for hypertension",
        paraphrase="what lisinopril dose controls hypertension",
        low_overlap="amount of blood pressure medicine to bring down elevated readings",
        neg_same_keyword="lisinopril induced cough evaluation in hypertension",
        neg_diff_entity="losartan dosing for hypertension",
        neg_adjacent="lifestyle modifications to manage hypertension",
        holdout="how much lisinopril for high blood pressure",
    ),
    SeedCluster(
        domain="clinical",
        condition="type two diabetes",
        entity_a="metformin",
        query_a="metformin initiation for type two diabetes",
        entity_b="glipizide",
        query_b="glipizide dosing for type two diabetes",
        paraphrase="how to start metformin in type two diabetes",
        low_overlap="beginning an oral pill to lower blood sugar in adult onset disease",
        neg_same_keyword="metformin contraindications in type two diabetes",
        neg_diff_entity="sitagliptin initiation for type two diabetes",
        neg_adjacent="carbohydrate counting education for type two diabetes",
        holdout="what starting metformin dose for a newly diagnosed diabetic",
    ),
    SeedCluster(
        domain="software",
        condition="python list deduplication",
        entity_a="order",
        query_a="remove duplicate items from a python list while preserving order",
        entity_b="set",
        query_b="remove duplicate items from a python list using a set",
        paraphrase="how to drop repeated entries from a python list and keep their order",
        low_overlap="eliminate repeated elements from a sequence in python without changing the arrangement",
        neg_same_keyword="how to find the duplicate items in a python list",
        neg_diff_entity="remove duplicate items from a python list using pandas",
        neg_adjacent="sort a python list of dictionaries by a key",
        holdout="how do i dedupe a list in python but keep the original order",
    ),
    SeedCluster(
        domain="software",
        condition="http status code meaning",
        entity_a="404",
        query_a="what does http status code 404 mean",
        entity_b="500",
        query_b="what does http status code 500 indicate",
        paraphrase="meaning of the 404 response status in http",
        low_overlap="which error is returned when a requested web page cannot be found",
        neg_same_keyword="how to fix a 404 not found error on my website",
        neg_diff_entity="what does http status code 301 mean",
        neg_adjacent="difference between the http and https protocols",
        holdout="what is a 404 error",
    ),
    SeedCluster(
        domain="software",
        condition="git branch deletion",
        entity_a="delete",
        query_a="how to delete a local git branch",
        entity_b="rename",
        query_b="how to rename a local git branch",
        paraphrase="what command will delete a local branch in git",
        low_overlap="getting rid of a development line in a repository on your own machine",
        neg_same_keyword="how to recover a deleted git branch",
        neg_diff_entity="how to merge a local git branch",
        neg_adjacent="how to list all of the remote git branches",
        holdout="delete a branch in git",
    ),
    SeedCluster(
        domain="personal-finance",
        condition="ira contribution limit",
        entity_a="roth",
        query_a="what is the annual contribution limit for a roth ira",
        entity_b="traditional",
        query_b="what is the annual contribution limit for a traditional ira",
        paraphrase="how much can i put into a roth ira each year",
        low_overlap="yearly cap on money added to an after tax retirement account",
        neg_same_keyword="what is the income limit to contribute to a roth ira",
        neg_diff_entity="what is the annual contribution limit for a 401k",
        neg_adjacent="how are roth ira withdrawals taxed in retirement",
        holdout="how much can i contribute to a roth ira this year",
    ),
    SeedCluster(
        domain="personal-finance",
        condition="credit card debt payoff",
        entity_a="avalanche",
        query_a="how does the debt avalanche method pay off credit cards",
        entity_b="snowball",
        query_b="how does the debt snowball method pay off credit cards",
        paraphrase="how to use the avalanche strategy to clear credit card balances",
        low_overlap="approach that tackles the highest interest balances first to become debt free",
        neg_same_keyword="how does a balance transfer help pay off credit card debt",
        neg_diff_entity="how does debt consolidation pay off credit cards",
        neg_adjacent="how is credit card interest calculated each month",
        holdout="what is the avalanche method for paying off credit cards",
    ),
    SeedCluster(
        domain="how-to",
        condition="red wine stain removal",
        entity_a="carpet",
        query_a="how to remove a red wine stain from carpet",
        entity_b="shirt",
        query_b="how to remove a red wine stain from a cotton shirt",
        paraphrase="best way to get a red wine stain out of carpet",
        low_overlap="lifting a dark spilled drink mark out of floor covering fibers",
        neg_same_keyword="how to prevent red wine stains on a carpet",
        neg_diff_entity="how to remove a red wine stain from a wooden table",
        neg_adjacent="how to remove a coffee stain from carpet",
        holdout="get a red wine stain out of the carpet",
    ),
    SeedCluster(
        domain="how-to",
        condition="moving files to a new computer",
        entity_a="network",
        query_a=(
            "what is the most reliable way to transfer a large folder of files from my old "
            "laptop to a new computer over a home network"
        ),
        entity_b="drive",
        query_b=(
            "what is the most reliable way to transfer a large folder of files from my old "
            "laptop to a new computer using an external drive"
        ),
        paraphrase=(
            "how can i move a big directory of documents from one machine to another across my "
            "local home network without losing anything"
        ),
        low_overlap=(
            "copying a sizable collection of personal data between two desktops connected to the "
            "same router at home"
        ),
        neg_same_keyword=(
            "why is transferring a large folder of files over the home network so slow"
        ),
        neg_diff_entity=(
            "what is the most reliable way to transfer a large folder of files from my old laptop "
            "to a new computer with a usb cable"
        ),
        neg_adjacent="how do i back up the files on my old laptop before i replace it",
        holdout="best way to move all my files to a new laptop over wifi",
    ),
)


# Context-disambiguation seed clusters (deterministic path). One shared query and key set
# with two or three context arms, each its own canonical now that EntryId hashes context.
# Per arm, the eval context shares the arm's distinctive tokens and is token-disjoint from
# the sibling arms, so the lexical context gate accepts the matching arm and rejects the
# others; the query never names the disambiguator.
DISAMBIG_CLUSTERS: tuple[DisambigSpec, ...] = (
    DisambigSpec(
        cluster_id="d00-atrial-fibrillation",
        domain="clinical",
        keys=["atrial fibrillation"],
        entity="fibrillation",
        query="what is the recommended anticoagulant dose for this patient with atrial fibrillation",
        paraphrases=[
            "recommended anticoagulation dosing for a patient who has atrial fibrillation",
            "how should the anticoagulant be dosed for this atrial fibrillation patient",
        ],
        variants=[
            DisambigVariant(
                payload_suffix="warfarin",
                context="warfarin vitamin k antagonist",
                eval_context="currently prescribed warfarin a vitamin k antagonist",
            ),
            DisambigVariant(
                payload_suffix="apixaban",
                context="apixaban direct factor xa inhibitor",
                eval_context="currently prescribed apixaban a factor xa inhibitor",
            ),
        ],
    ),
    DisambigSpec(
        cluster_id="d01-persistent-asthma",
        domain="clinical",
        keys=["persistent asthma"],
        entity="asthma",
        query="which controller medication should be added for this patient with persistent asthma",
        paraphrases=[
            "what controller therapy should be added on for a patient with persistent asthma",
            "recommended add on controller treatment for this persistent asthma patient",
        ],
        variants=[
            DisambigVariant(
                payload_suffix="ics",
                context="low dose inhaled corticosteroid monotherapy",
                eval_context="currently using a low dose inhaled corticosteroid",
            ),
            DisambigVariant(
                payload_suffix="laba",
                context="combination inhaler adding long acting beta agonist",
                eval_context="already on a combination inhaler with a long acting beta agonist",
            ),
        ],
    ),
    DisambigSpec(
        cluster_id="d02-configuration-loading",
        domain="software",
        keys=["configuration loading at startup"],
        entity="configuration",
        query="how do i read a configuration value at startup",
        paraphrases=[
            "what is the best way to read a configuration value at service startup",
            "how should the app read a configuration value during startup",
        ],
        variants=[
            DisambigVariant(
                payload_suffix="python",
                context=(
                    "this backend runs on python and relies on the pydantic settings library "
                    "reading values from environment variables and a dotenv file at process startup"
                ),
                eval_context=(
                    "our python service uses pydantic settings to read values from environment "
                    "variables and a dotenv file"
                ),
            ),
            DisambigVariant(
                payload_suffix="rust",
                context=(
                    "this server is written in rust using the serde crate with figment to merge "
                    "toml defaults and shell exported overrides into typed structs"
                ),
                eval_context=(
                    "our rust service uses serde and figment to merge toml defaults and shell "
                    "exported overrides into typed structs"
                ),
            ),
        ],
    ),
    DisambigSpec(
        cluster_id="d03-run-the-test-suite",
        domain="software",
        keys=["run the test suite"],
        entity="test",
        query="how do i run the test suite for this project",
        paraphrases=[
            "what command runs the full test suite in this project",
            "how to execute the whole test suite for this project",
        ],
        variants=[
            DisambigVariant(
                payload_suffix="pytest",
                context="python pytest",
                eval_context="written in python and tested with pytest",
            ),
            DisambigVariant(
                payload_suffix="cargo",
                context="rust cargo",
                eval_context="written in rust and tested with cargo",
            ),
            DisambigVariant(
                payload_suffix="jest",
                context="javascript jest",
                eval_context="written in javascript and tested with jest",
            ),
        ],
    ),
    DisambigSpec(
        cluster_id="d04-retirement-contribution",
        domain="personal-finance",
        keys=["retirement account contribution"],
        entity="contribution",
        query="how is my retirement contribution taxed for this account",
        paraphrases=[
            "how does the tax treatment work for my retirement contribution in this account",
            "what are the tax rules on a contribution to this retirement account",
        ],
        variants=[
            DisambigVariant(
                payload_suffix="roth",
                context="roth after tax",
                eval_context="this account is a roth funded with after tax money",
            ),
            DisambigVariant(
                payload_suffix="traditional",
                context="traditional pre tax deductible",
                eval_context="this account is a traditional one with pre tax deductible money",
            ),
        ],
    ),
    DisambigSpec(
        cluster_id="d05-mortgage-payment",
        domain="personal-finance",
        keys=["mortgage interest rate"],
        entity="mortgage",
        query=(
            "if market interest rates move over the next several years how will the monthly "
            "payment on this mortgage change across the full loan term"
        ),
        paraphrases=[
            "assuming interest rates shift over the coming years how much will my monthly "
            "mortgage payment change across the entire loan term",
            "over the full term of this mortgage how will my monthly payment respond if market "
            "interest rates rise or fall in the coming years",
        ],
        variants=[
            DisambigVariant(
                payload_suffix="fixed",
                context=(
                    "the interest is locked for the entire thirty year term so principal and "
                    "interest never change and the monthly amount stays identical"
                ),
                eval_context=(
                    "my loan has interest locked for the entire thirty year term so the monthly "
                    "amount stays identical"
                ),
            ),
            DisambigVariant(
                payload_suffix="arm",
                context=(
                    "the rate resets every year against a published index after a five year teaser "
                    "so payments can climb sharply once the intro window ends"
                ),
                eval_context=(
                    "my loan resets every year against a published index after the five year teaser "
                    "so payments can climb"
                ),
            ),
        ],
    ),
    DisambigSpec(
        cluster_id="d06-ingredient-substitute",
        domain="how-to",
        keys=["baking ingredient substitute"],
        entity="substitute",
        query="what can i substitute for this ingredient in the recipe",
        paraphrases=[
            "what is a good substitute for this ingredient when baking the recipe",
            "what should i use as a substitute for this missing ingredient",
        ],
        variants=[
            DisambigVariant(
                payload_suffix="butter",
                context="butter as the fat",
                eval_context="i ran out of butter the main fat here",
            ),
            DisambigVariant(
                payload_suffix="eggs",
                context="eggs as the binder",
                eval_context="i ran out of eggs the binder here",
            ),
            DisambigVariant(
                payload_suffix="buttermilk",
                context="buttermilk for acidity",
                eval_context="i ran out of buttermilk needed for acidity here",
            ),
        ],
    ),
    DisambigSpec(
        cluster_id="d07-houseplant-watering",
        domain="how-to",
        keys=["houseplant watering schedule"],
        entity="houseplant",
        query="how often should i water this houseplant",
        paraphrases=[
            "what is the right watering frequency for this houseplant",
            "how frequently does this houseplant need to be watered",
        ],
        variants=[
            DisambigVariant(
                payload_suffix="succulent",
                context=(
                    "a desert succulent storing water in thick fleshy leaves that prefers long dry "
                    "spells and rots if its roots stay wet"
                ),
                eval_context=(
                    "my plant is a desert succulent with thick fleshy leaves that prefers long dry "
                    "spells between waterings"
                ),
            ),
            DisambigVariant(
                payload_suffix="fern",
                context=(
                    "a tropical rainforest fern that demands constantly damp footing and elevated "
                    "humidity and browns at the fronds whenever it is left thirsty"
                ),
                eval_context=(
                    "my plant is a tropical rainforest fern that demands constantly damp footing "
                    "and high humidity"
                ),
            ),
        ],
    ),
)


def _seed_to_spec(index: int, seed: SeedCluster) -> ClusterSpec:
    cluster_id = f"c{index:02d}-{_slugify(seed.condition)}"
    return ClusterSpec(
        cluster_id=cluster_id,
        domain=seed.domain,
        keys=[seed.condition],
        canon_a=CanonSpec(
            query=seed.query_a,
            entity=seed.entity_a,
            payload=f"{cluster_id}:a",
        ),
        canon_b=CanonSpec(
            query=seed.query_b,
            entity=seed.entity_b,
            payload=f"{cluster_id}:b",
        ),
        seed_positives=[seed.paraphrase, seed.low_overlap],
        hard_negatives=[
            HardNegSpec(seed.neg_same_keyword, "same_keyword_diff_meaning"),
            HardNegSpec(seed.neg_diff_entity, "same_template_diff_entity"),
            HardNegSpec(seed.neg_adjacent, "topically_adjacent"),
        ],
        holdout_query=seed.holdout,
    )


def _draft_to_spec(index: int, draft: ClusterDraft) -> ClusterSpec:
    cluster_id = f"c{index:02d}-{_slugify(draft.condition)}"
    return ClusterSpec(
        cluster_id=cluster_id,
        domain="clinical",
        keys=[draft.condition],
        canon_a=CanonSpec(
            query=draft.canonical_a.query,
            entity=draft.canonical_a.entity,
            payload=f"{cluster_id}:a",
        ),
        canon_b=CanonSpec(
            query=draft.canonical_b.query,
            entity=draft.canonical_b.entity,
            payload=f"{cluster_id}:b",
        ),
        seed_positives=list(draft.paraphrases),
        hard_negatives=[
            HardNegSpec(neg.text, neg.subtype) for neg in draft.hard_negatives
        ],
        holdout_query=draft.holdout_query,
    )


# Distractor corpora


def build_distractor_specs(seed: int, count: int) -> list[DistractorSpec]:
    combos = [
        (drug, cond, template)
        for template in DISTRACTOR_TEMPLATES
        for drug in DISTRACTOR_DRUGS
        for cond in DISTRACTOR_CONDITIONS
    ]
    if count > len(combos):
        raise ValueError(f"requested {count} distractors but only {len(combos)} combos exist")
    rng = random.Random(seed ^ 0xD15)
    rng.shuffle(combos)
    specs: list[DistractorSpec] = []
    for i, (drug, cond, template) in enumerate(combos[:count]):
        specs.append(
            DistractorSpec(
                payload=f"distractor-{i:05d}",
                query=template.format(drug=drug, cond=cond),
                keys=[cond],
            )
        )
    return specs


# LLM authoring path


def _author_prompt(index: int, theme: str) -> str:
    return (
        "You are authoring ONE labeled cluster for a clinical-QA semantic-cache benchmark.\n"
        f"Scenario theme: {theme}.\n\n"
        "Produce a structured cluster with:\n"
        "- canonical_a: a concise drug-dosing/selection lookup query and its single primary drug.\n"
        "- canonical_b: the SAME clinical condition with a DIFFERENT drug, phrased with a different\n"
        "  descriptive word than canonical_a (e.g. 'dosing' vs 'titration') so the two queries\n"
        "  differ by more than just the drug name.\n"
        "- paraphrases: 2-3 same-meaning rewordings of canonical_a; at least one sharing very few\n"
        "  words with it.\n"
        "- hard_negatives: exactly three, one of each subtype (same_keyword_diff_meaning,\n"
        "  same_template_diff_entity, topically_adjacent); the diff_entity one names a drug that is\n"
        "  NOT canonical_a or canonical_b's drug.\n"
        "- holdout_query: a natural realistic phrasing a clinician would type that matches canonical_a.\n"
        "Keep every string short (a clinician's search query), lowercase, no punctuation."
    )


def _structured_call(prompt: str, model: str) -> ClusterDraft:
    return extract_sync(
        prompt,
        ClusterDraft,
        backend=ClaudeCliBackend(),
        model=model,
        timeout=LLM_TIMEOUT_SECS,
    )


def _cache_key(version: str, index: int, seed: int) -> str:
    return f"{version}:{index}:{seed}"


def author_clusters_llm(
    *, version: str, n_clusters: int, seed: int, model: str, run_dir: Path
) -> list[ClusterSpec]:
    if n_clusters > len(CLINICAL_THEMES):
        raise ValueError(
            f"LLM path has {len(CLINICAL_THEMES)} clinical themes; requested {n_clusters}"
        )
    cache_path = run_dir / "_llm_cache.json"
    cache: dict[str, dict] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    drafts: dict[int, ClusterDraft] = {}
    pending: list[int] = []
    for index in range(n_clusters):
        key = _cache_key(version, index, seed)
        if key in cache:
            drafts[index] = ClusterDraft.model_validate(cache[key])
        else:
            pending.append(index)

    if pending:

        async def author(index: int) -> ClusterDraft:
            prompt = _author_prompt(index, CLINICAL_THEMES[index])
            return await asyncio.to_thread(_structured_call, prompt, model)

        results = asyncio.run(map_concurrent(pending, author, limit=LLM_CONCURRENCY))
        for index, draft in zip(pending, results):
            drafts[index] = draft
            cache[_cache_key(version, index, seed)] = draft.model_dump()
        run_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))

    return [_draft_to_spec(index, drafts[index]) for index in range(n_clusters)]


def author_clusters_deterministic(n_clusters: int) -> list[ClusterSpec]:
    if n_clusters > len(SEED_CLUSTERS):
        raise ValueError(
            f"deterministic path has {len(SEED_CLUSTERS)} pair seed clusters; requested {n_clusters}"
        )
    return [_seed_to_spec(i, SEED_CLUSTERS[i]) for i in range(n_clusters)]


def author_disambig_deterministic() -> list[DisambigSpec]:
    return list(DISAMBIG_CLUSTERS)


def _assert_distinct_canonicals(
    clusters: list[ClusterSpec], disambig: list[DisambigSpec]
) -> None:
    """Reject any two canonicals that share a ``(query, keys, context)`` triple.

    That triple is exactly what ``EntryId::derive`` (``src/newtype.rs``) hashes, so a repeated
    triple is a genuine entry-id collision: the two payloads overwrite each other on one
    lookup target and confound every recall/wrong-entry metric. Same query and keys with a
    *different* context now derive distinct ids -- the disambiguation clusters depend on this,
    so they pass. Authoring fails loud here rather than emit a poisoned dataset.
    """
    seen: dict[tuple[str, tuple[str, ...], str | None], str] = {}
    offenders: list[str] = []

    def check(query: str, keys: list[str], ctx: str | None, payload: str) -> None:
        signature = (query, tuple(sorted(keys)), ctx)
        if signature in seen:
            offenders.append(f"{payload} duplicates {seen[signature]} on {signature}")
        else:
            seen[signature] = payload

    for cluster in clusters:
        for canon in (cluster.canon_a, cluster.canon_b):
            check(canon.query, cluster.keys, None, canon.payload)
    for spec in disambig:
        for variant in spec.variants:
            check(
                spec.query,
                spec.keys,
                variant.context,
                f"{spec.cluster_id}:{variant.payload_suffix}",
            )
    if offenders:
        raise ValueError(
            "duplicate canonical (query, keys, context) triples: " + "; ".join(offenders)
        )


# Materialization + labeling pipeline


class _TextTable:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def add(self, text: str) -> int:
        index = len(self.texts)
        self.texts.append(text)
        return index


@dataclass
class _PendingQuery:
    query_id: str
    cluster_id: str
    domain: str
    kind: QueryKind
    query: str
    context: str | None
    keys: list[str]
    expected: str
    negative_subtype: str | None
    vector_ref: int
    context_vector_ref: int | None
    ref_vector_ref: int
    ref_query: str
    ref_entity: str


@dataclass
class _Materialized:
    canonicals: list[CanonicalEntry]
    queries: list[_PendingQuery]
    holdout: list[_PendingQuery]
    distractors: list[Distractor]
    main: _TextTable
    context: _TextTable


def _positive_variants(spec: ClusterSpec, rng: random.Random) -> list[str]:
    canon = spec.canon_a.query
    return [
        *spec.seed_positives,
        _near_duplicate(canon),
        _reorder(canon, rng),
        _filler(canon, rng),
        _synonym_swap(canon),
    ]


def _materialize(
    clusters: list[ClusterSpec],
    disambig: list[DisambigSpec],
    distractors: list[DistractorSpec],
    seed: int,
) -> _Materialized:
    main = _TextTable()
    context = _TextTable()
    canonicals: list[CanonicalEntry] = []
    queries: list[_PendingQuery] = []
    canon_refs: dict[str, tuple[int, str, str]] = {}

    for cluster in clusters:
        for canon in (cluster.canon_a, cluster.canon_b):
            vector_ref = main.add(canon.query)
            canonicals.append(
                CanonicalEntry(
                    schema_version=SCHEMA_VERSION,
                    cluster_id=cluster.cluster_id,
                    domain=cluster.domain,
                    canonical_key=canon.payload,
                    query=canon.query,
                    context=None,
                    keys=cluster.keys,
                    payload=canon.payload,
                    vector_ref=vector_ref,
                    context_vector_ref=None,
                )
            )
            canon_refs[canon.payload] = (vector_ref, canon.query, canon.entity)

    for cluster in clusters:
        rng = random.Random(seed * 1000 + int(cluster.cluster_id[1:3]))
        a_ref, a_query, a_entity = canon_refs[cluster.canon_a.payload]

        for i, text in enumerate(_positive_variants(cluster, rng)):
            queries.append(
                _PendingQuery(
                    query_id=f"{cluster.cluster_id}-pos-{i}",
                    cluster_id=cluster.cluster_id,
                    domain=cluster.domain,
                    kind="positive",
                    query=text,
                    context=None,
                    keys=cluster.keys,
                    expected=f"hit:{cluster.canon_a.payload}",
                    negative_subtype=None,
                    vector_ref=main.add(text),
                    context_vector_ref=None,
                    ref_vector_ref=a_ref,
                    ref_query=a_query,
                    ref_entity=a_entity,
                )
            )

        for i, neg in enumerate(cluster.hard_negatives):
            queries.append(
                _PendingQuery(
                    query_id=f"{cluster.cluster_id}-neg-{i}",
                    cluster_id=cluster.cluster_id,
                    domain=cluster.domain,
                    kind="hard_negative",
                    query=neg.text,
                    context=None,
                    keys=cluster.keys,
                    expected="miss",
                    negative_subtype=neg.subtype,
                    vector_ref=main.add(neg.text),
                    context_vector_ref=None,
                    ref_vector_ref=a_ref,
                    ref_query=a_query,
                    ref_entity=a_entity,
                )
            )

    _materialize_disambig(disambig, queries, canonicals, main, context)

    holdout = _materialize_holdout(clusters, canon_refs, main)

    distractor_records = [
        Distractor(
            schema_version=SCHEMA_VERSION,
            distractor_id=spec.payload,
            query=spec.query,
            context=None,
            keys=spec.keys,
            payload=spec.payload,
            vector_ref=main.add(spec.query),
        )
        for spec in distractors
    ]

    return _Materialized(canonicals, queries, holdout, distractor_records, main, context)


def _materialize_disambig(
    specs: list[DisambigSpec],
    queries: list[_PendingQuery],
    canonicals: list[CanonicalEntry],
    main: _TextTable,
    context: _TextTable,
) -> None:
    """Append one canonical per context arm and one eval query per (paraphrase, arm) pair.

    Every arm of a cluster stores the same query ``Q`` (one shared vector) under a distinct
    context, so the arms are distinct entries that the dense and entity gates cannot tell
    apart -- only the context gate can. Each paraphrase is paired with each arm's eval
    context, so a paraphrase resolved to the wrong arm is a measurable ``wrong_entry_hit``.
    """
    for spec in specs:
        query_ref = main.add(spec.query)
        for variant in spec.variants:
            payload = f"{spec.cluster_id}:{variant.payload_suffix}"
            canonicals.append(
                CanonicalEntry(
                    schema_version=SCHEMA_VERSION,
                    cluster_id=spec.cluster_id,
                    domain=spec.domain,
                    canonical_key=payload,
                    query=spec.query,
                    context=variant.context,
                    keys=spec.keys,
                    payload=payload,
                    vector_ref=query_ref,
                    context_vector_ref=context.add(variant.context),
                )
            )
        eval_context_refs = {
            variant.payload_suffix: context.add(variant.eval_context)
            for variant in spec.variants
        }
        for para_index, paraphrase in enumerate(spec.paraphrases):
            paraphrase_ref = main.add(paraphrase)
            for variant in spec.variants:
                payload = f"{spec.cluster_id}:{variant.payload_suffix}"
                queries.append(
                    _PendingQuery(
                        query_id=f"{spec.cluster_id}-ctx-{variant.payload_suffix}-{para_index}",
                        cluster_id=spec.cluster_id,
                        domain=spec.domain,
                        kind="context_pair",
                        query=paraphrase,
                        context=variant.eval_context,
                        keys=spec.keys,
                        expected=f"hit:{payload}",
                        negative_subtype=None,
                        vector_ref=paraphrase_ref,
                        context_vector_ref=eval_context_refs[variant.payload_suffix],
                        ref_vector_ref=query_ref,
                        ref_query=spec.query,
                        ref_entity=spec.entity,
                    )
                )


FIXED_HOLDOUT_MISSES: tuple[tuple[str, str], ...] = (
    ("what are common side effects of ibuprofen", "general pharmacology"),
    ("how is a sprained ankle treated", "musculoskeletal injury"),
    ("which vaccines are recommended for healthy adults", "preventive care"),
)


def _materialize_holdout(
    clusters: list[ClusterSpec],
    canon_refs: dict[str, tuple[int, str, str]],
    main: _TextTable,
) -> list[_PendingQuery]:
    holdout: list[_PendingQuery] = []
    for cluster in clusters:
        a_ref, a_query, a_entity = canon_refs[cluster.canon_a.payload]
        holdout.append(
            _PendingQuery(
                query_id=f"holdout-{cluster.cluster_id}",
                cluster_id=cluster.cluster_id,
                domain=cluster.domain,
                kind="positive",
                query=cluster.holdout_query,
                context=None,
                keys=cluster.keys,
                expected=f"hit:{cluster.canon_a.payload}",
                negative_subtype=None,
                vector_ref=main.add(cluster.holdout_query),
                context_vector_ref=None,
                ref_vector_ref=a_ref,
                ref_query=a_query,
                ref_entity=a_entity,
            )
        )
    ref_ref, ref_query, ref_entity = canon_refs[clusters[0].canon_a.payload]
    for i, (text, key) in enumerate(FIXED_HOLDOUT_MISSES):
        holdout.append(
            _PendingQuery(
                query_id=f"holdout-miss-{i}",
                cluster_id=clusters[0].cluster_id,
                domain=clusters[0].domain,
                kind="hard_negative",
                query=text,
                context=None,
                keys=[key],
                expected="miss",
                negative_subtype="topically_adjacent",
                vector_ref=main.add(text),
                context_vector_ref=None,
                ref_vector_ref=ref_ref,
                ref_query=ref_query,
                ref_entity=ref_entity,
            )
        )
    return holdout


def _label_query(pending: _PendingQuery, vectors: np.ndarray) -> LabeledQuery:
    cosine = round(float(np.dot(vectors[pending.vector_ref], vectors[pending.ref_vector_ref])), 6)
    jaccard = round(_jaccard(_token_set(pending.query), _token_set(pending.ref_query)), 6)
    return LabeledQuery(
        schema_version=SCHEMA_VERSION,
        query_id=pending.query_id,
        cluster_id=pending.cluster_id,
        domain=pending.domain,
        kind=pending.kind,
        query=pending.query,
        context=pending.context,
        keys=pending.keys,
        expected=pending.expected,
        lexical_overlap_jaccard=jaccard,
        lexical_overlap_band=_lexical_band(jaccard),
        semantic_cosine=cosine,
        semantic_cosine_band=_semantic_band(cosine),
        entity_overlap=_entity_overlap(pending.query, pending.ref_query, pending.ref_entity),
        has_context=pending.context is not None,
        negative_subtype=pending.negative_subtype,
        vector_ref=pending.vector_ref,
        context_vector_ref=pending.context_vector_ref,
    )


# Output


def _write_jsonl(path: Path, rows: list[BaseModel]) -> bytes:
    payload = "".join(
        json.dumps(row.model_dump(), sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    path.write_bytes(payload)
    return payload


def _histogram(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _build_manifest(
    *,
    version: str,
    generator: str,
    seed: int,
    n_clusters: int,
    canonicals: list[CanonicalEntry],
    queries: list[LabeledQuery],
    holdout: list[LabeledQuery],
    distractor_count: int,
    corpus_sizes: list[int],
    content_sha256: str,
) -> Manifest:
    counts = {
        "canonicals": len(canonicals),
        "positives": sum(1 for q in queries if q.kind == "positive"),
        "hard_negatives": sum(1 for q in queries if q.kind == "hard_negative"),
        "context_pairs": sum(1 for q in queries if q.kind == "context_pair"),
        "queries_total": len(queries),
        "holdout": len(holdout),
        "distractor_pool": distractor_count,
        **{f"corpus_{_size_tag(size)}": len(canonicals) + size for size in corpus_sizes},
    }
    axis_histograms = {
        "kind": _histogram([q.kind for q in queries]),
        "domain": _histogram([canon.domain for canon in canonicals]),
        "query_length": _histogram([_query_length_band(q.query) for q in queries]),
        "context_length": _histogram([_context_length_band(q.context) for q in queries]),
        "semantic_cosine_band": _histogram([q.semantic_cosine_band for q in queries]),
        "lexical_overlap_band": _histogram([q.lexical_overlap_band for q in queries]),
        "entity_overlap": _histogram([q.entity_overlap for q in queries]),
        "negative_subtype": _histogram(
            [q.negative_subtype for q in queries if q.negative_subtype is not None]
        ),
    }
    return Manifest(
        schema_version=SCHEMA_VERSION,
        dataset_version=version,
        generator=generator,
        rng_seed=seed,
        embed_model_id=BGE_MODEL_ID,
        embed_dim=BGE_DIM,
        query_instruction=BGE_QUERY_PREFIX,
        normalized=True,
        n_clusters=n_clusters,
        counts=counts,
        axis_histograms=axis_histograms,
        content_sha256=content_sha256,
        harness_git_sha=_harness_git_sha(),
    )


def generate(
    *,
    version: str,
    n_clusters: int,
    corpus_sizes: list[int],
    seed: int,
    deterministic: bool,
    llm_model: str,
) -> Path:
    run_dir = DATA_ROOT / version
    run_dir.mkdir(parents=True, exist_ok=True)

    if deterministic:
        clusters = author_clusters_deterministic(n_clusters)
        disambig = author_disambig_deterministic()
        generator = "deterministic"
    else:
        clusters = author_clusters_llm(
            version=version, n_clusters=n_clusters, seed=seed, model=llm_model, run_dir=run_dir
        )
        disambig = []
        generator = f"llm:claude-{llm_model}"

    _assert_distinct_canonicals(clusters, disambig)
    total_clusters = len(clusters) + len(disambig)

    pool_size = max(corpus_sizes)
    distractor_specs = build_distractor_specs(seed, pool_size)

    materialized = _materialize(clusters, disambig, distractor_specs, seed)

    embedder = ParityEmbedder()
    vectors = embedder.embed(materialized.main.texts)
    context_vectors = embedder.embed(materialized.context.texts)

    labeled_queries = [_label_query(q, vectors) for q in materialized.queries]
    labeled_holdout = [_label_query(q, vectors) for q in materialized.holdout]

    np.save(run_dir / "vectors.npy", vectors)
    np.save(run_dir / "context_vectors.npy", context_vectors)

    entries_bytes = _write_jsonl(run_dir / "entries.jsonl", materialized.canonicals)
    queries_bytes = _write_jsonl(run_dir / "queries.jsonl", labeled_queries)
    holdout_bytes = _write_jsonl(run_dir / "holdout_real.jsonl", labeled_holdout)

    canon_as_distractors = [
        Distractor(
            schema_version=SCHEMA_VERSION,
            distractor_id=canon.canonical_key,
            query=canon.query,
            context=canon.context,
            keys=canon.keys,
            payload=canon.payload,
            vector_ref=canon.vector_ref,
        )
        for canon in materialized.canonicals
    ]
    corpus_dir = run_dir / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    corpus_bytes: list[bytes] = []
    for size in corpus_sizes:
        rows = [*canon_as_distractors, *materialized.distractors[:size]]
        corpus_bytes.append(_write_jsonl(corpus_dir / f"distractors_{_size_tag(size)}.jsonl", rows))

    digest = hashlib.sha256()
    for chunk in (entries_bytes, queries_bytes, holdout_bytes, *corpus_bytes):
        digest.update(chunk)
    content_sha256 = digest.hexdigest()

    manifest = _build_manifest(
        version=version,
        generator=generator,
        seed=seed,
        n_clusters=total_clusters,
        canonicals=materialized.canonicals,
        queries=labeled_queries,
        holdout=labeled_holdout,
        distractor_count=pool_size,
        corpus_sizes=corpus_sizes,
        content_sha256=content_sha256,
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest.model_dump(), indent=2, sort_keys=True))
    return run_dir


def _size_tag(size: int) -> str:
    return f"{size // 1000}k" if size >= 1000 and size % 1000 == 0 else str(size)


# CLI


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bench.gen_dataset", description=__doc__)
    parser.add_argument("--version", required=True, help="dataset version, e.g. v0")
    parser.add_argument(
        "--clusters",
        type=int,
        required=True,
        help="number of standard pair clusters; context-disambiguation clusters are always included",
    )
    parser.add_argument("--corpus", default="10,100,1k,10k", help="comma-separated corpus sizes")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="hand-authored seed clusters, no network (reproducibility gate)",
    )
    parser.add_argument(
        "--llm-model",
        default="large",
        choices=("small", "medium", "large"),
        help="Claude tier for the LLM authoring path (ignored with --deterministic)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    corpus_sizes = _parse_corpus_sizes(args.corpus)
    run_dir = generate(
        version=args.version,
        n_clusters=args.clusters,
        corpus_sizes=corpus_sizes,
        seed=args.seed,
        deterministic=args.deterministic,
        llm_model=args.llm_model,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    print(f"wrote dataset to {run_dir}")
    print(json.dumps({"counts": manifest["counts"], "axis_histograms": manifest["axis_histograms"]}, indent=2))


if __name__ == "__main__":
    main()
