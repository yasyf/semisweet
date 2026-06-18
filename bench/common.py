"""Shared spine for the empirical-sweep harness.

Schemas (dataset/manifest/sweep records), the parity embedder that reproduces the
daemon's BGE ``embed_query`` path on the harness side, the direct mapping from
:class:`SweepParams` to the hard-gate :class:`semisweet.Scoring`, the confusion-matrix
classifier/metrics, and the daemon-keepalive helper. Everything here is pure config and
math; the sweep/analyze modules layer the I/O and the optimizer on top.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel
from semisweet import MemoryVectors, Scoring

if TYPE_CHECKING:
    from fastembed import TextEmbedding

# Constants

BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"
BGE_DIM = 384
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TAU = 0.01

KEEPALIVE_IDLE_SECS = 3600

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_CACHE = _REPO_ROOT / ".fastembed_cache"

DATA_ROOT = Path(__file__).resolve().parent / "data"

# Type aliases

QueryKind = Literal["positive", "hard_negative", "context_pair"]
LexicalBand = Literal["high", "mid", "low", "zero"]
EntityOverlap = Literal["full", "partial", "none"]
ContextMode = Literal["ignore", "gate"]
ClassLabel = Literal[
    "correct_hit", "wrong_entry_hit", "correct_miss", "should_hit_miss", "false_hit"
]


# Helpers


def _model_cache_dir() -> str:
    override = os.environ.get("SEMISWEET_MODEL_CACHE")
    return override if override is not None else str(_DEFAULT_MODEL_CACHE)


def _size_tag(size: int) -> str:
    return f"{size // 1000}k" if size >= 1000 and size % 1000 == 0 else str(size)


def _read_jsonl(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line]


# Schemas


class CanonicalEntry(BaseModel):
    schema_version: int
    cluster_id: str
    domain: str
    canonical_key: str
    query: str
    context: str | None
    keys: list[str]
    payload: str
    vector_ref: int
    context_vector_ref: int | None


class LabeledQuery(BaseModel):
    schema_version: int
    query_id: str
    cluster_id: str
    domain: str
    kind: QueryKind
    query: str
    context: str | None
    keys: list[str]
    expected: str
    lexical_overlap_jaccard: float
    lexical_overlap_band: LexicalBand
    semantic_cosine: float
    semantic_cosine_band: str
    entity_overlap: EntityOverlap
    has_context: bool
    negative_subtype: str | None
    vector_ref: int
    context_vector_ref: int | None


class Distractor(BaseModel):
    schema_version: int
    distractor_id: str
    query: str
    context: str | None
    keys: list[str]
    payload: str
    vector_ref: int


class Manifest(BaseModel):
    schema_version: int
    dataset_version: str
    generator: str
    rng_seed: int
    embed_model_id: str
    embed_dim: int
    query_instruction: str
    normalized: bool
    n_clusters: int
    counts: dict[str, int]
    axis_histograms: dict[str, dict[str, int]]
    content_sha256: str
    harness_git_sha: str | None


class LoadedDataset(BaseModel):
    version: str
    corpus_size: int
    manifest: Manifest
    canonicals: list[CanonicalEntry]
    queries: list[LabeledQuery]
    corpus: list[Distractor]


class SweepParams(BaseModel):
    threshold: float
    context_gate: float
    context: ContextMode
    entity_filter: bool
    top_k: int
    k1: float | None
    b: float | None


class Confusion(BaseModel):
    correct_hit: int = 0
    wrong_entry_hit: int = 0
    correct_miss: int = 0
    should_hit_miss: int = 0
    false_hit: int = 0

    def metrics(self, n_should_miss: int, n_should_hit: int) -> dict[str, float]:
        return metrics(self, n_should_miss, n_should_hit)


# Classes


class ParityEmbedder:
    """Reproduce the daemon's BGE ``embed_query`` path on the harness side.

    Prepends :data:`BGE_QUERY_PREFIX`, runs ``fastembed`` BGE-small, and L2-normalizes
    each row so a dot product is a cosine. Stored entries and lookup queries go through
    this same path in the daemon, so a cosine computed here matches the daemon's dense
    score. The model loads lazily on the first :meth:`embed` and downloads to the
    ``SEMISWEET_MODEL_CACHE`` dir (or ``<repo>/.fastembed_cache``) if not cached.
    """

    def __init__(self) -> None:
        self._model: TextEmbedding | None = None

    def _ensure_model(self) -> TextEmbedding:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=BGE_MODEL_ID, cache_dir=_model_cache_dir()
            )
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        prefixed = [f"{BGE_QUERY_PREFIX}{text}" for text in texts]
        raw = np.asarray(list(model.embed(prefixed)), dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return (raw / norms).astype(np.float32)


class DaemonKeepalive:
    """Pin the daemon's idle timeout and runtime paths for the duration of a sweep run.

    On enter, raise ``SEMISWEET_IDLE_SECS`` (default 3600) and point the
    ``SEMISWEET_SOCKET``/``SEMISWEET_LOCK``/``SEMISWEET_LOG``/``SEMISWEET_MODEL_CACHE``
    paths at ``run_dir`` — but only where unset, so an outer harness's choices win
    (mirrors ``tests/conftest.py``'s per-run env). The idle pin alone does not keep the
    daemon warm: the connection pin is per-live-cache, so the caller must hold one
    long-lived, connected :class:`semisweet.SemanticCache` for the whole run.
    """

    def __init__(self, run_dir: Path, idle_secs: int = KEEPALIVE_IDLE_SECS) -> None:
        self._run_dir = run_dir
        self._idle_secs = idle_secs

    def __enter__(self) -> DaemonKeepalive:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("SEMISWEET_IDLE_SECS", str(self._idle_secs))
        os.environ.setdefault("SEMISWEET_SOCKET", str(self._run_dir / "d.sock"))
        os.environ.setdefault("SEMISWEET_LOCK", str(self._run_dir / "d.lock"))
        os.environ.setdefault("SEMISWEET_LOG", str(self._run_dir / "d.log"))
        os.environ.setdefault("SEMISWEET_MODEL_CACHE", str(_DEFAULT_MODEL_CACHE))
        return self

    def __exit__(self, *exc: object) -> None:
        return None


# Functions


def config_hash(params: SweepParams) -> str:
    payload = json.dumps(params.model_dump(), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def namespace_for(dataset_ver: str, corpus_size: int, params: SweepParams) -> str:
    return f"sweep-{dataset_ver}-{corpus_size}-{config_hash(params)}"


def build_scoring(params: SweepParams) -> Scoring:
    return Scoring(
        threshold=params.threshold,
        context_gate=params.context_gate,
        context=params.context,
        entity_filter=params.entity_filter,
        top_k=params.top_k,
    )


def load_dataset(
    version: str, corpus_size: int, data_root: Path = DATA_ROOT
) -> LoadedDataset:
    """Read a generated dataset and the distractor corpus for one ``corpus_size``.

    The corpus file already carries every canonical (as a :class:`Distractor` with the
    canonical payload) plus ``corpus_size`` off-topic distractors, so ``corpus`` is the
    complete ``set()`` list for a sweep run.
    """
    run_dir = data_root / version
    manifest = Manifest.model_validate_json((run_dir / "manifest.json").read_text())
    canonicals = [
        CanonicalEntry.model_validate_json(line)
        for line in _read_jsonl(run_dir / "entries.jsonl")
    ]
    queries = [
        LabeledQuery.model_validate_json(line)
        for line in _read_jsonl(run_dir / "queries.jsonl")
    ]
    corpus_path = run_dir / "corpus" / f"distractors_{_size_tag(corpus_size)}.jsonl"
    corpus = [
        Distractor.model_validate_json(line) for line in _read_jsonl(corpus_path)
    ]
    return LoadedDataset(
        version=version,
        corpus_size=corpus_size,
        manifest=manifest,
        canonicals=canonicals,
        queries=queries,
        corpus=corpus,
    )


def split_clusters(
    queries: list[LabeledQuery], seed: int, frac: float = 0.5
) -> tuple[set[str], set[str]]:
    """Partition cluster ids into (train, test) so a cluster's paraphrases never straddle.

    The split is on cluster id, not query, so every canonical's positives, hard
    negatives, and context pairs land on the same side. ``seed`` makes it reproducible,
    which lets the analyze step re-derive the identical test split.
    """
    cluster_ids = sorted({query.cluster_id for query in queries})
    rng = random.Random(seed)
    rng.shuffle(cluster_ids)
    n_train = max(1, round(len(cluster_ids) * frac))
    return set(cluster_ids[:n_train]), set(cluster_ids[n_train:])


def build_memory_vectors(params: SweepParams) -> MemoryVectors:
    try:
        return MemoryVectors(k1=params.k1, b=params.b)
    except TypeError:
        return MemoryVectors()


def classify(expected: str, returned: object, miss_sentinel: object) -> ClassLabel:
    if expected == "miss":
        return "correct_miss" if returned is miss_sentinel else "false_hit"
    canonical_key = expected.removeprefix("hit:")
    if returned is miss_sentinel:
        return "should_hit_miss"
    return "correct_hit" if returned == canonical_key else "wrong_entry_hit"


def metrics(
    confusion: Confusion, n_should_miss: int, n_should_hit: int
) -> dict[str, float]:
    true_positive = confusion.correct_hit
    false_positive = confusion.false_hit + confusion.wrong_entry_hit
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = true_positive / n_should_hit if n_should_hit else 0.0
    denominator = 0.25 * precision + recall
    f0_5 = 1.25 * precision * recall / denominator if denominator else 0.0
    false_hit_rate = confusion.false_hit / n_should_miss if n_should_miss else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f0_5": f0_5,
        "false_hit_rate": false_hit_rate,
    }
