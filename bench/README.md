# Benchmark harness

Dev tooling that produces and validates `semisweet`'s scoring defaults. It is not
packaged into the published wheel — the harness lives at the repo root, outside
maturin's `python-source`.

## Setup

Build the extension into a virtualenv, then install the harness dependencies:

```bash
uv venv
uvx maturin develop --uv
uv pip install --group bench
```

## Generate a dataset

The deterministic path is the reproducibility gate — hand-authored seed clusters, no
network. This reproduces the `v3` dataset the current defaults were locked against:

```bash
python -m bench.gen_dataset --version v3 --clusters 10 --deterministic --corpus 10,100
```

It writes `bench/data/v3/`: canonical entries to `set()`, labeled lookup queries, the
float32 vectors each record indexes into, distractor corpora, and a manifest.
`--clusters` counts the standard pair clusters; the context-disambiguation clusters are
always included. Drop `--deterministic` to author clinical pair-clusters through the
Claude CLI via `spawnllm` instead (cached, so re-runs are API-free).

`bench/data/` and `bench/results/` are gitignored — regenerate them rather than
committing the artifacts.

## Layout

- `gen_dataset.py` — the authoring (deterministic seeds or LLM) and labeling pipeline.
  Every axis label (semantic cosine, lexical overlap, entity overlap) is computed from
  the real embeddings, never an authored self-label.
- `common.py` — the schemas, the `ParityEmbedder` that reproduces the daemon's BGE
  `embed_query` path for offline labeling, `build_scoring` (maps tuning params to
  `semisweet.Scoring`), the confusion-matrix `classify`/`metrics`, and the dataset loader.

## What it exercises

The dataset spans four domains (clinical, software, personal-finance, how-to) and three
query kinds:

- **positive** — paraphrases of a canonical query that should hit it.
- **hard_negative** — same-template or same-keyword queries that should miss.
- **context_pair** — the same query and keys with different stored context, which are
  distinct entries because `EntryId` hashes context. The disambiguator appears only in
  the context, so the lexical context gate is the sole signal; returning the wrong
  variant is a measurable `wrong_entry_hit`.

Together they stress every gate in the scoring model: the dense threshold, the entity
hard-gate, the lexical context hard-gate, and the context-present dense floor.
