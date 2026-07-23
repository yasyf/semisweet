# semisweet Development Guide

An async, in-memory semantic cache with pluggable backends.

## Repository Structure

```
semisweet/
├── src/
│   └── lib.rs          # pyo3 module entry point (the `semisweet` extension)
├── docs/
│   └── assets/         # brand images (logo, README banner, social card)
├── .github/
│   └── workflows/      # CI — cargo fmt/clippy/test + maturin build
├── Cargo.toml          # Rust crate manifest + pyo3 dependency
├── pyproject.toml      # maturin build backend + Python package metadata
├── rustfmt.toml        # formatter config (edition 2024)
├── .python-version     # local interpreter pin (3.13)
├── AGENTS.md           # This file — shared conventions
├── STYLEGUIDE.md       # Concrete style rules
├── CLAUDE.md           # Claude-specific rules (imports AGENTS.md)
├── CHANGELOG.md        # Keep a Changelog
└── README.md           # Project overview
```
