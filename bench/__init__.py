"""Benchmark harness backing semisweet's scoring defaults.

The modules here are dev tooling, run as ``python -m bench.<tool>``; maturin does
not package them into the published wheel (the package lives under ``python/``).
Package import stays light: the heavy dependencies (``fastembed``, ``numpy``) are
pulled in by the submodules that need them, not at ``import bench``.
"""
