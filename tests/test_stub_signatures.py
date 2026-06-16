"""Drift guard between ``semisweet.pyi`` and the compiled module.

pyo3 abi3 classes expose little introspection, so this checks names, not deep
signatures: every class/function declared in the stub must exist at runtime, and
every public runtime name must be declared in the stub. A new export on either
side fails the build until the stub catches up.
"""

import ast
from pathlib import Path

import semisweet


def _stub_path() -> Path:
    # Source of truth at the repo root; fall back to the copy maturin ships next to
    # the compiled module in site-packages.
    repo_root = Path(__file__).resolve().parent.parent / "semisweet.pyi"
    if repo_root.is_file():
        return repo_root
    installed = Path(semisweet.__file__).resolve().parent / "semisweet.pyi"
    if installed.is_file():
        return installed
    raise FileNotFoundError(
        "semisweet.pyi not found at the repo root or alongside the installed module"
    )


def _declared_names() -> set[str]:
    tree = ast.parse(_stub_path().read_text())
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _public_runtime_names() -> set[str]:
    exported = getattr(semisweet, "__all__", None)
    if exported is not None:
        return set(exported)
    return {name for name in dir(semisweet) if not name.startswith("_")}


def test_every_stub_name_exists_at_runtime():
    missing = {name for name in _declared_names() if not hasattr(semisweet, name)}
    assert missing == set(), f"declared in stub but absent at runtime: {sorted(missing)}"


def test_every_public_runtime_name_is_declared_in_stub():
    declared = _declared_names()
    undocumented = {name for name in _public_runtime_names() if name not in declared}
    assert undocumented == set(), (
        f"public at runtime but missing from stub: {sorted(undocumented)}"
    )
