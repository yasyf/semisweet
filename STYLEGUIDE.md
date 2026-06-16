# semisweet Style Guide

The concrete style rules for this repository.

## Core Principles

1. **Fail fast, fail loud.** No defensive coding: no fallbacks, shims, or
   backwards-compat layers, and no guards against impossible states. No sentinel
   values, no silent defaults. If unused, delete it. Crash on the unexpected.
2. **Make invalid states unrepresentable.** Branded/newtype primitives, immutable
   data structures, required fields over optionals.
3. **Minimal changes.** Stay within scope. Make the test pass, then stop. Improve
   only the code you touch.
4. **Match surrounding code.** Follow this guide first, then the file you're in,
   then the module. If surrounding code violates this guide, fix it.

## Error Handling

Keep error-handling blocks minimal: only the operation that can fail belongs
inside. No catch-all handlers that swallow everything; use dedicated error types.
Read required configuration so a missing key fails at startup. No sentinel return
values; raise, or return a typed result.

## Code Organization

Order each module: imports, constants, type aliases, helpers, classes, then
functions. Constants sit immediately after imports, before any class or function.
Use the language's export-control mechanism instead of underscore/naming
conventions to hide internals.

## Comments & Docstrings

Code documents itself through names, types, and organization. No comments except
TODOs, non-obvious workarounds, or disabled code. Document the public API only;
a doc comment that restates the signature is clutter to delete.

## Testing

Write strict assertions against specific expected values; a test that can't fail
uncovers nothing. Mock the boundaries your code talks to, such as the network,
filesystem, and clock, and leave the function under test real. A database (or any
stateful service) is not a mock boundary: when a test needs one, start a real
ephemeral instance with testcontainers rather than mocking the driver or using an
in-memory fake. Parameterize repeated test bodies, giving each case a descriptive
id and its own expected values.

## Rust

Rust-specific rules. The house principles above still bind; this section makes
them concrete for `semisweet` (a Rust library exposing Python bindings via pyo3).

### Naming

`snake_case` for functions, methods, variables, and modules. `UpperCamelCase` for
types, traits, and enum variants. `SCREAMING_SNAKE_CASE` for `const`/`static`.
Don't abbreviate past recognition, and don't prefix to fake namespaces — modules
already namespace.

```rust
// Good
const MAX_ENTRIES: usize = 10_000;

pub struct CacheEntry { embedding: Embedding }

fn nearest_neighbor(query: &Embedding) -> Option<&CacheEntry> { /* ... */ }

// Bad
const maxEntries: usize = 10_000;          // const is SCREAMING_SNAKE
pub struct cache_entry { /* ... */ }        // type is UpperCamelCase
fn NearestNeighbor(q: &Embedding) {}        // fn is snake_case
fn cache_get_entry_fn() {}                  // redundant `cache_`/`_fn` noise
```

### Module organization

Order each module top-to-bottom: imports, then constants, type aliases, helpers,
types (`struct`/`enum`/`trait`), then free functions. Group imports `std` →
external crates → `crate`. Default to private; widen deliberately with `pub(crate)`
for crate-internal API and `pub` only for the public surface. Never use a
leading-underscore name to "hide" an item — visibility is the mechanism.

```rust
// Good — visibility, not naming, controls the boundary
use std::collections::HashMap;

use pyo3::prelude::*;

const DEFAULT_CAPACITY: usize = 1024;

type Store = HashMap<Key, Embedding>;

pub(crate) fn normalize(v: &mut [f32]) { /* ... */ }   // crate-internal helper

pub struct Cache { store: Store }                       // public type

// Bad
fn _normalize(v: &mut [f32]) {}   // underscore-as-private; use no `pub` instead
pub fn normalize(v: &mut [f32]) {} // leaks an internal helper to the public API
```

### Error handling

One `thiserror` enum is the crate's error type. Pair it with a `Result` alias.
Library code returns `Result`; it does not panic. No `.unwrap()`, `.expect()`,
`panic!`, `todo!`, or array-index-into-untrusted-len outside `#[cfg(test)]`.
Propagate with `?`. Match every variant — no catch-all `_ =>` arm that swallows
cases the compiler would otherwise force you to handle. No sentinel returns: an
absence that is an error is a typed variant, not `-1`, `""`, or a bare `Option`
standing in for failure.

```rust
// Good
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no entry for key `{0}`")]
    Missing(Key),
    #[error("embedding dimension {got} != index dimension {want}")]
    DimMismatch { got: usize, want: usize },
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn get(&self, key: &Key) -> Result<&Embedding> {
    self.store.get(key).ok_or_else(|| Error::Missing(key.clone()))
}

// Bad
pub fn get(&self, key: &Key) -> Embedding {
    self.store.get(key).cloned().unwrap()   // panics in library code
}
pub fn dim_of(&self, key: &Key) -> i64 {
    self.store.get(key).map(|e| e.len() as i64).unwrap_or(-1)  // sentinel return
}
```

The crate enforces the no-panic rule mechanically: `src/lib.rs` carries
`#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]`, so
`.unwrap()`/`.expect()` in non-test code fails `clippy -D warnings`.

### Make invalid states unrepresentable

Wrap meaningful primitives in newtypes at API edges so a caller can't pass the
wrong `String`/`usize`. Prefer required fields and enums over `Option` plus a
runtime check. Construct-time validation beats per-call validation.

```rust
// Good
pub struct Key(String);
pub struct Dim(std::num::NonZeroUsize);   // a zero-dim index is unrepresentable

// Bad
fn query(key: String, dim: usize) { /* dim == 0 must be checked everywhere */ }
```

### pyo3 FFI boundary

`#[pyfunction]` and `#[pymethods]` are thin wrappers: convert Python types in, call
into plain Rust, convert results out. No business logic in the binding layer — it
lives in normal `impl` blocks and free functions that know nothing about Python.
Map the crate `Error` to Python exceptions exactly once with
`impl From<Error> for PyErr`; methods then return `Result<T, Error>` and rely on
`?` and the `From` conversion. Never hand-build a `PyErr` inside a method body.

```rust
// Good — one conversion site, thin wrapper
impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        match err {
            Error::Missing(key) => PyKeyError::new_err(key.into_inner()),
            Error::DimMismatch { .. } => PyValueError::new_err(err.to_string()),
        }
    }
}

#[pymethods]
impl Cache {
    fn get(&self, key: String) -> Result<String, Error> {
        self.lookup(&Key::new(key))   // real work in a non-pyo3 method
    }
}

// Bad — logic + ad hoc error construction in the binding
#[pymethods]
impl Cache {
    fn get(&self, key: String) -> PyResult<String> {
        match self.store.get(&key) {
            Some(v) => Ok(v.clone()),
            None => Err(PyKeyError::new_err(format!("missing {key}"))), // bypasses Error
        }
    }
}
```
