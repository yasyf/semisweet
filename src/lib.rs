use pyo3::prelude::*;

/// The `semisweet` Python extension module.
#[pymodule]
fn semisweet(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
