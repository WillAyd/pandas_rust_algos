use numpy::ndarray::{ArrayView1, ArrayViewMut1};
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use pyo3::PyResult;

#[pymodule]
fn pandas_rust_algos(_py: Python, m: &PyModule) -> PyResult<()> {
    fn take_1d(
        values: ArrayView1<u8>,
        indexer: ArrayView1<i64>,
        mut out: ArrayViewMut1<u8>,
        fill_value: u8,
    ) {
        for (i, idx) in indexer.iter().enumerate() {
            match idx {
                -1 => {
                    out[i] = fill_value;
                }
                _ => {
                    let uidx = usize::try_from(*idx).unwrap();
                    out[i] = values[uidx]
                }
            }
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "take_1d")]
    fn take_1d_py<'py>(
        values: PyReadonlyArray1<u8>,
        indexer: PyReadonlyArray1<i64>,
        mut out: PyReadwriteArray1<u8>,
        fill_value: u8,
    ) {
        take_1d(
            values.as_array(),
            indexer.as_array(),
            out.as_array_mut(),
            fill_value,
        )
    }

    Ok(())
}
