use numpy::ndarray::{ArrayView1, ArrayViewMut1};
use numpy::{PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use pyo3::PyResult;
use std::fmt::Debug;

#[pymodule]
fn pandas_rust_algos(_py: Python, m: &PyModule) -> PyResult<()> {
    #[derive(FromPyObject)]
    enum TakeArrayInType<'py> {
        U8(&'py PyArray1<u8>),
        I8(&'py PyArray1<i8>),
        I16(&'py PyArray1<i16>),
        I32(&'py PyArray1<i32>),
        I64(&'py PyArray1<i64>),
        F32(&'py PyArray1<f32>),
        F64(&'py PyArray1<f64>),
    }

    fn take_1d<InT, OutT>(
        values: ArrayView1<InT>,
        indexer: ArrayView1<i64>,
        mut out: ArrayViewMut1<OutT>,
        fill_value: OutT,
    ) where
        InT: std::marker::Copy,
        OutT: TryFrom<InT> + std::marker::Copy,
        <OutT as TryFrom<InT>>::Error: Debug,
    {
        for (i, idx) in indexer.iter().enumerate() {
            if *idx == -1 {
                out[i] = fill_value;
            } else {
                let uidx = usize::try_from(*idx).unwrap();
                out[i] = OutT::try_from(values[uidx]).unwrap();
            }
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "take_1d")]
    fn take_1d_py<'py>(
        values: TakeArrayInType<'py>,
        indexer: PyReadonlyArray1<i64>,
        mut out: PyReadwriteArray1<u8>,
        fill_value: u8,
    ) {
        match values {
            TakeArrayInType::U8(values) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.as_array_mut(),
                fill_value,
            ),
            TakeArrayInType::I8(values) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.as_array_mut(),
                fill_value,
            ),
            TakeArrayInType::I16(values) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.as_array_mut(),
                fill_value,
            ),
            TakeArrayInType::I32(values) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.as_array_mut(),
                fill_value,
            ),
            TakeArrayInType::I64(values) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.as_array_mut(),
                fill_value,
            ),
            _ => {}
        }
    }

    Ok(())
}
