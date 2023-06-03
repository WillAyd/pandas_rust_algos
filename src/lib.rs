mod algos;
mod groupby;
mod types;

use crate::algos::take_2d_axis1;
use crate::groupby::{
    group_any_all, group_cumprod, group_cumsum, group_fillna_indexer, group_median_float64,
    group_prod, group_shift_indexer, group_skew, group_sum, group_var,
};
use crate::types::NumericArray2;
use ndarray::parallel::prelude::*;
use numpy::ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis, Zip};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;
use pyo3::PyResult;
use std::cell::UnsafeCell;
use std::fmt::Debug;

#[pymodule]
fn pandas_rust_algos(_py: Python, m: &PyModule) -> PyResult<()> {
    #[derive(FromPyObject)]
    enum TakeType<'py> {
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
    fn take_1d_py<'py>(values: TakeType<'py>, indexer: PyReadonlyArray1<i64>, out: TakeType<'py>) {
        match (values, out) {
            (TakeType::U8(values), TakeType::U8(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I8(values), TakeType::I8(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I8(values), TakeType::I32(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I8(values), TakeType::I64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I8(values), TakeType::F64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0.0,
            ),
            (TakeType::I16(values), TakeType::I32(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I16(values), TakeType::I64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I16(values), TakeType::F64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0.0,
            ),
            (TakeType::I32(values), TakeType::I32(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I32(values), TakeType::I64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            (TakeType::I32(values), TakeType::F64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0.0,
            ),
            (TakeType::I64(values), TakeType::I64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0,
            ),
            // TODO: the trait `From<i64>` is not implemented for `f64`
            /*
                (TakeType::I64(values), TakeType::F64(out)) => take_1d(
                    values.readonly().as_array(),
                    indexer.as_array(),
                    out.readwrite().as_array_mut(),
                    0.0,
            ),
             */
            (TakeType::F32(values), TakeType::F32(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0.0,
            ),
            (TakeType::F32(values), TakeType::F64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0.0,
            ),
            (TakeType::F64(values), TakeType::F64(out)) => take_1d(
                values.readonly().as_array(),
                indexer.as_array(),
                out.readwrite().as_array_mut(),
                0.0,
            ),
            (_, _) => panic!("Types not supported"),
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "take_2d_axis1")]
    fn take_2d_axis1_py<'py>(
        values: PyReadonlyArray2<i64>,
        indexer: PyReadonlyArray1<i64>,
        mut out: PyReadwriteArray2<i64>,
    ) {
        take_2d_axis1(values.as_array(), indexer.as_array(), out.as_array_mut())
    }

    #[derive(Copy, Clone)]
    struct UnsafeArrayView2<'a> {
        array: &'a UnsafeCell<ArrayViewMut2<'a, i64>>,
    }

    unsafe impl<'a> Send for UnsafeArrayView2<'a> {}
    unsafe impl<'a> Sync for UnsafeArrayView2<'a> {}

    impl<'a> UnsafeArrayView2<'a> {
        pub fn new(array: &'a mut ArrayViewMut2<i64>) -> Self {
            let ptr = array as *mut ArrayViewMut2<i64> as *const UnsafeCell<ArrayViewMut2<i64>>;
            Self {
                array: unsafe { &*ptr },
            }
        }

        /// SAFETY: It is UB if two threads write to the same index without
        /// synchronization.
        pub unsafe fn write(&self, i: usize, j: usize, value: i64) {
            let ptr = self.array.get();
            *(*ptr).uget_mut((i, j)) = value;
        }
    }

    fn take_2d_unsafe(
        values: ArrayView2<i64>,
        indexer: ArrayView1<i64>,
        mut out: ArrayViewMut2<i64>,
    ) {
        let ncols = indexer.raw_dim()[0];
        let uout = UnsafeArrayView2::new(&mut out);

        Zip::indexed(values.axis_iter(Axis(0)))
            .into_par_iter()
            .for_each(|(i, val_row)| {
                for j in 0..ncols {
                    unsafe {
                        let idx = *indexer.uget(j);
                        let val = *val_row.uget(idx as usize);
                        uout.write(i, j, val);
                    }
                }
            });
    }

    #[pyfn(m)]
    #[pyo3(name = "take_2d_unsafe")]
    fn take_2d_unsafe_py<'py>(
        values: PyReadonlyArray2<i64>,
        indexer: PyReadonlyArray1<i64>,
        mut out: PyReadwriteArray2<i64>,
    ) {
        take_2d_unsafe(values.as_array(), indexer.as_array(), out.as_array_mut())
    }

    #[pyfn(m)]
    #[pyo3(name = "group_median_float64")]
    fn group_median_float64_py<'py>(
        mut out: PyReadwriteArray2<f64>,
        mut counts: PyReadwriteArray1<i64>,
        values: PyReadonlyArray2<f64>,
        labels: PyReadonlyArray1<i64>,
        min_count: isize,
        mask: Option<PyReadonlyArray2<u8>>,
        result_mask: Option<PyReadwriteArray2<u8>>,
    ) {
        group_median_float64(
            out.as_array_mut(),
            counts.as_array_mut(),
            values.as_array(),
            labels.as_array(),
            min_count,
            mask,
            result_mask,
        )
    }

    #[pyfn(m)]
    #[pyo3(name = "group_cumprod")]
    // TODO: pandas has a generic implementation for int64 / float
    // this is currently just float; we likely want a custom type trait
    // to serve the NA value for ints versus float objects
    fn group_cumprod_py<'py>(
        out: NumericArray2,
        values: NumericArray2,
        labels: PyReadonlyArray1<i64>,
        ngroups: i64,
        is_datetimelike: bool,
        skipna: bool,
        mask: Option<PyReadonlyArray2<u8>>,
        result_mask: Option<PyReadwriteArray2<u8>>,
    ) {
        match (out, values) {
            // TODO: pretty hard to dispatch here; PyO3 does not allow for a generic type
            // match arms must all have the same resulting expression types, and a
            // closure does not seem to work; so we repeat the same function for
            // all allowed values...
            (NumericArray2::I64(out), NumericArray2::I64(values)) => group_cumprod(
                out.readwrite().as_array_mut(),
                values.readonly().as_array(),
                labels.as_array(),
                ngroups,
                is_datetimelike,
                skipna,
                mask,
                result_mask,
            ),
            (NumericArray2::F32(out), NumericArray2::F32(values)) => group_cumprod(
                out.readwrite().as_array_mut(),
                values.readonly().as_array(),
                labels.as_array(),
                ngroups,
                is_datetimelike,
                skipna,
                mask,
                result_mask,
            ),
            (NumericArray2::F64(out), NumericArray2::F64(values)) => group_cumprod(
                out.readwrite().as_array_mut(),
                values.readonly().as_array(),
                labels.as_array(),
                ngroups,
                is_datetimelike,
                skipna,
                mask,
                result_mask,
            ),
            _ => panic!("Unsupported argument types to cumprod!"),
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "group_cumsum")]
    // TODO: pandas has a generic implementation for int64 / float
    // this is currently just float; we likely want a custom type trait
    // to serve the NA value for ints versus float objects
    fn group_cumsum_py<'py>(
        out: NumericArray2,
        values: NumericArray2,
        labels: PyReadonlyArray1<i64>,
        ngroups: i64,
        is_datetimelike: bool,
        skipna: bool,
        mask: Option<PyReadonlyArray2<u8>>,
        result_mask: Option<PyReadwriteArray2<u8>>,
    ) {
        match (out, values) {
            // TODO: pretty hard to dispatch here; PyO3 does not allow for a generic type
            // match arms must all have the same resulting expression types, and a
            // closure does not seem to work; so we repeat the same function for
            // all allowed values...
            (NumericArray2::I64(out), NumericArray2::I64(values)) => group_cumsum(
                out.readwrite().as_array_mut(),
                values.readonly().as_array(),
                labels.as_array(),
                ngroups,
                is_datetimelike,
                skipna,
                mask,
                result_mask,
            ),
            (NumericArray2::F32(out), NumericArray2::F32(values)) => group_cumsum(
                out.readwrite().as_array_mut(),
                values.readonly().as_array(),
                labels.as_array(),
                ngroups,
                is_datetimelike,
                skipna,
                mask,
                result_mask,
            ),
            (NumericArray2::F64(out), NumericArray2::F64(values)) => group_cumsum(
                out.readwrite().as_array_mut(),
                values.readonly().as_array(),
                labels.as_array(),
                ngroups,
                is_datetimelike,
                skipna,
                mask,
                result_mask,
            ),
            _ => panic!("Unsupported argument types to cumsum!"),
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "group_shift_indexer")]
    fn group_shift_indexer_py<'py>(
        mut out: PyReadwriteArray1<i64>,
        labels: PyReadonlyArray1<i64>,
        ngroups: i64,
        periods: i64,
    ) {
        group_shift_indexer(out.as_array_mut(), labels.as_array(), ngroups, periods);
    }

    #[pyfn(m)]
    #[pyo3(name = "group_fillna_indexer")]
    fn group_fillna_indexer_py<'py>(
        mut out: PyReadwriteArray1<i64>,
        labels: PyReadonlyArray1<i64>,
        sorted_labels: PyReadonlyArray1<i64>,
        mask: PyReadonlyArray1<u8>,
        limit: i64,
        dropna: bool,
    ) {
        group_fillna_indexer(
            out.as_array_mut(),
            labels.as_array(),
            sorted_labels.as_array(),
            mask.as_array(),
            limit,
            dropna,
        );
    }

    #[pyfn(m)]
    #[pyo3(name = "group_any_all")]
    fn group_any_all_py<'py>(
        mut out: PyReadwriteArray2<i8>,
        values: PyReadonlyArray2<i8>,
        labels: PyReadonlyArray1<i64>,
        mask: PyReadonlyArray2<u8>,
        val_test: String,
        skipna: bool,
        py_result_mask: Option<PyReadwriteArray2<u8>>,
    ) {
        group_any_all(
            out.as_array_mut(),
            values.as_array(),
            labels.as_array(),
            mask.as_array(),
            val_test,
            skipna,
            py_result_mask,
        );
    }

    #[pyfn(m)]
    #[pyo3(name = "group_sum")]
    fn group_sum_py<'py>(
        out: NumericArray2,
        mut counts: PyReadwriteArray1<i64>,
        values: NumericArray2,
        labels: PyReadonlyArray1<i64>,
        mask: Option<PyReadonlyArray2<u8>>,
        result_mask: Option<PyReadwriteArray2<u8>>,
        min_count: isize,
        is_datetimelike: bool,
    ) {
        match (out, values) {
            (NumericArray2::I64(out), NumericArray2::I64(values)) => group_sum(
                out.readwrite().as_array_mut(),
                counts.as_array_mut(),
                values.readonly().as_array(),
                labels.as_array(),
                mask,
                result_mask,
                min_count,
                is_datetimelike,
            ),
            (NumericArray2::F32(out), NumericArray2::F32(values)) => group_sum(
                out.readwrite().as_array_mut(),
                counts.as_array_mut(),
                values.readonly().as_array(),
                labels.as_array(),
                mask,
                result_mask,
                min_count,
                is_datetimelike,
            ),
            (NumericArray2::F64(out), NumericArray2::F64(values)) => group_sum(
                out.readwrite().as_array_mut(),
                counts.as_array_mut(),
                values.readonly().as_array(),
                labels.as_array(),
                mask,
                result_mask,
                min_count,
                is_datetimelike,
            ),
            _ => panic!("Unsupported argument types to group_sum!"),
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "group_prod")]
    fn group_prod_py<'py>(
        out: NumericArray2,
        mut counts: PyReadwriteArray1<i64>,
        values: NumericArray2,
        labels: PyReadonlyArray1<i64>,
        mask: Option<PyReadonlyArray2<u8>>,
        result_mask: Option<PyReadwriteArray2<u8>>,
        min_count: isize,
    ) {
        match (out, values) {
            (NumericArray2::I64(out), NumericArray2::I64(values)) => group_prod(
                out.readwrite().as_array_mut(),
                counts.as_array_mut(),
                values.readonly().as_array(),
                labels.as_array(),
                mask,
                result_mask,
                min_count,
            ),
            (NumericArray2::F32(out), NumericArray2::F32(values)) => group_prod(
                out.readwrite().as_array_mut(),
                counts.as_array_mut(),
                values.readonly().as_array(),
                labels.as_array(),
                mask,
                result_mask,
                min_count,
            ),
            (NumericArray2::F64(out), NumericArray2::F64(values)) => group_prod(
                out.readwrite().as_array_mut(),
                counts.as_array_mut(),
                values.readonly().as_array(),
                labels.as_array(),
                mask,
                result_mask,
                min_count,
            ),
            _ => panic!("Unsupported argument types to group_prod!"),
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "group_var")]
    fn group_var_py<'py>(
        out: NumericArray2,
        mut counts: PyReadwriteArray1<i64>,
        values: NumericArray2,
        labels: PyReadonlyArray1<i64>,
        min_count: isize,
        ddof: i64,
        py_mask: Option<PyReadonlyArray2<u8>>,
        py_result_mask: Option<PyReadwriteArray2<u8>>,
        is_datetimelike: bool,
        name: String,
    ) {
        match (out, values) {
            // TODO: we aren't using a platform int so rust doesn't like 32bit ->
            // f32; change to c platform int and can likely get that specialization
            (NumericArray2::F64(out), NumericArray2::F64(values)) => group_var(
                out.readwrite().as_array_mut(),
                counts.as_array_mut(),
                values.readonly().as_array(),
                labels.as_array(),
                min_count,
                ddof,
                py_mask,
                py_result_mask,
                is_datetimelike,
                name,
            ),
            _ => panic!("Unsupported argument types to group_var!"),
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "group_skew")]
    fn group_skew_py<'py>(
        mut out: PyReadwriteArray2<f64>,
        mut counts: PyReadwriteArray1<i64>,
        values: PyReadonlyArray2<f64>,
        labels: PyReadonlyArray1<i64>,
        py_mask: Option<PyReadonlyArray2<u8>>,
        py_result_mask: Option<PyReadwriteArray2<u8>>,
        skipna: bool,
    ) {
        group_skew(
            out.as_array_mut(),
            counts.as_array_mut(),
            values.as_array(),
            labels.as_array(),
            py_mask,
            py_result_mask,
            skipna,
        )
    }

    Ok(())
}
