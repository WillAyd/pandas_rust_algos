use std::alloc::{alloc, dealloc, Layout};
use std::mem::size_of;

use crate::algos::{groupsort_indexer, kth_smallest_c, take_2d_axis1};
use numpy::ndarray::{s, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use numpy::{PyReadonlyArray2, PyReadwriteArray2};

unsafe fn calc_median_linear(a: *const f64, n: i64, na_count: i64) -> f64 {
    let result;
    let halfway = (n / 2) as usize;
    if (n % 2) > 0 {
        result = kth_smallest_c(a, halfway, n as usize);
    } else {
        result = (kth_smallest_c(a, halfway, n as usize)
            + kth_smallest_c(a, halfway - 1, n as usize))
            / 2.;
    }

    result
}

unsafe fn median_linear_mask(a: *const f64, mut n: i64, mask: *const u8) -> f64 {
    let mut na_count = 0;

    if n == 0 {
        return f64::NAN;
    }

    for i in 0..n {
        if *mask.add(i as usize) == 1 {
            na_count += 1;
        }
    }

    let result;
    if na_count > 0 {
        if na_count == n {
            return f64::NAN;
        }

        let count = n - na_count;
        // TODO: better method than having to specify alignment?
        let layout = Layout::from_size_align(size_of::<f64>() * count as usize, 8);
        let ptr = alloc(layout.clone().unwrap());

        let mut j = 0;
        for i in 0..n {
            if *mask.add(i as usize) == 0 {
                *(ptr as *mut f64).add(j) = *a.add(i as usize);
                j += 1;
            }
        }

        n -= na_count;
        result = calc_median_linear(ptr as *const f64, n, na_count);
        dealloc(ptr, layout.unwrap());
    } else {
        result = calc_median_linear(a, n, 0);
    }

    result
}

unsafe fn median_linear(a: *const f64, mut n: i64) -> f64 {
    let mut na_count = 0;

    if n == 0 {
        return f64::NAN;
    }

    for i in 0..n {
        if (*a.add(i as usize)).is_nan() {
            na_count += 1;
        }
    }

    let result;
    if na_count > 0 {
        if na_count == n {
            return f64::NAN;
        }

        let count = n - na_count;
        // TODO: better method than having to specify alignment?
        let layout = Layout::from_size_align(size_of::<f64>() * count as usize, 8);
        let ptr = alloc(layout.clone().unwrap());

        let mut j = 0;
        for i in 0..n {
            if (*a.add(i as usize)).is_finite() {
                *(ptr as *mut f64).add(j) = *a.add(i as usize);
                j += 1;
            }
        }

        n -= na_count;
        result = calc_median_linear(ptr as *const f64, n, na_count);
        dealloc(ptr, layout.unwrap());
    } else {
        result = calc_median_linear(a, n, 0);
    }

    result
}

pub fn group_median_float64(
    mut out: ArrayViewMut2<f64>,
    mut counts: ArrayViewMut1<i64>,
    values: ArrayView2<f64>,
    labels: ArrayView1<i64>,
    min_count: isize,
    py_mask: Option<PyReadonlyArray2<u8>>,
    py_result_mask: Option<PyReadwriteArray2<u8>>,
) {
    if min_count != -1 {
        panic!("'min_count' only used in sum and prod");
    }

    let ngroups = counts.len();
    let dim = values.raw_dim();
    let n = dim[0];
    let k = dim[1];

    let (indexer, _counts) = groupsort_indexer(labels, ngroups);
    counts.assign(&_counts.slice(s![1..]));

    let mut data = Array2::<f64>::default((k, n));
    let mut ptr = data.as_ptr();

    take_2d_axis1(values.t(), indexer.view(), data.view_mut());

    match (py_mask, py_result_mask) {
        (Some(py_mask), Some(mut py_result_mask)) => {
            let mask = py_mask.as_array();
            let mut result_mask = py_result_mask.as_array_mut();
            let mut data_mask = Array2::<u8>::default((k, n));
            let mut ptr_mask = data_mask.as_ptr();

            take_2d_axis1(mask.t(), indexer.view(), data_mask.view_mut());

            for i in 0..k {
                unsafe {
                    let increment = _counts[0] as usize;
                    ptr = ptr.add(increment);
                    ptr_mask = ptr_mask.add(increment);

                    for j in 0..ngroups {
                        let size = _counts[j + 1];
                        let result = median_linear_mask(ptr, size, ptr_mask);
                        *out.uget_mut((j, i)) = result;

                        if result.is_nan() {
                            *result_mask.uget_mut((j, i)) = 1
                        }
                        ptr = ptr.add(size as usize);
                        ptr_mask = ptr_mask.add(size as usize);
                    }
                }
            }
        }
        (_, _) => {
            for i in 0..k {
                unsafe {
                    let increment = *_counts.uget(0) as usize;
                    ptr = ptr.add(increment);
                    for j in 0..ngroups {
                        let size = *_counts.uget(j + 1);
                        let result = median_linear(ptr, size);
                        *out.uget_mut((j, i)) = result;
                        ptr = ptr.add(size as usize);
                    }
                }
            }
        }
    }
}

///
/// Cumulative product of columns of `values`, in row groups `labels`.
///
/// Parameters
/// ----------
/// out : np.ndarray[np.float64, ndim=2]
///     Array to store cumprod in.
/// values : np.ndarray[np.float64, ndim=2]
///     Values to take cumprod of.
/// labels : np.ndarray[np.intp]
///     Labels to group by.
/// ngroups : int
///     Number of groups, larger than all entries of `labels`.
/// is_datetimelike : bool
///     Always false, `values` is never datetime-like.
/// skipna : bool
///     If true, ignore nans in `values`.
/// mask : np.ndarray[uint8], optional
///     Mask of values
/// result_mask : np.ndarray[int8], optional
///     Mask of out array
///
/// Notes
/// -----
/// This method modifies the `out` parameter, rather than returning an object.
pub fn group_cumprod(
    mut out: ArrayViewMut2<f64>,
    values: ArrayView2<f64>,
    labels: ArrayView1<i64>,
    ngroups: i64,
    is_datetimelike: bool,
    skipna: bool,
    py_mask: Option<PyReadonlyArray2<u8>>,
    py_result_mask: Option<PyReadwriteArray2<u8>>,
) {
    let dim = values.raw_dim();
    let n = dim[0];
    let k = dim[1];

    let mut accum = Array2::<f64>::ones((ngroups as usize, k));
    let na_val = f64::NAN;
    let mut accum_mask = Array2::<u8>::zeros((ngroups as usize, k));

    match (py_mask, py_result_mask) {
        (Some(py_mask), Some(mut py_result_mask)) => {
            let mask = py_mask.as_array();
            let mut result_mask = py_result_mask.as_array_mut();
            for i in 0..n {
                unsafe {
                    let lab = *labels.uget(i);
                    if lab < 0 {
                        continue;
                    }

                    for j in 0..k {
                        let val = *values.uget((i, j));
                        let isna_entry = *mask.uget((i, j)) == 0;
                        if !isna_entry {
                            let isna_prev = *accum_mask.uget((lab as usize, j)) == 0;
                            if isna_prev {
                                *out.uget_mut((i, j)) = na_val;
                                *result_mask.uget_mut((i, j)) = 1;
                            } else {
                                *accum.uget_mut((lab as usize, j)) *= val;
                                *out.uget_mut((i, j)) = val;
                            }
                        } else {
                            *result_mask.uget_mut((i, j)) = 1;
                            *out.uget_mut((i, j)) = 0.;

                            if !skipna {
                                *accum.uget_mut((lab as usize, j)) *= na_val;
                                *accum_mask.uget_mut((lab as usize, j)) = 1;
                            }
                        }
                    }
                }
            }
        }
        (_, _) => {
            for i in 0..n {
                unsafe {
                    let lab = *labels.uget(i);
                    if lab < 0 {
                        continue;
                    }

                    for j in 0..k {
                        let val = *values.uget((i, j));
                        let isna_entry = val.is_nan();
                        if !isna_entry {
                            let isna_prev = *accum_mask.uget((lab as usize, j)) == 0;
                            if isna_prev {
                                *out.uget_mut((i, j)) = na_val;
                            } else {
                                *accum.uget_mut((lab as usize, j)) *= val;
                                *out.uget_mut((i, j)) = val;
                            }
                        } else {
                            *out.uget_mut((i, j)) = f64::NAN;

                            if !skipna {
                                *accum.uget_mut((lab as usize, j)) *= na_val;
                                *accum_mask.uget_mut((lab as usize, j)) = 1;
                            }
                        }
                    }
                }
            }
        }
    }
}
