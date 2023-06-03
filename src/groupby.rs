use crate::algos::{groupsort_indexer, kth_smallest_c, take_2d_axis1};
use num::traits::{NumCast, One, Zero};
use numpy::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use std::alloc::{alloc, dealloc, Layout};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::mem::size_of;

pub trait PandasNA {
    fn na_val(is_datetimelike: bool) -> Self;
    fn isna(&self, is_datetimelike: bool) -> bool;
}

impl PandasNA for f64 {
    fn na_val(_is_datetimelike: bool) -> Self {
        f64::NAN
    }

    fn isna(&self, _is_datetimelike: bool) -> bool {
        self.is_nan()
    }
}

impl PandasNA for f32 {
    fn na_val(_is_datetimelike: bool) -> Self {
        f32::NAN
    }

    fn isna(&self, _is_datetimelike: bool) -> bool {
        self.is_nan()
    }
}

impl PandasNA for i64 {
    fn na_val(is_datetimelike: bool) -> Self {
        if is_datetimelike {
            i64::MIN
        } else {
            0
        }
    }

    fn isna(&self, is_datetimelike: bool) -> bool {
        if is_datetimelike {
            *self == i64::MIN
        } else {
            *self == 0
        }
    }
}

unsafe fn calc_median_linear(a: *const f64, n: i64) -> f64 {
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
        result = calc_median_linear(ptr as *const f64, n);
        dealloc(ptr, layout.unwrap());
    } else {
        result = calc_median_linear(a, n);
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
        result = calc_median_linear(ptr as *const f64, n);
        dealloc(ptr, layout.unwrap());
    } else {
        result = calc_median_linear(a, n);
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
pub fn group_cumprod<T>(
    mut out: ArrayViewMut2<T>,
    values: ArrayView2<T>,
    labels: ArrayView1<i64>,
    ngroups: i64,
    is_datetimelike: bool,
    skipna: bool,
    py_mask: Option<PyReadonlyArray2<u8>>,
    py_result_mask: Option<PyReadwriteArray2<u8>>,
) where
    T: Zero + One + Clone + Copy + PandasNA + std::ops::MulAssign,
{
    let dim = values.raw_dim();
    let n = dim[0];
    let k = dim[1];

    let mut accum = Array2::<T>::ones((ngroups as usize, k));
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
                            let isna_prev = *accum_mask.uget((lab as usize, j)) != 0;
                            if isna_prev {
                                *out.uget_mut((i, j)) = <T as PandasNA>::na_val(is_datetimelike);
                                *result_mask.uget_mut((i, j)) = 1;
                            } else {
                                *accum.uget_mut((lab as usize, j)) *= val;
                                *out.uget_mut((i, j)) = *accum.uget((lab as usize, j));
                            }
                        } else {
                            *result_mask.uget_mut((i, j)) = 1;
                            *out.uget_mut((i, j)) = <T as Zero>::zero();

                            if !skipna {
                                *accum.uget_mut((lab as usize, j)) =
                                    <T as PandasNA>::na_val(is_datetimelike);
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
                        let isna_entry = val.isna(is_datetimelike);
                        if !isna_entry {
                            let isna_prev = *accum_mask.uget((lab as usize, j)) != 0;
                            if isna_prev {
                                *out.uget_mut((i, j)) = <T as PandasNA>::na_val(is_datetimelike);
                            } else {
                                *accum.uget_mut((lab as usize, j)) *= val;
                                *out.uget_mut((i, j)) = *accum.uget((lab as usize, j));
                            }
                        } else {
                            *out.uget_mut((i, j)) = <T as PandasNA>::na_val(is_datetimelike);

                            if !skipna {
                                *accum.uget_mut((lab as usize, j)) =
                                    <T as PandasNA>::na_val(is_datetimelike);
                                *accum_mask.uget_mut((lab as usize, j)) = 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

pub trait CumSumAccumulator {
    fn acummulate<T>(
        val: T,
        accum: ArrayView2<T>,
        compensation: ArrayViewMut2<T>,
        lab: usize,
        j: usize,
    ) -> T
    where
        T: std::ops::Sub<Output = T> + std::ops::Add<Output = T> + Copy;
}

impl CumSumAccumulator for f64 {
    // For floats, use Kahan summation to reduce floating-point
    // error (https://en.wikipedia.org/wiki/Kahan_summation_algorithm)

    fn acummulate<T>(
        val: T,
        accum: ArrayView2<T>,
        mut compensation: ArrayViewMut2<T>,
        lab: usize,
        j: usize,
    ) -> T
    where
        T: std::ops::Sub<Output = T> + std::ops::Add<Output = T> + Copy,
    {
        let t;
        unsafe {
            let y = val - *compensation.uget((lab, j));
            t = *accum.uget((lab, j)) + y;
            *compensation.uget_mut((lab, j)) = t - *accum.uget((lab, j)) - y;
        }
        t
    }
}

impl CumSumAccumulator for f32 {
    // For floats, use Kahan summation to reduce floating-point
    // error (https://en.wikipedia.org/wiki/Kahan_summation_algorithm)

    fn acummulate<T>(
        val: T,
        accum: ArrayView2<T>,
        mut compensation: ArrayViewMut2<T>,
        lab: usize,
        j: usize,
    ) -> T
    where
        T: std::ops::Sub<Output = T> + std::ops::Add<Output = T> + Copy,
    {
        let t;
        unsafe {
            let y = val - *compensation.uget((lab, j));
            t = *accum.uget((lab, j)) + y;
            *compensation.uget_mut((lab, j)) = t - *accum.uget((lab, j)) - y;
        }
        t
    }
}

impl CumSumAccumulator for i64 {
    fn acummulate<T>(
        val: T,
        accum: ArrayView2<T>,
        mut _compensation: ArrayViewMut2<T>,
        lab: usize,
        j: usize,
    ) -> T
    where
        T: std::ops::Sub<Output = T> + std::ops::Add<Output = T> + Copy,
    {
        let t;
        unsafe {
            t = val + *accum.uget((lab, j));
        }
        t
    }
}

/// Cumulative sum of columns of `values`, in row groups `labels`.
///
/// Parameters
/// ----------
/// out : np.ndarray[ndim=2]
///     Array to store cumsum in.
/// values : np.ndarray[ndim=2]
///     Values to take cumsum of.
/// labels : np.ndarray[np.intp]
///     Labels to group by.
/// ngroups : int
///     Number of groups, larger than all entries of `labels`.
/// is_datetimelike : bool
///     True if `values` contains datetime-like entries.
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
pub fn group_cumsum<T>(
    mut out: ArrayViewMut2<T>,
    values: ArrayView2<T>,
    labels: ArrayView1<i64>,
    ngroups: i64,
    is_datetimelike: bool,
    skipna: bool,
    py_mask: Option<PyReadonlyArray2<u8>>,
    py_result_mask: Option<PyReadwriteArray2<u8>>,
) where
    T: Zero + One + Clone + Copy + PandasNA + std::ops::Sub<Output = T> + CumSumAccumulator,
{
    let dim = values.raw_dim();
    let n = dim[0];
    let k = dim[1];

    let mut accum = Array2::<T>::zeros((ngroups as usize, k));
    let mut compensation = Array2::<T>::zeros((ngroups as usize, k));

    match (py_mask, py_result_mask) {
        (Some(py_mask), Some(mut py_result_mask)) => {
            let mask = py_mask.as_array();
            let mut result_mask = py_result_mask.as_array_mut();
            let mut accum_mask = Array2::<u8>::zeros((ngroups as usize, k));

            for i in 0..n {
                unsafe {
                    let lab = *labels.uget(i);
                    if lab < 0 {
                        continue;
                    }

                    for j in 0..k {
                        let val = *values.uget((i, j));
                        let isna_entry = *mask.uget((i, j)) == 0;

                        if !skipna {
                            let isna_prev = *accum_mask.uget((lab as usize, j)) != 0;
                            if isna_prev {
                                *result_mask.uget_mut((i, j)) = 1;
                                // Be determinisitc, out was initialized as empty
                                *out.uget_mut((i, j)) = <T as Zero>::zero();
                                continue;
                            }
                        }

                        if isna_entry {
                            *result_mask.uget_mut((i, j)) = 1;
                            // Be determinisitc, out was initialized as empty
                            *out.uget_mut((i, j)) = <T as Zero>::zero();

                            if !skipna {
                                *accum_mask.uget_mut((lab as usize, j)) = 1;
                            }
                        } else {
                            let t = T::acummulate(
                                val,
                                accum.view(),
                                compensation.view_mut(),
                                lab as usize,
                                j,
                            );
                            *accum.uget_mut((lab as usize, j)) = t;
                            *out.uget_mut((i, j)) = t;
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
                        let isna_entry = val.isna(is_datetimelike);

                        if !skipna {
                            let isna_prev = (*accum.uget((lab as usize, j))).isna(is_datetimelike);
                            if isna_prev {
                                *out.uget_mut((i, j)) = <T as PandasNA>::na_val(is_datetimelike);
                                continue;
                            }
                        }

                        if isna_entry {
                            *out.uget_mut((i, j)) = <T as PandasNA>::na_val(is_datetimelike);

                            if !skipna {
                                *accum.uget_mut((lab as usize, j)) =
                                    <T as PandasNA>::na_val(is_datetimelike);
                            }
                        } else {
                            let t = T::acummulate(
                                val,
                                accum.view(),
                                compensation.view_mut(),
                                lab as usize,
                                j,
                            );
                            *accum.uget_mut((lab as usize, j)) = t;
                            *out.uget_mut((i, j)) = t;
                        }
                    }
                }
            }
        }
    }
}

pub fn group_shift_indexer(
    mut out: ArrayViewMut1<i64>,
    labels: ArrayView1<i64>,
    ngroups: i64,
    periods: i64,
) {
    let dim = labels.shape();
    let n = dim[0];
    let mut periods_m = periods;
    let offset: isize;
    let sign: isize;

    if periods_m < 0 {
        periods_m = -periods_m;
        offset = (n - 1) as isize;
        sign = -1;
    } else {
        offset = 0;
        sign = 1;
    }

    if periods_m == 0 {
        for i in 0..n {
            unsafe {
                *out.uget_mut(i) = i as i64;
            }
        }
    } else {
        let mut label_seen = Array1::<i64>::zeros(ngroups as usize);
        let mut label_indexer = Array2::<i64>::zeros((ngroups as usize, periods_m as usize));
        for i in 0..n {
            // reverse iterator if shifting backwards
            let ii = offset + sign * (i as isize);
            unsafe {
                let lab = *labels.uget(ii as usize);

                // skip null keys
                if lab == -1 {
                    *out.uget_mut(ii as usize) = -1;
                    continue;
                }

                *label_seen.uget_mut(lab as usize) += 1;

                let idxer_slot = *label_seen.uget(lab as usize) % periods_m;
                let idxer = *label_indexer.uget((lab as usize, idxer_slot as usize));

                if *label_seen.uget(lab as usize) > periods_m {
                    *out.uget_mut(ii as usize) = idxer;
                } else {
                    *out.uget_mut(ii as usize) = -1;
                }

                *label_indexer.uget_mut((lab as usize, idxer_slot as usize)) = ii as i64;
            }
        }
    }
}

/// Indexes how to fill values forwards or backwards within a group.
///
/// Parameters
/// ----------
/// out : np.ndarray[np.intp]
///     Values into which this method will write its results.
/// labels : np.ndarray[np.intp]
///     Array containing unique label for each group, with its ordering
///     matching up to the corresponding record in `values`.
/// sorted_labels : np.ndarray[np.intp]
///     obtained by `np.argsort(labels, kind="mergesort")`; reversed if
///     direction == "bfill"
/// values : np.ndarray[np.uint8]
///     Containing the truth value of each element.
/// mask : np.ndarray[np.uint8]
///     Indicating whether a value is na or not.
/// direction : {'ffill', 'bfill'}
///     Direction for fill to be applied (forwards or backwards, respectively)
/// limit : Consecutive values to fill before stopping, or -1 for no limit
/// dropna : Flag to indicate if NaN groups should return all NaN values
///
/// Notes
/// -----
/// This method modifies the `out` parameter rather than returning an object
pub fn group_fillna_indexer(
    mut out: ArrayViewMut1<i64>,
    labels: ArrayView1<i64>,
    sorted_labels: ArrayView1<i64>,
    mask: ArrayView1<u8>,
    limit: i64,
    dropna: bool,
) {
    let n = out.len();

    // make sure all arrays are the same size
    if !((n == labels.len()) & (n == mask.len())) {
        panic!("Not all arrays are the same size!");
    }

    let mut filled_vals = 0;
    let mut curr_fill_idx = -1;

    for i in 0..n {
        unsafe {
            let idx = *sorted_labels.uget(1);
            if dropna & (*labels.uget(idx as usize) == -1) {
                curr_fill_idx = -1;
            } else if *mask.uget(idx as usize) == 1 {
                // is missing
                // Stop filling once we've hit the limit
                if (filled_vals >= limit) & (limit != -1) {
                    curr_fill_idx = -1;
                }
                filled_vals += 1;
            } else {
                // reset items when not missing
                filled_vals = 0;
                curr_fill_idx = idx;
            }

            *out.uget_mut(idx as usize) = curr_fill_idx;

            // If we move to the next group, reset
            // the fill_idx and counter
            if (i == n)
                | (*labels.uget(idx as usize) != *labels.uget(*sorted_labels.uget(i + 1) as usize))
            {
                curr_fill_idx = -1;
                filled_vals = 0;
            }
        }
    }
}

/// Aggregated boolean values to show truthfulness of group elements. If the
/// input is a nullable type (result_mask is not None), the result will be computed
/// using Kleene logic.
/// Parameters
/// ----------
/// out : np.ndarray[np.int8]
///     Values into which this method will write its results.
/// labels : np.ndarray[np.intp]
///     Array containing unique label for each group, with its
///     ordering matching up to the corresponding record in `values`
/// values : np.ndarray[np.int8]
///     Containing the truth value of each element.
/// mask : np.ndarray[np.uint8]
///     Indicating whether a value is na or not.
/// val_test : {'any', 'all'}
///     String object dictating whether to use any or all truth testing
/// skipna : bool
///     Flag to ignore nan values during truth testing
/// result_mask : ndarray[bool, ndim=2], optional
///     If not None, these specify locations in the output that are NA.
///     Modified in-place.
///
/// Notes
/// -----
/// This method modifies the `out` parameter rather than returning an object.
/// The returned values will either be 0, 1 (False or True, respectively), or
/// -1 to signify a masked position in the case of a nullable input.
pub fn group_any_all(
    mut out: ArrayViewMut2<i8>,
    values: ArrayView2<i8>,
    labels: ArrayView1<i64>,
    mask: ArrayView2<u8>,
    val_test: String,
    skipna: bool,
    py_result_mask: Option<PyReadwriteArray2<u8>>,
) {
    let n = labels.len();
    let out_dim = out.shape();
    let k = out_dim[1];

    let flag_val: bool;
    if val_test == "all" {
        flag_val = false;
        out.fill(1);
    } else if val_test == "any" {
        flag_val = true;
        out.fill(0);
    } else {
        panic!("'val_test' must be either 'any' or 'all'!");
    }

    match py_result_mask {
        Some(mut py_result_mask) => {
            let mut result_mask = py_result_mask.as_array_mut();
            for i in 0..n {
                unsafe {
                    let lab = *labels.uget(i);
                    if lab < 0 {
                        continue;
                    }

                    for j in 0..k {
                        if skipna & (*mask.uget((i, j)) == 1) {
                            continue;
                        }

                        if *mask.uget((i, j)) == 1 {
                            // Set the position as masked if `out[lab] != flag_val`, which
                            // would indicate True/False has not yet been seen for any/all,
                            // so by Kleene logic the result is currently unknown
                            if *out.uget((lab as usize, j)) != flag_val as i8 {
                                *result_mask.uget_mut((lab as usize, j)) = 1;
                            }
                            continue;
                        }

                        let val = *values.uget((i, j));

                        // If True and 'any' or False and 'all', the result is
                        // already determined
                        if val == flag_val as i8 {
                            *out.uget_mut((lab as usize, j)) = flag_val as i8;
                            *result_mask.uget_mut((lab as usize, j)) = 0;
                        }
                    }
                }
            }
        }
        _ => {
            for i in 0..n {
                unsafe {
                    let lab = *labels.uget(i);
                    if lab < 0 {
                        continue;
                    }

                    for j in 0..k {
                        if skipna & (*mask.uget((i, j)) == 1) {
                            continue;
                        }

                        let val = *values.uget((i, j));

                        // If True and 'any' or False and 'all', the result is
                        // already determined
                        if val == flag_val as i8 {
                            *out.uget_mut((lab as usize, j)) = flag_val as i8;
                        }
                    }
                }
            }
        }
    }
}

/// Check if the number of observations for a group is below min_count,
/// and if so set the result for that group to the appropriate NA-like value.
fn check_below_mincount<T>(
    mut out: ArrayViewMut2<T>,
    _uses_mask: bool,
    py_result_mask: Option<PyReadwriteArray2<u8>>,
    ncounts: isize,
    k: isize,
    nobs: ArrayView2<i64>,
    min_count: i64,
    resx: ArrayView2<T>,
) where
    T: PandasNA + Zero + Copy,
{
    match py_result_mask {
        Some(mut py_result_mask) => {
            let mut result_mask = py_result_mask.as_array_mut();

            for i in 0..ncounts {
                for j in 0..k {
                    unsafe {
                        if *nobs.uget((i as usize, j as usize)) < min_count {
                            //  if we are integer dtype, not is_datetimelike, and
                            //  not uses_mask, then getting here implies that
                            //  counts[i] < min_count, which means we will
                            //  be cast to float64 and masked at the end
                            //  of WrappedCythonOp._call_cython_op. So we can safely
                            //  set a placeholder value in out[i, j].
                            *result_mask.uget_mut((i as usize, j as usize)) = 1;
                            // set out[i, j] to 0 to be deterministic, as
                            // it was initialized with np.empty. Also ensures
                            //  we can downcast out if appropriate.
                            *out.uget_mut((i as usize, j as usize)) = <T as Zero>::zero();
                        } else {
                            *out.uget_mut((i as usize, j as usize)) =
                                *resx.uget((i as usize, j as usize));
                        }
                    }
                }
            }
        }
        None => {
            for i in 0..ncounts {
                for j in 0..k {
                    unsafe {
                        if *nobs.uget((i as usize, j as usize)) < min_count {
                            // TODO: is_datetimelike never makes it here, but why not?
                            *out.uget_mut((i as usize, j as usize)) = T::na_val(false);
                        } else {
                            *out.uget_mut((i as usize, j as usize)) =
                                *resx.uget((i as usize, j as usize));
                        }
                    }
                }
            }
        }
    }
}

/// Only aggregates on axis=0 using Kahan summation
pub fn group_sum<T>(
    mut out: ArrayViewMut2<T>,
    mut counts: ArrayViewMut1<i64>,
    values: ArrayView2<T>,
    labels: ArrayView1<i64>,
    py_mask: Option<PyReadonlyArray2<u8>>,
    mut py_result_mask: Option<PyReadwriteArray2<u8>>,
    min_count: isize,
    is_datetimelike: bool,
) where
    T: PandasNA + Zero + One + Clone + Copy + std::ops::Sub<Output = T> + std::ops::Add<Output = T>,
{
    if values.len() != labels.len() {
        panic!("len(index) != len(labels)");
    }

    let ncounts = counts.len();
    let out_dim = out.shape();
    let mut nobs = Array2::<i64>::zeros((out_dim[0], out_dim[1]));
    // the below is equivalent to `np.zeros_like(out)` but faster
    let mut sumx = Array2::<T>::zeros((out_dim[0], out_dim[1]));
    let mut compensation = Array2::<T>::zeros((out_dim[0], out_dim[1]));

    let values_shape = values.shape();
    let n = values_shape[0];
    let k = values_shape[1];

    // For now we haven't implemented the PyObject case - do we need to?
    match (&py_mask, py_result_mask.as_mut()) {
        (Some(py_mask), Some(py_result_mask)) => {
            let mask = py_mask.as_array();
            let mut result_mask = py_result_mask.as_array_mut();

            for i in 0..n {
                unsafe {
                    let lab = *labels.uget(i);
                    if lab < 0 {
                        continue;
                    }

                    *counts.uget_mut(lab as usize) += 1;
                    for j in 0..k {
                        let val = *values.uget((i, j));
                        if *mask.uget((i, j)) == 0 {
                            *nobs.uget_mut((lab as usize, j)) += 1;
                            let y = val - *compensation.uget((lab as usize, j));
                            let t = *sumx.uget((lab as usize, j)) + y;
                            *compensation.uget_mut((lab as usize, j)) =
                                t - *sumx.uget((lab as usize, j)) - y;
                            *sumx.uget_mut((lab as usize, j)) = t;
                        }
                    }
                }
            }

            for i in 0..ncounts {
                for j in 0..k {
                    unsafe {
                        if *nobs.uget((i, j)) < min_count as i64 {
                            *result_mask.uget_mut((i, j)) = 1;
                        } else {
                            *out.uget_mut((i, j)) = *sumx.uget((i, j));
                        }
                    }
                }
            }
        }
        _ => {
            for i in 0..n {
                unsafe {
                    let lab = *labels.uget(i);
                    if lab < 0 {
                        continue;
                    }

                    *counts.uget_mut(lab as usize) += 1;
                    for j in 0..k {
                        let val = *values.uget((i, j));
                        if !val.isna(is_datetimelike) {
                            *nobs.uget_mut((lab as usize, j)) += 1;
                            let y = val - *compensation.uget((lab as usize, j));
                            let t = *sumx.uget((lab as usize, j)) + y;
                            *compensation.uget_mut((lab as usize, j)) =
                                t - *sumx.uget((lab as usize, j)) - y;
                            *sumx.uget_mut((lab as usize, j)) = t;
                        }
                    }
                }
            }

            for i in 0..ncounts {
                for j in 0..k {
                    unsafe {
                        if *nobs.uget((i, j)) < min_count as i64 {
                            *out.uget_mut((i, j)) = <T as PandasNA>::na_val(is_datetimelike);
                        } else {
                            *out.uget_mut((i, j)) = *sumx.uget((i, j));
                        }
                    }
                }
            }
        }
    }

    check_below_mincount(
        out,
        !py_mask.is_none(),
        py_result_mask,
        counts.len() as isize,
        k as isize,
        nobs.view(),
        min_count.try_into().unwrap(),
        sumx.view(),
    );
}

pub fn group_prod<T>(
    out: ArrayViewMut2<T>,
    mut counts: ArrayViewMut1<i64>,
    values: ArrayView2<T>,
    labels: ArrayView1<i64>,
    py_mask: Option<PyReadonlyArray2<u8>>,
    py_result_mask: Option<PyReadwriteArray2<u8>>,
    min_count: isize,
) where
    T: PandasNA + Zero + One + Clone + Copy + std::ops::MulAssign + std::ops::Add<Output = T>,
{
    if values.len() != labels.len() {
        panic!("len(index) != len(labels)");
    }

    let out_dim = out.shape();
    let mut nobs = Array2::<i64>::zeros((out_dim[0], out_dim[1]));
    let mut prodx = Array2::<T>::zeros((out_dim[0], out_dim[1]));

    let values_shape = values.shape();
    let n = values_shape[0];
    let k = values_shape[1];

    match py_mask {
        Some(ref py_mask) => {
            let mask = py_mask.as_array();

            for i in 0..n {
                unsafe {
                    let lab = *labels.uget(i);
                    if lab < 0 {
                        continue;
                    }

                    *counts.uget_mut(lab as usize) += 1;
                    for j in 0..k {
                        let val = *values.uget((i, j));
                        if *mask.uget((i, j)) == 0 {
                            *nobs.uget_mut((lab as usize, j)) += 1;
                            *prodx.uget_mut((lab as usize, j)) *= val;
                        }
                    }
                }
            }
        }
        _ => {
            for i in 0..n {
                unsafe {
                    let lab = *labels.uget(i);
                    if lab < 0 {
                        continue;
                    }

                    *counts.uget_mut(lab as usize) += 1;
                    for j in 0..k {
                        let val = *values.uget((i, j));
                        // No is_datetimelike in group_prod signature...
                        if !val.isna(false) {
                            *nobs.uget_mut((lab as usize, j)) += 1;
                            *prodx.uget_mut((lab as usize, j)) *= val;
                        }
                    }
                }
            }
        }
    }

    check_below_mincount(
        out,
        !py_mask.is_none(),
        py_result_mask,
        counts.len() as isize,
        k as isize,
        nobs.view(),
        min_count.try_into().unwrap(),
        prodx.view(),
    );
}

pub fn group_var<T>(
    mut out: ArrayViewMut2<T>,
    mut counts: ArrayViewMut1<i64>,
    values: ArrayView2<T>,
    labels: ArrayView1<i64>,
    min_count: isize,
    ddof: i64,
    py_mask: Option<PyReadonlyArray2<u8>>,
    py_result_mask: Option<PyReadwriteArray2<u8>>,
    is_datetimelike: bool,
    name: String,
) where
    T: Zero
        + Clone
        + Copy
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::AddAssign
        + std::ops::DivAssign
        + PandasNA
        + From<i32> // see note for nobs type
        + num::Float,
{
    if min_count != -1 {
        panic!("'min_count' only used in sum and prod");
    }

    if values.len() != labels.len() {
        panic!("len(index) != len(labels)");
    }

    let ncounts = counts.len();
    let is_std = name == "std";
    let is_sem = name == "sem";

    let out_dim = out.shape();

    // TODO: pandas has this as an i32, but the conversion from
    // i32 to f64 is not lossless, so rust does not like it during division
    let mut nobs = Array2::<i32>::zeros((out_dim[0], out_dim[1]));
    let mut mean = Array2::<T>::zeros((out_dim[0], out_dim[1]));

    let values_shape = values.shape();
    let n = values_shape[0];
    let k = values_shape[1];

    out.fill(<T as Zero>::zero());

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

                    *counts.uget_mut(lab as usize) += 1;

                    for j in 0..k {
                        let val = *values.uget((i, j));
                        if *mask.uget((i, j)) == 0 {
                            *nobs.uget_mut((lab as usize, j)) += 1;
                            let oldmean = *mean.uget((lab as usize, j));
                            let temp = (val - oldmean) / (*nobs.uget((lab as usize, j))).into();
                            *mean.uget_mut((lab as usize, j)) += temp;

                            *out.uget_mut((lab as usize, j)) += (val - temp) * (val - oldmean);
                        }
                    }

                    for i in 0..ncounts {
                        for j in 0..k {
                            let ct = *nobs.uget((i, j));
                            if ct < ddof as i32 {
                                *result_mask.uget_mut((i, j)) = 1;
                            } else {
                                if is_std {
                                    *out.uget_mut((i, j)) =
                                        (*out.uget((i, j)) / (ct - ddof as i32).into()).sqrt();
                                } else if is_sem {
                                    *out.uget_mut((i, j)) =
                                        (*out.uget((i, j)) / (ct - ddof as i32).into() / ct.into())
                                            .sqrt();
                                } else {
                                    // just "var"
                                    *out.uget_mut((i, j)) /= (ct - ddof as i32).into();
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => {
            for i in 0..n {
                unsafe {
                    let lab = *labels.uget(i);
                    if lab < 0 {
                        continue;
                    }

                    *counts.uget_mut(lab as usize) += 1;

                    for j in 0..k {
                        let val = *values.uget((i, j));
                        // TODO: Cython has the following note can't replicated here
                        // With group_var, we cannot just use _treat_as_na bc
                        // datetimelike dtypes get cast to float64 instead of
                        // to int64.
                        if !val.isna(is_datetimelike) {
                            *nobs.uget_mut((lab as usize, j)) += 1;
                            let oldmean = *mean.uget((lab as usize, j));
                            let temp = (val - oldmean) / (*nobs.uget((lab as usize, j))).into();
                            *mean.uget_mut((lab as usize, j)) += temp;

                            *out.uget_mut((lab as usize, j)) += (val - temp) * (val - oldmean);
                        }
                    }

                    for i in 0..ncounts {
                        for j in 0..k {
                            let ct = *nobs.uget((i, j));
                            if ct < ddof as i32 {
                                *out.uget_mut((i, j)) = <T as PandasNA>::na_val(false);
                            } else {
                                if is_std {
                                    *out.uget_mut((i, j)) =
                                        (*out.uget((i, j)) / (ct - ddof as i32).into()).sqrt();
                                } else if is_sem {
                                    *out.uget_mut((i, j)) =
                                        (*out.uget((i, j)) / (ct - ddof as i32).into() / ct.into())
                                            .sqrt();
                                } else {
                                    // just "var"
                                    *out.uget_mut((i, j)) /= (ct - ddof as i32).into();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn group_skew(
    mut out: ArrayViewMut2<f64>,
    mut counts: ArrayViewMut1<i64>,
    values: ArrayView2<f64>,
    labels: ArrayView1<i64>,
    py_mask: Option<PyReadonlyArray2<u8>>,
    py_result_mask: Option<PyReadwriteArray2<u8>>,
    skipna: bool,
) {
    if values.len() != labels.len() {
        panic!("len(index) != len(labels)");
    }

    let out_dim = out.shape();
    let mut nobs = Array2::<i64>::zeros((out_dim[0], out_dim[1]));

    // M1, M2 and M3 correspond to 1st, 2nd and third Moments
    let mut m1 = Array2::<f64>::zeros((out_dim[0], out_dim[1]));
    let mut m2 = Array2::<f64>::zeros((out_dim[0], out_dim[1]));
    let mut m3 = Array2::<f64>::zeros((out_dim[0], out_dim[1]));

    let values_shape = values.shape();
    let n = values_shape[0];
    let k = values_shape[1];

    out.fill(0.);

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

                    *counts.uget_mut(lab as usize) += 1;

                    for j in 0..k {
                        let val = *values.uget((i, j));

                        let isna_entry = *mask.uget((i, j));
                        if isna_entry == 0 {
                            // Based on Runningsats::Push from
                            // https://www.johndcook.com/blog/skewness_kurtosis/
                            let n1 = *nobs.uget((lab as usize, j));
                            let n_ = n1 + 1;

                            *nobs.uget_mut((lab as usize, j)) = n_;
                            let delta = val - *m1.uget((lab as usize, j));
                            let delta_n = delta / n_ as f64;
                            let term1 = delta * delta_n * n1 as f64;

                            *m1.uget_mut((lab as usize, j)) += delta_n;
                            *m3.uget_mut((lab as usize, j)) += term1 * delta_n * (n - 2) as f64
                                - 3 as f64 * delta_n * *m2.uget((lab as usize, j));
                            *m2.uget_mut((lab as usize, j)) += term1;
                        } else if !skipna {
                            *m1.uget_mut((lab as usize, j)) = f64::NAN;
                            *m2.uget_mut((lab as usize, j)) = f64::NAN;
                            *m3.uget_mut((lab as usize, j)) = f64::NAN;
                        }
                    }

                    for i in 0..counts.len() {
                        for j in 0..k {
                            let ct = *nobs.uget((i, j));
                            if ct < 3 {
                                *result_mask.uget_mut((i, j)) = 1;
                                *out.uget_mut((i, j)) = f64::NAN;
                            } else if *m2.uget((i, j)) == 0. {
                                *out.uget_mut((i, j)) = 0.;
                            } else {
                                *out.uget_mut((i, j)) = ((ct as f64) * ((ct - 1) as f64).powf(0.5)
                                    / ((ct - 2) as f64))
                                    * (*m3.uget((i, j)) / (*m2.uget((i, j)) as f64).powf(1.5));
                            }
                        }
                    }
                }
            }
        }
        _ => {
            for i in 0..n {
                unsafe {
                    let lab = *labels.uget(i);
                    if lab < 0 {
                        continue;
                    }

                    *counts.uget_mut(lab as usize) += 1;

                    for j in 0..k {
                        let val = *values.uget((i, j));

                        let isna_entry = val.isna(false);
                        if !isna_entry {
                            // Based on Runningsats::Push from
                            // https://www.johndcook.com/blog/skewness_kurtosis/
                            let n1 = *nobs.uget((lab as usize, j));
                            let n_ = n1 + 1;

                            *nobs.uget_mut((lab as usize, j)) = n_;
                            let delta = val - *m1.uget((lab as usize, j));
                            let delta_n = delta / n_ as f64;
                            let term1 = delta * delta_n * n1 as f64;

                            *m1.uget_mut((lab as usize, j)) += delta_n;
                            *m3.uget_mut((lab as usize, j)) += term1 * delta_n * (n - 2) as f64
                                - 3 as f64 * delta_n * *m2.uget((lab as usize, j));
                            *m2.uget_mut((lab as usize, j)) += term1;
                        } else if !skipna {
                            *m1.uget_mut((lab as usize, j)) = f64::NAN;
                            *m2.uget_mut((lab as usize, j)) = f64::NAN;
                            *m3.uget_mut((lab as usize, j)) = f64::NAN;
                        }
                    }

                    for i in 0..counts.len() {
                        for j in 0..k {
                            let ct = *nobs.uget((i, j));
                            if ct < 3 {
                                *out.uget_mut((i, j)) = f64::NAN;
                            } else if *m2.uget((i, j)) == 0. {
                                *out.uget_mut((i, j)) = 0.;
                            } else {
                                *out.uget_mut((i, j)) = ((ct as f64) * ((ct - 1) as f64).powf(0.5)
                                    / ((ct - 2) as f64))
                                    * (*m3.uget((i, j)) / (*m2.uget((i, j)) as f64).powf(1.5));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Only aggregates on axis=0 using Kahan summation
pub fn group_mean<T>(
    mut out: ArrayViewMut2<T>,
    mut counts: ArrayViewMut1<i64>,
    values: ArrayView2<T>,
    labels: ArrayView1<i64>,
    min_count: isize,
    is_datetimelike: bool,
    py_mask: Option<PyReadonlyArray2<u8>>,
    mut py_result_mask: Option<PyReadwriteArray2<u8>>,
) where
    T: PandasNA
        + Zero
        + One
        + Clone
        + Copy
        + std::ops::Sub<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + num::NumCast,
{
    if values.len() != labels.len() {
        panic!("len(index) != len(labels)");
    }

    let ncounts = counts.len();
    let out_dim = out.shape();
    let mut nobs = Array2::<i64>::zeros((out_dim[0], out_dim[1]));
    // the below is equivalent to `np.zeros_like(out)` but faster
    let mut sumx = Array2::<T>::zeros((out_dim[0], out_dim[1]));
    let mut compensation = Array2::<T>::zeros((out_dim[0], out_dim[1]));

    let values_shape = values.shape();
    let n = values_shape[0];
    let k = values_shape[1];

    // For now we haven't implemented the PyObject case - do we need to?

    match (&py_mask, py_result_mask.as_mut()) {
        (Some(py_mask), Some(py_result_mask)) => {
            let mask = py_mask.as_array();
            let mut result_mask = py_result_mask.as_array_mut();

            for i in 0..n {
                unsafe {
                    let lab = *labels.uget(i);
                    if lab < 0 {
                        continue;
                    }

                    *counts.uget_mut(lab as usize) += 1;
                    for j in 0..k {
                        let val = *values.uget((i, j));
                        if *mask.uget((i, j)) == 0 {
                            *nobs.uget_mut((lab as usize, j)) += 1;
                            let y = val - *compensation.uget((lab as usize, j));
                            let t = *sumx.uget((lab as usize, j)) + y;
                            *compensation.uget_mut((lab as usize, j)) =
                                t - *sumx.uget((lab as usize, j)) - y;
                            *sumx.uget_mut((lab as usize, j)) = t;
                        }
                    }
                }
            }

            for i in 0..ncounts {
                for j in 0..k {
                    unsafe {
                        let count = *nobs.uget((i, j));
                        if *nobs.uget((i, j)) < min_count as i64 {
                            *result_mask.uget_mut((i, j)) = 1;
                        } else {
                            *out.uget_mut((i, j)) =
                                *sumx.uget((i, j)) / NumCast::from(count).unwrap();
                        }
                    }
                }
            }
        }
        _ => {
            for i in 0..n {
                unsafe {
                    let lab = *labels.uget(i);
                    if lab < 0 {
                        continue;
                    }

                    *counts.uget_mut(lab as usize) += 1;
                    for j in 0..k {
                        let val = *values.uget((i, j));
                        if !val.isna(is_datetimelike) {
                            *nobs.uget_mut((lab as usize, j)) += 1;
                            let y = val - *compensation.uget((lab as usize, j));
                            let t = *sumx.uget((lab as usize, j)) + y;
                            *compensation.uget_mut((lab as usize, j)) =
                                t - *sumx.uget((lab as usize, j)) - y;
                            *sumx.uget_mut((lab as usize, j)) = t;
                        }
                    }
                }
            }

            for i in 0..ncounts {
                for j in 0..k {
                    unsafe {
                        let count = *nobs.uget((i, j));
                        if *nobs.uget((i, j)) < min_count as i64 {
                            *out.uget_mut((i, j)) = <T as PandasNA>::na_val(is_datetimelike);
                        } else {
                            *out.uget_mut((i, j)) =
                                *sumx.uget((i, j)) / NumCast::from(count).unwrap();
                        }
                    }
                }
            }
        }
    }

    check_below_mincount(
        out,
        !py_mask.is_none(),
        py_result_mask,
        counts.len() as isize,
        k as isize,
        nobs.view(),
        min_count.try_into().unwrap(),
        sumx.view(),
    );
}

pub fn group_ohlc<T>(
    mut out: ArrayViewMut2<T>,
    mut counts: ArrayViewMut1<i64>,
    values: ArrayView2<T>,
    labels: ArrayView1<i64>,
    min_count: isize,
    py_mask: Option<PyReadonlyArray2<u8>>,
    mut py_result_mask: Option<PyReadwriteArray2<u8>>,
) where
    T: PandasNA + Zero + Clone + Copy + PartialOrd,
{
    if min_count != -1 {
        panic!("'min_count' only used in sum and prod");
    }

    if labels.len() == 0 {
        return;
    }

    let values_shape = values.shape();
    let n = values_shape[0];
    let k = values_shape[1];

    if out.shape()[1] != 4 {
        panic!("Output array must have 4 columns");
    }

    if k > 1 {
        panic!("Argument 'values' must have only one dimension");
    }

    let mut first_element_set = Array1::<u8>::zeros(counts.len());

    match (&py_mask, py_result_mask.as_mut()) {
        (Some(py_mask), Some(py_result_mask)) => {
            let mask = py_mask.as_array();
            let mut result_mask = py_result_mask.as_array_mut();
            result_mask.fill(1);

            for i in 0..n {
                unsafe {
                    let lab = *labels.uget(i);
                    if lab == -1 {
                        continue;
                    }

                    *counts.uget_mut(lab as usize) += 1;
                    let val = *values.uget((i, 0));
                    let isna_entry = *mask.uget((i, 0));
                    if isna_entry == 1 {
                        continue;
                    }

                    if *first_element_set.uget(lab as usize) == 0 {
                        *out.uget_mut((lab as usize, 0)) = val;
                        *out.uget_mut((lab as usize, 1)) = val;
                        *out.uget_mut((lab as usize, 2)) = val;
                        *out.uget_mut((lab as usize, 3)) = val;
                        *first_element_set.uget_mut(lab as usize) = 1;

                        // TODO: can we replace this with a slice?
                        *result_mask.uget_mut((lab as usize, 0)) = 0;
                        *result_mask.uget_mut((lab as usize, 1)) = 0;
                        *result_mask.uget_mut((lab as usize, 2)) = 0;
                        *result_mask.uget_mut((lab as usize, 3)) = 0;
                    } else {
                        let result1 = (*out.uget_mut((lab as usize, 1)))
                            .partial_cmp(&val)
                            .expect("trying to compare NA");
                        if result1 == Ordering::Less {
                            *out.uget_mut((lab as usize, 1)) = val;
                        }

                        let result2 = (*out.uget((lab as usize, 1)))
                            .partial_cmp(&val)
                            .expect("tried to compare nan");
                        if result2 == Ordering::Less {
                            *out.uget_mut((lab as usize, 2)) = val;
                        }
                        *out.uget_mut((lab as usize, 3)) = val;
                    }
                }
            }
        }
        _ => {
            for i in 0..n {
                unsafe {
                    let lab = *labels.uget(i);
                    if lab == -1 {
                        continue;
                    }

                    *counts.uget_mut(lab as usize) += 1;
                    let val = *values.uget((i, 0));
                    let isna_entry = val.isna(false); // no is_datetimelike arg
                    if isna_entry {
                        continue;
                    }

                    if *first_element_set.uget(lab as usize) == 0 {
                        *out.uget_mut((lab as usize, 0)) = val;
                        *out.uget_mut((lab as usize, 1)) = val;
                        *out.uget_mut((lab as usize, 2)) = val;
                        *out.uget_mut((lab as usize, 3)) = val;
                        *first_element_set.uget_mut(lab as usize) = 1;
                    } else {
                        let result1 = (*out.uget_mut((lab as usize, 1)))
                            .partial_cmp(&val)
                            .expect("trying to compare NA");
                        if result1 == Ordering::Less {
                            *out.uget_mut((lab as usize, 1)) = val;
                        }

                        let result2 = (*out.uget((lab as usize, 1)))
                            .partial_cmp(&val)
                            .expect("tried to compare nan");
                        if result2 == Ordering::Less {
                            *out.uget_mut((lab as usize, 2)) = val;
                        }
                        *out.uget_mut((lab as usize, 3)) = val;
                    }
                }
            }
        }
    }
}

pub fn group_quantile<T>(
    mut out: ArrayViewMut2<f64>,
    values: ArrayView1<T>,
    labels: ArrayView1<i64>,
    mut mask: ArrayViewMut1<u8>,
    sort_indexer: ArrayView1<i64>,
    qs: ArrayView1<f64>,
    interpolation: String,
    mut py_result_mask: Option<PyReadwriteArray2<u8>>,
) where
    T: Zero + Copy + NumCast + std::ops::Sub<Output = T>,
{
    let n = labels.len();
    if values.shape()[0] != n {
        panic!("shape mismatch");
    }

    for q in qs.into_iter() {
        let val = *q;
        if (val < 0.) | (val > 1.) {
            panic!("Each 'q' must be between 0 and 1. Got '{}' instead", val);
        }
    }

    let inter_methods = HashMap::from([
        ("linear", 0),
        ("lower", 1),
        ("higher", 2),
        ("nearest", 3),
        ("midpoint", 4),
    ]);
    let interp = inter_methods
        .get(&interpolation as &str)
        .expect("could not find interpolation method");

    let mut grp_start = 0;
    let mut grp_sz;
    let nqs = qs.len();
    let ngroups = out.len();
    let mut counts = Array1::<i64>::zeros(ngroups);
    let mut non_na_counts = Array1::<i64>::zeros(ngroups);

    // First figure out the size of every group
    for i in 0..n {
        unsafe {
            let lab = *labels.uget(i);
            if lab == -1 {
                // NA group label
                continue;
            }

            *counts.uget_mut(lab as usize) += 1;
            if *mask.uget_mut(i) == 0 {
                *non_na_counts.uget_mut(lab as usize) += 1;
            }
        }
    }

    for i in 0..ngroups {
        unsafe {
            // Figure out how many group elements there are
            grp_sz = *counts.uget(i);
            let non_na_sz = *non_na_counts.uget(i);

            if non_na_sz == 0 {
                for k in 0..nqs {
                    match py_result_mask.as_mut() {
                        Some(py_result_mask) => {
                            let mut result_mask = py_result_mask.as_array_mut();
                            *result_mask.uget_mut((i, k)) = 1;
                        }
                        _ => {
                            *out.uget_mut((i, k)) = f64::NAN;
                        }
                    }
                }
            } else {
                for k in 0..nqs {
                    let q_val = *qs.uget(k);

                    // Calculate where to retrieve the desired value
                    // Casting to int will intentionally truncate result
                    let idx = grp_start + (q_val * (non_na_sz - 1) as f64) as i64;

                    let val = *values.uget(*sort_indexer.uget(idx as usize) as usize);
                    // If requested quantile falls evenly on a particular index
                    // then write that index's value out. Otherwise interpolate
                    let q_idx = q_val * (non_na_sz - 1) as f64;
                    let frac = q_idx % 1.;

                    // TODO: we should create an enum to manage this instead
                    // of using integral codes
                    if (frac == 0.) || (*interp == 1) {
                        // LOWER
                        *out.uget_mut((i, k)) = NumCast::from(val).unwrap();
                    } else {
                        let next_val =
                            *values.uget(*sort_indexer.uget((idx + 1) as usize) as usize);
                        if *interp == 0 {
                            // LINEAR
                            // Rust does not implicitly allow i64 -> f64 conversions
                            // so need extra hackery
                            let f_next_val: f64 = NumCast::from(next_val).unwrap();
                            let f_val: f64 = NumCast::from(val).unwrap();
                            let result = f_val + (f_next_val - f_val) * frac;
                            *out.uget_mut((i, k)) = NumCast::from(result).unwrap();
                        } else if *interp == 2 {
                            // HIGHER
                            *out.uget_mut((i, k)) = NumCast::from(next_val).unwrap();
                        } else if *interp == 4 {
                            // MIDPOINT
                            let f_sum: f64 = NumCast::from(val + next_val).unwrap();
                            *out.uget_mut((i, k)) = NumCast::from(f_sum / 2.).unwrap();
                        } else if *interp == 3 {
                            // NEAREST
                            if (frac > 0.5) || (frac == 0.5 && q_val > 0.5) {
                                *out.uget_mut((i, k)) = NumCast::from(next_val).unwrap();
                            } else {
                                *out.uget_mut((i, k)) = NumCast::from(val).unwrap();
                            }
                        }
                    }
                }
            }
        }

        grp_start += grp_sz;
    }
}
