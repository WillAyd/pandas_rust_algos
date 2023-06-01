use numpy::ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut2, Axis};
use std::ptr;

pub unsafe fn kth_smallest_c<T>(arr: *const T, k: usize, n: usize) -> T
where
    T: PartialOrd + Copy,
{
    let mut left = 0;
    let mut m: isize = (n - 1) as isize;

    while left < m {
        let x = *arr.add(k);
        let mut i = left;
        let mut j = m;

        loop {
            while *arr.add(i as usize) < x {
                i += 1;
            }
            while x < *arr.add(j as usize) {
                j -= 1;
            }
            if i <= j {
                ptr::swap(
                    arr.add(i as usize).cast_mut(),
                    arr.add(j as usize).cast_mut(),
                );
                i += 1;
                j -= 1;
            }

            if i > j {
                break;
            }
        }

        if j < k as isize {
            left = i;
        }
        if k < i as usize {
            m = j;
        }
    }

    *arr.add(k)
}

pub fn take_2d_axis1<T>(values: ArrayView2<T>, indexer: ArrayView1<i64>, mut out: ArrayViewMut2<T>)
where
    T: Copy,
{
    let ncols = indexer.raw_dim()[0];

    for (i, val_row) in values.axis_iter(Axis(0)).enumerate() {
        for j in 0..ncols {
            unsafe {
                let idx = *indexer.uget(j);
                *out.uget_mut((i, j)) = val_row[idx as usize];
            }
        }
    }
}

pub fn groupsort_indexer(index: ArrayView1<i64>, ngroups: usize) -> (Array1<i64>, Array1<i64>) {
    let mut counts = Array1::<i64>::zeros(ngroups + 1);
    let n = index.len();
    let mut indexer = Array1::<i64>::zeros(n);
    let mut where_ = Array1::<i64>::zeros(ngroups + 1);

    for i in 0..n {
        let idx = index[i];
        counts[(idx + 1) as usize] += 1;
    }

    for i in 1..ngroups + 1 {
        where_[i] = where_[i - 1] + counts[i - 1];
    }

    for i in 0..n {
        let label = index[i] + 1;
        indexer[where_[label as usize] as usize] = i as i64;
        where_[label as usize] += 1;
    }

    (indexer, counts)
}
