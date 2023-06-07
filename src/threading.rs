use numpy::ndarray::{ArrayViewMut1, ArrayViewMut2};
use std::cell::UnsafeCell;

#[derive(Copy, Clone)]
pub struct UnsafeArrayView1<'a, T> {
    pub array: &'a UnsafeCell<ArrayViewMut1<'a, T>>,
}

unsafe impl<'a, T: Copy> Send for UnsafeArrayView1<'a, T> {}
unsafe impl<'a, T: Copy> Sync for UnsafeArrayView1<'a, T> {}

impl<'a, T> UnsafeArrayView1<'a, T> {
    pub fn new(array: &'a mut ArrayViewMut1<T>) -> Self {
        let ptr = array as *mut ArrayViewMut1<T> as *const UnsafeCell<ArrayViewMut1<T>>;
        Self {
            array: unsafe { &*ptr },
        }
    }

    pub unsafe fn read(&self, i: usize) -> T
    where
        T: Copy,
    {
        let ptr = self.array.get();
        *(*ptr).uget(i)
    }

    /// SAFETY: It is UB if two threads write to the same index without
    /// synchronization.
    pub unsafe fn write(&self, i: usize, value: T) {
        let ptr = self.array.get();
        *(*ptr).uget_mut(i) = value;
    }
}

#[derive(Copy, Clone)]
pub struct UnsafeArrayView2<'a, T> {
    pub array: &'a UnsafeCell<ArrayViewMut2<'a, T>>,
}

unsafe impl<'a, T> Send for UnsafeArrayView2<'a, T> {}
unsafe impl<'a, T> Sync for UnsafeArrayView2<'a, T> {}

impl<'a, T> UnsafeArrayView2<'a, T> {
    pub fn new(array: &'a mut ArrayViewMut2<T>) -> Self {
        let ptr = array as *mut ArrayViewMut2<T> as *const UnsafeCell<ArrayViewMut2<T>>;
        Self {
            array: unsafe { &*ptr },
        }
    }

    pub unsafe fn read(&self, i: usize, j: usize) -> T
    where
        T: Copy,
    {
        let ptr = self.array.get();
        *(*ptr).uget((i, j))
    }

    /// SAFETY: It is UB if two threads write to the same index without
    /// synchronization.
    pub unsafe fn write(&self, i: usize, j: usize, value: T) {
        let ptr = self.array.get();
        *(*ptr).uget_mut((i, j)) = value;
    }
}
