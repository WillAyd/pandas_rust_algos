use numpy::ndarray::ArrayViewMut2;
use std::cell::UnsafeCell;

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

    /// SAFETY: It is UB if two threads write to the same index without
    /// synchronization.
    pub unsafe fn write(&self, i: usize, j: usize, value: T) {
        let ptr = self.array.get();
        *(*ptr).uget_mut((i, j)) = value;
    }
}
