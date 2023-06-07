use num::traits::Float;

pub trait PandasNA {
    fn na_val(is_datetimelike: bool) -> Self;
    fn isna(&self, is_datetimelike: bool) -> bool;
    fn is_finite(&self) -> bool;
}

impl PandasNA for i8 {
    fn na_val(is_datetimelike: bool) -> Self {
        if is_datetimelike {
            i8::MIN
        } else {
            0
        }
    }

    fn isna(&self, is_datetimelike: bool) -> bool {
        if is_datetimelike {
            *self == i8::MIN
        } else {
            *self == 0
        }
    }

    fn is_finite(&self) -> bool {
        true
    }
}

impl PandasNA for i16 {
    fn na_val(is_datetimelike: bool) -> Self {
        if is_datetimelike {
            i16::MIN
        } else {
            0
        }
    }

    fn isna(&self, is_datetimelike: bool) -> bool {
        if is_datetimelike {
            *self == i16::MIN
        } else {
            *self == 0
        }
    }

    fn is_finite(&self) -> bool {
        true
    }
}

impl PandasNA for i32 {
    fn na_val(is_datetimelike: bool) -> Self {
        if is_datetimelike {
            i32::MIN
        } else {
            0
        }
    }

    fn isna(&self, is_datetimelike: bool) -> bool {
        if is_datetimelike {
            *self == i32::MIN
        } else {
            *self == 0
        }
    }

    fn is_finite(&self) -> bool {
        true
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

    fn is_finite(&self) -> bool {
        true
    }
}

impl PandasNA for u8 {
    fn na_val(is_datetimelike: bool) -> Self {
        if is_datetimelike {
            u8::MAX // TOOD: this might not be correct
        } else {
            0
        }
    }

    fn isna(&self, is_datetimelike: bool) -> bool {
        if is_datetimelike {
            *self == u8::MIN
        } else {
            *self == 0
        }
    }

    fn is_finite(&self) -> bool {
        true
    }
}

impl PandasNA for u16 {
    fn na_val(is_datetimelike: bool) -> Self {
        if is_datetimelike {
            u16::MAX // TOOD: this might not be correct
        } else {
            0
        }
    }

    fn isna(&self, is_datetimelike: bool) -> bool {
        if is_datetimelike {
            *self == u16::MIN
        } else {
            *self == 0
        }
    }

    fn is_finite(&self) -> bool {
        true
    }
}

impl PandasNA for u32 {
    fn na_val(is_datetimelike: bool) -> Self {
        if is_datetimelike {
            u32::MAX // TOOD: this might not be correct
        } else {
            0
        }
    }

    fn isna(&self, is_datetimelike: bool) -> bool {
        if is_datetimelike {
            *self == u32::MIN
        } else {
            *self == 0
        }
    }

    fn is_finite(&self) -> bool {
        true
    }
}

impl PandasNA for u64 {
    fn na_val(is_datetimelike: bool) -> Self {
        if is_datetimelike {
            u64::MAX // TOOD: this might not be correct
        } else {
            0
        }
    }

    fn isna(&self, is_datetimelike: bool) -> bool {
        if is_datetimelike {
            *self == u64::MIN
        } else {
            *self == 0
        }
    }

    fn is_finite(&self) -> bool {
        true
    }
}

impl PandasNA for f32 {
    fn na_val(_is_datetimelike: bool) -> Self {
        f32::NAN
    }

    fn isna(&self, _is_datetimelike: bool) -> bool {
        self.is_nan()
    }

    fn is_finite(&self) -> bool {
        <f32 as Float>::is_finite(*self)
    }
}

impl PandasNA for f64 {
    fn na_val(_is_datetimelike: bool) -> Self {
        f64::NAN
    }

    fn isna(&self, _is_datetimelike: bool) -> bool {
        self.is_nan()
    }

    fn is_finite(&self) -> bool {
        <f64 as Float>::is_finite(*self)
    }
}
