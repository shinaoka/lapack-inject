//! LAPACKE utility functions.
//!
//! Following OpenBLAS LAPACKE implementation exactly.

use crate::types::{lapack_int, lapack_min, LAPACK_COL_MAJOR, LAPACK_ROW_MAJOR};
use num_complex::{Complex32, Complex64};

// =============================================================================
// General matrix transpose (following OpenBLAS lapacke_*ge_trans.c exactly)
// =============================================================================

/// LAPACKE_sge_trans - Transpose general single precision matrix.
/// Converts input general matrix from row-major(C) to column-major(Fortran)
/// layout or vice versa.
#[inline]
pub fn lapacke_sge_trans(
    matrix_layout: i32,
    m: lapack_int,
    n: lapack_int,
    input: *const f32,
    ldin: lapack_int,
    output: *mut f32,
    ldout: lapack_int,
) {
    if input.is_null() || output.is_null() {
        return;
    }

    let (x, y) = if matrix_layout == LAPACK_COL_MAJOR {
        (n, m)
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        (m, n)
    } else {
        return;
    };

    unsafe {
        for i in 0..lapack_min(y, ldin) {
            for j in 0..lapack_min(x, ldout) {
                *output.offset((i as isize) * (ldout as isize) + (j as isize)) =
                    *input.offset((j as isize) * (ldin as isize) + (i as isize));
            }
        }
    }
}

/// LAPACKE_dge_trans - Transpose general double precision matrix.
#[inline]
pub fn lapacke_dge_trans(
    matrix_layout: i32,
    m: lapack_int,
    n: lapack_int,
    input: *const f64,
    ldin: lapack_int,
    output: *mut f64,
    ldout: lapack_int,
) {
    if input.is_null() || output.is_null() {
        return;
    }

    let (x, y) = if matrix_layout == LAPACK_COL_MAJOR {
        (n, m)
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        (m, n)
    } else {
        return;
    };

    unsafe {
        for i in 0..lapack_min(y, ldin) {
            for j in 0..lapack_min(x, ldout) {
                *output.offset((i as isize) * (ldout as isize) + (j as isize)) =
                    *input.offset((j as isize) * (ldin as isize) + (i as isize));
            }
        }
    }
}

/// LAPACKE_cge_trans - Transpose general single precision complex matrix.
#[inline]
pub fn lapacke_cge_trans(
    matrix_layout: i32,
    m: lapack_int,
    n: lapack_int,
    input: *const Complex32,
    ldin: lapack_int,
    output: *mut Complex32,
    ldout: lapack_int,
) {
    if input.is_null() || output.is_null() {
        return;
    }

    let (x, y) = if matrix_layout == LAPACK_COL_MAJOR {
        (n, m)
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        (m, n)
    } else {
        return;
    };

    unsafe {
        for i in 0..lapack_min(y, ldin) {
            for j in 0..lapack_min(x, ldout) {
                *output.offset((i as isize) * (ldout as isize) + (j as isize)) =
                    *input.offset((j as isize) * (ldin as isize) + (i as isize));
            }
        }
    }
}

/// LAPACKE_zge_trans - Transpose general double precision complex matrix.
#[inline]
pub fn lapacke_zge_trans(
    matrix_layout: i32,
    m: lapack_int,
    n: lapack_int,
    input: *const Complex64,
    ldin: lapack_int,
    output: *mut Complex64,
    ldout: lapack_int,
) {
    if input.is_null() || output.is_null() {
        return;
    }

    let (x, y) = if matrix_layout == LAPACK_COL_MAJOR {
        (n, m)
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        (m, n)
    } else {
        return;
    };

    unsafe {
        for i in 0..lapack_min(y, ldin) {
            for j in 0..lapack_min(x, ldout) {
                *output.offset((i as isize) * (ldout as isize) + (j as isize)) =
                    *input.offset((j as isize) * (ldin as isize) + (i as isize));
            }
        }
    }
}

// =============================================================================
// Memory allocation/deallocation (using standard allocator)
// =============================================================================

/// Allocate memory (equivalent to LAPACKE_malloc)
#[inline]
pub fn lapacke_malloc<T>(count: usize) -> *mut T {
    if count == 0 {
        return std::ptr::null_mut();
    }
    let layout = std::alloc::Layout::array::<T>(count).unwrap();
    unsafe { std::alloc::alloc(layout) as *mut T }
}

/// Free memory (equivalent to LAPACKE_free)
#[inline]
pub fn lapacke_free<T>(ptr: *mut T) {
    if !ptr.is_null() {
        // Note: We need to know the size to properly deallocate.
        // In practice, we'll use Vec for safe memory management.
    }
}

// =============================================================================
// Triangular/symmetric matrix transpose (following OpenBLAS exactly)
// =============================================================================

/// LAPACKE_dtr_trans - Transpose triangular matrix.
#[inline]
pub fn lapacke_dtr_trans(
    matrix_layout: i32,
    uplo: u8,
    diag: u8,
    n: lapack_int,
    input: *const f64,
    ldin: lapack_int,
    output: *mut f64,
    ldout: lapack_int,
) {
    use crate::types::lapacke_lsame;

    if input.is_null() || output.is_null() {
        return;
    }

    let colmaj = matrix_layout == LAPACK_COL_MAJOR;
    let lower = lapacke_lsame(uplo, b'l');
    let unit = lapacke_lsame(diag, b'u');

    if (!colmaj && matrix_layout != LAPACK_ROW_MAJOR)
        || (!lower && !lapacke_lsame(uplo, b'u'))
        || (!unit && !lapacke_lsame(diag, b'n'))
    {
        return;
    }

    let st = if unit { 1 } else { 0 };

    unsafe {
        if (colmaj || lower) && !(colmaj && lower) {
            for j in st..lapack_min(n, ldout) {
                for i in 0..lapack_min(j + 1 - st, ldin) {
                    *output.offset((j as isize) + (i as isize) * (ldout as isize)) =
                        *input.offset((i as isize) + (j as isize) * (ldin as isize));
                }
            }
        } else {
            for j in 0..lapack_min(n - st, ldout) {
                for i in (j + st)..lapack_min(n, ldin) {
                    *output.offset((j as isize) + (i as isize) * (ldout as isize)) =
                        *input.offset((i as isize) + (j as isize) * (ldin as isize));
                }
            }
        }
    }
}

/// LAPACKE_dpo_trans - Transpose symmetric/positive-definite matrix.
#[inline]
pub fn lapacke_dpo_trans(
    matrix_layout: i32,
    uplo: u8,
    n: lapack_int,
    input: *const f64,
    ldin: lapack_int,
    output: *mut f64,
    ldout: lapack_int,
) {
    lapacke_dtr_trans(matrix_layout, uplo, b'n', n, input, ldin, output, ldout);
}

/// LAPACKE_str_trans - Transpose triangular matrix (single precision).
#[inline]
pub fn lapacke_str_trans(
    matrix_layout: i32,
    uplo: u8,
    diag: u8,
    n: lapack_int,
    input: *const f32,
    ldin: lapack_int,
    output: *mut f32,
    ldout: lapack_int,
) {
    use crate::types::lapacke_lsame;

    if input.is_null() || output.is_null() {
        return;
    }

    let colmaj = matrix_layout == LAPACK_COL_MAJOR;
    let lower = lapacke_lsame(uplo, b'l');
    let unit = lapacke_lsame(diag, b'u');

    if (!colmaj && matrix_layout != LAPACK_ROW_MAJOR)
        || (!lower && !lapacke_lsame(uplo, b'u'))
        || (!unit && !lapacke_lsame(diag, b'n'))
    {
        return;
    }

    let st = if unit { 1 } else { 0 };

    unsafe {
        if (colmaj || lower) && !(colmaj && lower) {
            for j in st..lapack_min(n, ldout) {
                for i in 0..lapack_min(j + 1 - st, ldin) {
                    *output.offset((j as isize) + (i as isize) * (ldout as isize)) =
                        *input.offset((i as isize) + (j as isize) * (ldin as isize));
                }
            }
        } else {
            for j in 0..lapack_min(n - st, ldout) {
                for i in (j + st)..lapack_min(n, ldin) {
                    *output.offset((j as isize) + (i as isize) * (ldout as isize)) =
                        *input.offset((i as isize) + (j as isize) * (ldin as isize));
                }
            }
        }
    }
}

/// LAPACKE_spo_trans - Transpose symmetric/positive-definite matrix (single precision).
#[inline]
pub fn lapacke_spo_trans(
    matrix_layout: i32,
    uplo: u8,
    n: lapack_int,
    input: *const f32,
    ldin: lapack_int,
    output: *mut f32,
    ldout: lapack_int,
) {
    lapacke_str_trans(matrix_layout, uplo, b'n', n, input, ldin, output, ldout);
}

/// LAPACKE_ctr_trans - Transpose triangular matrix (complex single precision).
#[inline]
pub fn lapacke_ctr_trans(
    matrix_layout: i32,
    uplo: u8,
    diag: u8,
    n: lapack_int,
    input: *const Complex32,
    ldin: lapack_int,
    output: *mut Complex32,
    ldout: lapack_int,
) {
    use crate::types::lapacke_lsame;

    if input.is_null() || output.is_null() {
        return;
    }

    let colmaj = matrix_layout == LAPACK_COL_MAJOR;
    let lower = lapacke_lsame(uplo, b'l');
    let unit = lapacke_lsame(diag, b'u');

    if (!colmaj && matrix_layout != LAPACK_ROW_MAJOR)
        || (!lower && !lapacke_lsame(uplo, b'u'))
        || (!unit && !lapacke_lsame(diag, b'n'))
    {
        return;
    }

    let st = if unit { 1 } else { 0 };

    unsafe {
        if (colmaj || lower) && !(colmaj && lower) {
            for j in st..lapack_min(n, ldout) {
                for i in 0..lapack_min(j + 1 - st, ldin) {
                    *output.offset((j as isize) + (i as isize) * (ldout as isize)) =
                        *input.offset((i as isize) + (j as isize) * (ldin as isize));
                }
            }
        } else {
            for j in 0..lapack_min(n - st, ldout) {
                for i in (j + st)..lapack_min(n, ldin) {
                    *output.offset((j as isize) + (i as isize) * (ldout as isize)) =
                        *input.offset((i as isize) + (j as isize) * (ldin as isize));
                }
            }
        }
    }
}

/// LAPACKE_cpo_trans - Transpose symmetric/positive-definite matrix (complex single precision).
#[inline]
pub fn lapacke_cpo_trans(
    matrix_layout: i32,
    uplo: u8,
    n: lapack_int,
    input: *const Complex32,
    ldin: lapack_int,
    output: *mut Complex32,
    ldout: lapack_int,
) {
    lapacke_ctr_trans(matrix_layout, uplo, b'n', n, input, ldin, output, ldout);
}

/// LAPACKE_ztr_trans - Transpose triangular matrix (complex double precision).
#[inline]
pub fn lapacke_ztr_trans(
    matrix_layout: i32,
    uplo: u8,
    diag: u8,
    n: lapack_int,
    input: *const Complex64,
    ldin: lapack_int,
    output: *mut Complex64,
    ldout: lapack_int,
) {
    use crate::types::lapacke_lsame;

    if input.is_null() || output.is_null() {
        return;
    }

    let colmaj = matrix_layout == LAPACK_COL_MAJOR;
    let lower = lapacke_lsame(uplo, b'l');
    let unit = lapacke_lsame(diag, b'u');

    if (!colmaj && matrix_layout != LAPACK_ROW_MAJOR)
        || (!lower && !lapacke_lsame(uplo, b'u'))
        || (!unit && !lapacke_lsame(diag, b'n'))
    {
        return;
    }

    let st = if unit { 1 } else { 0 };

    unsafe {
        if (colmaj || lower) && !(colmaj && lower) {
            for j in st..lapack_min(n, ldout) {
                for i in 0..lapack_min(j + 1 - st, ldin) {
                    *output.offset((j as isize) + (i as isize) * (ldout as isize)) =
                        *input.offset((i as isize) + (j as isize) * (ldin as isize));
                }
            }
        } else {
            for j in 0..lapack_min(n - st, ldout) {
                for i in (j + st)..lapack_min(n, ldin) {
                    *output.offset((j as isize) + (i as isize) * (ldout as isize)) =
                        *input.offset((i as isize) + (j as isize) * (ldin as isize));
                }
            }
        }
    }
}

/// LAPACKE_zpo_trans - Transpose symmetric/positive-definite matrix (complex double precision).
#[inline]
pub fn lapacke_zpo_trans(
    matrix_layout: i32,
    uplo: u8,
    n: lapack_int,
    input: *const Complex64,
    ldin: lapack_int,
    output: *mut Complex64,
    ldout: lapack_int,
) {
    lapacke_ztr_trans(matrix_layout, uplo, b'n', n, input, ldin, output, ldout);
}

// =============================================================================
// Safe memory allocation using Vec (preferred approach)
// =============================================================================

/// Allocate a temporary f32 array and return as Vec
#[inline]
pub fn alloc_f32(count: usize) -> Option<Vec<f32>> {
    if count == 0 {
        return Some(Vec::new());
    }
    Some(vec![0.0f32; count])
}

/// Allocate a temporary f64 array and return as Vec
#[inline]
pub fn alloc_f64(count: usize) -> Option<Vec<f64>> {
    if count == 0 {
        return Some(Vec::new());
    }
    Some(vec![0.0f64; count])
}

/// Allocate a temporary Complex32 array and return as Vec
#[inline]
pub fn alloc_c32(count: usize) -> Option<Vec<Complex32>> {
    if count == 0 {
        return Some(Vec::new());
    }
    Some(vec![Complex32::new(0.0, 0.0); count])
}

/// Allocate a temporary Complex64 array and return as Vec
#[inline]
pub fn alloc_c64(count: usize) -> Option<Vec<Complex64>> {
    if count == 0 {
        return Some(Vec::new());
    }
    Some(vec![Complex64::new(0.0, 0.0); count])
}
