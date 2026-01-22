//! GELS - Least squares solution using QR/LQ factorization.
//!
//! LAPACKE interface following OpenBLAS implementation exactly.

use crate::backend::{get_cgels, get_dgels, get_sgels, get_zgels};
use crate::types::{
    lapack_int, lapack_max, LAPACK_COL_MAJOR, LAPACK_ROW_MAJOR, LAPACK_TRANSPOSE_MEMORY_ERROR,
    LAPACK_WORK_MEMORY_ERROR,
};
use crate::utils::{alloc_c32, alloc_c64, alloc_f32, alloc_f64, lapacke_cge_trans, lapacke_dge_trans, lapacke_sge_trans, lapacke_zge_trans};
use crate::xerbla::lapacke_xerbla_internal;
use num_complex::{Complex32, Complex64};
use std::ffi::c_char;

// =============================================================================
// LAPACKE high-level interface
// =============================================================================

/// LAPACKE_sgels - Solves overdetermined/underdetermined linear systems (single precision).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_sgels(
    matrix_layout: i32,
    trans: u8,
    m: lapack_int,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut f32,
    lda: lapack_int,
    b: *mut f32,
    ldb: lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_sgels", -1);
        return -1;
    }

    // Query optimal workspace size
    let mut work_query: f32 = 0.0;
    let info = LAPACKE_sgels_work(matrix_layout, trans, m, n, nrhs, a, lda, b, ldb, &mut work_query, -1);
    if info != 0 {
        return info;
    }

    let lwork = work_query as lapack_int;
    let mut work = match alloc_f32(lwork as usize) {
        Some(v) => v,
        None => {
            lapacke_xerbla_internal("LAPACKE_sgels", LAPACK_WORK_MEMORY_ERROR);
            return LAPACK_WORK_MEMORY_ERROR;
        }
    };

    LAPACKE_sgels_work(matrix_layout, trans, m, n, nrhs, a, lda, b, ldb, work.as_mut_ptr(), lwork)
}

/// LAPACKE_dgels - Solves overdetermined/underdetermined linear systems (double precision).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgels(
    matrix_layout: i32,
    trans: u8,
    m: lapack_int,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut f64,
    lda: lapack_int,
    b: *mut f64,
    ldb: lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_dgels", -1);
        return -1;
    }

    // Query optimal workspace size
    let mut work_query: f64 = 0.0;
    let info = LAPACKE_dgels_work(matrix_layout, trans, m, n, nrhs, a, lda, b, ldb, &mut work_query, -1);
    if info != 0 {
        return info;
    }

    let lwork = work_query as lapack_int;
    let mut work = match alloc_f64(lwork as usize) {
        Some(v) => v,
        None => {
            lapacke_xerbla_internal("LAPACKE_dgels", LAPACK_WORK_MEMORY_ERROR);
            return LAPACK_WORK_MEMORY_ERROR;
        }
    };

    LAPACKE_dgels_work(matrix_layout, trans, m, n, nrhs, a, lda, b, ldb, work.as_mut_ptr(), lwork)
}

/// LAPACKE_cgels - Solves overdetermined/underdetermined linear systems (single complex).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_cgels(
    matrix_layout: i32,
    trans: u8,
    m: lapack_int,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut Complex32,
    lda: lapack_int,
    b: *mut Complex32,
    ldb: lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_cgels", -1);
        return -1;
    }

    // Query optimal workspace size
    let mut work_query = Complex32::new(0.0, 0.0);
    let info = LAPACKE_cgels_work(matrix_layout, trans, m, n, nrhs, a, lda, b, ldb, &mut work_query, -1);
    if info != 0 {
        return info;
    }

    let lwork = work_query.re as lapack_int;
    let mut work = match alloc_c32(lwork as usize) {
        Some(v) => v,
        None => {
            lapacke_xerbla_internal("LAPACKE_cgels", LAPACK_WORK_MEMORY_ERROR);
            return LAPACK_WORK_MEMORY_ERROR;
        }
    };

    LAPACKE_cgels_work(matrix_layout, trans, m, n, nrhs, a, lda, b, ldb, work.as_mut_ptr(), lwork)
}

/// LAPACKE_zgels - Solves overdetermined/underdetermined linear systems (double complex).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_zgels(
    matrix_layout: i32,
    trans: u8,
    m: lapack_int,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut Complex64,
    lda: lapack_int,
    b: *mut Complex64,
    ldb: lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_zgels", -1);
        return -1;
    }

    // Query optimal workspace size
    let mut work_query = Complex64::new(0.0, 0.0);
    let info = LAPACKE_zgels_work(matrix_layout, trans, m, n, nrhs, a, lda, b, ldb, &mut work_query, -1);
    if info != 0 {
        return info;
    }

    let lwork = work_query.re as lapack_int;
    let mut work = match alloc_c64(lwork as usize) {
        Some(v) => v,
        None => {
            lapacke_xerbla_internal("LAPACKE_zgels", LAPACK_WORK_MEMORY_ERROR);
            return LAPACK_WORK_MEMORY_ERROR;
        }
    };

    LAPACKE_zgels_work(matrix_layout, trans, m, n, nrhs, a, lda, b, ldb, work.as_mut_ptr(), lwork)
}

// =============================================================================
// LAPACKE work interface (following OpenBLAS exactly)
// =============================================================================

/// LAPACKE_sgels_work - Work interface for least squares (single precision).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_sgels_work(
    matrix_layout: i32,
    trans: u8,
    m: lapack_int,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut f32,
    lda: lapack_int,
    b: *mut f32,
    ldb: lapack_int,
    work: *mut f32,
    lwork: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;
    let trans_char = trans as c_char;

    if matrix_layout == LAPACK_COL_MAJOR {
        let sgels = get_sgels();
        sgels(&trans_char, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, m);
        let ldb_t = lapack_max(1, lapack_max(m, n));

        if lda < n {
            info = -7;
            lapacke_xerbla_internal("LAPACKE_sgels_work", info);
            return info;
        }
        if ldb < nrhs {
            info = -9;
            lapacke_xerbla_internal("LAPACKE_sgels_work", info);
            return info;
        }

        // Workspace query
        if lwork == -1 {
            let sgels = get_sgels();
            sgels(&trans_char, &m, &n, &nrhs, a, &lda_t, b, &ldb_t, work, &lwork, &mut info);
            return if info < 0 { info - 1 } else { info };
        }

        let mut a_t = match alloc_f32((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_sgels_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        let mut b_t = match alloc_f32((ldb_t * lapack_max(1, nrhs)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_sgels_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_sge_trans(matrix_layout, m, n, a, lda, a_t.as_mut_ptr(), lda_t);
        lapacke_sge_trans(matrix_layout, lapack_max(m, n), nrhs, b, ldb, b_t.as_mut_ptr(), ldb_t);

        let sgels = get_sgels();
        sgels(&trans_char, &m, &n, &nrhs, a_t.as_mut_ptr(), &lda_t, b_t.as_mut_ptr(), &ldb_t, work, &lwork, &mut info);
        if info < 0 {
            info -= 1;
        }

        lapacke_sge_trans(LAPACK_COL_MAJOR, m, n, a_t.as_ptr(), lda_t, a, lda);
        lapacke_sge_trans(LAPACK_COL_MAJOR, lapack_max(m, n), nrhs, b_t.as_ptr(), ldb_t, b, ldb);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_sgels_work", info);
    }

    info
}

/// LAPACKE_dgels_work - Work interface for least squares (double precision).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgels_work(
    matrix_layout: i32,
    trans: u8,
    m: lapack_int,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut f64,
    lda: lapack_int,
    b: *mut f64,
    ldb: lapack_int,
    work: *mut f64,
    lwork: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;
    let trans_char = trans as c_char;

    if matrix_layout == LAPACK_COL_MAJOR {
        let dgels = get_dgels();
        dgels(&trans_char, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, m);
        let ldb_t = lapack_max(1, lapack_max(m, n));

        if lda < n {
            info = -7;
            lapacke_xerbla_internal("LAPACKE_dgels_work", info);
            return info;
        }
        if ldb < nrhs {
            info = -9;
            lapacke_xerbla_internal("LAPACKE_dgels_work", info);
            return info;
        }

        // Workspace query
        if lwork == -1 {
            let dgels = get_dgels();
            dgels(&trans_char, &m, &n, &nrhs, a, &lda_t, b, &ldb_t, work, &lwork, &mut info);
            return if info < 0 { info - 1 } else { info };
        }

        let mut a_t = match alloc_f64((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_dgels_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        let mut b_t = match alloc_f64((ldb_t * lapack_max(1, nrhs)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_dgels_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_dge_trans(matrix_layout, m, n, a, lda, a_t.as_mut_ptr(), lda_t);
        lapacke_dge_trans(matrix_layout, lapack_max(m, n), nrhs, b, ldb, b_t.as_mut_ptr(), ldb_t);

        let dgels = get_dgels();
        dgels(&trans_char, &m, &n, &nrhs, a_t.as_mut_ptr(), &lda_t, b_t.as_mut_ptr(), &ldb_t, work, &lwork, &mut info);
        if info < 0 {
            info -= 1;
        }

        lapacke_dge_trans(LAPACK_COL_MAJOR, m, n, a_t.as_ptr(), lda_t, a, lda);
        lapacke_dge_trans(LAPACK_COL_MAJOR, lapack_max(m, n), nrhs, b_t.as_ptr(), ldb_t, b, ldb);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_dgels_work", info);
    }

    info
}

/// LAPACKE_cgels_work - Work interface for least squares (single complex).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_cgels_work(
    matrix_layout: i32,
    trans: u8,
    m: lapack_int,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut Complex32,
    lda: lapack_int,
    b: *mut Complex32,
    ldb: lapack_int,
    work: *mut Complex32,
    lwork: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;
    let trans_char = trans as c_char;

    if matrix_layout == LAPACK_COL_MAJOR {
        let cgels = get_cgels();
        cgels(&trans_char, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, m);
        let ldb_t = lapack_max(1, lapack_max(m, n));

        if lda < n {
            info = -7;
            lapacke_xerbla_internal("LAPACKE_cgels_work", info);
            return info;
        }
        if ldb < nrhs {
            info = -9;
            lapacke_xerbla_internal("LAPACKE_cgels_work", info);
            return info;
        }

        // Workspace query
        if lwork == -1 {
            let cgels = get_cgels();
            cgels(&trans_char, &m, &n, &nrhs, a, &lda_t, b, &ldb_t, work, &lwork, &mut info);
            return if info < 0 { info - 1 } else { info };
        }

        let mut a_t = match alloc_c32((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_cgels_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        let mut b_t = match alloc_c32((ldb_t * lapack_max(1, nrhs)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_cgels_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_cge_trans(matrix_layout, m, n, a, lda, a_t.as_mut_ptr(), lda_t);
        lapacke_cge_trans(matrix_layout, lapack_max(m, n), nrhs, b, ldb, b_t.as_mut_ptr(), ldb_t);

        let cgels = get_cgels();
        cgels(&trans_char, &m, &n, &nrhs, a_t.as_mut_ptr(), &lda_t, b_t.as_mut_ptr(), &ldb_t, work, &lwork, &mut info);
        if info < 0 {
            info -= 1;
        }

        lapacke_cge_trans(LAPACK_COL_MAJOR, m, n, a_t.as_ptr(), lda_t, a, lda);
        lapacke_cge_trans(LAPACK_COL_MAJOR, lapack_max(m, n), nrhs, b_t.as_ptr(), ldb_t, b, ldb);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_cgels_work", info);
    }

    info
}

/// LAPACKE_zgels_work - Work interface for least squares (double complex).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_zgels_work(
    matrix_layout: i32,
    trans: u8,
    m: lapack_int,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut Complex64,
    lda: lapack_int,
    b: *mut Complex64,
    ldb: lapack_int,
    work: *mut Complex64,
    lwork: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;
    let trans_char = trans as c_char;

    if matrix_layout == LAPACK_COL_MAJOR {
        let zgels = get_zgels();
        zgels(&trans_char, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, m);
        let ldb_t = lapack_max(1, lapack_max(m, n));

        if lda < n {
            info = -7;
            lapacke_xerbla_internal("LAPACKE_zgels_work", info);
            return info;
        }
        if ldb < nrhs {
            info = -9;
            lapacke_xerbla_internal("LAPACKE_zgels_work", info);
            return info;
        }

        // Workspace query
        if lwork == -1 {
            let zgels = get_zgels();
            zgels(&trans_char, &m, &n, &nrhs, a, &lda_t, b, &ldb_t, work, &lwork, &mut info);
            return if info < 0 { info - 1 } else { info };
        }

        let mut a_t = match alloc_c64((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_zgels_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        let mut b_t = match alloc_c64((ldb_t * lapack_max(1, nrhs)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_zgels_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_zge_trans(matrix_layout, m, n, a, lda, a_t.as_mut_ptr(), lda_t);
        lapacke_zge_trans(matrix_layout, lapack_max(m, n), nrhs, b, ldb, b_t.as_mut_ptr(), ldb_t);

        let zgels = get_zgels();
        zgels(&trans_char, &m, &n, &nrhs, a_t.as_mut_ptr(), &lda_t, b_t.as_mut_ptr(), &ldb_t, work, &lwork, &mut info);
        if info < 0 {
            info -= 1;
        }

        lapacke_zge_trans(LAPACK_COL_MAJOR, m, n, a_t.as_ptr(), lda_t, a, lda);
        lapacke_zge_trans(LAPACK_COL_MAJOR, lapack_max(m, n), nrhs, b_t.as_ptr(), ldb_t, b, ldb);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_zgels_work", info);
    }

    info
}
