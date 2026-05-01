//! Tests for the LAPACKE-compatible C entry points.

extern crate lapack_inject;

use lapack_inject::*;
use std::ffi::c_char;

#[cfg(not(feature = "ilp64"))]
type DgesvCurrentFnPtr = DgesvLp64FnPtr;
#[cfg(feature = "ilp64")]
type DgesvCurrentFnPtr = DgesvIlp64FnPtr;
#[cfg(not(feature = "ilp64"))]
type DgetrfCurrentFnPtr = DgetrfLp64FnPtr;
#[cfg(feature = "ilp64")]
type DgetrfCurrentFnPtr = DgetrfIlp64FnPtr;
#[cfg(not(feature = "ilp64"))]
type DgetriCurrentFnPtr = DgetriLp64FnPtr;
#[cfg(feature = "ilp64")]
type DgetriCurrentFnPtr = DgetriIlp64FnPtr;
#[cfg(not(feature = "ilp64"))]
type DpotrfCurrentFnPtr = DpotrfLp64FnPtr;
#[cfg(feature = "ilp64")]
type DpotrfCurrentFnPtr = DpotrfIlp64FnPtr;

unsafe fn register_dgesv_current(f: DgesvCurrentFnPtr) {
    #[cfg(not(feature = "ilp64"))]
    let status = register_dgesv_lp64(f);
    #[cfg(feature = "ilp64")]
    let status = register_dgesv_ilp64(f);
    assert!(matches!(status, 0 | 2));
}

unsafe fn register_dgetrf_current(f: DgetrfCurrentFnPtr) {
    #[cfg(not(feature = "ilp64"))]
    let status = register_dgetrf_lp64(f);
    #[cfg(feature = "ilp64")]
    let status = register_dgetrf_ilp64(f);
    assert!(matches!(status, 0 | 2));
}

unsafe fn register_dgetri_current(f: DgetriCurrentFnPtr) {
    #[cfg(not(feature = "ilp64"))]
    let status = register_dgetri_lp64(f);
    #[cfg(feature = "ilp64")]
    let status = register_dgetri_ilp64(f);
    assert!(matches!(status, 0 | 2));
}

unsafe fn register_dpotrf_current(f: DpotrfCurrentFnPtr) {
    #[cfg(not(feature = "ilp64"))]
    let status = register_dpotrf_lp64(f);
    #[cfg(feature = "ilp64")]
    let status = register_dpotrf_ilp64(f);
    assert!(matches!(status, 0 | 2));
}

unsafe extern "C" fn fake_dgesv(
    n: *const lapackint,
    nrhs: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    b: *mut f64,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    assert_eq!(*n, 2);
    assert_eq!(*nrhs, 1);
    assert_eq!(*lda, 2);
    assert_eq!(*ldb, 2);
    assert_eq!(std::slice::from_raw_parts(a, 4), &[1.0, 3.0, 2.0, 4.0]);
    assert_eq!(std::slice::from_raw_parts(b, 2), &[5.0, 11.0]);

    *ipiv.add(0) = 1;
    *ipiv.add(1) = 2;
    *b.add(0) = 1.0;
    *b.add(1) = 2.0;
    *info = 0;
}

unsafe extern "C" fn fake_dgetrf(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    info: *mut lapackint,
) {
    assert_eq!(*m, 2);
    assert_eq!(*n, 2);
    assert_eq!(*lda, 2);
    assert_eq!(std::slice::from_raw_parts(a, 4), &[1.0, 3.0, 2.0, 4.0]);

    std::slice::from_raw_parts_mut(a, 4).copy_from_slice(&[10.0, 30.0, 20.0, 40.0]);
    *ipiv.add(0) = 2;
    *ipiv.add(1) = 1;
    *info = 0;
}

unsafe extern "C" fn fake_dgetri(
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    ipiv: *const lapackint,
    _work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    assert_eq!(*n, 2);
    assert_eq!(*lda, 2);
    assert!(*lwork >= 2);
    assert_eq!(std::slice::from_raw_parts(a, 4), &[1.0, 3.0, 2.0, 4.0]);
    assert_eq!(std::slice::from_raw_parts(ipiv, 2), &[2, 1]);

    std::slice::from_raw_parts_mut(a, 4).copy_from_slice(&[-2.0, 1.5, 1.0, -0.5]);
    *info = 0;
}

unsafe extern "C" fn fake_dpotrf(
    uplo: *const c_char,
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    info: *mut lapackint,
) {
    assert_eq!(*uplo as u8, b'L');
    assert_eq!(*n, 2);
    assert_eq!(*lda, 2);
    assert_eq!(std::slice::from_raw_parts(a, 4), &[4.0, 2.0, 2.0, 5.0]);

    std::slice::from_raw_parts_mut(a, 4).copy_from_slice(&[2.0, 1.0, 0.0, 2.0]);
    *info = 0;
}

#[test]
fn lapacke_dgesv_row_major_transposes_inputs_and_solution() {
    unsafe {
        register_dgesv_current(fake_dgesv);
    }

    let mut a = [1.0, 2.0, 3.0, 4.0];
    let mut b = [5.0, 11.0];
    let mut ipiv = [0; 2];

    let info = unsafe {
        LAPACKE_dgesv(
            LAPACK_ROW_MAJOR,
            2,
            1,
            a.as_mut_ptr(),
            2,
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            1,
        )
    };

    assert_eq!(info, 0);
    assert_eq!(ipiv, [1, 2]);
    assert_eq!(b, [1.0, 2.0]);
}

#[test]
fn lapacke_dgetrf_row_major_transposes_factorization_back() {
    unsafe {
        register_dgetrf_current(fake_dgetrf);
    }

    let mut a = [1.0, 2.0, 3.0, 4.0];
    let mut ipiv = [0; 2];

    let info =
        unsafe { LAPACKE_dgetrf(LAPACK_ROW_MAJOR, 2, 2, a.as_mut_ptr(), 2, ipiv.as_mut_ptr()) };

    assert_eq!(info, 0);
    assert_eq!(ipiv, [2, 1]);
    assert_eq!(a, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn lapacke_dgetri_row_major_transposes_inverse_back() {
    unsafe {
        register_dgetri_current(fake_dgetri);
    }

    let mut a = [1.0, 2.0, 3.0, 4.0];
    let ipiv = [2, 1];

    let info = unsafe { LAPACKE_dgetri(LAPACK_ROW_MAJOR, 2, a.as_mut_ptr(), 2, ipiv.as_ptr()) };

    assert_eq!(info, 0);
    assert_eq!(a, [-2.0, 1.0, 1.5, -0.5]);
}

#[test]
fn lapacke_dpotrf_row_major_transposes_factor_back() {
    unsafe {
        register_dpotrf_current(fake_dpotrf);
    }

    let mut a = [4.0, 2.0, 2.0, 5.0];

    let info = unsafe { LAPACKE_dpotrf(LAPACK_ROW_MAJOR, b'L' as c_char, 2, a.as_mut_ptr(), 2) };

    assert_eq!(info, 0);
    assert_eq!(a, [2.0, 0.0, 1.0, 2.0]);
}

#[test]
fn lapacke_rejects_unknown_layout() {
    let mut a = [0.0; 4];
    let mut ipiv = [0; 2];

    let info = unsafe { LAPACKE_dgetrf(999, 2, 2, a.as_mut_ptr(), 2, ipiv.as_mut_ptr()) };

    assert_eq!(info, -1);
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn fake_dgetrf_ilp64(
    m: *const i64,
    n: *const i64,
    a: *mut f64,
    lda: *const i64,
    ipiv: *mut i64,
    info: *mut i64,
) {
    assert_eq!(*m, 2);
    assert_eq!(*n, 2);
    assert_eq!(*lda, 2);
    assert_eq!(std::slice::from_raw_parts(a, 4), &[1.0, 3.0, 2.0, 4.0]);

    std::slice::from_raw_parts_mut(a, 4).copy_from_slice(&[10.0, 30.0, 20.0, 40.0]);
    *ipiv.add(0) = 2;
    *ipiv.add(1) = 1;
    *info = 0;
}

#[cfg(not(feature = "ilp64"))]
#[test]
fn lapacke_dgetrf_64_uses_ilp64_provider_without_ilp64_feature() {
    unsafe {
        assert!(matches!(register_dgetrf_ilp64(fake_dgetrf_ilp64), 0 | 2));
    }

    let mut a = [1.0, 2.0, 3.0, 4.0];
    let mut ipiv = [0_i64; 2];

    let info = unsafe {
        LAPACKE_dgetrf_64(
            LAPACK_ROW_MAJOR as i64,
            2,
            2,
            a.as_mut_ptr(),
            2,
            ipiv.as_mut_ptr(),
        )
    };

    assert_eq!(info, 0);
    assert_eq!(ipiv, [2, 1]);
    assert_eq!(a, [10.0, 20.0, 30.0, 40.0]);
}
