//! Tests for QR factorization (GEQRF, ORGQR)

extern crate blas_src;
extern crate lapack_sys;
extern crate openblas_src;

use lapack_inject::*;

fn register_dgeqrf_from_lapack() {
    extern "C" {
        fn dgeqrf_(
            m: *const i32,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            tau: *mut f64,
            work: *mut f64,
            lwork: *const i32,
            info: *mut i32,
        );
    }
    unsafe {
        register_dgeqrf(std::mem::transmute(dgeqrf_ as *const ()));
    }
}

fn register_dorgqr_from_lapack() {
    extern "C" {
        fn dorgqr_(
            m: *const i32,
            n: *const i32,
            k: *const i32,
            a: *mut f64,
            lda: *const i32,
            tau: *const f64,
            work: *mut f64,
            lwork: *const i32,
            info: *mut i32,
        );
    }
    unsafe {
        register_dorgqr(std::mem::transmute(dorgqr_ as *const ()));
    }
}

#[test]
fn test_dgeqrf() {
    register_dgeqrf_from_lapack();

    // QR factorization of A = [[1, 2], [3, 4], [5, 6]]
    let m = 3i32;
    let n = 2i32;
    let mut a = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // column-major
    let lda = 3i32;
    let mut tau = vec![0.0f64; 2];
    let mut info = 0i32;

    // Query workspace size
    let lwork = -1i32;
    let mut work_query = vec![0.0f64; 1];
    unsafe {
        lapack_dgeqrf(m, n, a.as_mut_ptr(), lda, tau.as_mut_ptr(), work_query.as_mut_ptr(), lwork, &mut info);
    }
    assert_eq!(info, 0, "DGEQRF workspace query should succeed");

    let lwork = work_query[0] as i32;
    let mut work = vec![0.0f64; lwork as usize];

    // Perform QR factorization
    unsafe {
        lapack_dgeqrf(m, n, a.as_mut_ptr(), lda, tau.as_mut_ptr(), work.as_mut_ptr(), lwork, &mut info);
    }
    assert_eq!(info, 0, "DGEQRF should succeed");
}

#[test]
fn test_dorgqr() {
    register_dgeqrf_from_lapack();
    register_dorgqr_from_lapack();

    // QR factorization of A = [[1, 2], [3, 4], [5, 6]]
    let m = 3i32;
    let n = 2i32;
    let k = 2i32;
    let mut a = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // column-major
    let lda = 3i32;
    let mut tau = vec![0.0f64; 2];
    let mut info = 0i32;

    // Perform QR factorization first
    let lwork = -1i32;
    let mut work_query = vec![0.0f64; 1];
    unsafe {
        lapack_dgeqrf(m, n, a.as_mut_ptr(), lda, tau.as_mut_ptr(), work_query.as_mut_ptr(), lwork, &mut info);
    }
    let lwork = work_query[0] as i32;
    let mut work = vec![0.0f64; lwork as usize];
    unsafe {
        lapack_dgeqrf(m, n, a.as_mut_ptr(), lda, tau.as_mut_ptr(), work.as_mut_ptr(), lwork, &mut info);
    }
    assert_eq!(info, 0, "DGEQRF should succeed");

    // Generate Q
    let lwork = -1i32;
    unsafe {
        lapack_dorgqr(m, n, k, a.as_mut_ptr(), lda, tau.as_ptr(), work_query.as_mut_ptr(), lwork, &mut info);
    }
    let lwork = work_query[0] as i32;
    let mut work = vec![0.0f64; lwork as usize];
    unsafe {
        lapack_dorgqr(m, n, k, a.as_mut_ptr(), lda, tau.as_ptr(), work.as_mut_ptr(), lwork, &mut info);
    }
    assert_eq!(info, 0, "DORGQR should succeed");

    // Check that Q is orthonormal (Q^T * Q = I)
    // First column
    let q1_norm = (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]).sqrt();
    assert!((q1_norm - 1.0).abs() < 1e-10, "First column of Q should have norm 1");
}
