//! Tests for GESV (linear solve)

extern crate blas_src;
extern crate lapack_sys;
extern crate openblas_src;

use lapack_inject::*;
use num_complex::Complex64;

fn register_dgesv_from_lapack() {
    extern "C" {
        fn dgesv_(
            n: *const i32,
            nrhs: *const i32,
            a: *mut f64,
            lda: *const i32,
            ipiv: *mut i32,
            b: *mut f64,
            ldb: *const i32,
            info: *mut i32,
        );
    }
    unsafe {
        register_dgesv(std::mem::transmute(dgesv_ as *const ()));
    }
}

fn register_zgesv_from_lapack() {
    extern "C" {
        fn zgesv_(
            n: *const i32,
            nrhs: *const i32,
            a: *mut Complex64,
            lda: *const i32,
            ipiv: *mut i32,
            b: *mut Complex64,
            ldb: *const i32,
            info: *mut i32,
        );
    }
    unsafe {
        register_zgesv(std::mem::transmute(zgesv_ as *const ()));
    }
}

#[test]
fn test_dgesv() {
    register_dgesv_from_lapack();

    // Solve A * X = B where A = [[1, 2], [3, 4]], B = [[5], [11]]
    // Solution: X = [[1], [2]]
    let n = 2i32;
    let nrhs = 1i32;
    let mut a = vec![1.0, 3.0, 2.0, 4.0]; // column-major
    let lda = 2i32;
    let mut ipiv = vec![0i32; 2];
    let mut b = vec![5.0, 11.0];
    let ldb = 2i32;
    let mut info = 0i32;

    unsafe {
        lapack_dgesv(n, nrhs, a.as_mut_ptr(), lda, ipiv.as_mut_ptr(), b.as_mut_ptr(), ldb, &mut info);
    }

    assert_eq!(info, 0, "DGESV should succeed");
    assert!((b[0] - 1.0).abs() < 1e-10, "x[0] should be 1.0, got {}", b[0]);
    assert!((b[1] - 2.0).abs() < 1e-10, "x[1] should be 2.0, got {}", b[1]);
}

#[test]
fn test_zgesv() {
    register_zgesv_from_lapack();

    // Solve A * X = B with complex numbers
    let n = 2i32;
    let nrhs = 1i32;
    let mut a = vec![
        Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0),
        Complex64::new(2.0, 0.0), Complex64::new(4.0, 0.0),
    ]; // column-major
    let lda = 2i32;
    let mut ipiv = vec![0i32; 2];
    let mut b = vec![Complex64::new(5.0, 0.0), Complex64::new(11.0, 0.0)];
    let ldb = 2i32;
    let mut info = 0i32;

    unsafe {
        lapack_zgesv(n, nrhs, a.as_mut_ptr(), lda, ipiv.as_mut_ptr(), b.as_mut_ptr(), ldb, &mut info);
    }

    assert_eq!(info, 0, "ZGESV should succeed");
    assert!((b[0].re - 1.0).abs() < 1e-10, "x[0] real should be 1.0");
    assert!((b[1].re - 2.0).abs() < 1e-10, "x[1] real should be 2.0");
}
