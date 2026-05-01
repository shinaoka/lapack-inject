extern crate lapack_inject;

use core::ptr::{read_unaligned, write_unaligned};
use lapack_inject::*;
use num_complex::Complex64;
use std::ffi::c_char;

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn fake_dgeqrf_ilp64(
    m: *const i64,
    n: *const i64,
    _a: *mut f64,
    lda: *const i64,
    tau: *mut f64,
    work: *mut f64,
    lwork: *const i64,
    info: *mut i64,
) {
    assert_eq!(read_unaligned(m), 3);
    assert_eq!(read_unaligned(n), 2);
    assert_eq!(read_unaligned(lda), 3);
    assert_eq!(read_unaligned(lwork), -1);
    *work = 128.0;
    *tau = 1.0;
    write_unaligned(info, 0);
}

#[cfg(not(feature = "ilp64"))]
#[test]
fn lp64_dgeqrf_symbol_dispatches_to_ilp64_provider() {
    unsafe {
        assert!(matches!(register_dgeqrf_ilp64(fake_dgeqrf_ilp64), 0 | 2));
    }
    let m = 3_i32;
    let n = 2_i32;
    let mut a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let lda = 3_i32;
    let mut tau = [0.0_f64];
    let lwork = -1_i32;
    let mut work = [0.0_f64];
    let mut info = -1_i32;
    unsafe {
        lapack_sys::dgeqrf_(
            &m, &n, a.as_mut_ptr(), &lda, tau.as_mut_ptr(),
            work.as_mut_ptr(), &lwork, &mut info,
        );
    }
    assert_eq!(info, 0);
    assert_eq!(work[0], 128.0);
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn fake_dorgqr_ilp64(
    m: *const i64,
    n: *const i64,
    k: *const i64,
    _a: *mut f64,
    lda: *const i64,
    tau: *const f64,
    work: *mut f64,
    lwork: *const i64,
    info: *mut i64,
) {
    assert_eq!(read_unaligned(m), 3);
    assert_eq!(read_unaligned(n), 2);
    assert_eq!(read_unaligned(k), 1);
    assert_eq!(read_unaligned(lda), 3);
    assert_eq!(read_unaligned(lwork), -1);
    assert_eq!(*tau, 0.5);
    *work = 64.0;
    write_unaligned(info, 0);
}

#[cfg(not(feature = "ilp64"))]
#[test]
fn lp64_dorgqr_symbol_dispatches_to_ilp64_provider() {
    unsafe {
        assert!(matches!(register_dorgqr_ilp64(fake_dorgqr_ilp64), 0 | 2));
    }
    let m = 3_i32;
    let n = 2_i32;
    let k = 1_i32;
    let mut a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let lda = 3_i32;
    let tau = [0.5_f64];
    let lwork = -1_i32;
    let mut work = [0.0_f64];
    let mut info = -1_i32;
    unsafe {
        lapack_sys::dorgqr_(
            &m, &n, &k, a.as_mut_ptr(), &lda, tau.as_ptr(),
            work.as_mut_ptr(), &lwork, &mut info,
        );
    }
    assert_eq!(info, 0);
    assert_eq!(work[0], 64.0);
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn fake_dtrtrs_ilp64(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const i64,
    nrhs: *const i64,
    _a: *const f64,
    lda: *const i64,
    b: *mut f64,
    ldb: *const i64,
    info: *mut i64,
) {
    assert_eq!(*uplo as u8, b'U');
    assert_eq!(*trans as u8, b'N');
    assert_eq!(*diag as u8, b'N');
    assert_eq!(read_unaligned(n), 2);
    assert_eq!(read_unaligned(nrhs), 1);
    assert_eq!(read_unaligned(lda), 2);
    assert_eq!(read_unaligned(ldb), 2);
    std::slice::from_raw_parts_mut(b, 2).copy_from_slice(&[11.0, 13.0]);
    write_unaligned(info, 0);
}

#[cfg(not(feature = "ilp64"))]
#[test]
fn lp64_dtrtrs_symbol_dispatches_to_ilp64_provider() {
    unsafe {
        assert!(matches!(register_dtrtrs_ilp64(fake_dtrtrs_ilp64), 0 | 2));
    }
    let uplo = b'U' as c_char;
    let trans = b'N' as c_char;
    let diag = b'N' as c_char;
    let n = 2_i32;
    let nrhs = 1_i32;
    let a = [1.0, 0.0, 0.0, 1.0];
    let lda = 2_i32;
    let mut b = [1.0, 2.0];
    let ldb = 2_i32;
    let mut info = -1_i32;
    unsafe {
        lapack_sys::dtrtrs_(
            &uplo, &trans, &diag, &n, &nrhs, a.as_ptr(), &lda,
            b.as_mut_ptr(), &ldb, &mut info,
        );
    }
    assert_eq!(info, 0);
    assert_eq!(b, [11.0, 13.0]);
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn fake_zheev_ilp64(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const i64,
    _a: *mut Complex64,
    lda: *const i64,
    w: *mut f64,
    work: *mut Complex64,
    lwork: *const i64,
    rwork: *mut f64,
    info: *mut i64,
) {
    assert_eq!(*jobz as u8, b'N');
    assert_eq!(*uplo as u8, b'U');
    assert_eq!(read_unaligned(n), 2);
    assert_eq!(read_unaligned(lda), 2);
    assert_eq!(read_unaligned(lwork), -1);
    *w = 3.0;
    *work = Complex64::new(64.0, 0.0);
    *rwork = 1.0;
    write_unaligned(info, 0);
}

#[cfg(not(feature = "ilp64"))]
#[test]
fn lp64_zheev_symbol_dispatches_to_ilp64_provider() {
    use lapack_sys::lapack_complex_double;
    unsafe {
        assert!(matches!(register_zheev_ilp64(fake_zheev_ilp64), 0 | 2));
    }
    let jobz = b'N' as c_char;
    let uplo = b'U' as c_char;
    let n = 2_i32;
    let mut a = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let lda = 2_i32;
    let mut w = [0.0_f64; 2];
    let lwork = -1_i32;
    let mut work = [Complex64::new(0.0, 0.0)];
    let mut rwork = [0.0_f64];
    let mut info = -1_i32;
    unsafe {
        lapack_sys::zheev_(
            &jobz, &uplo, &n, a.as_mut_ptr() as *mut lapack_complex_double,
            &lda, w.as_mut_ptr(),
            work.as_mut_ptr() as *mut lapack_complex_double,
            &lwork, rwork.as_mut_ptr(), &mut info,
        );
    }
    assert_eq!(info, 0);
    assert_eq!(w[0], 3.0);
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn fake_dgeev_ilp64(
    jobvl: *const c_char,
    jobvr: *const c_char,
    n: *const i64,
    _a: *mut f64,
    lda: *const i64,
    wr: *mut f64,
    wi: *mut f64,
    vl: *mut f64,
    ldvl: *const i64,
    vr: *mut f64,
    ldvr: *const i64,
    work: *mut f64,
    lwork: *const i64,
    info: *mut i64,
) {
    assert_eq!(*jobvl as u8, b'N');
    assert_eq!(*jobvr as u8, b'N');
    assert_eq!(read_unaligned(n), 2);
    assert_eq!(read_unaligned(lda), 2);
    assert_eq!(read_unaligned(lwork), -1);
    *wr = 4.0;
    *wi = 0.0;
    *work = 128.0;
    write_unaligned(info, 0);
}

#[cfg(not(feature = "ilp64"))]
#[test]
fn lp64_dgeev_symbol_dispatches_to_ilp64_provider() {
    unsafe {
        assert!(matches!(register_dgeev_ilp64(fake_dgeev_ilp64), 0 | 2));
    }
    let jobvl = b'N' as c_char;
    let jobvr = b'N' as c_char;
    let n = 2_i32;
    let mut a = [1.0, 2.0, 3.0, 4.0];
    let lda = 2_i32;
    let mut wr = [0.0_f64; 2];
    let mut wi = [0.0_f64; 2];
    let mut vl = [0.0_f64; 1];
    let ldvl = 1_i32;
    let mut vr = [0.0_f64; 1];
    let ldvr = 1_i32;
    let lwork = -1_i32;
    let mut work = [0.0_f64];
    let mut info = -1_i32;
    unsafe {
        lapack_sys::dgeev_(
            &jobvl, &jobvr, &n, a.as_mut_ptr(), &lda,
            wr.as_mut_ptr(), wi.as_mut_ptr(),
            vl.as_mut_ptr(), &ldvl,
            vr.as_mut_ptr(), &ldvr,
            work.as_mut_ptr(), &lwork, &mut info,
        );
    }
    assert_eq!(info, 0);
    assert_eq!(wr[0], 4.0);
}
