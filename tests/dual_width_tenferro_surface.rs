extern crate lapack_inject;

#[cfg(not(feature = "ilp64"))]
use core::ptr::{read_unaligned, write_unaligned};
#[cfg(not(feature = "ilp64"))]
use lapack_inject::*;
#[cfg(not(feature = "ilp64"))]
use num_complex::Complex64;
#[cfg(not(feature = "ilp64"))]
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
            &m,
            &n,
            a.as_mut_ptr(),
            &lda,
            tau.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            &mut info,
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
            &m,
            &n,
            &k,
            a.as_mut_ptr(),
            &lda,
            tau.as_ptr(),
            work.as_mut_ptr(),
            &lwork,
            &mut info,
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
            &uplo,
            &trans,
            &diag,
            &n,
            &nrhs,
            a.as_ptr(),
            &lda,
            b.as_mut_ptr(),
            &ldb,
            &mut info,
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
            &jobz,
            &uplo,
            &n,
            a.as_mut_ptr() as *mut lapack_complex_double,
            &lda,
            w.as_mut_ptr(),
            work.as_mut_ptr() as *mut lapack_complex_double,
            &lwork,
            rwork.as_mut_ptr(),
            &mut info,
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
    _vl: *mut f64,
    _ldvl: *const i64,
    _vr: *mut f64,
    _ldvr: *const i64,
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
            &jobvl,
            &jobvr,
            &n,
            a.as_mut_ptr(),
            &lda,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vl.as_mut_ptr(),
            &ldvl,
            vr.as_mut_ptr(),
            &ldvr,
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    assert_eq!(info, 0);
    assert_eq!(wr[0], 4.0);
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn fake_dgetrf_ilp64(
    m: *const i64,
    n: *const i64,
    _a: *mut f64,
    lda: *const i64,
    ipiv: *mut i64,
    info: *mut i64,
) {
    assert_eq!(read_unaligned(m), 3);
    assert_eq!(read_unaligned(n), 3);
    assert_eq!(read_unaligned(lda), 3);
    // Write pivot array [1, 3, 2]
    *ipiv.add(0) = 1i64;
    *ipiv.add(1) = 3i64;
    *ipiv.add(2) = 2i64;
    write_unaligned(info, 0);
}

#[cfg(not(feature = "ilp64"))]
#[test]
fn lp64_dgetrf_symbol_bridges_ipiv_array_to_ilp64_provider() {
    unsafe {
        assert!(matches!(register_dgetrf_ilp64(fake_dgetrf_ilp64), 0 | 2));
    }
    let m = 3_i32;
    let n = 3_i32;
    let mut a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let lda = 3_i32;
    let mut ipiv = [0_i32; 3];
    let mut info = -1_i32;
    unsafe {
        lapack_sys::dgetrf_(&m, &n, a.as_mut_ptr(), &lda, ipiv.as_mut_ptr(), &mut info);
    }
    assert_eq!(info, 0);
    assert_eq!(ipiv, [1, 3, 2], "all pivot elements should be bridged back");
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn fake_dgetc2_ilp64(
    n: *const i64,
    _a: *mut f64,
    lda: *const i64,
    ipiv: *mut i64,
    jpiv: *mut i64,
    info: *mut i64,
) {
    assert_eq!(read_unaligned(n), 3);
    assert_eq!(read_unaligned(lda), 3);
    *ipiv.add(0) = 3i64;
    *ipiv.add(1) = 1i64;
    *ipiv.add(2) = 2i64;
    *jpiv.add(0) = 2i64;
    *jpiv.add(1) = 3i64;
    *jpiv.add(2) = 1i64;
    write_unaligned(info, 0);
}

#[cfg(not(feature = "ilp64"))]
#[test]
fn lp64_dgetc2_symbol_bridges_ipiv_jpiv_arrays_to_ilp64_provider() {
    extern "C" {
        fn dgetc2_(
            n: *const lapackint,
            a: *mut f64,
            lda: *const lapackint,
            ipiv: *mut lapackint,
            jpiv: *mut lapackint,
            info: *mut lapackint,
        );
    }
    unsafe {
        assert!(matches!(register_dgetc2_ilp64(fake_dgetc2_ilp64), 0 | 2));
    }
    let n: lapackint = 3;
    let lda: lapackint = 3;
    let mut a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut ipiv: [lapackint; 3] = [0; 3];
    let mut jpiv: [lapackint; 3] = [0; 3];
    let mut info: lapackint = -1;
    unsafe {
        dgetc2_(
            &n,
            a.as_mut_ptr(),
            &lda,
            ipiv.as_mut_ptr(),
            jpiv.as_mut_ptr(),
            &mut info,
        );
    }
    assert_eq!(info, 0);
    assert_eq!(ipiv, [3, 1, 2], "all ipiv elements bridged back");
    assert_eq!(jpiv, [2, 3, 1], "all jpiv elements bridged back");
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn fake_dgesc2_ilp64(
    n: *const i64,
    a: *const f64,
    lda: *const i64,
    rhs: *mut f64,
    ipiv: *const i64,
    jpiv: *const i64,
    scale: *mut f64,
) {
    assert_eq!(read_unaligned(n), 3);
    assert_eq!(read_unaligned(lda), 3);
    assert_eq!(*ipiv.add(0), 3i64, "ipiv[0] bridged");
    assert_eq!(*ipiv.add(1), 1i64, "ipiv[1] bridged");
    assert_eq!(*ipiv.add(2), 2i64, "ipiv[2] bridged");
    assert_eq!(*jpiv.add(0), 2i64, "jpiv[0] bridged");
    assert_eq!(*jpiv.add(1), 3i64, "jpiv[1] bridged");
    assert_eq!(*jpiv.add(2), 1i64, "jpiv[2] bridged");
    *rhs = *a;
    *scale = 1.0;
}

#[cfg(not(feature = "ilp64"))]
#[test]
fn lp64_dgesc2_symbol_bridges_ipiv_jpiv_arrays_to_ilp64_provider() {
    extern "C" {
        fn dgesc2_(
            n: *const lapackint,
            a: *const f64,
            lda: *const lapackint,
            rhs: *mut f64,
            ipiv: *const lapackint,
            jpiv: *const lapackint,
            scale: *mut f64,
        );
    }
    unsafe {
        assert!(matches!(register_dgesc2_ilp64(fake_dgesc2_ilp64), 0 | 2));
    }
    let n: lapackint = 3;
    let lda: lapackint = 3;
    let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut rhs = [0.0_f64];
    let ipiv: [lapackint; 3] = [3, 1, 2];
    let jpiv: [lapackint; 3] = [2, 3, 1];
    let mut scale = 0.0_f64;
    unsafe {
        dgesc2_(
            &n,
            a.as_ptr(),
            &lda,
            rhs.as_mut_ptr(),
            ipiv.as_ptr(),
            jpiv.as_ptr(),
            &mut scale,
        );
    }
    assert_eq!(rhs[0], 1.0);
    assert_eq!(scale, 1.0);
}
