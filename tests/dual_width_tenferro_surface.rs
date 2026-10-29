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

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn fake_dgeqp3_ilp64(
    m: *const i64,
    n: *const i64,
    _a: *mut f64,
    lda: *const i64,
    jpvt: *mut i64,
    _tau: *mut f64,
    work: *mut f64,
    lwork: *const i64,
    info: *mut i64,
) {
    assert_eq!(read_unaligned(m), 2);
    assert_eq!(read_unaligned(n), 3);
    assert_eq!(read_unaligned(lda), 2);
    assert_eq!(read_unaligned(lwork), 8);
    // JPVT is read on input as zero/nonzero flags (nonzero fixes a leading column).
    let jpvt = std::slice::from_raw_parts_mut(jpvt, 3);
    assert_eq!(jpvt, &[0, 1, 0]);
    jpvt.copy_from_slice(&[3, 1, 2]);
    *work = 8.0;
    write_unaligned(info, 0);
}

#[cfg(not(feature = "ilp64"))]
#[test]
fn lp64_dgeqp3_symbol_bridges_jpvt_both_ways_to_ilp64_provider() {
    unsafe {
        assert!(matches!(register_dgeqp3_ilp64(fake_dgeqp3_ilp64), 0 | 2));
    }
    let (m, n, lda, lwork) = (2_i32, 3_i32, 2_i32, 8_i32);
    let mut a = [0.0_f64; 6];
    let mut jpvt = [0_i32, 2, 0];
    let mut tau = [0.0_f64; 2];
    let mut work = [0.0_f64; 8];
    let mut info = -1_i32;
    unsafe {
        lapack_sys::dgeqp3_(
            &m,
            &n,
            a.as_mut_ptr(),
            &lda,
            jpvt.as_mut_ptr(),
            tau.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    assert_eq!(info, 0);
    assert_eq!(jpvt, [3, 1, 2]);
    assert_eq!(work[0], 8.0);
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn fake_dsyevd_ilp64(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const i64,
    _a: *mut f64,
    lda: *const i64,
    _w: *mut f64,
    work: *mut f64,
    lwork: *const i64,
    iwork: *mut i64,
    liwork: *const i64,
    info: *mut i64,
) {
    assert_eq!(*jobz as u8, b'V');
    assert_eq!(*uplo as u8, b'L');
    assert_eq!(read_unaligned(n), 2);
    assert_eq!(read_unaligned(lda), 2);
    if read_unaligned(liwork) == -1 {
        // Workspace query: the optimal sizes come back in work[0] and iwork[0].
        assert_eq!(read_unaligned(lwork), -1);
        *work = 9.0;
        write_unaligned(iwork, 13);
    } else {
        assert_eq!(read_unaligned(liwork), 3);
        std::slice::from_raw_parts_mut(iwork, 3).copy_from_slice(&[7, 8, 9]);
    }
    write_unaligned(info, 0);
}

#[cfg(not(feature = "ilp64"))]
#[test]
fn lp64_dsyevd_symbol_bridges_iwork_for_query_and_call() {
    unsafe {
        assert!(matches!(register_dsyevd_ilp64(fake_dsyevd_ilp64), 0 | 2));
    }
    let jobz = b'V' as c_char;
    let uplo = b'L' as c_char;
    let (n, lda) = (2_i32, 2_i32);
    let mut a = [0.0_f64; 4];
    let mut w = [0.0_f64; 2];
    let mut info = -1_i32;

    let mut work = [0.0_f64];
    let mut iwork = [0_i32];
    unsafe {
        lapack_sys::dsyevd_(
            &jobz,
            &uplo,
            &n,
            a.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &-1,
            iwork.as_mut_ptr(),
            &-1,
            &mut info,
        );
    }
    assert_eq!(info, 0);
    assert_eq!((work[0], iwork[0]), (9.0, 13));

    let mut work = [0.0_f64; 9];
    let mut iwork = [0_i32; 3];
    info = -1;
    unsafe {
        lapack_sys::dsyevd_(
            &jobz,
            &uplo,
            &n,
            a.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &9,
            iwork.as_mut_ptr(),
            &3,
            &mut info,
        );
    }
    assert_eq!(info, 0);
    assert_eq!(iwork, [7, 8, 9]);
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn fake_zunmqr_ilp64(
    side: *const c_char,
    trans: *const c_char,
    m: *const i64,
    n: *const i64,
    k: *const i64,
    _a: *const Complex64,
    lda: *const i64,
    _tau: *const Complex64,
    c: *mut Complex64,
    ldc: *const i64,
    _work: *mut Complex64,
    lwork: *const i64,
    info: *mut i64,
) {
    assert_eq!(*side as u8, b'L');
    assert_eq!(*trans as u8, b'C');
    assert_eq!(
        [m, n, k, lda, ldc, lwork].map(|p| read_unaligned(p)),
        [3, 1, 2, 3, 3, 4]
    );
    *c = Complex64::new(1.0, -1.0);
    write_unaligned(info, 0);
}

#[cfg(not(feature = "ilp64"))]
#[test]
fn lp64_zunmqr_symbol_dispatches_to_ilp64_provider() {
    unsafe {
        assert!(matches!(register_zunmqr_ilp64(fake_zunmqr_ilp64), 0 | 2));
    }
    let side = b'L' as c_char;
    let trans = b'C' as c_char;
    let a = [Complex64::new(0.0, 0.0); 6];
    let tau = [Complex64::new(0.0, 0.0); 2];
    let mut c = [Complex64::new(0.0, 0.0); 3];
    let mut work = [Complex64::new(0.0, 0.0); 4];
    let mut info = -1_i32;
    unsafe {
        lapack_sys::zunmqr_(
            &side,
            &trans,
            &3,
            &1,
            &2,
            a.as_ptr().cast(),
            &3,
            tau.as_ptr().cast(),
            c.as_mut_ptr().cast(),
            &3,
            work.as_mut_ptr().cast(),
            &4,
            &mut info,
        );
    }
    assert_eq!(info, 0);
    assert_eq!(c[0], Complex64::new(1.0, -1.0));
}

// ILP64 consumer -> LP64 provider: the narrowing direction of the integer-array bridges.

#[cfg(feature = "ilp64")]
mod ilp64_consumer {
    use core::ptr::read_unaligned;
    use lapack_inject::*;
    use std::ffi::c_char;
    use std::sync::atomic::{AtomicBool, Ordering};

    // With `ilp64`, lapack-inject exports the Fortran symbols with 64-bit integers.
    extern "C" {
        fn dgeqp3_(
            m: *const i64,
            n: *const i64,
            a: *mut f64,
            lda: *const i64,
            jpvt: *mut i64,
            tau: *mut f64,
            work: *mut f64,
            lwork: *const i64,
            info: *mut i64,
        );
        fn dsyevd_(
            jobz: *const c_char,
            uplo: *const c_char,
            n: *const i64,
            a: *mut f64,
            lda: *const i64,
            w: *mut f64,
            work: *mut f64,
            lwork: *const i64,
            iwork: *mut i64,
            liwork: *const i64,
            info: *mut i64,
        );
    }

    static GEQP3_QUERY_SEEN: AtomicBool = AtomicBool::new(false);

    unsafe extern "C" fn fake_dgeqp3_lp64(
        _m: *const i32,
        n: *const i32,
        _a: *mut f64,
        _lda: *const i32,
        jpvt: *mut i32,
        _tau: *mut f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    ) {
        if read_unaligned(lwork) == -1 {
            // LAPACK does not touch JPVT in a workspace query; neither may the bridge.
            GEQP3_QUERY_SEEN.store(true, Ordering::SeqCst);
            *work = 4.0;
        } else {
            assert_eq!(read_unaligned(n), 2);
            let jpvt = std::slice::from_raw_parts_mut(jpvt, 2);
            // 1 << 32 must stay a nonzero (fixed-column) flag after narrowing.
            assert_eq!(jpvt, &[1, 0]);
            jpvt.copy_from_slice(&[1, 2]);
        }
        *info = 0;
    }

    #[test]
    fn dgeqp3_narrows_jpvt_flags_and_skips_jpvt_on_query() {
        unsafe {
            assert!(matches!(register_dgeqp3_lp64(fake_dgeqp3_lp64), 0 | 2));
        }
        let (m, n, lda) = (2_i64, 2_i64, 2_i64);
        let mut a = [0.0_f64; 4];
        let mut tau = [0.0_f64; 2];
        let mut work = [0.0_f64; 4];
        let mut info = -1_i64;

        // The query passes a JPVT the bridge must not read or write.
        let mut sentinel = [i64::MIN; 2];
        unsafe {
            dgeqp3_(
                &m,
                &n,
                a.as_mut_ptr(),
                &lda,
                sentinel.as_mut_ptr(),
                tau.as_mut_ptr(),
                work.as_mut_ptr(),
                &-1,
                &mut info,
            );
        }
        assert!(GEQP3_QUERY_SEEN.load(Ordering::SeqCst));
        assert_eq!((info, work[0], sentinel), (0, 4.0, [i64::MIN; 2]));

        let mut jpvt = [1_i64 << 32, 0];
        info = -1;
        unsafe {
            dgeqp3_(
                &m,
                &n,
                a.as_mut_ptr(),
                &lda,
                jpvt.as_mut_ptr(),
                tau.as_mut_ptr(),
                work.as_mut_ptr(),
                &4,
                &mut info,
            );
        }
        assert_eq!((info, jpvt), (0, [1, 2]));
    }

    unsafe extern "C" fn fake_dsyevd_lp64(
        _jobz: *const c_char,
        _uplo: *const c_char,
        _n: *const i32,
        _a: *mut f64,
        _lda: *const i32,
        _w: *mut f64,
        work: *mut f64,
        _lwork: *const i32,
        iwork: *mut i32,
        liwork: *const i32,
        info: *mut i32,
    ) {
        assert_eq!(read_unaligned(liwork), -1);
        *work = 9.0;
        *iwork = 13;
        *info = 0;
    }

    #[test]
    fn dsyevd_query_returns_liwork_through_the_narrowing_bridge() {
        unsafe {
            assert!(matches!(register_dsyevd_lp64(fake_dsyevd_lp64), 0 | 2));
        }
        let jobz = b'V' as c_char;
        let uplo = b'L' as c_char;
        let mut a = [0.0_f64; 4];
        let mut w = [0.0_f64; 2];
        let mut work = [0.0_f64];
        let mut iwork = [0_i64];
        let mut info = -1_i64;
        unsafe {
            dsyevd_(
                &jobz,
                &uplo,
                &2,
                a.as_mut_ptr(),
                &2,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                &-1,
                iwork.as_mut_ptr(),
                &-1,
                &mut info,
            );
        }
        assert_eq!((info, work[0], iwork[0]), (0, 9.0, 13));
    }
}
