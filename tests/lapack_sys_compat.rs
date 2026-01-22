//! Test compatibility with lapack-sys.
//!
//! This test verifies that lapack-inject provides all the symbols
//! that lapack-sys expects.

// Link lapack-inject to provide the symbols
extern crate lapack_inject;

// Now lapack-sys functions should resolve to lapack-inject's exports
use lapack_sys;

/// Test that we can take the address of lapack-sys functions
/// (they resolve to lapack-inject's symbols)
#[test]
fn symbols_are_linkable() {
    // These function pointer assignments verify that the symbols exist
    // and have the correct type signature.

    // GESV
    let _ = lapack_sys::sgesv_ as *const ();
    let _ = lapack_sys::dgesv_ as *const ();
    let _ = lapack_sys::cgesv_ as *const ();
    let _ = lapack_sys::zgesv_ as *const ();

    // GETRF
    let _ = lapack_sys::sgetrf_ as *const ();
    let _ = lapack_sys::dgetrf_ as *const ();
    let _ = lapack_sys::cgetrf_ as *const ();
    let _ = lapack_sys::zgetrf_ as *const ();

    // GETRI
    let _ = lapack_sys::sgetri_ as *const ();
    let _ = lapack_sys::dgetri_ as *const ();
    let _ = lapack_sys::cgetri_ as *const ();
    let _ = lapack_sys::zgetri_ as *const ();

    // POTRF
    let _ = lapack_sys::spotrf_ as *const ();
    let _ = lapack_sys::dpotrf_ as *const ();
    let _ = lapack_sys::cpotrf_ as *const ();
    let _ = lapack_sys::zpotrf_ as *const ();

    // GEQRF
    let _ = lapack_sys::sgeqrf_ as *const ();
    let _ = lapack_sys::dgeqrf_ as *const ();
    let _ = lapack_sys::cgeqrf_ as *const ();
    let _ = lapack_sys::zgeqrf_ as *const ();

    // GESVD
    let _ = lapack_sys::sgesvd_ as *const ();
    let _ = lapack_sys::dgesvd_ as *const ();
    let _ = lapack_sys::cgesvd_ as *const ();
    let _ = lapack_sys::zgesvd_ as *const ();

    // GESDD
    let _ = lapack_sys::sgesdd_ as *const ();
    let _ = lapack_sys::dgesdd_ as *const ();
    let _ = lapack_sys::cgesdd_ as *const ();
    let _ = lapack_sys::zgesdd_ as *const ();

    // GEEV
    let _ = lapack_sys::sgeev_ as *const ();
    let _ = lapack_sys::dgeev_ as *const ();
    let _ = lapack_sys::cgeev_ as *const ();
    let _ = lapack_sys::zgeev_ as *const ();

    // SYEV
    let _ = lapack_sys::ssyev_ as *const ();
    let _ = lapack_sys::dsyev_ as *const ();

    // HEEV
    let _ = lapack_sys::cheev_ as *const ();
    let _ = lapack_sys::zheev_ as *const ();

    // GELS
    let _ = lapack_sys::sgels_ as *const ();
    let _ = lapack_sys::dgels_ as *const ();
    let _ = lapack_sys::cgels_ as *const ();
    let _ = lapack_sys::zgels_ as *const ();

    println!("All tested symbols are linkable!");
}
