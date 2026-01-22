//! # lapack-inject
//!
//! LAPACK compatible interface backed by Fortran LAPACK function pointers.
//!
//! This crate allows you to use LAPACK functions while the actual computation
//! is performed by Fortran LAPACK functions provided at runtime.
//! This is useful for integrating with Python (scipy) or Julia (libblastrampoline)
//! where Fortran LAPACK pointers are available.
//!
//! ## Usage
//!
//! ```ignore
//! use lapack_inject::{register_dgesv, lapack_dgesv};
//!
//! // Register Fortran dgesv pointer (e.g., from scipy or Julia)
//! unsafe {
//!     register_dgesv(dgesv_ptr);
//! }
//!
//! // Use LAPACK-style interface
//! unsafe {
//!     lapack_dgesv(n, nrhs, a, lda, ipiv, b, ldb, &mut info);
//! }
//! ```
//!
//! ## Supported Functions
//!
//! - **Linear Solve**: GESV (s,d,c,z)
//! - **LU Factorization**: GETRF, GETRI (s,d,c,z)
//! - **Cholesky**: POTRF (s,d,c,z)
//! - **QR**: GEQRF (s,d,c,z), ORGQR (s,d), UNGQR (c,z)
//! - **SVD**: GESVD, GESDD (s,d,c,z)
//! - **Eigenvalue**: GEEV (s,d,c,z), SYEV (s,d), HEEV (c,z), GEES (s,d,c,z)

mod backend;
mod types;
pub mod utils;
mod xerbla;

#[cfg(feature = "openblas")]
mod autoregister;

pub mod chol;
pub mod eig;
pub mod lu;
pub mod qr;
pub mod solve;
pub mod svd;

pub use backend::*;
pub use types::*;

// Re-export LAPACKE functions at crate root

// Linear Solve (LAPACKE interface)
pub use solve::gesv::{
    LAPACKE_cgesv, LAPACKE_cgesv_work, LAPACKE_dgesv, LAPACKE_dgesv_work, LAPACKE_sgesv,
    LAPACKE_sgesv_work, LAPACKE_zgesv, LAPACKE_zgesv_work,
};

// Least Squares (LAPACKE interface)
pub use solve::gels::{
    LAPACKE_cgels, LAPACKE_cgels_work, LAPACKE_dgels, LAPACKE_dgels_work, LAPACKE_sgels,
    LAPACKE_sgels_work, LAPACKE_zgels, LAPACKE_zgels_work,
};

// LU Factorization (LAPACKE interface)
pub use lu::getrf::{
    LAPACKE_cgetrf, LAPACKE_cgetrf_work, LAPACKE_dgetrf, LAPACKE_dgetrf_work, LAPACKE_sgetrf,
    LAPACKE_sgetrf_work, LAPACKE_zgetrf, LAPACKE_zgetrf_work,
};
pub use lu::getri::{
    LAPACKE_cgetri, LAPACKE_cgetri_work, LAPACKE_dgetri, LAPACKE_dgetri_work, LAPACKE_sgetri,
    LAPACKE_sgetri_work, LAPACKE_zgetri, LAPACKE_zgetri_work,
};

// Cholesky (LAPACKE interface)
pub use chol::potrf::{
    LAPACKE_cpotrf, LAPACKE_cpotrf_work, LAPACKE_dpotrf, LAPACKE_dpotrf_work, LAPACKE_spotrf,
    LAPACKE_spotrf_work, LAPACKE_zpotrf, LAPACKE_zpotrf_work,
};

// QR (legacy interface - will be converted)
pub use qr::geqrf::{lapack_cgeqrf, lapack_dgeqrf, lapack_sgeqrf, lapack_zgeqrf};
pub use qr::orgqr::{lapack_dorgqr, lapack_sorgqr};
pub use qr::ungqr::{lapack_cungqr, lapack_zungqr};

// SVD (legacy interface - will be converted)
pub use svd::gesdd::{lapack_cgesdd, lapack_dgesdd, lapack_sgesdd, lapack_zgesdd};
pub use svd::gesvd::{lapack_cgesvd, lapack_dgesvd, lapack_sgesvd, lapack_zgesvd};

// Eigenvalue (legacy interface - will be converted)
pub use eig::gees::{lapack_cgees, lapack_dgees, lapack_sgees, lapack_zgees};
pub use eig::geev::{lapack_cgeev, lapack_dgeev, lapack_sgeev, lapack_zgeev};
pub use eig::heev::{lapack_cheev, lapack_zheev};
pub use eig::syev::{lapack_dsyev, lapack_ssyev};

// Error handling (LAPACKE interface)
pub use xerbla::LAPACKE_xerbla;
