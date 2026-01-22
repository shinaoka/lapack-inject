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
//! use lapack_inject::register_dgesv;
//!
//! // Register Fortran dgesv pointer (e.g., from scipy or Julia)
//! unsafe {
//!     register_dgesv(dgesv_ptr);
//! }
//!
//! // Now lapack_inject exports dgesv_ symbol that can be used by other crates
//! ```
//!
//! ## lapack-src/lapack-sys Compatibility
//!
//! This crate exports all Fortran-style LAPACK symbols defined in lapack-sys
//! (e.g., `dgesv_`, `dgetrf_`, etc.). These symbols are compatible with the
//! lapack-src crate. Register function pointers at runtime, and this crate
//! will provide the symbols that other crates expect from lapack-src.
//!
//! ## Supported Functions
//!
//! All 1315 LAPACK functions from lapack-sys are supported, including:
//! - Linear solvers (GESV, GBSV, POSV, SYSV, etc.)
//! - Factorizations (GETRF, POTRF, GEQRF, etc.)
//! - SVD (GESVD, GESDD, etc.)
//! - Eigenvalue problems (GEEV, SYEV, HEEV, GEES, etc.)
//! - And many more...

mod backend;
pub mod fortran;
mod types;

pub use backend::*;
pub use fortran::*;
pub use types::*;
