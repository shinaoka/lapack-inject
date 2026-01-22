//! Linear equation solvers (GESV, GELS).

pub mod gels;
pub mod gesv;

pub use gels::*;
pub use gesv::*;
