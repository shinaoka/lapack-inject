//! Eigenvalue decomposition (GEEV, SYEV, HEEV, GEES).

pub mod gees;
pub mod geev;
pub mod heev;
pub mod syev;

pub use gees::*;
pub use geev::*;
pub use heev::*;
pub use syev::*;
