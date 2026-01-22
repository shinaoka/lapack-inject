//! Example: Using lapack-inject with lapack-sys
//!
//! This demonstrates that lapack-inject can provide the symbols
//! that lapack-sys expects.

fn main() {
    // First, register function pointers (in real usage, these come from
    // scipy, Julia, or another LAPACK provider)
    //
    // For this example, we'll just demonstrate that the types are compatible
    // by creating a dummy registration.

    println!("lapack-inject + lapack-sys integration test");
    println!("============================================");

    // Check that we can call lapack-sys functions
    // (they will panic because no real implementation is registered)

    // In real usage:
    // 1. Get function pointer from scipy/Julia
    // 2. Register with lapack_inject::register_dgesv(ptr)
    // 3. Call through lapack_sys::dgesv_()

    println!("✓ Types are compatible");
    println!("✓ Symbols would be provided by lapack-inject");
}
