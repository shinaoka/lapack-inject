# LP64/ILP64 Dual ABI C API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dual LP64/ILP64 LAPACKE C API layer with row-major support to lapack-inject.

**Architecture:** Generator reads lapack-sys bindings, outputs dual LP64/ILP64 fn pointer types + `define_dual_backend!` macro calls. Fortran symbols dispatch dynamically. LAPACKE wrappers pass through to Fortran with int widening. Row-major handled via transposition helpers.

**Tech Stack:** Rust, Python 3 (dev-time only), num-complex, paste

---

### Task 0: Create macro infrastructure + fix generator

**Files:**
- Create/Modify: `src/backend.rs` (top-level macros only)
- Modify: `scripts/generate_lapack_bindings.py`

**Step 0.1: Write `define_dual_backend!` macro**

Replace the old macro infrastructure in `src/backend.rs` with the new dual backend macro. Keep the select function types, remove old `define_lapack_ffi!`.

```rust
macro_rules! define_dual_backend {
    ($name:ident, $lp64_type:ty, $ilp64_type:ty) => {
        paste::paste! {
            static [<$name:upper _LP64>]: OnceLock<$lp64_type> = OnceLock::new();
            static [<$name:upper _ILP64>]: OnceLock<$ilp64_type> = OnceLock::new();

            #[no_mangle]
            pub unsafe extern "C" fn [<register_ $name _lp64>](f: $lp64_type) -> i32 {
                if f.is_null() { return 1; }
                match [<$name:upper _LP64>].set(f) {
                    Ok(_) => 0,
                    Err(_) => 2,
                }
            }

            #[no_mangle]
            pub unsafe extern "C" fn [<register_ $name _ilp64>](f: $ilp64_type) -> i32 {
                if f.is_null() { return 1; }
                match [<$name:upper _ILP64>].set(f) {
                    Ok(_) => 0,
                    Err(_) => 2,
                }
            }

            pub enum [<$name:camel Provider>] {
                Lp64($lp64_type),
                Ilp64($ilp64_type),
            }

            #[allow(dead_code)]
            pub(crate) fn [<get_ $name _for_lp64>]() -> [<$name:camel Provider>] {
                if let Some(f) = [<$name:upper _LP64>].get() {
                    return [<$name:camel Provider>]::Lp64(*f);
                }
                if let Some(f) = [<$name:upper _ILP64>].get() {
                    return [<$name:camel Provider>]::Ilp64(*f);
                }
                panic!(concat!("lapack function `", stringify!($name), "` is not registered (call register_", stringify!($name), "_lp64 or register_", stringify!($name), "_ilp64 first)"));
            }

            #[allow(dead_code)]
            pub(crate) fn [<get_ $name _for_ilp64>]() -> [<$name:camel Provider>] {
                if let Some(f) = [<$name:upper _ILP64>].get() {
                    return [<$name:camel Provider>]::Ilp64(*f);
                }
                if let Some(f) = [<$name:upper _LP64>].get() {
                    return [<$name:camel Provider>]::Lp64(*f);
                }
                panic!(concat!("lapack function `", stringify!($name), "` is not registered (call register_", stringify!($name), "_lp64 or register_", stringify!($name), "_ilp64 first)"));
            }
        }
    };
}
```

**Step 0.2: Fix generator path + update output format**

Update `scripts/generate_lapack_bindings.py`:
- Add command-line args for input/output paths (defaulting to cargo registry and src/)
- Change output: for each function, emit LP64 type (`c_int`→`i32`) and ILP64 type (`c_int`→`i64`), then `define_dual_backend!(name, Lp64FnPtr, Ilp64FnPtr);`

Key generator changes:
```python
# Type conversion now produces concrete types
def convert_type_to_concrete(rust_type: str, int_ty: str) -> str:
    """Convert lapack-sys types to concrete i32/i64 types."""
    type_map = {
        '*const c_int': f'*const {int_ty}',
        '*mut c_int': f'*mut {int_ty}',
        # ... other type mappings
    }
    for old, new in type_map.items():
        rust_type = rust_type.replace(old, new)
    return rust_type
```

Output format:
```python
def generate_backend_types(func, name):
    lp64_params = [f"    {p.name}: {convert_type_to_concrete(p.type_, 'i32')}" for p in func.params]
    ilp64_params = [f"    {p.name}: {convert_type_to_concrete(p.type_, 'i64')}" for p in func.params]
    return f"""
pub type {name}Lp64FnPtr = unsafe extern "C" fn(
{chr(10).join(lp64_params)},
);
pub type {name}Ilp64FnPtr = unsafe extern "C" fn(
{chr(10).join(ilp64_params)},
);
define_dual_backend!({func.name.rstrip('_')}, {name}Lp64FnPtr, {name}Ilp64FnPtr);
"""
```

Run generator:
```bash
python3 scripts/generate_lapack_bindings.py \
  --lapack-sys-path ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/lapack-sys-0.15.0/src/lapack.rs \
  --output-dir src/
```

---

### Task 1: Regenerate backend.rs

**Files:**
- Modify: `scripts/generate_lapack_bindings.py`
- Regenerate: `src/backend.rs`

Run:
```bash
python3 scripts/generate_lapack_bindings.py \
  --lapack-sys-path /home/shinaoka/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/lapack-sys-0.15.0/src/lapack.rs \
  --output-dir src/
```

---

### Task 2: Regenerate fortran.rs with dual dispatch

**Files:**
- Modify: `scripts/generate_lapack_bindings.py`
- Regenerate: `src/fortran.rs`

Fortran export generator:
```python
def generate_fortran_export(func, name):
    params = [f"    {p.name}: {convert_type(func.type_)}" for p in func.params]
    return f"""
#[no_mangle]
pub unsafe extern "C" fn {func.name}(
{chr(10).join(params)},
) {{
    let provider = get_{name}_for_lp64();
    match provider {{
        {name}Provider::Lp64(f) => f({', '.join(p.name for p in func.params)}),
        {name}Provider::Ilp64(f) => {{
            // Widening cast for integer params
            f({', '.join(widen_cast(p, func.name) for p in func.params)})
        }}
    }}
}}
"""
```

---

### Task 3: Update types.rs

**Files:**
- Modify: `src/types.rs`

Add:
```rust
pub const LAPACK_ROW_MAJOR: lapack_int = 101;
pub const LAPACK_COL_MAJOR: lapack_int = 102;
```

---

### Task 4: Create lapacke.rs with LAPACKE wrappers

**Files:**
- Create: `src/lapacke.rs` (hand-written, ~200 lines)
- Modify: `scripts/generate_lapack_bindings.py` (generator produces lapacke function invocations)

The LAPACKE module is structured as:
1. Hand-written: transposition helpers + `dispatch_lapacke_colmajor!` macro
2. Generated: full LAPACKE wrapper function bodies (produced by Python generator)

**Step 4.1: Transposition helpers (hand-written in lapacke.rs)**

```rust
pub(crate) unsafe fn lapacke_dge_trans(
    layout: lapack_int, m: lapack_int, n: lapack_int,
    src: *const f64, ld_src: lapack_int,
    dst: *mut f64, ld_dst: lapack_int,
) {
    for i in 0..m {
        for j in 0..n {
            if layout == LAPACK_COL_MAJOR {
                *dst.add((i * ld_dst + j) as usize) = *src.add((j * ld_src + i) as usize);
            } else {
                *dst.add((j * ld_dst + i) as usize) = *src.add((i * ld_src + j) as usize);
            }
        }
    }
}
// Same for sge, cge, zge
```

**Step 4.2: Dispatch macro (hand-written in lapacke.rs)**

```rust
/// Dispatch a LAPACKE call to the registered Fortran provider with i32→i64 widening.
macro_rules! dispatch_colmajor {
    ($provider:expr, $getter:expr, $info:ident, $( $param:ident : $param_ty:ty ),*) => {
        let mut $info: lapack_int = 0;
        let provider = $getter();
        match provider {
            $provider::Lp64(f) => {
                f($(&$param),*, &mut $info);
                $info
            }
            $provider::Ilp64(f) => {
                $(let [<$param _w>]: i64 = $param as i64;)*
                let mut info_i64: i64 = 0;
                f($(&[<$param _w>]),*, &mut info_i64);
                info_i64 as lapack_int
            }
        }
    };
}
```

**Step 4.3: Generator produces full function bodies**

Generator produces LAPACKE function bodies directly (no complex macro):

```python
def generate_lapacke_func(func, name, matrix_type):
    """Generate a LAPACKE wrapper function for col-major dispatch."""
    # Determine base name for provider/getter
    lp64_name = name + "Lp64FnPtr"
    ilp64_name = name + "Ilp64FnPtr"
    provider = name + "Provider"
    getter_lp64 = f"get_{name}_for_lp64"
    getter_ilp64 = f"get_{name}_for_ilp64"

    # Build param list (remove last info param, add matrix_layout)
    params = func.params[:-1]  # remove info
    lapacke_params = ["matrix_layout: lapack_int"]
    for p in params:
        ct = convert_to_lapacke_type(p.type_)
        lapacke_params.append(f"{p.name}: {ct}")

    # Build dispatch params
    dispatch_params = []
    for p in params:
        if p.type_.startswith('*const c_int') or p.type_.startswith('*mut c_int'):
            dispatch_params.append(f"{p.name}: lapack_int")
        else:
            dispatch_params.append(f"{p.name}: {p.type_}")

    # Convert c_char to char for LAPACKE
    # ...

    return f"""
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_{name}(
    {', '.join(lapacke_params)},
) -> lapack_int {{
    if matrix_layout == LAPACK_COL_MAJOR {{
        dispatch_colmajor!({provider}, {getter_lp64}, info,
            {', '.join(f'{p.name}: {p.type_}' for p in params)})
    }} else if matrix_layout == LAPACK_ROW_MAJOR {{
        // Row-major: allocate temp, transpose, dispatch, transpose back
        lapacke_dge_rowmajor_dispatch(
            matrix_layout, {', '.join(p.name for p in params)},
            {getter_lp64},
        )
    }} else {{
        -1
    }}
}}
"""
```

---

### Task 5: Update lib.rs

**Files:**
- Modify: `src/lib.rs`

```rust
mod backend;
pub mod fortran;
mod types;
pub mod lapacke;  // NEW

pub use backend::*;
pub use fortran::*;
pub use types::*;
pub use lapacke::*;  // NEW
```

---

### Task 6: Build and fix

```bash
cargo build 2>&1 | head -50
cargo build --features ilp64 2>&1 | head -50
```

Fix compilation errors iteratively. Main issues expected:
- Dead code warnings for unused provider types
- Name collisions between modules
- Import issues

---

### Task 7: Write tests

**Files:**
- Modify: `tests/functional_test.rs`

Add tests:
```rust
#[test]
fn test_dual_registration_lp64() {
    // Register a mock LP64 dgesv, call via get_dgesv_for_lp64
}

#[test]
fn test_dual_registration_ilp64() {
    // Register ILP64 only, verify LP64 fallback works
}

#[test]
fn test_lapacke_dgetrf_colmajor() {
    // Call LAPACKE_dgetrf with LAPACK_COL_MAJOR, verify against reference
}

#[test]
fn test_lapacke_dgetrf_rowmajor() {
    // Call LAPACKE_dgetrf with LAPACK_ROW_MAJOR, verify against reference
}
```

---

### Task 8: Update README.md

**Files:**
- Modify: `README.md`

Add:
- LAPACKE C API section with code examples
- LP64/ILP64 dual registration explanation
- Registration status codes
- Row-major support notes
- Updated features section
