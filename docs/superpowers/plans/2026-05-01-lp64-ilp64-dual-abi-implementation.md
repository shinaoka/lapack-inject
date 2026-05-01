# LP64/ILP64 Dual ABI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Finish the dual LP64/ILP64 Fortran LAPACK registration layer first, then add LAPACKE C wrappers in a later phase.

**Architecture:** `src/backend.rs` and `src/fortran.rs` are handwritten preambles that include generated files. `scripts/generate_lapack_bindings.py` reads `lapack-sys` plus supplemental routines and emits `src/backend_gen.rs` and `src/fortran_gen.rs`. Fortran exports prefer the provider matching the crate's `lapackint` ABI and cast integer pointer parameters when falling back to the other provider width.

**Tech Stack:** Rust 2021, Python 3 generator, `num-complex`, `paste`, `lapack-sys` for tests.

---

## Current Scope

This plan is intentionally split into phases. Do not start LAPACKE or row-major support until Phase 1 is green.

### Phase 1: Dual Fortran Backend For Existing Test Surface

Supported generated functions:

- GESV, GETRF, GETRS, GETC2, GESC2 for `s/d/c/z`
- GETRI, POTRF, GESVD for `s/d/c/z`
- SYEV for `s/d`

Phase 1 success criteria:

- `cargo build` passes.
- `cargo build --features ilp64` passes.
- `cargo test --no-run` passes.
- `cargo test --no-run --features ilp64` passes, or any remaining failure is documented with an exact root cause.

### Phase 2: LAPACKE C API

Add `src/lapacke.rs`, `LAPACK_ROW_MAJOR`, `LAPACK_COL_MAJOR`, and exported `LAPACKE_*` entry points after Phase 1 is stable. Keep this phase separate because correct row-major handling depends on routine-specific matrix layout rules.

---

## Task 1: Fix The Generator Function Set

**Files:**

- Modify: `scripts/generate_lapack_bindings.py`
- Regenerate: `src/backend_gen.rs`
- Regenerate: `src/fortran_gen.rs`

**Step 1.1: Extend `CORE_FUNCTIONS` To Match Tests**

Add the test-covered routines that currently fail to resolve:

```python
"sgetri_", "dgetri_", "cgetri_", "zgetri_",
"spotrf_", "dpotrf_", "cpotrf_", "zpotrf_",
"ssyev_", "dsyev_",
"sgesvd_", "dgesvd_", "cgesvd_", "zgesvd_",
```

**Step 1.2: Regenerate Bindings**

Run:

```bash
python3 scripts/generate_lapack_bindings.py \
  --lapack-sys-path ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/lapack-sys-0.15.0/src/lapack.rs \
  --output-dir src/
```

Expected:

- `src/backend_gen.rs` contains `DgetriLp64FnPtr`, `DpotrfLp64FnPtr`, `DsyevLp64FnPtr`, and `DgesvdLp64FnPtr`.
- `cargo test --no-run` no longer fails because these type aliases are missing.

---

## Task 2: Make Fortran Dispatch Compile For Both ABIs

**Files:**

- Modify: `scripts/generate_lapack_bindings.py`
- Regenerate: `src/fortran_gen.rs`

**Step 2.1: Capture The Current Failure**

Run:

```bash
cargo build --features ilp64
```

Expected before the fix:

- Fails with `E0308`.
- Errors point at `src/fortran_gen.rs` LP64 match arms passing `*const i64` or `*mut i64` where LP64 providers expect `*const i32` or `*mut i32`.

**Step 2.2: Generate Provider Selection By Build ABI**

For each export, generate:

```rust
#[cfg(feature = "ilp64")]
let provider = get_dgesv_for_ilp64();
#[cfg(not(feature = "ilp64"))]
let provider = get_dgesv_for_lp64();
```

**Step 2.3: Generate Both LP64 And ILP64 Integer Pointer Views**

For each `*const c_int` parameter, emit casts to `*const i32` in the LP64 arm and `*const i64` in the ILP64 arm. For each `*mut c_int` parameter, emit casts to `*mut i32` and `*mut i64` respectively. Non-integer parameters are passed through unchanged.

Example:

```rust
match provider {
    DgesvProvider::Lp64(fun) => {
        let n_lp64: *const i32 = n as *const i32;
        let ipiv_lp64: *mut i32 = ipiv as *mut i32;
        fun(n_lp64, nrhs_lp64, A, lda_lp64, ipiv_lp64, B, ldb_lp64, info_lp64)
    }
    DgesvProvider::Ilp64(fun) => {
        let n_ilp64: *const i64 = n as *const i64;
        let ipiv_ilp64: *mut i64 = ipiv as *mut i64;
        fun(n_ilp64, nrhs_ilp64, A, lda_ilp64, ipiv_ilp64, B, ldb_ilp64, info_ilp64)
    }
}
```

**Step 2.4: Verify**

Run:

```bash
cargo build
cargo build --features ilp64
```

Expected:

- Both commands pass.

---

## Task 3: Update Rust Tests For The New API

**Files:**

- Modify: `tests/getc2_gesc2_symbols.rs`
- Modify: `tests/functional_test.rs`

**Step 3.1: Replace Old Type Aliases**

Use ABI-specific generated type aliases:

- default build: `DgesvLp64FnPtr`, `Dgetc2Lp64FnPtr`, ...
- `ilp64` build: `DgesvIlp64FnPtr`, `Dgetc2Ilp64FnPtr`, ...

**Step 3.2: Replace Old Registration Functions**

Use build-ABI-specific registration helpers in tests:

- default build: `register_dgesv_lp64`, `register_dgetc2_lp64`, ...
- `ilp64` build: `register_dgesv_ilp64`, `register_dgetc2_ilp64`, ...

Assert the status code is either `0` for first registration or `2` for already registered in tests that may share global `OnceLock` state.

**Step 3.3: Verify**

Run:

```bash
cargo test --no-run
cargo test --no-run --features ilp64
```

Expected:

- Tests compile in both configurations.

---

## Task 4: Refresh Handoff

**Files:**

- Modify: `docs/superpowers/HANDOFF.md`

Update the handoff after verification with:

- Commands run and their outcomes.
- Remaining Phase 2 LAPACKE work.
- Any known ABI fallback limitations discovered during Phase 1.
