# LP64/ILP64 Dual ABI - 引き継ぎ書

## 参照ドキュメント

- **仕様書**: `docs/superpowers/specs/2026-05-01-lp64-ilp64-dual-abi-c-api-design.md`
- **実装計画**: `docs/superpowers/plans/2026-05-01-lp64-ilp64-dual-abi-implementation.md`

## 現在の状態

Phase 1 の dual Fortran backend と、Phase 2 の最小 LAPACKE wrapper はビルド・テストが通る状態です。

### 完了したこと

1. **PLANを修正**
   - 旧PLANは「全LAPACK + LAPACKE + row-major」を一度に扱っていて実行不能だったため、Phase 1（dual Fortran backend）と Phase 2（LAPACKE/row-major）に分離しました。

2. **`scripts/generate_lapack_bindings.py` を更新**
   - 出力先は `src/backend_gen.rs` / `src/fortran_gen.rs`。
   - 生成対象を既存テスト範囲まで拡張しました。
   - `fortran_gen.rs` はビルドABIに合わせて provider を選択します。
     - default: `get_*_for_lp64()`
     - `--features ilp64`: `get_*_for_ilp64()`
   - LP64/ILP64 provider arms それぞれで `c_int` pointer 引数を `i32` / `i64` pointer にキャストします。

3. **生成済みファイルを更新**
   - `src/backend_gen.rs`
   - `src/fortran_gen.rs`

4. **テストを新APIへ更新**
   - `*FnPtr` から `*Lp64FnPtr` / `*Ilp64FnPtr` へ変更。
   - `register_*()` から `register_*_lp64()` / `register_*_ilp64()` へ変更。
   - default build と `ilp64` build の両方でコンパイル可能にしました。

5. **README / crate docs を現状に合わせて更新**
   - 旧API例と「全1315関数対応」の記述を削除。
   - 現在の対応範囲を明記。

6. **`src/lapacke.rs` を追加**
   - `LAPACKE_dgesv`, `LAPACKE_dgetrf`, `LAPACKE_dgetri`, `LAPACKE_dpotrf` を追加。
   - 各関数に `_64` variant を追加。
   - row-major / column-major の両方に対応。
   - LAPACKE C APIは `_64` の別名を持てるため、`ilp64` featureなしで64-bit integer entrypointを公開します。

7. **C/Fortran外部テストを新APIへ更新**
   - `ctest` はローカルの最小 `lapacke.h` と `lapacke_example_aux.c` を使うように変更。
   - `ctest` / `ftest` のOpenBLAS初期化は `register_*_lp64()` を使うように変更。

## 現在の生成対象

- `xGESV`
- `xGETRF`
- `xGETRS`
- `xGETRI`
- `xPOTRF`
- `xGESVD`
- `sSYEV`, `dSYEV`
- supplemental `xGETC2`, `xGESC2`

合計 34 function bindings。

## 現在のLAPACKE対象

- `LAPACKE_dgesv`, `LAPACKE_dgesv_64`
- `LAPACKE_dgetrf`, `LAPACKE_dgetrf_64`
- `LAPACKE_dgetri`, `LAPACKE_dgetri_64`
- `LAPACKE_dpotrf`, `LAPACKE_dpotrf_64`

`LAPACK_ROW_MAJOR` と `LAPACK_COL_MAJOR` の両方に対応しています。

## 検証結果

以下は通過済みです。

```bash
cargo build
cargo build --features ilp64
cargo test --no-run
cargo test --no-run --features ilp64
cargo test
cargo test --features ilp64
make -C ctest test
make -C ftest test
```

`functional_test.rs` の実LAPACK連携テストは `#[ignore]` のままです。

## 既知の注意点

- Cross-ABI fallback は現在 pointer cast です。ビルドABIとprovider ABIが一致する通常パスは通りますが、`i32` caller から ILP64 provider、または `i64` caller から LP64 provider へ実データを変換する完全なmarshal実装ではありません。
- `register_*_lp64()` / `register_*_ilp64()` はRustのfn pointer型を受け取るため、null pointer検査はしていません。戻り値は `0` 成功、`2` 登録済みです。
- Fortran互換シンボル（例: `dgesv_`）はLP64/ILP64で同じシンボル名を使うため、`ilp64` featureによるcompile-time ABI選択を残しています。provider登録自体はfeatureなしでLP64/ILP64両方を同時に持てます。
- LAPACKE C APIは `_64` entrypointを持つため、featureなしでLP64/ILP64のC ABIを同時公開できます。
- LAPACKE wrapperは現時点ではdouble precisionの4関数だけです。全LAPACKE関数への拡張は未実装です。

## 生成スクリプトの実行方法

```bash
python3 scripts/generate_lapack_bindings.py \
  --lapack-sys-path ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/lapack-sys-0.15.0/src/lapack.rs \
  --output-dir src/
```

## 関連ファイル

| ファイル | 役割 |
|----------|------|
| `src/backend.rs` | プリアンブル（`define_dual_backend!` マクロ, select型）+ `include!("backend_gen.rs")` |
| `src/backend_gen.rs` | 自動生成: `Lp64FnPtr` / `Ilp64FnPtr` 型 + `define_dual_backend!` 呼び出し |
| `src/fortran.rs` | プリアンブル（imports）+ `include!("fortran_gen.rs")` |
| `src/fortran_gen.rs` | 自動生成: `#[no_mangle]` Fortran export with dual dispatch |
| `src/lapacke.rs` | 手書き: 最小LAPACKE wrapper + row-major変換 + `_64` variants |
| `scripts/generate_lapack_bindings.py` | コード生成スクリプト |
| `tests/functional_test.rs` | 実LAPACK連携テスト（ignore） |
| `tests/getc2_gesc2_symbols.rs` | supplemental symbol dispatch test |
| `tests/lapacke_test.rs` | LAPACKE row-major / `_64` tests |
