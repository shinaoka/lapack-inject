# LP64/ILP64 Dual ABI - 引き継ぎ書

## 参照ドキュメント

- **仕様書**: `docs/superpowers/specs/2026-05-01-lp64-ilp64-dual-abi-c-api-design.md`
- **実装計画**: `docs/superpowers/plans/2026-05-01-lp64-ilp64-dual-abi-implementation.md`

## 現在の状態

Phase 1 の dual Fortran backend はビルド・テストが通る状態です。

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
   - 現在のPhase 1対応範囲を明記。

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

## 検証結果

以下は通過済みです。

```bash
cargo build
cargo build --features ilp64
cargo test --no-run
cargo test --no-run --features ilp64
cargo test
cargo test --features ilp64
```

`functional_test.rs` の実LAPACK連携テストは `#[ignore]` のままです。

## 既知の注意点

- Cross-ABI fallback は現在 pointer cast です。ビルドABIとprovider ABIが一致する通常パスは通りますが、`i32` caller から ILP64 provider、または `i64` caller から LP64 provider へ実データを変換する完全なmarshal実装ではありません。
- `register_*_lp64()` / `register_*_ilp64()` はRustのfn pointer型を受け取るため、null pointer検査はしていません。戻り値は `0` 成功、`2` 登録済みです。
- `src/lapacke.rs`、`LAPACK_ROW_MAJOR` / `LAPACK_COL_MAJOR`、LAPACKE row-major wrapper はまだ未実装です。これは Phase 2 で扱います。

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
| `scripts/generate_lapack_bindings.py` | コード生成スクリプト |
| `tests/functional_test.rs` | 実LAPACK連携テスト（ignore） |
| `tests/getc2_gesc2_symbols.rs` | supplemental symbol dispatch test |
