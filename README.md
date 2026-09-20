# 単位正方形上の Poisson 方程式

単位正方形上の Dirichlet 問題を、Julia、Rust、Fortran で同じ格子・同じ右辺に対して解く。

解法は二系統ある。
5 点差分の **Jacobi 反復**と、同じ差分作用素の固有関数展開である **DST-I**（第一種離散正弦変換）である。

## 問題

領域 \(\Omega = (0,1)\times(0,1)\) で、次を満たす \(u\) を求める。

\[
-\Delta u = f \quad \text{in }\Omega, \qquad
u = 0 \quad \text{on }\partial\Omega.
\]

厳密解を

\[
u(x,y)=\sin(\pi x)\sin(\pi y)
\]

と置くと、右辺は

\[
f(x,y)=2\pi^2\sin(\pi x)\sin(\pi y)
\]

である。

格子点数は各方向 \(N=401\)、間隔は \(h=1/(N-1)\) である。
内部点は 5 点 Laplacian で離散化する。

## 解法

**Jacobi 反復**は、内部点ごとに近傍 4 点と右辺から新しい値を書く。
境界は 0 のままにする。
更新幅が \(10^{-10}\) を下回るか、10 万回に達するまで繰り返す。
この格子では許容誤差には届かず、反復回数の上限で止まる。

**DST-I**は、矩形上の斉次 Dirichlet 条件に対応する基底で \(f\) を展開し、固有値で割って逆変換する。
5 点 Laplacian の厳密な逆であり、反復しない。
実装は、内部点の奇関数延長に対する正規化なし FFT である。

`julia_sparse_cg/` と `rust_tenferro_sparse_cg/` で CG が 1 反復で収束するのは、右辺に使う
\(\sin(\pi x)\sin(\pi y)\) が 5 点差分 Laplacian の固有ベクトルだからである。
任意の Poisson 問題で CG が常に 1 反復で収束するわけではない。

格子点が \(N^2\) 個のとき、Jacobi の収束にはだいたい \(O(N^2)\) 回かかるので計算量は \(O(N^4)\) である。
DST は \(O(N^2\log N)\) である。

## ディレクトリ

| パス | 言語 | 解法 | ホットパス |
|---|---|---|---|
| `julia/` | Julia | Jacobi | ホットループを `LoopVectorization.@turbo` で展開。バッファは入れ替える。更新幅の還元と表示は1000反復ごと |
| `julia_unsafe/` | Julia | Jacobi | 生ポインタ + `@simd`。8反復の列方向 wavefront でキャッシュを再利用。[検証・計測手順](julia_unsafe/README.md) |
| `cxx/` | C++23 | Jacobi | 値型の `Grid` を `operator()(i, j)` で添字。バッファは O(1) swap。図は Fortran と同じ自前 PNG エンコード |
| `fortran/` | Fortran | Jacobi | 既定では境界チェックなし（`julia` 相当）。バッファはポインタの付け替え。更新幅の還元は1000反復ごと |
| `julia_fft/` | Julia | DST-I | FFTW。比較のため FFTW と BLAS は 1 スレッド |
| `julia_sparse_cg/` | Julia | 疎行列 CG | 内部点の 5 点 Laplacian を CSC 行列で構成し、`IterativeSolvers.cg` で解く |
| `rust_tenferro/` | Rust | Jacobi | `Vec<f64>` の安全な添字。バッファは `swap`。tenferro は求解後の誤差だけ |
| `rust_tenferro_sparse_cg/` | Rust | 疎行列 CG PoC | COO builder/import から固定 CSR pattern と tenferro values を構築し、共通の `LinearOperator` 上で CSR と matrix-free の CG を解く |
| `rust_tenferro_opt/` | Rust | Jacobi | 格子は `TypedTensor` のまま。[tenferro-rs#1736](https://github.com/tensor4all/tenferro-rs/issues/1736) の列優先ホストビューで一度検証し、内側は軸 0 のレーンを回す |
| `rust_hataori/` | Rust | Jacobi | `rust_tenferro` と同じステンシル。内部を 2×2 の四象限に分け、[Hataori](https://github.com/shinaoka/hataori-rs) の `map_in`（Rayon、`LocalMode::Outer`）で並列更新 |
| `rust_ndarray/` | Rust | Jacobi | `ndarray` のスライスと `Zip` |
| `rust_unsafe/` | Rust | Jacobi | `pulp` でNEON/x86 SIMD/Scalarへdispatch。8本の独立SIMDチェーンと2反復の行パイプライン。更新幅は1000反復ごとに還元 |
| `rust_tenferro_fft/` | Rust | DST-I | 奇関数延長の軸方向 FFT（tenferro-fft） |

`rust_tenferro/` と `rust_hataori/` は、求解中は `Vec<f64>` を回し、誤差の `sub` / `abs` / `reduce_sum` に tenferro を使う。
`rust_tenferro_opt/` は求解中も `TypedTensor` を保持し、検証済み列優先ビュー経由で同じステンシルを書く。
`rust_unsafe/` の2反復パイプラインは、1反復目で行 `j+1` を生成した直後、以後参照されない旧行 `j` に2反復目の結果を書き戻す。各反復の数値結果は通常のJacobi反復と同一である。
`rust_unsafe/` のSIMD幅と命令セットは、`pulp::Arch` が実行時に選択する。

`rust_tenferro/`、`rust_tenferro_opt/`、`rust_hataori/` は `target-cpu=native` でビルドする。

gfortran は `-fcheck=bounds` を付けない限り添字検査を入れない。
したがって `gfortran -O3` は、すでに Julia の `@inbounds` と同じ前提である。
C++ の `g++ -O3` も同様で、`std::vector::operator[]` に添字検査は入らない。

## 計測

以下は `julia_unsafe/` の temporal blocking 導入前の計測値。現在の実装の比較は `julia --project=julia_unsafe julia_unsafe/benchmark.jl 5` で再現できる。

単一スレッドの Jacobi 実装を同じ条件（N=401、10万反復）で計測した。
Apple M4 上で各実装を交互に実行し、最速値を示す。
熱・負荷で絶対値は大きく揺れる（同セッションでも後半は 20–40% 遅くなることがある）ため、同一セッション内の相対比較に留める。

| 実装 | 1回目 | 2回目 | 3回目 | 最速 | ホットパス |
|---|---|---|---|---|---|
| `cxx/` | 3.73 | 3.97 | 4.49 | **3.73s** | 値型 `Grid` と O(1) swap |
| `julia_unsafe/` | 3.86 | 4.09 | 4.84 | **3.86s** | 生ポインタ + `@simd` |
| `julia/` | 4.17 | 4.29 | 4.50 | **4.17s** | `@inbounds` 2D + `@turbo`（LoopVectorization） |
| `rust_unsafe/` | 4.44 | 4.60 | 5.07 | **4.44s** | pulp SIMD + 8チェーン + 2反復パイプライン |
| `fortran/` | 4.53 | 4.97 | 5.29 | **4.53s** | ポインタの付け替え |

計測区間は求解ループのみで、配列確保・誤差計算・作図は含まない。
数値は全実装で一致する（`max error = 4.575793e-02`、`L2 error = 1.142521e-03`）。

## 実行

Julia はプロジェクトディレクトリを `--project` で指定する。

```bash
julia --project=./julia ./julia/poisson.jl
julia --project=./julia_unsafe ./julia_unsafe/poisson.jl
julia --project=./julia_fft ./julia_fft/poisson.jl
julia --project=./julia_sparse_cg ./julia_sparse_cg/poisson.jl
```

初回は依存の取得が必要なら、先に `julia --project=<dir> -e 'using Pkg; Pkg.instantiate()'` を実行する。

`julia_unsafe/poisson.jl` は `unsafe_load` を使う自己完結の Jacobi 実装。`runtests.jl` と `benchmark.jl` で検証・比較できる。
`julia/` のホットループは `LoopVectorization` の `@turbo` で展開する（`julia/Project.toml` に依存を追加済み）。`@turbo` は生ポインタの `unsafe_load` を扱えないため、`julia_unsafe/` には適用していない。

Rust は各 crate のディレクトリでリリースビルドする。

```bash
cargo run --release --manifest-path rust_ndarray/Cargo.toml
cargo run --release --manifest-path rust_tenferro/Cargo.toml
cargo run --release --manifest-path rust_tenferro_sparse_cg/Cargo.toml
cargo run --release --manifest-path rust_tenferro_opt/Cargo.toml
cargo run --release --manifest-path rust_unsafe/Cargo.toml
cargo run --release --manifest-path rust_tenferro_fft/Cargo.toml
cargo run --release --manifest-path rust_hataori/Cargo.toml
# スレッド数は引数または HATAORI_THREADS（省略時は 4。領域は常に 2×2）
cargo run --release --manifest-path rust_hataori/Cargo.toml -- 4
```

Fortran は `gfortran -O3` でビルドする。

```bash
./fortran/build.sh
./fortran/poisson
```

C++ は `g++ -O3 -std=c++23` でビルドする。

```bash
./cxx/build.sh
./cxx/poisson
```

求解時間は標準出力の `time =` 行である。
図は各ディレクトリに PNG で保存する。
Fortran の図はヒートマップのみである。
