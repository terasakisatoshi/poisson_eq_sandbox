# 単位正方形上の Poisson 方程式

単位正方形上の Dirichlet 問題を、Julia、C++、Rust、Fortran で同じ格子と右辺に対して解く。

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
| `julia_unsafe/` | Julia | Jacobi | 8反復の列方向 wavefront と、`VecElement` による2反復の SIMD 処理。中間値をレジスタで再利用。[検証と計測の手順](julia_unsafe/README.md) |
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

## `julia_unsafe` の最適化

`julia_unsafe/` は、8反復を列ごとにずらして進める **wavefront** で、直前に読み書きした列をキャッシュから再利用する。
各段階では2反復分を同じ SIMD ループで計算し、1反復目の値をレジスタから2反復目へ渡す。
右辺の `h² * rhs` は求解の開始時に一度だけ計算する。

SIMD ループは `NTuple{16,VecElement{Float64}}` で16点ずつ処理する。
残りの点は8点、4点、2点、1点に分け、端数の大部分も SIMD で計算する。
Apple M4 上で `@code_native` と C++ の逆アセンブルを比較すると、変更前の Julia と C++ は8点ずつ処理していた。
今回の変更で、32点更新あたりの128ビットベクトルの読み込みは68本から58本に減った。
これは主ループの機械語から数えた値で、ループの準備と端数処理は含まない。

演算の順序は変更前の `julia_unsafe` に合わせ、`@fastmath` や積和演算の融合は使わない。
ランダムな右辺、非ゼロの境界値、SIMD の端数、奇数回の反復、収束判定を含む1,241件のテストが通っている。
401×401の格子で10万反復した比較でも、全5回で変更前の解とビット単位で一致し、最終更新幅と反復回数も一致した。

## 計測

### SIMD 最適化後の Julia と C++

Apple M4、Julia 1.13.0、Apple clang 21.0.0 で、単一スレッド、N=401、最大10万反復、許容誤差 `1e-10` を条件に計測した。
この環境の `g++` は Apple clang を呼び出し、C++ のビルド条件は `-O3 -std=c++23` である。
各回で実行順を変え、変更前の Julia、変更後の Julia、C++ を比較した。
「変更前」は8反復の wavefront を導入済みで、2反復を同じ SIMD ループにまとめる前の実装を指す。

| 回 | Julia 変更前 (秒) | Julia 変更後 (秒) | C++ (秒) |
|---|---:|---:|---:|
| 1 | 4.699281 | 3.385435 | 3.846381 |
| 2 | 3.637510 | 3.504544 | 3.913423 |
| 3 | 4.079874 | 3.818548 | 3.695129 |
| 4 | 3.846951 | 3.413451 | 3.707956 |
| 5 | 3.633087 | 3.517270 | 3.708979 |
| 中央値 | 3.846951 | **3.504544** | 3.708979 |
| 最速 | 3.633087 | 3.385435 | 3.695129 |

中央値では変更前の Julia 比で8.9%、C++ 比で5.5%短縮した。
5回中1回は C++ より遅く、変更前の Julia の初回にも大きな揺れがあるため、結果はこの環境での比較として扱う。

計測区間は求解処理で、Julia は小さい格子でウォームアップしてから測る。
両言語ともコンパイル、入力格子の準備、誤差計算、作図を除外する。
Julia の計測には、求解中の右辺の定数倍と配列確保、収束判定、進捗出力を含める。
比較スクリプトでは Julia の進捗出力を `/dev/null` に送り、C++ の標準出力から求解時間を読み取る。

現在の Julia と C++ の比較、および数値検証は次のコマンドで実行できる。

```bash
julia --project=julia_unsafe julia_unsafe/runtests.jl
julia --project=julia_unsafe julia_unsafe/benchmark.jl 5
```

### 最適化前の各言語の比較

以下は `julia_unsafe/` に wavefront を導入する前の記録である。

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
表示された誤差は全実装で一致する（`max error = 4.575793e-02`、`L2 error = 1.142521e-03`）。

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
