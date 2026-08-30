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

格子点が \(N^2\) 個のとき、Jacobi の収束にはだいたい \(O(N^2)\) 回かかるので計算量は \(O(N^4)\) である。
DST は \(O(N^2\log N)\) である。

## ディレクトリ

| パス | 言語 | 解法 | ホットパス |
|---|---|---|---|
| `julia/` | Julia | Jacobi | 境界チェック付きの二重ループ。毎反復 `u .= u_new` でコピーする |
| `julia_unsafe/` | Julia | Jacobi | `@inbounds`。バッファは入れ替える |
| `fortran/` | Fortran | Jacobi | 既定では境界チェックなし（`julia_unsafe` 相当）。バッファはポインタの付け替え |
| `julia_fft/` | Julia | DST-I | FFTW。比較のため FFTW と BLAS は 1 スレッド |
| `rust_tenferro/` | Rust | Jacobi | `Vec<f64>` の安全な添字。バッファは `swap`。tenferro は求解後の誤差だけ |
| `rust_ndarray/` | Rust | Jacobi | `ndarray` のスライスと `Zip` |
| `rust_tenferro_unsafe/` | Rust | Jacobi | 境界チェックなしのポインタ読み書き。tenferro は求解後の誤差だけ |
| `rust_tenferro_fft/` | Rust | DST-I | 奇関数延長の軸方向 FFT（tenferro-fft） |

Rust の Jacobi 実装は、求解中は生の配列を回す。
tenferro のテンソル演算は、誤差の `sub` / `abs` / `reduce_sum` に使う。

`rust/` と `rust_tenferro_unsafe/` は `target-cpu=native` でビルドする。

gfortran は `-fcheck=bounds` を付けない限り添字検査を入れない。
したがって `gfortran -O3` は、すでに Julia の `@inbounds` と同じ前提である。

## 実行

Julia はプロジェクトディレクトリを `--project` で指定する。

```bash
julia --project=./julia ./julia/poisson.jl
julia --project=./julia_unsafe ./julia_unsafe/poisson.jl
julia --project=./julia_fft ./julia_fft/poisson.jl
```

初回は依存の取得が必要なら、先に `julia --project=<dir> -e 'using Pkg; Pkg.instantiate()'` を実行する。

Rust は各 crate のディレクトリでリリースビルドする。

```bash
cargo run --release --manifest-path rust/Cargo.toml
cargo run --release --manifest-path rust_ndarray/Cargo.toml
cargo run --release --manifest-path rust_tenferro_unsafe/Cargo.toml
cargo run --release --manifest-path rust_tenferro_fft/Cargo.toml
```

Fortran は `gfortran -O3` でビルドする。

```bash
./fortran/build.sh
./fortran/poisson
```

求解時間は標準出力の `time =` 行である。
図は各ディレクトリに PNG で保存する。
Fortran の図はヒートマップのみである。
