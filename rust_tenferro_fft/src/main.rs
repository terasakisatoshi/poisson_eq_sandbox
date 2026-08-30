//! 2-D Poisson equation on the unit square, solved in one shot by DST-I.
//!
//! Same problem as `julia/poisson.jl`. Homogeneous Dirichlet on a rectangle
//! uses a discrete sine transform, implemented as an unnormalized FFT of the
//! odd extension (tenferro-fft). This is the exact inverse of the 5-point
//! Laplacian, not Jacobi iteration.
//!
//! ```text
//!   -Δu = f    in Ω = (0,1) × (0,1)
//!      u = 0   on ∂Ω
//!
//!   u(x,y) = sin(πx) sin(πy)
//!   f(x,y) = 2π² sin(πx) sin(πy)
//! ```

use std::error::Error;
use std::f64::consts::PI;
use std::path::PathBuf;
use std::time::Instant;

use num_complex::Complex64;
use plotters::prelude::*;
use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
use tenferro_fft::{FftNorm, TensorFftExt};
use tenferro_runtime::prelude::*;
use tenferro_runtime::BackendSessionHost;

fn u_exact(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}

fn f(x: f64, y: f64) -> f64 {
    2.0 * PI * PI * (PI * x).sin() * (PI * y).sin()
}

/// Column-major index for a compact `[n, n]` tensor (leftmost axis fastest).
fn idx(i: usize, j: usize, n: usize) -> usize {
    i + n * j
}

fn surface_style(z: &f64) -> ShapeStyle {
    HSLColor(0.65 * (1.0 - *z), 0.7, 0.45).filled()
}

fn viridis(t: f64) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    let stops = [
        (0.0, [68.0, 1.0, 84.0]),
        (0.25, [59.0, 82.0, 139.0]),
        (0.5, [33.0, 145.0, 140.0]),
        (0.75, [94.0, 201.0, 98.0]),
        (1.0, [253.0, 231.0, 37.0]),
    ];
    let mut i = 0;
    while i + 1 < stops.len() && t > stops[i + 1].0 {
        i += 1;
    }
    let (t0, c0) = stops[i];
    let (t1, c1) = stops[(i + 1).min(stops.len() - 1)];
    let a = if (t1 - t0).abs() < 1e-12 {
        0.0
    } else {
        (t - t0) / (t1 - t0)
    };
    RGBColor(
        (c0[0] + a * (c1[0] - c0[0])).round() as u8,
        (c0[1] + a * (c1[1] - c0[1])).round() as u8,
        (c0[2] + a * (c1[2] - c0[2])).round() as u8,
    )
}

fn fft_along_axis(
    backend: &mut CpuBackend,
    input: &Tensor,
    axis: isize,
) -> Result<Tensor, Box<dyn Error>> {
    backend
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |exec| {
                input.fft(None, axis, FftNorm::Backward, exec)
            })
            .expect("CpuBackend must expose a CPU execution session")
        })
        .map_err(|e| e.into())
}

/// Odd extension of length `2(M+1)` along axis 0 (contiguous columns).
///
/// `s[0]=0`, `s[1..=M]=x`, `s[L]=0`, `s[L+1..]=-reverse(x)`, `L=M+1`.
fn odd_extend_axis0(x: &[f64], m: usize, ncols: usize) -> (Vec<f64>, usize) {
    let l = m + 1;
    let nfft = 2 * l;
    let mut s = vec![0.0; nfft * ncols];
    for j in 0..ncols {
        let src = &x[j * m..(j + 1) * m];
        let dst = &mut s[j * nfft..(j + 1) * nfft];
        dst[1..=m].copy_from_slice(src);
        for k in 1..=m {
            dst[l + k] = -src[m - k];
        }
    }
    (s, nfft)
}

/// Odd extension of length `2(M+1)` along axis 1 (slow axis).
fn odd_extend_axis1(x: &[f64], nrows: usize, m: usize) -> (Vec<f64>, usize) {
    let l = m + 1;
    let nfft = 2 * l;
    let mut s = vec![0.0; nrows * nfft];
    for j in 0..m {
        for i in 0..nrows {
            s[i + nrows * (j + 1)] = x[i + nrows * j];
        }
    }
    for k in 1..=m {
        for i in 0..nrows {
            s[i + nrows * (l + k)] = -x[i + nrows * (m - k)];
        }
    }
    (s, nfft)
}

fn take_dst_axis0(spec: &[Complex64], nfft: usize, ncols: usize, m: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * ncols];
    for j in 0..ncols {
        for i in 0..m {
            out[i + m * j] = -spec[(i + 1) + nfft * j].im;
        }
    }
    out
}

fn take_dst_axis1(spec: &[Complex64], nrows: usize, _nfft: usize, m: usize) -> Vec<f64> {
    let mut out = vec![0.0; nrows * m];
    for j in 0..m {
        for i in 0..nrows {
            out[i + nrows * j] = -spec[i + nrows * (j + 1)].im;
        }
    }
    out
}

/// 2-D DST-I via batched 1-D FFTs of the odd extension along each axis.
///
/// Forward: `S_k = -Im(FFT(s)[k])` for `k = 1..=M`. Inverse is the same
/// transform divided by `2(M+1)` per axis.
fn dst2d(
    backend: &mut CpuBackend,
    x: &[f64],
    m: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let (s0, nfft) = odd_extend_axis0(x, m, m);
    let t0 = Tensor::from_vec_col_major(vec![nfft, m], s0)?;
    let spec0 = fft_along_axis(backend, &t0, 0)?;
    let y = take_dst_axis0(spec0.as_slice::<Complex64>()?, nfft, m, m);

    let (s1, nfft) = odd_extend_axis1(&y, m, m);
    let t1 = Tensor::from_vec_col_major(vec![m, nfft], s1)?;
    let spec1 = fft_along_axis(backend, &t1, 1)?;
    Ok(take_dst_axis1(
        spec1.as_slice::<Complex64>()?,
        m,
        nfft,
        m,
    ))
}

/// Exact solve of the 5-point Dirichlet Poisson system by 2-D DST-I.
fn solve_dst(
    backend: &mut CpuBackend,
    rhs: &[f64],
    n: usize,
    h: f64,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let m = n - 2;
    let l = m + 1;
    debug_assert_eq!(l, n - 1);

    let mut interior = vec![0.0; m * m];
    for j in 0..m {
        for i in 0..m {
            interior[i + m * j] = rhs[idx(i + 1, j + 1, n)];
        }
    }

    let mut fhat = dst2d(backend, &interior, m)?;

    // λ_{p,q} = [2-2cos(pπ/L) + 2-2cos(qπ/L)] / h², p,q = 1..=M.
    let h2 = h * h;
    for q in 1..=m {
        let lam_y = 2.0 - 2.0 * (q as f64 * PI / l as f64).cos();
        for p in 1..=m {
            let lam_x = 2.0 - 2.0 * (p as f64 * PI / l as f64).cos();
            fhat[(p - 1) + m * (q - 1)] /= (lam_x + lam_y) / h2;
        }
    }

    let mut uint = dst2d(backend, &fhat, m)?;
    let scale = (2.0 * l as f64).powi(2);
    for v in &mut uint {
        *v /= scale;
    }

    let mut u = vec![0.0; n * n];
    for j in 0..m {
        for i in 0..m {
            u[idx(i + 1, j + 1, n)] = uint[i + m * j];
        }
    }
    Ok(u)
}

fn grid_lookup(data: &[f64], n: usize, h: f64, x: f64, y: f64) -> f64 {
    let i = ((x / h).round() as usize).min(n - 1);
    let j = ((y / h).round() as usize).min(n - 1);
    data[idx(i, j, n)]
}

fn save_plot(
    path: &str,
    xs: &[f64],
    u: &[f64],
    ue: &[f64],
    err: &[f64],
    n: usize,
    h: f64,
    max_error: f64,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (2400, 520)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(4, 12, 4, 28);
    let (left, rest) = root.split_horizontally(740);
    let (mid, right) = rest.split_horizontally(740);

    let pitch = 30.0_f64.to_radians();
    let yaw = 45.0_f64.to_radians();

    let draw_surface = |area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
                        title: &str,
                        values: &[f64],
                        analytic: bool|
     -> Result<(), Box<dyn Error>> {
        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 18))
            .margin(8)
            .build_cartesian_3d(0.0..1.0, 0.0..1.05, 0.0..1.0)?;

        chart.with_projection(|mut p| {
            p.pitch = pitch;
            p.yaw = yaw;
            p.scale = 0.85;
            p.into_matrix()
        });

        chart
            .configure_axes()
            .light_grid_style(BLACK.mix(0.15))
            .max_light_lines(4)
            .x_labels(5)
            .y_labels(5)
            .z_labels(5)
            .draw()?;

        chart.draw_series(
            SurfaceSeries::xoz(
                xs.iter().copied(),
                xs.iter().copied(),
                |x, y| {
                    if analytic {
                        u_exact(x, y)
                    } else {
                        grid_lookup(values, n, h, x, y)
                    }
                },
            )
            .style_func(&surface_style),
        )?;
        Ok(())
    };

    draw_surface(&left, "Exact solution", ue, true)?;
    draw_surface(&mid, "Numerical solution", u, false)?;

    let (heat, bar) = right.split_horizontally(700);
    let mut chart = ChartBuilder::on(&heat)
        .caption("Absolute error", ("sans-serif", 18))
        .margin(8)
        .x_label_area_size(28)
        .y_label_area_size(36)
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)?;

    chart
        .configure_mesh()
        .x_desc("x")
        .y_desc("y")
        .draw()?;

    let scale = if max_error > 0.0 { max_error } else { 1.0 };
    chart.draw_series((0..n).flat_map(|i| {
        (0..n).map(move |j| {
            let t = (err[idx(i, j, n)] / scale).clamp(0.0, 1.0);
            Rectangle::new(
                [(xs[i], xs[j]), (xs[i] + h, xs[j] + h)],
                viridis(t).filled(),
            )
        })
    }))?;

    let mut colorbar = ChartBuilder::on(&bar)
        .caption("|u - u_exact|", ("sans-serif", 14))
        .margin_left(8)
        .margin_right(28)
        .margin_top(28)
        .margin_bottom(36)
        .y_label_area_size(72)
        .build_cartesian_2d(0.0..1.0, 0.0..scale)?;

    colorbar
        .configure_mesh()
        .disable_x_mesh()
        .disable_x_axis()
        .y_label_formatter(&|v| format!("{v:.1e}"))
        .axis_style(BLACK)
        .draw()?;

    const BARS: usize = 64;
    colorbar.draw_series((0..BARS).map(|k| {
        let y0 = scale * k as f64 / BARS as f64;
        let y1 = scale * (k + 1) as f64 / BARS as f64;
        let t = (k as f64 + 0.5) / BARS as f64;
        Rectangle::new([(0.0, y0), (1.0, y1)], viridis(t).filled())
    }))?;

    root.present()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    const N: usize = 401;

    let h = 1.0 / (N - 1) as f64;
    let xs: Vec<f64> = (0..N).map(|i| i as f64 * h).collect();

    println!("N = {N}");
    println!("h = {h:.6e}");

    let mut rhs = vec![0.0; N * N];
    for j in 0..N {
        for i in 0..N {
            rhs[idx(i, j, N)] = f(xs[i], xs[j]);
        }
    }

    let mut backend = CpuBackend::new();

    let start_time = Instant::now();
    let u = solve_dst(&mut backend, &rhs, N, h)?;
    let duration = start_time.elapsed();
    println!();
    println!("time = {:.6} seconds", duration.as_secs_f64());

    let u = TypedTensor::<f64>::from_vec_col_major(vec![N, N], u)?;

    let mut ue = TypedTensor::<f64>::zeros(vec![N, N])?;
    {
        let data = ue.host_data_mut()?;
        for j in 0..N {
            for i in 0..N {
                data[idx(i, j, N)] = u_exact(xs[i], xs[j]);
            }
        }
    }

    let diff = u.sub(&ue, &mut backend)?;
    let abs_err = diff.abs(&mut backend)?;
    let sq = diff.mul(&diff, &mut backend)?;
    let sum_sq = sq.reduce_sum(&[0, 1], &mut backend)?;
    let max_error = abs_err
        .as_slice()?
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let l2_error = (sum_sq.as_slice()?[0] * h * h).sqrt();

    println!();
    println!("max error = {max_error:.6e}");
    println!("L2 error  = {:.6e}", l2_error / (N as f64).sqrt());

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("poisson_fft.png");
    let out_str = out.to_string_lossy();
    save_plot(
        &out_str,
        &xs,
        u.as_slice()?,
        ue.as_slice()?,
        abs_err.as_slice()?,
        N,
        h,
        max_error,
    )?;
    println!("saved {out_str}");

    Ok(())
}
