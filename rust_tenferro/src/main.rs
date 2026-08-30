//! 2-D Poisson equation on the unit square, solved by Jacobi iteration.
//!
//! Same problem and algorithm as `julia/poisson.jl`, using tenferro tensors.
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

use plotters::prelude::*;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::prelude::*;

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

fn jacobi<const N: usize>(
    u: &mut Vec<f64>,
    u_new: &mut Vec<f64>,
    rhs: &[f64],
    h: f64,
    tol: f64,
    maxiter: usize,
) -> (usize, f64) {
    let h2 = h * h;
    let mut update_error = f64::INFINITY;
    let mut iterations = 0;

    for iter in 1..=maxiter {
        update_error = 0.0;

        // Same 5-point Jacobi stencil as julia/poisson.jl.
        // Column j is the contiguous range `j*N .. (j+1)*N`; the interior
        // is `start..end`, so the inner loop walks memory in order.
        for j in 1..N - 1 {
            let start = j * N + 1;
            let end = j * N + (N - 1);
            for k in start..end {
                let val = 0.25
                    * (u[k + 1] + u[k - 1] + u[k + N] + u[k - N] + h2 * rhs[k]);
                update_error = update_error.max((val - u[k]).abs());
                u_new[k] = val;
            }
        }

        std::mem::swap(u, u_new);
        iterations = iter;

        if iter % 1000 == 0 {
            println!("iteration = {iter:6}, update error = {update_error:.6e}");
        }

        if update_error < tol {
            break;
        }
    }

    (iterations, update_error)
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
                [
                    (xs[i], xs[j]),
                    (xs[i] + h, xs[j] + h),
                ],
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
    const TOL: f64 = 1e-10;
    const MAXITER: usize = 100_000;

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

    let mut u = vec![0.0; N * N];
    let mut u_new = vec![0.0; N * N];

    let start_time = Instant::now();
    let (iterations, update_error) = jacobi::<N>(&mut u, &mut u_new, &rhs, h, TOL, MAXITER);
    let duration = start_time.elapsed();
    println!();
    println!("time = {:.6} seconds", duration.as_secs_f64());
    println!("Jacobi iterations = {iterations}");
    println!("final update error = {update_error:.6e}");

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

    let mut backend = CpuBackend::new();
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

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("poisson_jacobi.png");
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
