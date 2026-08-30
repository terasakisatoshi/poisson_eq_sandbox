//! 2-D Poisson equation on the unit square, solved by Jacobi iteration.
//!
//! Same problem and algorithm as `julia/poisson.jl`, using ndarray arrays.
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

use ndarray::{s, Array1, Array2, Zip};
use plotters::prelude::*;

fn u_exact(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}

fn f(x: f64, y: f64) -> f64 {
    2.0 * PI * PI * (PI * x).sin() * (PI * y).sin()
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

fn jacobi(
    u: &mut Array2<f64>,
    u_new: &mut Array2<f64>,
    rhs: &Array2<f64>,
    h: f64,
    tol: f64,
    maxiter: usize,
) -> (usize, f64) {
    let n = u.nrows();
    debug_assert_eq!(u.ncols(), n);
    debug_assert_eq!(u_new.dim(), u.dim());
    debug_assert_eq!(rhs.dim(), u.dim());

    let h2 = h * h;
    let mut update_error = f64::INFINITY;
    let mut iterations = 0;

    for iter in 1..=maxiter {
        // Interior points only. Boundary points remain zero (Dirichlet).
        {
            let mut interior = u_new.slice_mut(s![1..n - 1, 1..n - 1]);
            Zip::from(&mut interior)
                .and(u.slice(s![2..n, 1..n - 1]))
                .and(u.slice(s![0..n - 2, 1..n - 1]))
                .and(u.slice(s![1..n - 1, 2..n]))
                .and(u.slice(s![1..n - 1, 0..n - 2]))
                .and(rhs.slice(s![1..n - 1, 1..n - 1]))
                .for_each(|un, &up, &um, &ur, &ul, &r| {
                    *un = 0.25 * (up + um + ur + ul + h2 * r);
                });
        }

        update_error = Zip::from(u_new.slice(s![1..n - 1, 1..n - 1]))
            .and(u.slice(s![1..n - 1, 1..n - 1]))
            .fold(0.0_f64, |acc, &un, &uo| acc.max((un - uo).abs()));

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

fn grid_lookup(data: &Array2<f64>, h: f64, x: f64, y: f64) -> f64 {
    let n = data.nrows();
    let i = ((x / h).round() as usize).min(n - 1);
    let j = ((y / h).round() as usize).min(n - 1);
    data[[i, j]]
}

fn save_plot(
    path: &str,
    xs: &Array1<f64>,
    u: &Array2<f64>,
    ue: &Array2<f64>,
    err: &Array2<f64>,
    h: f64,
    max_error: f64,
) -> Result<(), Box<dyn Error>> {
    let n = xs.len();
    let root = BitMapBackend::new(path, (2400, 520)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(4, 12, 4, 28);
    let (left, rest) = root.split_horizontally(740);
    let (mid, right) = rest.split_horizontally(740);

    let pitch = 30.0_f64.to_radians();
    let yaw = 45.0_f64.to_radians();

    let draw_surface = |area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
                        title: &str,
                        values: &Array2<f64>,
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
            SurfaceSeries::xoz(xs.iter().copied(), xs.iter().copied(), |x, y| {
                if analytic {
                    u_exact(x, y)
                } else {
                    grid_lookup(values, h, x, y)
                }
            })
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

    chart.configure_mesh().x_desc("x").y_desc("y").draw()?;

    let scale = if max_error > 0.0 { max_error } else { 1.0 };
    chart.draw_series((0..n).flat_map(|i| {
        (0..n).map(move |j| {
            let t = (err[[i, j]] / scale).clamp(0.0, 1.0);
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
    const TOL: f64 = 1e-10;
    const MAXITER: usize = 100_000;

    let h = 1.0 / (N - 1) as f64;
    let xs = Array1::linspace(0.0, 1.0, N);

    println!("N = {N}");
    println!("h = {h:.6e}");

    let rhs = Array2::from_shape_fn((N, N), |(i, j)| f(xs[i], xs[j]));
    let mut u = Array2::<f64>::zeros((N, N));
    let mut u_new = Array2::<f64>::zeros((N, N));

    let start_time = Instant::now();
    let (iterations, update_error) = jacobi(&mut u, &mut u_new, &rhs, h, TOL, MAXITER);
    let duration = start_time.elapsed();
    println!();
    println!("time = {:.6} seconds", duration.as_secs_f64());
    println!("Jacobi iterations = {iterations}");
    println!("final update error = {update_error:.6e}");

    let ue = Array2::from_shape_fn((N, N), |(i, j)| u_exact(xs[i], xs[j]));
    let diff = &u - &ue;
    let err = diff.mapv(f64::abs);
    let max_error = err.fold(0.0_f64, |a, &x| a.max(x));
    let l2_error = (diff.mapv(|x| x * x).sum() * h * h).sqrt();

    println!();
    println!("max error = {max_error:.6e}");
    println!("L2 error  = {:.6e}", l2_error / (N as f64).sqrt());

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("poisson_jacobi.png");
    let out_str = out.to_string_lossy();
    save_plot(&out_str, &xs, &u, &ue, &err, h, max_error)?;
    println!("saved {out_str}");

    Ok(())
}
