//! 2-D Poisson equation on the unit square, solved by Jacobi iteration.
//!
//! Same problem and Jacobi stencil as `julia_unsafe/poisson.jl`. The portable
//! `pulp` hot loop keeps four independent SIMD chains; runtime dispatch picks
//! NEON, an available x86 SIMD level, or scalar execution. Two sweeps are
//! pipelined by row so intermediate rows are consumed while still hot in
//! cache. The convergence reduction is evaluated every 1000 sweeps.
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

fn u_exact(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}

fn f(x: f64, y: f64) -> f64 {
    2.0 * PI * PI * (PI * x).sin() * (PI * y).sin()
}

/// Column-major index for a compact `[n, n]` buffer (leftmost axis fastest).
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

#[inline(always)]
unsafe fn jacobi_row<S: pulp::Simd, const N: usize, const TRACK_ERROR: bool>(
    simd: S,
    u_ptr: *const f64,
    out: *mut f64,
    rhs_ptr: *const f64,
    j: usize,
    h2: f64,
) -> f64 {
    let col = j * N;
    let width = N - 2;
    let u_j = u_ptr.add(col);
    let u_jm = u_ptr.add(col - N);
    let u_jp = u_ptr.add(col + N);
    let rhs_j = rhs_ptr.add(col);
    let dst = out.add(col);

    let left = std::slice::from_raw_parts(u_j, width);
    let center = std::slice::from_raw_parts(u_j.add(1), width);
    let right = std::slice::from_raw_parts(u_j.add(2), width);
    let below = std::slice::from_raw_parts(u_jm.add(1), width);
    let above = std::slice::from_raw_parts(u_jp.add(1), width);
    let rhs = std::slice::from_raw_parts(rhs_j.add(1), width);
    let output = std::slice::from_raw_parts_mut(dst.add(1), width);

    let (left_v, left_tail) = S::as_simd_f64s(left);
    let (center_v, center_tail) = S::as_simd_f64s(center);
    let (right_v, right_tail) = S::as_simd_f64s(right);
    let (below_v, below_tail) = S::as_simd_f64s(below);
    let (above_v, above_tail) = S::as_simd_f64s(above);
    let (rhs_v, rhs_tail) = S::as_simd_f64s(rhs);
    let (output_v, output_tail) = S::as_mut_simd_f64s(output);

    debug_assert_eq!(left_v.len(), output_v.len());
    debug_assert_eq!(left_tail.len(), output_tail.len());

    let zero = simd.splat_f64s(0.0);
    let h2v = simd.splat_f64s(h2);
    let qtr = simd.splat_f64s(0.25);
    let mut e0 = zero;
    let mut e1 = zero;
    let mut e2 = zero;
    let mut e3 = zero;

    macro_rules! point_vector {
        ($k:expr, $error:ident) => {{
            let k = $k;
            let mut value = simd.add_f64s(*right_v.get_unchecked(k), *left_v.get_unchecked(k));
            value = simd.add_f64s(value, *above_v.get_unchecked(k));
            value = simd.add_f64s(value, *below_v.get_unchecked(k));
            value = simd.add_f64s(value, simd.mul_f64s(*rhs_v.get_unchecked(k), h2v));
            value = simd.mul_f64s(value, qtr);
            if TRACK_ERROR {
                $error = simd.max_f64s(
                    $error,
                    simd.abs_f64s(simd.sub_f64s(value, *center_v.get_unchecked(k))),
                );
            }
            *output_v.get_unchecked_mut(k) = value;
        }};
    }

    let vector_groups = left_v.len() / 4;
    for group in 0..vector_groups {
        let k = group * 4;
        point_vector!(k, e0);
        point_vector!(k + 1, e1);
        point_vector!(k + 2, e2);
        point_vector!(k + 3, e3);
    }
    for k in vector_groups * 4..left_v.len() {
        point_vector!(k, e0);
    }

    let mut local = if TRACK_ERROR {
        let e01 = simd.max_f64s(e0, e1);
        let e23 = simd.max_f64s(e2, e3);
        simd.reduce_max_f64s(simd.max_f64s(e01, e23))
    } else {
        0.0
    };

    for k in 0..left_tail.len() {
        let value = 0.25
            * (*left_tail.get_unchecked(k)
                + *right_tail.get_unchecked(k)
                + *above_tail.get_unchecked(k)
                + *below_tail.get_unchecked(k)
                + h2 * *rhs_tail.get_unchecked(k));
        if TRACK_ERROR {
            let error = (value - *center_tail.get_unchecked(k)).abs();
            if error > local {
                local = error;
            }
        }
        *output_tail.get_unchecked_mut(k) = value;
    }

    local
}

#[inline(always)]
unsafe fn jacobi_sweep<S: pulp::Simd, const N: usize, const TRACK_ERROR: bool>(
    simd: S,
    u: &[f64],
    u_new: &mut [f64],
    rhs: &[f64],
    h2: f64,
) -> f64 {
    let mut update_error = 0.0_f64;
    for j in 1..N - 1 {
        let local = jacobi_row::<S, N, TRACK_ERROR>(
            simd,
            u.as_ptr(),
            u_new.as_mut_ptr(),
            rhs.as_ptr(),
            j,
            h2,
        );
        if local > update_error {
            update_error = local;
        }
    }
    update_error
}

#[inline(always)]
unsafe fn jacobi_two_sweeps<S: pulp::Simd, const N: usize, const TRACK_SECOND: bool>(
    simd: S,
    u: &mut [f64],
    tmp: &mut [f64],
    rhs: &[f64],
    h2: f64,
) -> f64 {
    let u_ptr = u.as_mut_ptr();
    let tmp_ptr = tmp.as_mut_ptr();
    let rhs_ptr = rhs.as_ptr();
    let mut update_error = 0.0_f64;

    jacobi_row::<S, N, false>(simd, u_ptr, tmp_ptr, rhs_ptr, 1, h2);
    for j in 1..N - 1 {
        if j < N - 2 {
            jacobi_row::<S, N, false>(simd, u_ptr, tmp_ptr, rhs_ptr, j + 1, h2);
        }
        let local = jacobi_row::<S, N, TRACK_SECOND>(simd, tmp_ptr, u_ptr, rhs_ptr, j, h2);
        if local > update_error {
            update_error = local;
        }
    }

    update_error
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_sweep<const N: usize>(u: &[f64], out: &mut [f64], rhs: &[f64], h2: f64) -> f64 {
        let mut update_error = 0.0_f64;
        for j in 1..N - 1 {
            for i in 1..N - 1 {
                let k = i + N * j;
                let value = 0.25 * (u[k + 1] + u[k - 1] + u[k + N] + u[k - N] + h2 * rhs[k]);
                let error = (value - u[k]).abs();
                if error > update_error {
                    update_error = error;
                }
                out[k] = value;
            }
        }
        update_error
    }

    #[test]
    fn pipelined_pair_matches_two_separate_sweeps() {
        const N: usize = 25;
        let h2 = 1.0 / ((N - 1) * (N - 1)) as f64;
        let mut initial = vec![0.0; N * N];
        let mut rhs = vec![0.0; N * N];
        for j in 1..N - 1 {
            for i in 1..N - 1 {
                let k = i + N * j;
                initial[k] = ((3 * i + 5 * j) % 17) as f64 * 1e-3;
                rhs[k] = ((7 * i + 11 * j) % 23) as f64 * 1e-2;
            }
        }

        let mut first = vec![0.0; N * N];
        let mut expected = vec![0.0; N * N];
        scalar_sweep::<N>(&initial, &mut first, &rhs, h2);
        let expected_error = scalar_sweep::<N>(&first, &mut expected, &rhs, h2);

        let mut pulp_actual = initial;

        struct PulpPair<'a, const N: usize> {
            u: &'a mut [f64],
            tmp: &'a mut [f64],
            rhs: &'a [f64],
            h2: f64,
        }
        impl<const N: usize> pulp::WithSimd for PulpPair<'_, N> {
            type Output = f64;

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                unsafe {
                    jacobi_two_sweeps::<S, N, true>(simd, self.u, self.tmp, self.rhs, self.h2)
                }
            }
        }

        let mut pulp_tmp = vec![0.0; N * N];
        let pulp_error = pulp::Arch::new().dispatch(PulpPair::<N> {
            u: &mut pulp_actual,
            tmp: &mut pulp_tmp,
            rhs: &rhs,
            h2,
        });
        assert_eq!(pulp_actual, expected);
        assert_eq!(pulp_error, expected_error);
    }
}

#[inline(always)]
fn jacobi_impl<S: pulp::Simd, const N: usize>(
    simd: S,
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

    while iterations + 2 <= maxiter {
        let iter = iterations + 2;
        let check_error = iter % 1000 == 0 || iter == maxiter;
        if check_error {
            update_error = unsafe { jacobi_two_sweeps::<S, N, true>(simd, u, u_new, rhs, h2) };
        } else {
            unsafe { jacobi_two_sweeps::<S, N, false>(simd, u, u_new, rhs, h2) };
        }
        iterations = iter;

        if check_error {
            println!("iteration = {iter:6}, update error = {update_error:.6e}");
            if update_error < tol {
                return (iterations, update_error);
            }
        }
    }

    if iterations < maxiter {
        let iter = iterations + 1;
        update_error = unsafe { jacobi_sweep::<S, N, true>(simd, u, u_new, rhs, h2) };
        std::mem::swap(u, u_new);
        iterations = iter;
        println!("iteration = {iter:6}, update error = {update_error:.6e}");
    }

    (iterations, update_error)
}

struct JacobiOp<'a, const N: usize> {
    u: &'a mut Vec<f64>,
    u_new: &'a mut Vec<f64>,
    rhs: &'a [f64],
    h: f64,
    tol: f64,
    maxiter: usize,
}

impl<const N: usize> pulp::WithSimd for JacobiOp<'_, N> {
    type Output = (usize, f64);

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        jacobi_impl::<S, N>(
            simd,
            self.u,
            self.u_new,
            self.rhs,
            self.h,
            self.tol,
            self.maxiter,
        )
    }
}

fn jacobi<const N: usize>(
    u: &mut Vec<f64>,
    u_new: &mut Vec<f64>,
    rhs: &[f64],
    h: f64,
    tol: f64,
    maxiter: usize,
) -> (usize, f64) {
    pulp::Arch::new().dispatch(JacobiOp::<N> {
        u,
        u_new,
        rhs,
        h,
        tol,
        maxiter,
    })
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
            SurfaceSeries::xoz(xs.iter().copied(), xs.iter().copied(), |x, y| {
                if analytic {
                    u_exact(x, y)
                } else {
                    grid_lookup(values, n, h, x, y)
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
    const TOL: f64 = 1e-10;
    const MAXITER: usize = 100_000;

    let h = 1.0 / (N - 1) as f64;
    let xs: Vec<f64> = (0..N).map(|i| i as f64 * h).collect();
    println!("N = {N}");
    println!("h = {h:.6e}");
    println!("backend = pulp");

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

    let mut ue = vec![0.0; N * N];
    for j in 0..N {
        for i in 0..N {
            ue[idx(i, j, N)] = u_exact(xs[i], xs[j]);
        }
    }

    let mut abs_err = vec![0.0; N * N];
    let mut max_error = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for k in 0..N * N {
        let d = u[k] - ue[k];
        abs_err[k] = d.abs();
        max_error = max_error.max(abs_err[k]);
        sum_sq += d * d;
    }
    let l2_error = (sum_sq * h * h).sqrt();

    println!();
    println!("max error = {max_error:.6e}");
    println!("L2 error  = {:.6e}", l2_error / (N as f64).sqrt());

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("poisson_jacobi.png");
    let out_str = out.to_string_lossy();
    save_plot(&out_str, &xs, &u, &ue, &abs_err, N, h, max_error)?;
    println!("saved {out_str}");

    Ok(())
}
