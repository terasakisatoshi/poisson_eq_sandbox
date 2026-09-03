//! PoC for adding sparse matrix-vector multiplication and CG to tenferro-rs.
//!
//! COO is used only while assembling the matrix. The solve uses a fixed CSR
//! pattern plus tenferro values through the same `LinearOperator` interface as
//! a matrix-free five-point Laplacian.

mod plot;

use std::error::Error;
use std::f64::consts::PI;
use std::path::PathBuf;
use std::time::Instant;

use poisson_sparse_cg::linear_operator::{FivePointLaplacian, LinearOperator};
use poisson_sparse_cg::solvers::{conjugate_gradient, relative_residual, CgOptions};
use poisson_sparse_cg::sparse::{SparseCooBuilder, SparseCsrTensor};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
use tenferro_tensor::BackendSessionHost;

const N: usize = 401;

fn u_exact(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}

fn forcing(x: f64, y: f64) -> f64 {
    2.0 * PI * PI * (PI * x).sin() * (PI * y).sin()
}

fn idx(i: usize, j: usize, size: usize) -> usize {
    i + size * j
}

fn poisson_system(h: f64) -> Result<(SparseCsrTensor, TypedTensor<f64>), Box<dyn Error>> {
    let interior_size = N - 2;
    let unknowns = interior_size * interior_size;
    let scale = 1.0 / (h * h);
    let mut builder = SparseCooBuilder::new(unknowns, unknowns);
    for j in 0..interior_size {
        for i in 0..interior_size {
            let row = idx(i, j, interior_size);
            builder.push(row, row, 4.0 * scale)?;
            if i > 0 {
                builder.push(row, idx(i - 1, j, interior_size), -scale)?;
            }
            if i + 1 < interior_size {
                builder.push(row, idx(i + 1, j, interior_size), -scale)?;
            }
            if j > 0 {
                builder.push(row, idx(i, j - 1, interior_size), -scale)?;
            }
            if j + 1 < interior_size {
                builder.push(row, idx(i, j + 1, interior_size), -scale)?;
            }
        }
    }
    let matrix = builder.build()?;

    let mut rhs = vec![0.0; unknowns];
    for j in 0..interior_size {
        for i in 0..interior_size {
            let x = (i + 1) as f64 * h;
            let y = (j + 1) as f64 * h;
            rhs[idx(i, j, interior_size)] = forcing(x, y);
        }
    }
    Ok((
        matrix,
        TypedTensor::from_vec_col_major(vec![unknowns], rhs)?,
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let h = 1.0 / (N - 1) as f64;
    let coordinates: Vec<f64> = (0..N).map(|index| index as f64 * h).collect();
    let (csr, rhs) = poisson_system(h)?;
    let matrix_free = FivePointLaplacian::new(N - 2, h)?;

    println!("N = {N}");
    println!("h = {h:.6e}");
    println!("unknowns = {}", csr.shape()[0]);
    println!("nnz(A) = {}", csr.pattern().nnz());

    // sin(πx)sin(πy) is an eigenvector of the discrete five-point
    // Laplacian, so this particular problem converges in one CG iteration.
    let mut backend = CpuBackend::new();
    let csr_started = Instant::now();
    let csr_report = backend.with_backend_session(|session| {
        conjugate_gradient(&csr, &rhs, CgOptions::default(), session)
    })?;
    let csr_duration = csr_started.elapsed();
    let csr_residual = backend.with_backend_session(|session| {
        relative_residual(&csr, &csr_report.solution, &rhs, session)
    })?;

    let matrix_free_started = Instant::now();
    let matrix_free_report = backend.with_backend_session(|session| {
        conjugate_gradient(&matrix_free, &rhs, CgOptions::default(), session)
    })?;
    let matrix_free_duration = matrix_free_started.elapsed();
    let matrix_free_residual = backend.with_backend_session(|session| {
        relative_residual(&matrix_free, &matrix_free_report.solution, &rhs, session)
    })?;

    let operator_solution_difference = csr_report
        .solution
        .host_data()?
        .iter()
        .zip(matrix_free_report.solution.host_data()?)
        .map(|(csr, matrix_free)| (csr - matrix_free).abs())
        .fold(0.0_f64, f64::max);

    let interior_size = N - 2;
    let mut solution_data = vec![0.0; N * N];
    for j in 0..interior_size {
        for i in 0..interior_size {
            solution_data[idx(i + 1, j + 1, N)] =
                csr_report.solution.host_data()?[idx(i, j, interior_size)];
        }
    }
    let solution = TypedTensor::from_vec_col_major(vec![N, N], solution_data)?;
    let exact = TypedTensor::from_vec_col_major(
        vec![N, N],
        coordinates
            .iter()
            .flat_map(|&y| coordinates.iter().map(move |&x| u_exact(x, y)))
            .collect(),
    )?;

    // Keep the post-solve tensor work in tenferro, as in rust_tenferro.
    let (absolute_error, sum_squared_error) =
        backend.with_backend_session(|session| -> tenferro_tensor::Result<_> {
            let difference = solution.sub(&exact, session)?;
            let absolute_error = difference.abs(session)?;
            let sum_squared_error = difference
                .mul(&difference, session)?
                .reduce_sum(&[0, 1], session)?;
            Ok((absolute_error, sum_squared_error))
        })?;
    let max_error = absolute_error
        .host_data()?
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let l2_error = (sum_squared_error.host_data()?[0] * h * h).sqrt() / (N as f64).sqrt();

    println!();
    println!("time = {:.6} seconds", csr_duration.as_secs_f64());
    println!("CSR CG iterations = {}", csr_report.iterations);
    println!("CSR relative residual = {csr_residual:.6e}");
    println!(
        "CSR recurrence residual = {:.6e}",
        csr_report.relative_residual
    );
    println!(
        "matrix-free time = {:.6} seconds",
        matrix_free_duration.as_secs_f64()
    );
    println!(
        "matrix-free CG iterations = {}",
        matrix_free_report.iterations
    );
    println!("matrix-free relative residual = {matrix_free_residual:.6e}");
    println!("CSR/matrix-free max difference = {operator_solution_difference:.6e}");
    println!();
    println!("max error = {max_error:.6e}");
    println!("L2 error  = {l2_error:.6e}");

    let output = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("poisson_sparse_cg.png");
    plot::save_plot(
        &output.to_string_lossy(),
        &coordinates,
        solution.host_data()?,
        exact.host_data()?,
        absolute_error.host_data()?,
        N,
        h,
        max_error,
    )?;
    println!("saved {}", output.display());
    Ok(())
}
