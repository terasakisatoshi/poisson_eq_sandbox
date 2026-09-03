//! Iterative solvers over the storage-independent `LinearOperator` interface.
//!
//! GMRES can be added beside CG without coupling it to CSR or matrix-free
//! storage; both algorithms consume only `LinearOperator::apply`.

use tenferro_tensor::{BackendSession, TypedTensor};
use thiserror::Error;

use crate::linear_operator::{require_vector, LinearOperator, OperatorError};

/// Parameters controlling conjugate-gradient convergence.
#[derive(Clone, Copy, Debug)]
pub struct CgOptions {
    /// Relative residual tolerance `||Ax-b||₂ / ||b||₂`.
    pub relative_tolerance: f64,
    /// Maximum number of CG iterations.
    pub maximum_iterations: usize,
}

impl Default for CgOptions {
    fn default() -> Self {
        Self {
            relative_tolerance: 1.0e-10,
            maximum_iterations: 100_000,
        }
    }
}

/// Successful CG output and convergence diagnostics.
#[derive(Debug)]
pub struct CgReport {
    /// Approximate solution of `Ax=b`.
    pub solution: TypedTensor<f64>,
    /// Number of iterations executed.
    pub iterations: usize,
    /// Residual norm tracked by the CG recurrence.
    pub relative_residual: f64,
}

/// Failures reported by conjugate gradients.
#[derive(Debug, Error)]
pub enum CgError {
    /// Constructing or applying the operator failed.
    #[error(transparent)]
    Operator(#[from] OperatorError),
    /// CG requires a square operator.
    #[error("CG requires a square operator, got shape {rows}x{columns}")]
    NonSquare {
        /// Operator row count.
        rows: usize,
        /// Operator column count.
        columns: usize,
    },
    /// CG detected non-positive curvature.
    #[error("CG detected a non-positive-definite operator at iteration {iteration}")]
    NonPositiveDefinite {
        /// Iteration where the check failed.
        iteration: usize,
    },
    /// The requested tolerance was not reached.
    #[error(
        "CG did not converge in {iterations} iterations; relative residual={relative_residual:e}"
    )]
    NoConvergence {
        /// Iteration limit.
        iterations: usize,
        /// Residual at the iteration limit.
        relative_residual: f64,
    },
}

/// Solve `Ax=b` for a symmetric positive-definite linear operator.
pub fn conjugate_gradient<A: LinearOperator + ?Sized>(
    operator: &A,
    rhs: &TypedTensor<f64>,
    options: CgOptions,
    session: &mut dyn BackendSession,
) -> Result<CgReport, CgError> {
    let [rows, columns] = operator.shape();
    if rows != columns {
        return Err(CgError::NonSquare { rows, columns });
    }
    require_vector("rhs", rhs, rows)?;
    let rhs = rhs.host_data().map_err(OperatorError::from)?;
    let rhs_norm_squared = dot(rhs, rhs);
    if rhs_norm_squared == 0.0 {
        return Ok(CgReport {
            solution: vector(vec![0.0; columns])?,
            iterations: 0,
            relative_residual: 0.0,
        });
    }

    let mut solution = vec![0.0; columns];
    let mut residual = rhs.to_vec();
    let mut direction = residual.clone();
    let mut residual_norm_squared = rhs_norm_squared;

    for iteration in 1..=options.maximum_iterations {
        let direction_tensor = vector(direction.clone())?;
        let operator_direction = operator.apply(&direction_tensor, session)?;
        let operator_direction = operator_direction
            .host_data()
            .map_err(OperatorError::from)?;
        let curvature = dot(&direction, operator_direction);
        if !curvature.is_finite() || curvature <= 0.0 {
            return Err(CgError::NonPositiveDefinite { iteration });
        }

        let alpha = residual_norm_squared / curvature;
        for index in 0..columns {
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * operator_direction[index];
        }

        let next_residual_norm_squared = dot(&residual, &residual);
        let relative_residual = (next_residual_norm_squared / rhs_norm_squared).sqrt();
        if relative_residual <= options.relative_tolerance {
            return Ok(CgReport {
                solution: vector(solution)?,
                iterations: iteration,
                relative_residual,
            });
        }

        let beta = next_residual_norm_squared / residual_norm_squared;
        for index in 0..columns {
            direction[index] = residual[index] + beta * direction[index];
        }
        residual_norm_squared = next_residual_norm_squared;
    }

    let relative_residual = (residual_norm_squared / rhs_norm_squared).sqrt();
    Err(CgError::NoConvergence {
        iterations: options.maximum_iterations,
        relative_residual,
    })
}

/// Explicitly evaluate `||Ax-b||₂ / ||b||₂`.
pub fn relative_residual<A: LinearOperator + ?Sized>(
    operator: &A,
    solution: &TypedTensor<f64>,
    rhs: &TypedTensor<f64>,
    session: &mut dyn BackendSession,
) -> Result<f64, OperatorError> {
    require_vector("solution", solution, operator.shape()[1])?;
    require_vector("rhs", rhs, operator.shape()[0])?;
    let applied = operator.apply(solution, session)?;
    let residual_squared = applied
        .host_data()?
        .iter()
        .zip(rhs.host_data()?)
        .map(|(applied, rhs)| (applied - rhs).powi(2))
        .sum::<f64>();
    let rhs_squared = dot(rhs.host_data()?, rhs.host_data()?);
    Ok(if rhs_squared == 0.0 {
        residual_squared.sqrt()
    } else {
        (residual_squared / rhs_squared).sqrt()
    })
}

fn vector(values: Vec<f64>) -> Result<TypedTensor<f64>, OperatorError> {
    Ok(TypedTensor::from_vec_col_major(vec![values.len()], values)?)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_operator::FivePointLaplacian;
    use crate::sparse::SparseCooBuilder;
    use tenferro_cpu::CpuBackend;
    use tenferro_tensor::BackendSessionHost;

    #[test]
    fn cg_accepts_csr_through_linear_operator() -> Result<(), Box<dyn std::error::Error>> {
        let mut builder = SparseCooBuilder::new(2, 2);
        builder.push(0, 0, 2.0)?;
        builder.push(1, 1, 4.0)?;
        let matrix = builder.build()?;
        let rhs = vector(vec![2.0, 8.0])?;
        let mut backend = CpuBackend::new();
        let report = backend.with_backend_session(|session| {
            conjugate_gradient(&matrix, &rhs, CgOptions::default(), session)
        })?;
        assert!((report.solution.host_data()?[0] - 1.0).abs() < 1.0e-12);
        assert!((report.solution.host_data()?[1] - 2.0).abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn cg_accepts_matrix_free_operator() -> Result<(), Box<dyn std::error::Error>> {
        let operator = FivePointLaplacian::new(2, 0.5)?;
        let rhs = vector(vec![1.0, 2.0, 3.0, 4.0])?;
        let mut backend = CpuBackend::new();
        let report = backend.with_backend_session(|session| {
            conjugate_gradient(&operator, &rhs, CgOptions::default(), session)
        })?;
        let residual = backend.with_backend_session(|session| {
            relative_residual(&operator, &report.solution, &rhs, session)
        })?;
        assert!(residual < 1.0e-12);
        Ok(())
    }
}
