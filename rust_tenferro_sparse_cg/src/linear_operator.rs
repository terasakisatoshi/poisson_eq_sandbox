//! Linear-operator abstraction shared by explicit sparse matrices and
//! matrix-free discretizations.

use tenferro_tensor::{BackendSession, TypedTensor};
use thiserror::Error;

/// Failure while constructing or applying a linear operator.
#[derive(Debug, Error)]
pub enum OperatorError {
    /// A tenferro tensor operation failed.
    #[error(transparent)]
    Tensor(#[from] tenferro_tensor::Error),
    /// Sparse structural metadata is invalid.
    #[error("invalid sparse pattern: {0}")]
    InvalidSparsePattern(String),
    /// A vector has the wrong length for the operator.
    #[error("{name} length mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Name of the incompatible vector.
        name: &'static str,
        /// Required vector length.
        expected: usize,
        /// Actual vector length.
        actual: usize,
    },
    /// Matrix-free operator parameters are invalid.
    #[error("invalid linear operator: {0}")]
    InvalidOperator(String),
}

pub(crate) fn require_vector(
    name: &'static str,
    input: &TypedTensor<f64>,
    expected: usize,
) -> Result<(), OperatorError> {
    let actual = input.host_data()?.len();
    if input.shape() != [expected] {
        return Err(OperatorError::DimensionMismatch {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Matrix-like object that can apply `y = A*x` inside a tenferro session.
///
/// Iterative solvers depend on this interface rather than a particular sparse
/// storage format. A future backend implementation can therefore dispatch CSR
/// SpMV or a matrix-free stencil without changing the solver API.
pub trait LinearOperator {
    /// Operator shape `[rows, columns]`.
    fn shape(&self) -> [usize; 2];

    /// Apply the operator to a rank-one dense tensor.
    fn apply(
        &self,
        input: &TypedTensor<f64>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<f64>, OperatorError>;
}

/// Matrix-free five-point discretization of `-Δ` on an interior square grid.
#[derive(Clone, Copy, Debug)]
pub struct FivePointLaplacian {
    interior_size: usize,
    inverse_h_squared: f64,
}

impl FivePointLaplacian {
    /// Construct the operator for an `interior_size × interior_size` grid.
    pub fn new(interior_size: usize, h: f64) -> Result<Self, OperatorError> {
        if interior_size == 0 {
            return Err(OperatorError::InvalidOperator(
                "interior grid must be non-empty".into(),
            ));
        }
        if !h.is_finite() || h <= 0.0 {
            return Err(OperatorError::InvalidOperator(
                "grid spacing must be positive and finite".into(),
            ));
        }
        Ok(Self {
            interior_size,
            inverse_h_squared: 1.0 / (h * h),
        })
    }
}

impl LinearOperator for FivePointLaplacian {
    fn shape(&self) -> [usize; 2] {
        let unknowns = self.interior_size * self.interior_size;
        [unknowns, unknowns]
    }

    fn apply(
        &self,
        input: &TypedTensor<f64>,
        _session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<f64>, OperatorError> {
        let unknowns = self.shape()[1];
        require_vector("input", input, unknowns)?;
        let input = input.host_data()?;
        let mut output = vec![0.0; unknowns];
        let size = self.interior_size;

        for j in 0..size {
            for i in 0..size {
                let index = i + size * j;
                let mut value = 4.0 * input[index];
                if i > 0 {
                    value -= input[index - 1];
                }
                if i + 1 < size {
                    value -= input[index + 1];
                }
                if j > 0 {
                    value -= input[index - size];
                }
                if j + 1 < size {
                    value -= input[index + size];
                }
                output[index] = self.inverse_h_squared * value;
            }
        }
        Ok(TypedTensor::from_vec_col_major(vec![unknowns], output)?)
    }
}
