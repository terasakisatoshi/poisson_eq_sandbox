//! COO construction/import and canonical fixed-pattern CSR storage.

use std::sync::Arc;

use tenferro_tensor::{BackendSession, DType, Tensor, TypedTensor};

use crate::linear_operator::{require_vector, LinearOperator, OperatorError};

/// Concrete COO tensor used at the import boundary.
///
/// Coordinates are an `I64` tensor with shape `[2, nnz]`; values are an
/// `F64` tensor with shape `[nnz]`. This is the small concrete subset copied
/// from tenferro's unpublished `ext/sparse` tutorial crate.
#[derive(Debug)]
pub struct SparseCooTensor {
    shape: Vec<usize>,
    coordinates: Tensor,
    values: Tensor,
}

impl SparseCooTensor {
    /// Validate fixed COO coordinates and values.
    pub fn from_parts(
        shape: Vec<usize>,
        coordinates: Tensor,
        values: Tensor,
    ) -> Result<Self, OperatorError> {
        if shape.len() != 2 {
            return Err(OperatorError::InvalidSparsePattern(format!(
                "COO shape must have rank 2, got rank {}",
                shape.len()
            )));
        }
        if coordinates.dtype() != DType::I64 {
            return Err(OperatorError::InvalidSparsePattern(
                "COO coordinates must have dtype I64".into(),
            ));
        }
        let coordinate_shape = coordinates.shape();
        if coordinate_shape.len() != 2 || coordinate_shape[0] != 2 {
            return Err(OperatorError::InvalidSparsePattern(format!(
                "COO coordinates must have shape [2, nnz], got {coordinate_shape:?}"
            )));
        }
        let nnz = coordinate_shape[1];
        if values.dtype() != DType::F64 || values.shape() != [nnz] {
            return Err(OperatorError::InvalidSparsePattern(format!(
                "COO values must be F64 with shape [{nnz}]"
            )));
        }
        for coordinate in coordinates.as_slice::<i64>()?.as_chunks::<2>().0 {
            let row = usize::try_from(coordinate[0]).map_err(|_| {
                OperatorError::InvalidSparsePattern("negative COO row coordinate".into())
            })?;
            let column = usize::try_from(coordinate[1]).map_err(|_| {
                OperatorError::InvalidSparsePattern("negative COO column coordinate".into())
            })?;
            if row >= shape[0] || column >= shape[1] {
                return Err(OperatorError::InvalidSparsePattern(format!(
                    "COO coordinate [{row}, {column}] is out of bounds for shape {shape:?}"
                )));
            }
        }
        Ok(Self {
            shape,
            coordinates,
            values,
        })
    }

    /// Sparse logical shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Fixed COO coordinates.
    pub fn coordinates(&self) -> &Tensor {
        &self.coordinates
    }

    /// COO nonzero values.
    pub fn values(&self) -> &Tensor {
        &self.values
    }
}

/// Immutable CSR structure shared independently of differentiable values.
#[derive(Clone, Debug)]
pub struct SparseCsrPattern {
    shape: [usize; 2],
    row_offsets: Arc<[usize]>,
    column_indices: Arc<[usize]>,
}

impl SparseCsrPattern {
    /// Validate and construct a canonical CSR pattern.
    pub fn new(
        shape: [usize; 2],
        row_offsets: Vec<usize>,
        column_indices: Vec<usize>,
    ) -> Result<Self, OperatorError> {
        let [rows, columns] = shape;
        if row_offsets.len() != rows + 1 {
            return Err(OperatorError::InvalidSparsePattern(format!(
                "row_offsets must have length {}, got {}",
                rows + 1,
                row_offsets.len()
            )));
        }
        if row_offsets.first() != Some(&0) {
            return Err(OperatorError::InvalidSparsePattern(
                "row_offsets must start at zero".into(),
            ));
        }
        if row_offsets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(OperatorError::InvalidSparsePattern(
                "row_offsets must be nondecreasing".into(),
            ));
        }
        if row_offsets.last() != Some(&column_indices.len()) {
            return Err(OperatorError::InvalidSparsePattern(
                "last row offset must equal nnz".into(),
            ));
        }
        for row in 0..rows {
            let columns_in_row = &column_indices[row_offsets[row]..row_offsets[row + 1]];
            if columns_in_row.iter().any(|&column| column >= columns) {
                return Err(OperatorError::InvalidSparsePattern(
                    "column index is out of bounds".into(),
                ));
            }
            if columns_in_row.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(OperatorError::InvalidSparsePattern(
                    "column indices in each row must be sorted and unique".into(),
                ));
            }
        }
        Ok(Self {
            shape,
            row_offsets: row_offsets.into(),
            column_indices: column_indices.into(),
        })
    }

    /// Matrix shape `[rows, columns]`.
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Number of structurally stored entries.
    pub fn nnz(&self) -> usize {
        self.column_indices.len()
    }

    /// CSR row offsets.
    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    /// CSR column indices.
    pub fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }
}

/// Fixed CSR structure plus tenferro-managed numeric values.
#[derive(Debug)]
pub struct SparseCsrTensor {
    pattern: Arc<SparseCsrPattern>,
    values: TypedTensor<f64>,
}

impl SparseCsrTensor {
    /// Combine a validated pattern with a rank-one value tensor.
    pub fn new(
        pattern: Arc<SparseCsrPattern>,
        values: TypedTensor<f64>,
    ) -> Result<Self, OperatorError> {
        require_vector("values", &values, pattern.nnz())?;
        Ok(Self { pattern, values })
    }

    /// Convert a validated COO import tensor into canonical CSR.
    pub fn from_coo(coo: &SparseCooTensor) -> Result<Self, OperatorError> {
        let [rows, columns] = *coo.shape() else {
            unreachable!("SparseCooTensor validates rank two at construction")
        };
        let mut builder = SparseCooBuilder::new(rows, columns);
        let coordinates = coo.coordinates().as_slice::<i64>()?;
        let values = coo.values().as_slice::<f64>()?;
        for (coordinate, &value) in coordinates.as_chunks::<2>().0.iter().zip(values) {
            builder.push(coordinate[0] as usize, coordinate[1] as usize, value)?;
        }
        builder.build()
    }

    /// Fixed CSR structure.
    pub fn pattern(&self) -> &Arc<SparseCsrPattern> {
        &self.pattern
    }

    /// Numeric nonzero values.
    pub fn values(&self) -> &TypedTensor<f64> {
        &self.values
    }
}

impl LinearOperator for SparseCsrTensor {
    fn shape(&self) -> [usize; 2] {
        self.pattern.shape()
    }

    fn apply(
        &self,
        input: &TypedTensor<f64>,
        _session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<f64>, OperatorError> {
        let [rows, columns] = self.shape();
        require_vector("input", input, columns)?;
        let input = input.host_data()?;
        let values = self.values.host_data()?;
        let offsets = self.pattern.row_offsets();
        let indices = self.pattern.column_indices();
        let mut output = vec![0.0; rows];
        for row in 0..rows {
            for position in offsets[row]..offsets[row + 1] {
                output[row] += values[position] * input[indices[position]];
            }
        }
        Ok(TypedTensor::from_vec_col_major(vec![rows], output)?)
    }
}

/// Triplet builder used only for sparse construction and import.
#[derive(Debug)]
pub struct SparseCooBuilder {
    rows: usize,
    columns: usize,
    entries: Vec<(usize, usize, f64)>,
}

impl SparseCooBuilder {
    /// Create an empty builder for the requested matrix shape.
    pub fn new(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            entries: Vec::new(),
        }
    }

    /// Append one COO entry. Duplicate coordinates are combined by `build`.
    pub fn push(&mut self, row: usize, column: usize, value: f64) -> Result<(), OperatorError> {
        if row >= self.rows || column >= self.columns {
            return Err(OperatorError::InvalidSparsePattern(format!(
                "coordinate [{row}, {column}] is out of bounds for {}x{}",
                self.rows, self.columns
            )));
        }
        self.entries.push((row, column, value));
        Ok(())
    }

    /// Sort and combine COO entries into canonical CSR.
    pub fn build(mut self) -> Result<SparseCsrTensor, OperatorError> {
        self.entries.sort_by_key(|&(row, column, _)| (row, column));
        let mut row_offsets = vec![0; self.rows + 1];
        let mut column_indices = Vec::with_capacity(self.entries.len());
        let mut values = Vec::with_capacity(self.entries.len());
        let mut position = 0;
        while position < self.entries.len() {
            let (row, column, mut value) = self.entries[position];
            position += 1;
            while position < self.entries.len()
                && self.entries[position].0 == row
                && self.entries[position].1 == column
            {
                value += self.entries[position].2;
                position += 1;
            }
            row_offsets[row + 1] += 1;
            column_indices.push(column);
            values.push(value);
        }
        for row in 0..self.rows {
            row_offsets[row + 1] += row_offsets[row];
        }
        let pattern = Arc::new(SparseCsrPattern::new(
            [self.rows, self.columns],
            row_offsets,
            column_indices,
        )?);
        SparseCsrTensor::new(
            pattern,
            TypedTensor::from_vec_col_major(vec![values.len()], values)?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_tensor::Tensor;

    #[test]
    fn coo_builder_sorts_and_combines_duplicates() -> Result<(), OperatorError> {
        let mut builder = SparseCooBuilder::new(2, 2);
        builder.push(1, 1, 4.0)?;
        builder.push(0, 0, 1.0)?;
        builder.push(0, 0, 2.0)?;
        let matrix = builder.build()?;
        assert_eq!(matrix.pattern().row_offsets(), &[0, 1, 2]);
        assert_eq!(matrix.pattern().column_indices(), &[0, 1]);
        assert_eq!(matrix.values().host_data()?, &[3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn imports_coo_tensor() -> Result<(), OperatorError> {
        let coo = SparseCooTensor::from_parts(
            vec![2, 2],
            Tensor::from_vec_col_major(vec![2, 2], vec![0_i64, 0, 1, 1])?,
            Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0])?,
        )?;
        let matrix = SparseCsrTensor::from_coo(&coo)?;
        assert_eq!(matrix.pattern().row_offsets(), &[0, 1, 2]);
        assert_eq!(matrix.pattern().column_indices(), &[0, 1]);
        assert_eq!(matrix.values().host_data()?, &[2.0, 4.0]);
        Ok(())
    }
}
