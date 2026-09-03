//! Sparse linear-algebra API proof of concept for tenferro-rs.
//!
//! COO is limited to assembly/import, CSR is the execution representation,
//! and iterative solvers depend only on [`linear_operator::LinearOperator`].

#![forbid(unsafe_code)]

pub mod linear_operator;
pub mod solvers;
pub mod sparse;
