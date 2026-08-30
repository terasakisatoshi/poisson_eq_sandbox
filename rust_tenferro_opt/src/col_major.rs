//! Validated const-generic column-major host views.
#![allow(dead_code)]
//!
//! Local stand-in for the API proposed in
//! [tenferro-rs#1736](https://github.com/tensor4all/tenferro-rs/issues/1736).
//! Construction checks host compactness, rank, and shape product once.
//! Hot loops then use first-axis lanes or `get_unchecked`, without per-element
//! rank, backend, layout, or `Result` work.
//!
//! The first index varies fastest (Fortran / Julia column-major order).

use std::fmt;
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewError {
    RankMismatch { expected: usize, actual: usize },
    ShapeProductOverflow,
    LengthMismatch { expected: usize, actual: usize },
}

impl fmt::Display for ViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch { expected, actual } => {
                write!(f, "rank mismatch: expected {expected}, got {actual}")
            }
            Self::ShapeProductOverflow => write!(f, "shape product overflows usize"),
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "buffer length {actual} does not match shape product {expected}"
                )
            }
        }
    }
}

impl std::error::Error for ViewError {}

fn checked_product<const N: usize>(shape: &[usize; N]) -> Result<usize, ViewError> {
    let mut n = 1usize;
    for &extent in shape {
        n = n
            .checked_mul(extent)
            .ok_or(ViewError::ShapeProductOverflow)?;
    }
    Ok(n)
}

/// Column-major linear offset. `N` is a const generic so this unrolls.
///
/// Construction has already proved that an in-bounds index cannot overflow
/// `usize` and lands inside the buffer, so the loop uses wrapping-free
/// arithmetic.
#[inline(always)]
fn offset<const N: usize>(shape: &[usize; N], index: [usize; N]) -> usize {
    let mut off = 0usize;
    let mut stride = 1usize;
    for axis in 0..N {
        off += index[axis] * stride;
        stride *= shape[axis];
    }
    off
}

#[inline(always)]
fn in_bounds<const N: usize>(shape: &[usize; N], index: [usize; N]) -> bool {
    let mut axis = 0usize;
    while axis < N {
        if index[axis] >= shape[axis] {
            return false;
        }
        axis += 1;
    }
    true
}

/// Shared host-resident compact column-major view of rank `N`.
pub struct ColMajorView<'a, T, const N: usize> {
    data: &'a [T],
    shape: [usize; N],
}

/// Exclusive host-resident compact column-major view of rank `N`.
pub struct ColMajorViewMut<'a, T, const N: usize> {
    data: &'a mut [T],
    shape: [usize; N],
}

impl<'a, T, const N: usize> ColMajorView<'a, T, N> {
    pub fn try_new(data: &'a [T], shape: [usize; N]) -> Result<Self, ViewError> {
        let expected = checked_product(&shape)?;
        if expected != data.len() {
            return Err(ViewError::LengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { data, shape })
    }

    #[inline(always)]
    pub fn shape(&self) -> &[usize; N] {
        &self.shape
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    #[inline(always)]
    fn axis0_extent(&self) -> usize {
        if N == 0 {
            1
        } else {
            self.shape[0]
        }
    }

    pub fn get(&self, index: [usize; N]) -> Option<&T> {
        if !in_bounds(&self.shape, index) {
            return None;
        }
        // SAFETY: `index` is in bounds for a compact buffer whose length equals
        // the checked shape product, so the offset is in range.
        Some(unsafe { self.get_unchecked(index) })
    }

    /// # Safety
    /// `index[axis] < shape[axis]` for every axis.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: [usize; N]) -> &T {
        let off = offset(&self.shape, index);
        unsafe { self.data.get_unchecked(off) }
    }

    /// Contiguous first-axis vectors in column-major order.
    #[inline(always)]
    pub fn axis0_lanes(&self) -> std::slice::ChunksExact<'a, T> {
        self.data.chunks_exact(self.axis0_extent().max(1))
    }

    #[inline(always)]
    pub fn axis0_lane(&self, lane: usize) -> Option<&[T]> {
        let n0 = self.axis0_extent();
        let start = lane.checked_mul(n0)?;
        let end = start.checked_add(n0)?;
        self.data.get(start..end)
    }

    /// # Safety
    /// `lane` must index a full first-axis vector: `lane * n0 + n0 <= len`.
    #[inline(always)]
    pub unsafe fn axis0_lane_unchecked(&self, lane: usize) -> &[T] {
        let n0 = self.axis0_extent();
        let start = lane * n0;
        unsafe { self.data.get_unchecked(start..start + n0) }
    }
}

impl<'a, T, const N: usize> ColMajorViewMut<'a, T, N> {
    pub fn try_new(data: &'a mut [T], shape: [usize; N]) -> Result<Self, ViewError> {
        let expected = checked_product(&shape)?;
        if expected != data.len() {
            return Err(ViewError::LengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { data, shape })
    }

    #[inline(always)]
    pub fn shape(&self) -> &[usize; N] {
        &self.shape
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    #[inline(always)]
    fn axis0_extent(&self) -> usize {
        if N == 0 {
            1
        } else {
            self.shape[0]
        }
    }

    pub fn get(&self, index: [usize; N]) -> Option<&T> {
        if !in_bounds(&self.shape, index) {
            return None;
        }
        Some(unsafe { self.get_unchecked(index) })
    }

    pub fn get_mut(&mut self, index: [usize; N]) -> Option<&mut T> {
        if !in_bounds(&self.shape, index) {
            return None;
        }
        Some(unsafe { self.get_unchecked_mut(index) })
    }

    /// # Safety
    /// `index[axis] < shape[axis]` for every axis.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: [usize; N]) -> &T {
        let off = offset(&self.shape, index);
        unsafe { self.data.get_unchecked(off) }
    }

    /// # Safety
    /// `index[axis] < shape[axis]` for every axis.
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, index: [usize; N]) -> &mut T {
        let off = offset(&self.shape, index);
        unsafe { self.data.get_unchecked_mut(off) }
    }

    #[inline(always)]
    pub fn axis0_lanes(&self) -> std::slice::ChunksExact<'_, T> {
        self.data.chunks_exact(self.axis0_extent().max(1))
    }

    #[inline(always)]
    pub fn axis0_lanes_mut(&mut self) -> std::slice::ChunksExactMut<'_, T> {
        let n0 = self.axis0_extent().max(1);
        self.data.chunks_exact_mut(n0)
    }

    #[inline(always)]
    pub fn axis0_lane(&self, lane: usize) -> Option<&[T]> {
        let n0 = self.axis0_extent();
        let start = lane.checked_mul(n0)?;
        let end = start.checked_add(n0)?;
        self.data.get(start..end)
    }

    #[inline(always)]
    pub fn axis0_lane_mut(&mut self, lane: usize) -> Option<&mut [T]> {
        let n0 = self.axis0_extent();
        let start = lane.checked_mul(n0)?;
        let end = start.checked_add(n0)?;
        self.data.get_mut(start..end)
    }

    /// # Safety
    /// `lane` must index a full first-axis vector: `lane * n0 + n0 <= len`.
    #[inline(always)]
    pub unsafe fn axis0_lane_unchecked(&self, lane: usize) -> &[T] {
        let n0 = self.axis0_extent();
        let start = lane * n0;
        unsafe { self.data.get_unchecked(start..start + n0) }
    }

    /// # Safety
    /// `lane` must index a full first-axis vector. The returned slice is the
    /// unique mutable borrow of that lane.
    #[inline(always)]
    pub unsafe fn axis0_lane_unchecked_mut(&mut self, lane: usize) -> &mut [T] {
        let n0 = self.axis0_extent();
        let start = lane * n0;
        unsafe { self.data.get_unchecked_mut(start..start + n0) }
    }
}

impl<T, const N: usize> Index<[usize; N]> for ColMajorView<'_, T, N> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: [usize; N]) -> &T {
        self.get(index)
            .unwrap_or_else(|| panic!("index {index:?} out of bounds for shape {:?}", self.shape))
    }
}

impl<T, const N: usize> Index<[usize; N]> for ColMajorViewMut<'_, T, N> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: [usize; N]) -> &T {
        self.get(index)
            .unwrap_or_else(|| panic!("index {index:?} out of bounds for shape {:?}", self.shape))
    }
}

impl<T, const N: usize> IndexMut<[usize; N]> for ColMajorViewMut<'_, T, N> {
    #[inline(always)]
    fn index_mut(&mut self, index: [usize; N]) -> &mut T {
        let shape = self.shape;
        self.get_mut(index)
            .unwrap_or_else(|| panic!("index {index:?} out of bounds for shape {shape:?}"))
    }
}
