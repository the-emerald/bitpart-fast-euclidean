#![feature(core_intrinsics)]

use bitpart::metric::Metric;
use std::intrinsics::{fadd_fast, fmul_fast, fsub_fast};
use std::ops::Deref;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Wrapper struct to apply Euclidean distance to an object set.
///
/// Unlike its [safe counterpart](bitpart::metric::euclidean), `FastEuclidean` is not IEEE-754 compliant, but allows
/// for better optimisations by the compiler.
/// # Example
/// ```
/// # use crate::bitpart::metric::{Metric, euclidean::Euclidean};
/// let point1: Euclidean<[f64; 2]> = Euclidean::new([0.0, 0.0]);
/// let point2: Euclidean<[f64; 2]> = Euclidean::new([1.0, 1.0]);
///
/// assert_eq!(point1.distance(&point2), 2.0_f64.sqrt());
/// ```
#[derive(Debug, Clone)]
pub struct FastEuclidean<T>(T);

impl<T> FastEuclidean<T> {
    /// Creates a new `FastEuclidean`.
    /// # Safety
    /// You **must** ensure that the euclidean distance measurements between
    /// any two points cannot return [`NaN`](f64::NAN) nor [`INF`](f64::INFINITY).
    pub unsafe fn new(t: T) -> Self {
        Self(t)
    }
}

impl<T> Deref for FastEuclidean<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> IntoIterator for FastEuclidean<T>
where
    T: IntoIterator,
{
    type Item = <T as IntoIterator>::Item;
    type IntoIter = <T as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a FastEuclidean<T>
where
    &'a T: IntoIterator,
{
    type Item = <&'a T as IntoIterator>::Item;
    type IntoIter = <&'a T as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T> Metric for FastEuclidean<T>
where
    for<'a> &'a T: IntoIterator<Item = &'a f64>,
    T: Clone,
{
    fn distance(&self, rhs: &FastEuclidean<T>) -> f64 {
        self.0
            .into_iter()
            .zip(rhs.0.into_iter())
            .map(|(&x, &y)| unsafe {
                let a = fsub_fast(x, y);
                fmul_fast(a, a)
            })
            .fold(0.0, |acc, v| unsafe { fadd_fast(acc, v) })
            .sqrt()
    }
}

#[cfg(feature = "serde")]
impl<T> Serialize for FastEuclidean<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T> Deserialize<'de> for FastEuclidean<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        unsafe { Ok(FastEuclidean::new(T::deserialize(deserializer)?)) }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn euclidean_2d() {
//         let point1: FastEuclidean<[f64; 2]> = FastEuclidean::new([0.0, 0.0]);
//         let point2: FastEuclidean<[f64; 2]> = FastEuclidean::new([1.0, 1.0]);

//         assert_eq!(point1.distance(&point2), 2.0_f64.sqrt());
//     }
// }
