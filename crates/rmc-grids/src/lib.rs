//! Grid types for sampling, binning, and interpolation support.
//!
//! The first slice covers one-dimensional grids. Constructors validate their inputs, point and bin
//! queries are bounds-checked, and standard iterators expose grid points, bin centers, and bin
//! widths.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, GridError>;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum GridError {
    #[error("grid size must be at least 2, got {0}")]
    InvalidSize(usize),
    #[error("grid endpoints must differ, got first={first} and last={last}")]
    DegenerateRange { first: f64, last: f64 },
    #[error("power must be > 0, got {0}")]
    InvalidPower(f64),
    #[error("symmetric power grid size must be odd and at least 3, got {0}")]
    InvalidSymmetricSize(usize),
    #[error("custom grid points must be strictly monotonic")]
    NonMonotonicPoints,
    #[error("grid dimension must be at least 1, got {0}")]
    InvalidDimension(usize),
    #[error("axis {axis} must have at least 2 points, got {len}")]
    InvalidAxisSize { axis: usize, len: usize },
}

/// Common interface for one-dimensional grids.
pub trait Grid1d {
    /// Number of grid points.
    fn len(&self) -> usize;

    /// First grid point.
    fn first(&self) -> f64;

    /// Last grid point.
    fn last(&self) -> f64;

    /// Grid point at `index`, or `None` when out of bounds.
    fn point(&self, index: usize) -> Option<f64>;

    /// Bin index containing `value`, or `None` when outside the grid range.
    ///
    /// Bins are treated as half-open except that the final endpoint belongs to the final bin.
    fn bin_index(&self, value: f64) -> Option<usize>;

    /// Whether the grid has no points.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of bins, equal to `len() - 1`.
    fn bin_count(&self) -> usize {
        self.len().saturating_sub(1)
    }

    /// Whether the value lies inside the closed grid interval.
    fn contains(&self, value: f64) -> bool {
        if self.first() <= self.last() {
            self.first() <= value && value <= self.last()
        } else {
            self.last() <= value && value <= self.first()
        }
    }

    /// Whether points increase with index.
    fn is_increasing(&self) -> bool {
        self.first() < self.last()
    }

    /// Whether points decrease with index.
    fn is_decreasing(&self) -> bool {
        self.first() > self.last()
    }

    /// Closed domain endpoints in index order.
    fn domain(&self) -> (f64, f64) {
        (self.first(), self.last())
    }

    /// Left and right grid points of bin `index`, in index order.
    fn bin_bounds(&self, index: usize) -> Option<(f64, f64)> {
        Some((self.point(index)?, self.point(index + 1)?))
    }

    /// Bin width/volume at `index`.
    fn bin_width(&self, index: usize) -> Option<f64> {
        let (left, right) = self.bin_bounds(index)?;
        Some((right - left).abs())
    }

    /// Bin center at `index`.
    fn bin_center(&self, index: usize) -> Option<f64> {
        let (left, right) = self.bin_bounds(index)?;
        Some(0.5 * (left + right))
    }

    /// Iterate over grid points.
    fn points(&self) -> GridPoints<'_, Self>
    where
        Self: Sized,
    {
        GridPoints {
            grid: self,
            index: 0,
        }
    }

    /// Iterate over bin centers.
    fn bin_centers(&self) -> BinCenters<'_, Self>
    where
        Self: Sized,
    {
        BinCenters {
            grid: self,
            index: 0,
        }
    }

    /// Iterate over bin widths.
    fn bin_widths(&self) -> BinWidths<'_, Self>
    where
        Self: Sized,
    {
        BinWidths {
            grid: self,
            index: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GridPoints<'a, G> {
    grid: &'a G,
    index: usize,
}

impl<G: Grid1d> Iterator for GridPoints<'_, G> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let point = self.grid.point(self.index)?;
        self.index += 1;
        Some(point)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.grid.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<G: Grid1d> ExactSizeIterator for GridPoints<'_, G> {}

#[derive(Clone, Debug)]
pub struct BinCenters<'a, G> {
    grid: &'a G,
    index: usize,
}

impl<G: Grid1d> Iterator for BinCenters<'_, G> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let center = self.grid.bin_center(self.index)?;
        self.index += 1;
        Some(center)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.grid.bin_count().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<G: Grid1d> ExactSizeIterator for BinCenters<'_, G> {}

#[derive(Clone, Debug)]
pub struct BinWidths<'a, G> {
    grid: &'a G,
    index: usize,
}

impl<G: Grid1d> Iterator for BinWidths<'_, G> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let width = self.grid.bin_width(self.index)?;
        self.index += 1;
        Some(width)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.grid.bin_count().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<G: Grid1d> ExactSizeIterator for BinWidths<'_, G> {}

/// Homogeneous const-generic product grid.
///
/// `NdGrid<G, N>` combines `N` one-dimensional grids of the same concrete type and traverses points
/// and bins in row-major order, with the last axis varying fastest.
#[derive(Clone, Debug, PartialEq)]
pub struct NdGrid<G, const N: usize> {
    axes: [G; N],
}

#[cfg(feature = "serde")]
impl<G, const N: usize> serde::Serialize for NdGrid<G, N>
where
    G: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("NdGrid", 1)?;
        state.serialize_field("axes", &self.axes.as_slice())?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, G, const N: usize> serde::Deserialize<'de> for NdGrid<G, N>
where
    G: Grid1d + serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct SerializedNdGrid<G> {
            axes: Vec<G>,
        }

        let serialized = SerializedNdGrid::deserialize(deserializer)?;
        let actual_len = serialized.axes.len();
        let axes: [G; N] = serialized.axes.try_into().map_err(|_| {
            serde::de::Error::invalid_length(actual_len, &"matching const dimension")
        })?;
        Self::new(axes).map_err(serde::de::Error::custom)
    }
}

impl<G: Grid1d, const N: usize> NdGrid<G, N> {
    pub fn new(axes: [G; N]) -> Result<Self> {
        validate_axes(&axes)?;
        Ok(Self { axes })
    }

    pub fn dim(&self) -> usize {
        N
    }

    pub fn axes(&self) -> &[G; N] {
        &self.axes
    }

    pub fn axis(&self, index: usize) -> Option<&G> {
        self.axes.get(index)
    }

    pub fn shape(&self) -> [usize; N] {
        std::array::from_fn(|axis| self.axes[axis].len())
    }

    pub fn bin_shape(&self) -> [usize; N] {
        std::array::from_fn(|axis| self.axes[axis].bin_count())
    }

    pub fn len(&self) -> usize {
        size_from_shape(self.shape()).expect("validated axes have non-empty finite shape")
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bin_count(&self) -> usize {
        size_from_shape(self.bin_shape()).expect("validated axes have non-empty finite bin shape")
    }

    pub fn first(&self) -> [f64; N] {
        std::array::from_fn(|axis| self.axes[axis].first())
    }

    pub fn last(&self) -> [f64; N] {
        std::array::from_fn(|axis| self.axes[axis].last())
    }

    pub fn contains(&self, value: [f64; N]) -> bool {
        (0..N).all(|axis| self.axes[axis].contains(value[axis]))
    }

    pub fn point(&self, indices: [usize; N]) -> Option<[f64; N]> {
        let mut point = [0.0; N];
        for axis in 0..N {
            point[axis] = self.axes[axis].point(indices[axis])?;
        }
        Some(point)
    }

    pub fn bin_index(&self, value: [f64; N]) -> Option<[usize; N]> {
        let mut indices = [0; N];
        for axis in 0..N {
            indices[axis] = self.axes[axis].bin_index(value[axis])?;
        }
        Some(indices)
    }

    pub fn bin_center(&self, indices: [usize; N]) -> Option<[f64; N]> {
        let mut center = [0.0; N];
        for axis in 0..N {
            center[axis] = self.axes[axis].bin_center(indices[axis])?;
        }
        Some(center)
    }

    pub fn bin_bounds(&self, indices: [usize; N]) -> Option<([f64; N], [f64; N])> {
        let mut lower = [0.0; N];
        let mut upper = [0.0; N];
        for axis in 0..N {
            let (axis_lower, axis_upper) = self.axes[axis].bin_bounds(indices[axis])?;
            lower[axis] = axis_lower;
            upper[axis] = axis_upper;
        }
        Some((lower, upper))
    }

    pub fn bin_volume(&self, indices: [usize; N]) -> Option<f64> {
        let mut volume = 1.0;
        for (axis, index) in indices.iter().copied().enumerate() {
            volume *= self.axes[axis].bin_width(index)?;
        }
        Some(volume)
    }

    pub fn flat_index(&self, indices: [usize; N]) -> Option<usize> {
        flat_index_row_major(indices, self.shape())
    }

    pub fn flat_bin_index(&self, indices: [usize; N]) -> Option<usize> {
        flat_index_row_major(indices, self.bin_shape())
    }

    pub fn point_at_flat(&self, flat_index: usize) -> Option<[f64; N]> {
        self.point(nd_index_row_major(flat_index, self.shape())?)
    }

    pub fn bin_center_at_flat(&self, flat_index: usize) -> Option<[f64; N]> {
        self.bin_center(nd_index_row_major(flat_index, self.bin_shape())?)
    }

    pub fn bin_volume_at_flat(&self, flat_index: usize) -> Option<f64> {
        self.bin_volume(nd_index_row_major(flat_index, self.bin_shape())?)
    }

    pub fn index_subrange(&self, width: usize, value: [f64; N]) -> Option<[usize; N]> {
        let mut indices = [0; N];
        for axis in 0..N {
            indices[axis] = index_subrange(&self.axes[axis], width, value[axis])?;
        }
        Some(indices)
    }

    pub fn point_indices(&self) -> NdGridIndices<N> {
        NdGridIndices {
            shape: self.shape(),
            flat_index: 0,
        }
    }

    pub fn bin_indices(&self) -> NdGridIndices<N> {
        NdGridIndices {
            shape: self.bin_shape(),
            flat_index: 0,
        }
    }

    pub fn points(&self) -> NdGridPoints<'_, G, N> {
        NdGridPoints {
            grid: self,
            flat_index: 0,
        }
    }

    pub fn bin_centers(&self) -> NdBinCenters<'_, G, N> {
        NdBinCenters {
            grid: self,
            flat_index: 0,
        }
    }

    pub fn bin_volumes(&self) -> NdBinVolumes<'_, G, N> {
        NdBinVolumes {
            grid: self,
            flat_index: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NdGridIndices<const N: usize> {
    shape: [usize; N],
    flat_index: usize,
}

impl<const N: usize> Iterator for NdGridIndices<N> {
    type Item = [usize; N];

    fn next(&mut self) -> Option<Self::Item> {
        let indices = nd_index_row_major(self.flat_index, self.shape)?;
        self.flat_index += 1;
        Some(indices)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = size_from_shape(self.shape).unwrap_or(0);
        let remaining = size.saturating_sub(self.flat_index);
        (remaining, Some(remaining))
    }
}

impl<const N: usize> ExactSizeIterator for NdGridIndices<N> {}

#[derive(Clone, Debug)]
pub struct NdGridPoints<'a, G, const N: usize> {
    grid: &'a NdGrid<G, N>,
    flat_index: usize,
}

impl<G: Grid1d, const N: usize> Iterator for NdGridPoints<'_, G, N> {
    type Item = [f64; N];

    fn next(&mut self) -> Option<Self::Item> {
        let point = self.grid.point_at_flat(self.flat_index)?;
        self.flat_index += 1;
        Some(point)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.grid.len().saturating_sub(self.flat_index);
        (remaining, Some(remaining))
    }
}

impl<G: Grid1d, const N: usize> ExactSizeIterator for NdGridPoints<'_, G, N> {}

#[derive(Clone, Debug)]
pub struct NdBinCenters<'a, G, const N: usize> {
    grid: &'a NdGrid<G, N>,
    flat_index: usize,
}

impl<G: Grid1d, const N: usize> Iterator for NdBinCenters<'_, G, N> {
    type Item = [f64; N];

    fn next(&mut self) -> Option<Self::Item> {
        let center = self.grid.bin_center_at_flat(self.flat_index)?;
        self.flat_index += 1;
        Some(center)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.grid.bin_count().saturating_sub(self.flat_index);
        (remaining, Some(remaining))
    }
}

impl<G: Grid1d, const N: usize> ExactSizeIterator for NdBinCenters<'_, G, N> {}

#[derive(Clone, Debug)]
pub struct NdBinVolumes<'a, G, const N: usize> {
    grid: &'a NdGrid<G, N>,
    flat_index: usize,
}

impl<G: Grid1d, const N: usize> Iterator for NdBinVolumes<'_, G, N> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let volume = self.grid.bin_volume_at_flat(self.flat_index)?;
        self.flat_index += 1;
        Some(volume)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.grid.bin_count().saturating_sub(self.flat_index);
        (remaining, Some(remaining))
    }
}

impl<G: Grid1d, const N: usize> ExactSizeIterator for NdBinVolumes<'_, G, N> {}

/// Tagged one-dimensional grid for mixed-axis N-D grids.
///
/// Rust does not have C++-style variadic generics, so `NdGrid<AxisGrid, N>` is the ergonomic path
/// for product grids whose axes use different concrete 1-D grid types.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum AxisGrid {
    Linear(LinearGrid),
    Power(PowerGrid),
    SymmetricPower(SymmetricPowerGrid),
    Custom(CustomGrid),
}

impl AxisGrid {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Linear(_) => "linear",
            Self::Power(_) => "power",
            Self::SymmetricPower(_) => "symmetric_power",
            Self::Custom(_) => "custom",
        }
    }
}

impl Default for AxisGrid {
    fn default() -> Self {
        Self::Linear(LinearGrid::default())
    }
}

impl From<LinearGrid> for AxisGrid {
    fn from(grid: LinearGrid) -> Self {
        Self::Linear(grid)
    }
}

impl From<PowerGrid> for AxisGrid {
    fn from(grid: PowerGrid) -> Self {
        Self::Power(grid)
    }
}

impl From<SymmetricPowerGrid> for AxisGrid {
    fn from(grid: SymmetricPowerGrid) -> Self {
        Self::SymmetricPower(grid)
    }
}

impl From<CustomGrid> for AxisGrid {
    fn from(grid: CustomGrid) -> Self {
        Self::Custom(grid)
    }
}

impl Grid1d for AxisGrid {
    fn len(&self) -> usize {
        match self {
            Self::Linear(grid) => grid.len(),
            Self::Power(grid) => grid.len(),
            Self::SymmetricPower(grid) => grid.len(),
            Self::Custom(grid) => grid.len(),
        }
    }

    fn first(&self) -> f64 {
        match self {
            Self::Linear(grid) => grid.first(),
            Self::Power(grid) => grid.first(),
            Self::SymmetricPower(grid) => grid.first(),
            Self::Custom(grid) => grid.first(),
        }
    }

    fn last(&self) -> f64 {
        match self {
            Self::Linear(grid) => grid.last(),
            Self::Power(grid) => grid.last(),
            Self::SymmetricPower(grid) => grid.last(),
            Self::Custom(grid) => grid.last(),
        }
    }

    fn point(&self, index: usize) -> Option<f64> {
        match self {
            Self::Linear(grid) => grid.point(index),
            Self::Power(grid) => grid.point(index),
            Self::SymmetricPower(grid) => grid.point(index),
            Self::Custom(grid) => grid.point(index),
        }
    }

    fn bin_index(&self, value: f64) -> Option<usize> {
        match self {
            Self::Linear(grid) => grid.bin_index(value),
            Self::Power(grid) => grid.bin_index(value),
            Self::SymmetricPower(grid) => grid.bin_index(value),
            Self::Custom(grid) => grid.bin_index(value),
        }
    }
}

/// Grid defined by user-supplied point coordinates.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct CustomGrid {
    points: Vec<f64>,
}

impl CustomGrid {
    pub fn new(points: impl Into<Vec<f64>>) -> Result<Self> {
        let points = points.into();
        validate_custom_points(&points)?;
        Ok(Self { points })
    }

    pub fn grid_points(&self) -> &[f64] {
        &self.points
    }

    fn is_increasing(&self) -> bool {
        self.first() < self.last()
    }
}

impl Default for CustomGrid {
    fn default() -> Self {
        Self::new([0.0, 1.0]).expect("default custom grid is valid")
    }
}

impl Grid1d for CustomGrid {
    fn len(&self) -> usize {
        self.points.len()
    }

    fn first(&self) -> f64 {
        self.points[0]
    }

    fn last(&self) -> f64 {
        self.points[self.points.len() - 1]
    }

    fn point(&self, index: usize) -> Option<f64> {
        self.points.get(index).copied()
    }

    fn bin_index(&self, value: f64) -> Option<usize> {
        if !self.contains(value) {
            return None;
        }

        let insertion_index = if self.is_increasing() {
            self.points.partition_point(|point| *point <= value)
        } else {
            self.points.partition_point(|point| *point >= value)
        };
        Some(insertion_index.saturating_sub(1).min(self.bin_count() - 1))
    }
}

/// Uniform one-dimensional grid.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearGrid {
    first: f64,
    last: f64,
    len: usize,
    step: f64,
}

impl LinearGrid {
    pub fn new(first: f64, last: f64, len: usize) -> Result<Self> {
        validate_grid(first, last, len)?;
        Ok(Self {
            first,
            last,
            len,
            step: (last - first) / (len - 1) as f64,
        })
    }

    pub fn step(&self) -> f64 {
        self.step
    }
}

impl Default for LinearGrid {
    fn default() -> Self {
        Self::new(0.0, 1.0, 2).expect("default linear grid is valid")
    }
}

impl Grid1d for LinearGrid {
    fn len(&self) -> usize {
        self.len
    }

    fn first(&self) -> f64 {
        self.first
    }

    fn last(&self) -> f64 {
        self.last
    }

    fn point(&self, index: usize) -> Option<f64> {
        (index < self.len).then_some(self.first + self.step * index as f64)
    }

    fn bin_index(&self, value: f64) -> Option<usize> {
        if !self.contains(value) {
            return None;
        }

        let raw = ((value - self.first) / self.step).floor();
        Some(clamp_bin_index(raw, self.bin_count()))
    }
}

/// Power-spaced one-dimensional grid.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PowerGrid {
    first: f64,
    last: f64,
    len: usize,
    power: f64,
    scale: f64,
}

impl PowerGrid {
    pub fn new(first: f64, last: f64, len: usize, power: f64) -> Result<Self> {
        validate_grid(first, last, len)?;
        validate_power(power)?;
        Ok(Self {
            first,
            last,
            len,
            power,
            scale: (last - first) / ((len - 1) as f64).powf(power),
        })
    }

    pub fn power(&self) -> f64 {
        self.power
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Default for PowerGrid {
    fn default() -> Self {
        Self::new(0.0, 1.0, 2, 1.0).expect("default power grid is valid")
    }
}

impl Grid1d for PowerGrid {
    fn len(&self) -> usize {
        self.len
    }

    fn first(&self) -> f64 {
        self.first
    }

    fn last(&self) -> f64 {
        self.last
    }

    fn point(&self, index: usize) -> Option<f64> {
        (index < self.len).then(|| self.first + self.scale * (index as f64).powf(self.power))
    }

    fn bin_index(&self, value: f64) -> Option<usize> {
        if !self.contains(value) {
            return None;
        }

        let ratio = (value - self.first) / self.scale;
        let raw = ratio.max(0.0).powf(1.0 / self.power).floor();
        Some(clamp_bin_index(raw, self.bin_count()))
    }
}

/// Power grid mirrored around the interval midpoint.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SymmetricPowerGrid {
    first: f64,
    last: f64,
    len: usize,
    midpoint: f64,
    lower: PowerGrid,
    upper: PowerGrid,
}

impl SymmetricPowerGrid {
    pub fn new(first: f64, last: f64, len: usize, power: f64) -> Result<Self> {
        if len < 3 || len % 2 == 0 {
            return Err(GridError::InvalidSymmetricSize(len));
        }
        validate_grid(first, last, len)?;
        validate_power(power)?;

        let midpoint = 0.5 * (first + last);
        let half_len = len / 2 + 1;
        Ok(Self {
            first,
            last,
            len,
            midpoint,
            lower: PowerGrid::new(first, midpoint, half_len, power)?,
            upper: PowerGrid::new(last, midpoint, half_len, power)?,
        })
    }

    pub fn midpoint(&self) -> f64 {
        self.midpoint
    }

    pub fn lower_grid(&self) -> PowerGrid {
        self.lower
    }

    pub fn upper_grid(&self) -> PowerGrid {
        self.upper
    }

    pub fn power(&self) -> f64 {
        self.lower.power()
    }
}

impl Default for SymmetricPowerGrid {
    fn default() -> Self {
        Self::new(0.0, 1.0, 3, 2.0).expect("default symmetric power grid is valid")
    }
}

impl Grid1d for SymmetricPowerGrid {
    fn len(&self) -> usize {
        self.len
    }

    fn first(&self) -> f64 {
        self.first
    }

    fn last(&self) -> f64 {
        self.last
    }

    fn point(&self, index: usize) -> Option<f64> {
        if index >= self.len {
            return None;
        }

        let midpoint_index = self.len / 2;
        if index <= midpoint_index {
            self.lower.point(index)
        } else {
            self.upper.point(self.len - 1 - index)
        }
    }

    fn bin_index(&self, value: f64) -> Option<usize> {
        if !self.contains(value) {
            return None;
        }
        if value == self.midpoint {
            return Some(self.len / 2);
        }

        let use_lower =
            if self.first <= self.last { value <= self.midpoint } else { value >= self.midpoint };

        if use_lower {
            self.lower.bin_index(value)
        } else {
            self.upper
                .bin_index(value)
                .map(|index| self.len - 2 - index)
        }
    }
}

/// Return the first index of a centered integer subrange.
///
/// The subrange has length `width` and is clamped to `0..len`. Even widths use the larger possible
/// start index when the center lies exactly between two choices, matching the C++ helper.
pub fn integer_subrange(center: usize, len: usize, width: usize) -> Option<usize> {
    if center >= len || width == 0 || width > len {
        return None;
    }

    let offset = if width % 2 == 0 { width / 2 - 1 } else { width / 2 };
    Some(center.saturating_sub(offset).min(len - width))
}

/// Return the first grid-point index of a centered subrange containing `value`.
pub fn index_subrange(grid: &(impl Grid1d + ?Sized), width: usize, value: f64) -> Option<usize> {
    integer_subrange(grid.bin_index(value)?, grid.len(), width)
}

/// Product of a non-empty array shape, returning `None` for empty/zero/overflowing shapes.
pub fn size_from_shape<const N: usize>(shape: [usize; N]) -> Option<usize> {
    if N == 0 || shape.contains(&0) {
        return None;
    }

    shape
        .iter()
        .try_fold(1usize, |size, extent| size.checked_mul(*extent))
}

/// Convert an N-dimensional row-major index to a flat index.
pub fn flat_index_row_major<const N: usize>(
    indices: [usize; N],
    shape: [usize; N],
) -> Option<usize> {
    size_from_shape(shape)?;
    let mut flat_index = 0usize;
    for axis in 0..N {
        if indices[axis] >= shape[axis] {
            return None;
        }
        flat_index = flat_index * shape[axis] + indices[axis];
    }
    Some(flat_index)
}

/// Convert a flat row-major index to an N-dimensional index.
pub fn nd_index_row_major<const N: usize>(
    flat_index: usize,
    shape: [usize; N],
) -> Option<[usize; N]> {
    let size = size_from_shape(shape)?;
    if flat_index >= size {
        return None;
    }

    let mut indices = [0; N];
    let mut remainder = flat_index;
    for axis in (0..N).rev() {
        indices[axis] = remainder % shape[axis];
        remainder /= shape[axis];
    }
    Some(indices)
}

fn validate_grid(first: f64, last: f64, len: usize) -> Result<()> {
    if len < 2 {
        return Err(GridError::InvalidSize(len));
    }
    if first == last {
        return Err(GridError::DegenerateRange { first, last });
    }
    Ok(())
}

fn validate_axes<G: Grid1d, const N: usize>(axes: &[G; N]) -> Result<()> {
    if N == 0 {
        return Err(GridError::InvalidDimension(N));
    }
    for (axis, grid) in axes.iter().enumerate() {
        if grid.len() < 2 {
            return Err(GridError::InvalidAxisSize {
                axis,
                len: grid.len(),
            });
        }
    }
    Ok(())
}

fn validate_custom_points(points: &[f64]) -> Result<()> {
    if points.len() < 2 {
        return Err(GridError::InvalidSize(points.len()));
    }

    let increasing = points[0] < points[1];
    let decreasing = points[0] > points[1];
    if !increasing && !decreasing {
        return Err(GridError::NonMonotonicPoints);
    }

    let monotonic = if increasing {
        points.windows(2).all(|window| window[0] < window[1])
    } else {
        points.windows(2).all(|window| window[0] > window[1])
    };
    if !monotonic {
        return Err(GridError::NonMonotonicPoints);
    }

    Ok(())
}

fn validate_power(power: f64) -> Result<()> {
    if !power.is_finite() || power <= 0.0 {
        return Err(GridError::InvalidPower(power));
    }
    Ok(())
}

fn clamp_bin_index(raw: f64, bin_count: usize) -> usize {
    if raw.is_sign_negative() {
        0
    } else {
        (raw as usize).min(bin_count - 1)
    }
}
