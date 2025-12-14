//! grid.rs
//! Logical / computational domain: structured Cartesian index space on [0,1]^D.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
  X,
  Y,
  Z,
}

impl Axis {
  pub fn from_usize(u: usize) -> Self {
    match u {
      0 => Axis::X,
      1 => Axis::Y,
      2 => Axis::Z,
      _ => panic!("Axis::from_usize: out of range"),
    }
  }

  pub fn as_usize(self) -> usize {
    match self {
      Axis::X => 0,
      Axis::Y => 1,
      Axis::Z => 2,
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
  Minus,
  Plus,
}

impl Side {
  pub fn sign(self) -> f64 {
    match self {
      Side::Minus => -1.0,
      Side::Plus => 1.0,
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellId<const D: usize> {
  pub ijk: [usize; D],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FaceId<const D: usize> {
  pub cell: CellId<D>,
  pub axis: Axis,
  pub side: Side,
}

#[derive(Debug, Clone)]
pub struct Grid<const D: usize> {
  pub n: [usize; D],
}

impl<const D: usize> Grid<D> {
  pub fn new(n: [usize; D]) -> Self {
    assert!(D == 2 || D == 3, "This skeleton supports D=2 or D=3.");
    for a in 0..D {
      assert!(n[a] > 0, "Grid extent must be > 0");
    }
    Self { n }
  }

  /// Computational cell sizes for [0,1]^D.
  pub fn dxi(&self) -> [f64; D] {
    let mut h = [0.0; D];
    for a in 0..D {
      h[a] = 1.0 / (self.n[a] as f64);
    }
    h
  }

  pub fn cell_count(&self) -> usize {
    self.n.iter().product()
  }

  pub fn cell_linear(&self, c: CellId<D>) -> usize {
    let mut idx = 0usize;
    let mut stride = 1usize;
    for a in 0..D {
      idx += c.ijk[a] * stride;
      stride *= self.n[a];
    }
    idx
  }

  /// Inverse of `cell_linear`: linear -> (i,j,k).
  pub fn cell_from_linear(&self, mut idx: usize) -> CellId<D> {
    let mut ijk = [0usize; D];
    for a in 0..D {
      let na = self.n[a];
      ijk[a] = idx % na;
      idx /= na;
    }
    CellId { ijk }
  }

  /// Iterator over all cells.
  pub fn iter_cells(&self) -> CellIter<'_, D> {
    CellIter {
      grid: self,
      next_linear: 0,
      end_linear: self.cell_count(),
    }
  }

  pub fn cell_center_xi(&self, c: CellId<D>) -> [f64; D] {
    let h = self.dxi();
    let mut xi = [0.0; D];
    for a in 0..D {
      xi[a] = (c.ijk[a] as f64 + 0.5) * h[a];
    }
    xi
  }

  pub fn face_center_xi(&self, f: FaceId<D>) -> [f64; D] {
    let h = self.dxi();
    let mut xi = self.cell_center_xi(f.cell);
    let ax = f.axis.as_usize();
    xi[ax] += 0.5 * h[ax] * f.side.sign();
    xi
  }

  /// Faces for a cell.
  pub fn cell_faces(&self, cell: CellId<D>) -> Vec<FaceId<D>> {
    let mut faces = Vec::with_capacity(2 * D);
    for axis_u in 0..D {
      let axis = Axis::from_usize(axis_u);
      faces.push(FaceId { cell, axis, side: Side::Minus });
      faces.push(FaceId { cell, axis, side: Side::Plus });
    }
    faces
  }
}

pub struct CellIter<'a, const D: usize> {
  grid: &'a Grid<D>,
  next_linear: usize,
  end_linear: usize,
}

impl<'a, const D: usize> Iterator for CellIter<'a, D> {
  type Item = CellId<D>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.next_linear >= self.end_linear {
      return None;
    }
    let c = self.grid.cell_from_linear(self.next_linear);
    self.next_linear += 1;
    Some(c)
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let remaining = self.end_linear.saturating_sub(self.next_linear);
    (remaining, Some(remaining))
  }
}

