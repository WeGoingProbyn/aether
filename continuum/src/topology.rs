//! topology.rs
//! Connectivity: neighbor cells / boundaries for each face.

use crate::grid::grid::{Axis, CellId, FaceId, Grid, Side};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundaryId {
    pub axis: Axis,
    pub side: Side,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Neighbor<const D: usize> {
    Cell(CellId<D>),
    Boundary(BoundaryId),
}

pub trait Topology<const D: usize> {
    fn grid(&self) -> &Grid<D>;

    fn faces_of_cell(&self, cell: CellId<D>) -> Vec<FaceId<D>> {
        self.grid().cell_faces(cell)
    }

    fn neighbor_across(&self, face: FaceId<D>) -> Neighbor<D>;
}

#[derive(Debug, Clone)]
pub struct StructuredTopology<const D: usize> {
    grid: Grid<D>,
    pub periodic: [bool; D],
}

impl<const D: usize> StructuredTopology<D> {
    pub fn new(grid: Grid<D>) -> Self {
        Self {
            grid,
            periodic: [false; D],
        }
    }

    pub fn with_periodic(mut self, axis: usize, on: bool) -> Self {
        self.periodic[axis] = on;
        self
    }
}

impl<const D: usize> Topology<D> for StructuredTopology<D> {
    fn grid(&self) -> &Grid<D> {
        &self.grid
    }

    fn neighbor_across(&self, face: FaceId<D>) -> Neighbor<D> {
        let ax = face.axis.as_usize();
        let mut n = face.cell.ijk;
        let extent = self.grid.n[ax];

        match face.side {
            Side::Minus => {
                if n[ax] == 0 {
                    if self.periodic[ax] {
                        n[ax] = extent - 1;
                        Neighbor::Cell(CellId { ijk: n })
                    } else {
                        Neighbor::Boundary(BoundaryId { axis: face.axis, side: face.side })
                    }
                } else {
                    n[ax] -= 1;
                    Neighbor::Cell(CellId { ijk: n })
                }
            }
            Side::Plus => {
                if n[ax] + 1 >= extent {
                    if self.periodic[ax] {
                        n[ax] = 0;
                        Neighbor::Cell(CellId { ijk: n })
                    } else {
                        Neighbor::Boundary(BoundaryId { axis: face.axis, side: face.side })
                    }
                } else {
                    n[ax] += 1;
                    Neighbor::Cell(CellId { ijk: n })
                }
            }
        }
    }
}
