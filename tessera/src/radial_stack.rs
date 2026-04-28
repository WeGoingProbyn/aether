// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use crate::coupling::{FacePair, MeshCoupler, Side};
use utility::domain::{CellId, FaceId};

/// Two cube-spheres sharing the same 6 panels and angular dims, where the
/// upper mesh's bottom radial layer touches the lower mesh's top radial
/// layer. Pairings are pure index arithmetic — no interpolation.
pub struct RadialStackCoupler {
  panel_count: usize,
  angular_dims: [usize; 2],
  lower_radial_layers: usize,
  upper_radial_layers: usize,
  lower_top_layer_idx: usize,
  upper_bottom_layer_idx: usize,
  pairs: Vec<FacePair>,
}

impl RadialStackCoupler {
  pub fn new(
    angular_dims: [usize; 2],
    lower_radial_layers: usize,
    upper_radial_layers: usize,
  ) -> Self {
    Self::with_panel_count(
      6,
      angular_dims,
      lower_radial_layers,
      upper_radial_layers,
    )
  }

  pub fn with_panel_count(
    panel_count: usize,
    angular_dims: [usize; 2],
    lower_radial_layers: usize,
    upper_radial_layers: usize,
  ) -> Self {
    assert!(panel_count > 0, "radial stack needs at least one panel");
    assert!(
      angular_dims[0] > 0 && angular_dims[1] > 0,
      "angular dims must be non-zero"
    );
    assert!(
      lower_radial_layers > 0 && upper_radial_layers > 0,
      "radial layer counts must be non-zero"
    );

    let lower_top_layer_idx = lower_radial_layers - 1;
    let upper_bottom_layer_idx = 0;
    let mut pairs =
      Vec::with_capacity(panel_count * angular_cell_count(angular_dims));

    for panel in 0..panel_count {
      for j in 0..angular_dims[1] {
        for i in 0..angular_dims[0] {
          pairs.push(FacePair::new(
            lower_top_face(panel, i, j, angular_dims, lower_radial_layers),
            upper_bottom_face(panel, i, j, angular_dims, upper_radial_layers),
          ));
        }
      }
    }

    Self {
      panel_count,
      angular_dims,
      lower_radial_layers,
      upper_radial_layers,
      lower_top_layer_idx,
      upper_bottom_layer_idx,
      pairs,
    }
  }

  pub fn panel_count(&self) -> usize {
    self.panel_count
  }

  pub fn angular_dims(&self) -> [usize; 2] {
    self.angular_dims
  }

  pub fn lower_radial_layers(&self) -> usize {
    self.lower_radial_layers
  }

  pub fn upper_radial_layers(&self) -> usize {
    self.upper_radial_layers
  }

  pub fn lower_top_layer_idx(&self) -> usize {
    self.lower_top_layer_idx
  }

  pub fn upper_bottom_layer_idx(&self) -> usize {
    self.upper_bottom_layer_idx
  }

  fn interface_cell(
    &self,
    side: Side,
    cell: CellId,
  ) -> Option<(usize, usize, usize)> {
    let radial_layers = match side {
      Side::A => self.lower_radial_layers,
      Side::B => self.upper_radial_layers,
    };
    let target_layer = match side {
      Side::A => self.lower_top_layer_idx,
      Side::B => self.upper_bottom_layer_idx,
    };

    let cells_per_panel =
      self.angular_dims[0] * self.angular_dims[1] * radial_layers;
    let panel = cell.index() / cells_per_panel;
    if panel >= self.panel_count {
      return None;
    }

    let local = cell.index() % cells_per_panel;
    let angular = self.angular_dims[0] * self.angular_dims[1];
    let k = local / angular;
    if k != target_layer {
      return None;
    }

    let local_angular = local % angular;
    let i = local_angular % self.angular_dims[0];
    let j = local_angular / self.angular_dims[0];
    Some((panel, i, j))
  }

  fn interface_face(
    &self,
    side: Side,
    face: FaceId,
  ) -> Option<(usize, usize, usize)> {
    let radial_layers = match side {
      Side::A => self.lower_radial_layers,
      Side::B => self.upper_radial_layers,
    };
    let kept_faces = kept_faces_per_panel(self.angular_dims, radial_layers);
    let panel = face.index() / kept_faces;
    if panel >= self.panel_count {
      return None;
    }

    let local = face.index() % kept_faces;
    let interior = interior_faces_per_panel(self.angular_dims, radial_layers);
    let angular = angular_cell_count(self.angular_dims);
    let interface_offset = match side {
      Side::A => local.checked_sub(interior + angular)?,
      Side::B => local.checked_sub(interior)?,
    };
    if interface_offset >= angular {
      return None;
    }

    let i = interface_offset % self.angular_dims[0];
    let j = interface_offset / self.angular_dims[0];
    Some((panel, i, j))
  }
}

impl MeshCoupler for RadialStackCoupler {
  fn paired_face(&self, side: Side, face: FaceId) -> Option<(Side, FaceId)> {
    let (panel, i, j) = self.interface_face(side, face)?;
    match side {
      Side::A => Some((
        Side::B,
        upper_bottom_face(
          panel,
          i,
          j,
          self.angular_dims,
          self.upper_radial_layers,
        ),
      )),
      Side::B => Some((
        Side::A,
        lower_top_face(
          panel,
          i,
          j,
          self.angular_dims,
          self.lower_radial_layers,
        ),
      )),
    }
  }

  fn paired_cell(&self, side: Side, cell: CellId) -> Option<(Side, CellId)> {
    let (panel, i, j) = self.interface_cell(side, cell)?;
    match side {
      Side::A => Some((
        Side::B,
        cell_id(
          panel,
          i,
          j,
          self.upper_bottom_layer_idx,
          self.angular_dims,
          self.upper_radial_layers,
        ),
      )),
      Side::B => Some((
        Side::A,
        cell_id(
          panel,
          i,
          j,
          self.lower_top_layer_idx,
          self.angular_dims,
          self.lower_radial_layers,
        ),
      )),
    }
  }

  fn pairs(&self) -> &[FacePair] {
    &self.pairs
  }
}

fn angular_cell_count(angular_dims: [usize; 2]) -> usize {
  angular_dims[0] * angular_dims[1]
}

fn interior_faces_per_panel(
  angular_dims: [usize; 2],
  radial_layers: usize,
) -> usize {
  let [nx, ny] = angular_dims;
  (nx - 1) * ny * radial_layers
    + nx * (ny - 1) * radial_layers
    + nx * ny * (radial_layers - 1)
}

fn kept_faces_per_panel(
  angular_dims: [usize; 2],
  radial_layers: usize,
) -> usize {
  interior_faces_per_panel(angular_dims, radial_layers)
    + 2 * angular_cell_count(angular_dims)
}

fn lower_top_face(
  panel: usize,
  i: usize,
  j: usize,
  angular_dims: [usize; 2],
  radial_layers: usize,
) -> FaceId {
  let angular = angular_cell_count(angular_dims);
  FaceId::from(
    panel * kept_faces_per_panel(angular_dims, radial_layers)
      + interior_faces_per_panel(angular_dims, radial_layers)
      + angular
      + i
      + j * angular_dims[0],
  )
}

fn upper_bottom_face(
  panel: usize,
  i: usize,
  j: usize,
  angular_dims: [usize; 2],
  radial_layers: usize,
) -> FaceId {
  FaceId::from(
    panel * kept_faces_per_panel(angular_dims, radial_layers)
      + interior_faces_per_panel(angular_dims, radial_layers)
      + i
      + j * angular_dims[0],
  )
}

fn cell_id(
  panel: usize,
  i: usize,
  j: usize,
  k: usize,
  angular_dims: [usize; 2],
  radial_layers: usize,
) -> CellId {
  let angular = angular_cell_count(angular_dims);
  CellId::from(
    panel * angular * radial_layers + i + j * angular_dims[0] + k * angular,
  )
}
