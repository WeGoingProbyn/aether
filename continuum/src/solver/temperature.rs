use crate::coords::cartesian::{ CartesianGrid2D, CoordinateSystem };
use crate::field::scalar::ScalarField;

pub struct TemperatureSolver2D {
  pub grid: CartesianGrid2D,
  pub temperature: ScalarField,
  pub velocity: [f32; 2],
  pub diffusivity: f32,
}

impl TemperatureSolver2D {
  pub fn step(&mut self, dt: f32) {
    let nx = self.grid.nx;
    let ny = self.grid.ny;

    let dx = self.grid.dx;
    let dy = self.grid.dy;

    let u = self.velocity[0];
    let v = self.velocity[1];

    let mut new_temp = self.temperature.field.clone();

    // Interior update (Dirichlet/unchanged boundaries for now)
    for j in 1..ny - 1 {
      for i in 1..nx - 1 {
        let idx = self.grid.linear_index((i, j));

        let t   = self.temperature.get(idx);
        let t_ip = self.temperature.get(self.grid.linear_index((i + 1, j)));
        let t_im = self.temperature.get(self.grid.linear_index((i - 1, j)));
        let t_jp = self.temperature.get(self.grid.linear_index((i, j + 1)));
        let t_jm = self.temperature.get(self.grid.linear_index((i, j - 1)));

        // ---------- Advection: first-order upwind ----------
        // dT/dx based on sign(u)
        let dtdx = if u >= 0.0 {
          (t - t_im) / dx
        } else {
          (t_ip - t) / dx
        };

        // dT/dy based on sign(v)
        let dtdy = if v >= 0.0 {
          (t - t_jm) / dy
        } else {
          (t_jp - t) / dy
        };

        let adv = u * dtdx + v * dtdy;

        // ---------- Diffusion: 5-point Laplacian ----------
        let lap = (t_ip - 2.0 * t + t_im) / (dx * dx)
        + (t_jp - 2.0 * t + t_jm) / (dy * dy);

        // ---------- Explicit update ----------
        new_temp[idx] = t + dt * (-adv + self.diffusivity * lap);
      }
    }

    self.temperature.field = new_temp;
  }
}

