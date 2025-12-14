#[derive(Debug, Clone)]
pub struct GmresConfig {
  pub restart: usize,
  pub max_iters: usize,
  pub tol: f64, // relative tol on preconditioned residual
}

pub trait LinearOperator {
  fn dim(&self) -> usize;
  fn apply(&self, x: &[f64], y: &mut [f64]); // y = A x
}

pub trait LeftPreconditioner {
  fn dim(&self) -> usize;
  fn apply_inv(&self, x: &[f64], y: &mut [f64]); // y = M^{-1} x
}

#[derive(Debug, Clone)]
pub struct IdentityPreconditioner {
  n: usize,
}

impl IdentityPreconditioner {
  pub fn new(n: usize) -> Self {
    Self { n }
  }
}

impl LeftPreconditioner for IdentityPreconditioner {
  fn dim(&self) -> usize {
    self.n
  }
  fn apply_inv(&self, x: &[f64], y: &mut [f64]) {
    y.copy_from_slice(x);
  }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
  let mut s = 0.0;
  for i in 0..a.len() {
    s += a[i] * b[i];
  }
  s
}
fn norm(a: &[f64]) -> f64 {
  dot(a, a).sqrt()
}

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
  for i in 0..y.len() {
    y[i] += a * x[i];
  }
}
fn scal(y: &mut [f64], a: f64) {
  for i in 0..y.len() {
    y[i] *= a;
  }
}

/// Convenience: unpreconditioned GMRES
pub fn gmres(op: &dyn LinearOperator, b: &[f64], x: &mut [f64], cfg: &GmresConfig) -> Result<(), String> {
  let m = IdentityPreconditioner::new(op.dim());
  gmres_left_precond(op, &m, b, x, cfg)
}

/// Restarted left-preconditioned GMRES:
/// Solve A x = b by applying GMRES to (M^{-1} A) x = (M^{-1} b).
pub fn gmres_left_precond(
  op: &dyn LinearOperator,
  m_inv: &dyn LeftPreconditioner,
  b: &[f64],
  x: &mut [f64],
  cfg: &GmresConfig,
) -> Result<(), String> {
  let n = op.dim();
  if b.len() != n || x.len() != n {
    return Err("gmres: dimension mismatch".into());
  }
  if m_inv.dim() != n {
    return Err("gmres: preconditioner dimension mismatch".into());
  }

  let mut ax = vec![0.0; n];
  let mut tmp = vec![0.0; n];

  // b_tilde = M^{-1} b
  m_inv.apply_inv(b, &mut tmp);
  let bnorm = norm(&tmp).max(1e-30);

  let mut r_tilde = vec![0.0; n];
  let mut iters_total = 0usize;

  loop {
    // r_tilde = M^{-1} (b - A x)
    op.apply(x, &mut ax);
    for i in 0..n {
      tmp[i] = b[i] - ax[i];
    }
    m_inv.apply_inv(&tmp, &mut r_tilde);

    let rnorm0 = norm(&r_tilde);
    if rnorm0 / bnorm <= cfg.tol {
      return Ok(());
    }
    if iters_total >= cfg.max_iters {
      return Err("gmres: max_iters reached".into());
    }

    let m = cfg.restart;

    // Krylov basis V and Hessenberg H for the preconditioned operator
    let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    let mut h = vec![vec![0.0; m]; m + 1];

    // Givens rotations
    let mut cs = vec![0.0; m];
    let mut sn = vec![0.0; m];
    let mut g = vec![0.0; m + 1];

    // v0 = r_tilde / ||r_tilde||
    let beta = rnorm0;
    let mut v0 = r_tilde.clone();
    scal(&mut v0, 1.0 / beta);
    v.push(v0);
    g[0] = beta;

    let mut k_done = 0usize;

    for k in 0..m {
      iters_total += 1;

      // w = M^{-1} A v_k
      op.apply(&v[k], &mut ax);
      let mut w = vec![0.0; n];
      m_inv.apply_inv(&ax, &mut w);

      // Arnoldi
      for j in 0..=k {
        h[j][k] = dot(&w, &v[j]);
        axpy(&mut w, -h[j][k], &v[j]);
      }
      h[k + 1][k] = norm(&w);
      if h[k + 1][k] != 0.0 {
        scal(&mut w, 1.0 / h[k + 1][k]);
      }
      v.push(w);

      // Apply previous Givens
      for j in 0..k {
        let temp = cs[j] * h[j][k] + sn[j] * h[j + 1][k];
        h[j + 1][k] = -sn[j] * h[j][k] + cs[j] * h[j + 1][k];
        h[j][k] = temp;
      }

      // New Givens rotation
      let (c, s) = givens(h[k][k], h[k + 1][k]);
      cs[k] = c;
      sn[k] = s;

      let temp = c * h[k][k] + s * h[k + 1][k];
      h[k + 1][k] = 0.0;
      h[k][k] = temp;

      // Apply to g
      let tempg = c * g[k] + s * g[k + 1];
      g[k + 1] = -s * g[k] + c * g[k + 1];
      g[k] = tempg;

      k_done = k + 1;
      let res = g[k_done].abs() / bnorm;
      if res <= cfg.tol || iters_total >= cfg.max_iters {
        break;
      }
    }

    // Solve upper triangular system for y
    let mut y = vec![0.0; k_done];
    for i in (0..k_done).rev() {
      let mut sum = g[i];
      for j in (i + 1)..k_done {
        sum -= h[i][j] * y[j];
      }
      y[i] = sum / h[i][i];
    }

    // x = x + V y
    for j in 0..k_done {
      axpy(x, y[j], &v[j]);
    }

    if iters_total >= cfg.max_iters {
      return Err("gmres: max_iters reached".into());
    }
  }
}

fn givens(a: f64, b: f64) -> (f64, f64) {
  if b == 0.0 {
    (1.0, 0.0)
  } else if a.abs() > b.abs() {
    let t = b / a;
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = c * t;
    (c, s)
  } else {
    let t = a / b;
    let s = 1.0 / (1.0 + t * t).sqrt();
    let c = s * t;
    (c, s)
  }
}

