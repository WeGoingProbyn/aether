#[derive(Debug, Clone)]
pub struct GmresConfig {
    pub restart: usize,     // e.g. 30
    pub max_iters: usize,   // e.g. 200
    pub tol: f64,           // relative residual tol, e.g. 1e-8
}

pub trait LinearOperator {
    fn dim(&self) -> usize;
    fn apply(&self, x: &[f64], y: &mut [f64]); // y = A x
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() { s += a[i] * b[i]; }
    s
}
fn norm(a: &[f64]) -> f64 { dot(a, a).sqrt() }

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for i in 0..y.len() { y[i] += a * x[i]; }
}
fn scal(y: &mut [f64], a: f64) {
    for i in 0..y.len() { y[i] *= a; }
}
fn copy(dst: &mut [f64], src: &[f64]) {
    dst.copy_from_slice(src);
}

/// Basic restarted GMRES. No preconditioner (yet).
pub fn gmres(op: &dyn LinearOperator, b: &[f64], x: &mut [f64], cfg: &GmresConfig) -> Result<(), String> {
    let n = op.dim();
    if b.len() != n || x.len() != n { return Err("gmres: dimension mismatch".into()); }

    let mut r = vec![0.0; n];
    let mut ax = vec![0.0; n];
    let bnorm = norm(b).max(1e-30);

    let mut iters_total = 0usize;

    loop {
        // r = b - A x
        op.apply(x, &mut ax);
        for i in 0..n { r[i] = b[i] - ax[i]; }
        let rnorm0 = norm(&r);
        if rnorm0 / bnorm <= cfg.tol { return Ok(()); }
        if iters_total >= cfg.max_iters { return Err("gmres: max_iters reached".into()); }

        // Krylov basis V and Hessenberg H
        let m = cfg.restart;
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        let mut h = vec![vec![0.0; m]; m + 1];

        // Givens rotations
        let mut cs = vec![0.0; m];
        let mut sn = vec![0.0; m];
        let mut g = vec![0.0; m + 1];

        // v0 = r / ||r||
        let beta = rnorm0;
        let mut v0 = r.clone();
        scal(&mut v0, 1.0 / beta);
        v.push(v0);
        g[0] = beta;

        let mut k_done = 0usize;

        for k in 0..m {
            iters_total += 1;

            // w = A v_k
            let mut w = vec![0.0; n];
            op.apply(&v[k], &mut w);

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

            // Apply previous Givens rotations to new column of H
            for j in 0..k {
                let temp = cs[j] * h[j][k] + sn[j] * h[j + 1][k];
                h[j + 1][k] = -sn[j] * h[j][k] + cs[j] * h[j + 1][k];
                h[j][k] = temp;
            }

            // New Givens rotation to zero h[k+1][k]
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

            let res = g[k + 1].abs() / bnorm;
            k_done = k + 1;
            if res <= cfg.tol {
                break;
            }
            if iters_total >= cfg.max_iters {
                break;
            }
        }

        // Solve upper triangular system for y: H_hat y = g_hat
        let mut y = vec![0.0; k_done];
        for i in (0..k_done).rev() {
            let mut sum = g[i];
            for j in (i + 1)..k_done {
                sum -= h[i][j] * y[j];
            }
            y[i] = sum / h[i][i];
        }

        // x = x + V_k y
        for j in 0..k_done {
            axpy(x, y[j], &v[j]);
        }

        // Check convergence after restart
        op.apply(x, &mut ax);
        for i in 0..n { r[i] = b[i] - ax[i]; }
        let rnorm = norm(&r);
        if rnorm / bnorm <= cfg.tol { return Ok(()); }

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
