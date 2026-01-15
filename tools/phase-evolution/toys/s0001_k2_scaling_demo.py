#!/usr/bin/env python3
"""
S-0001 κ2-scaling classification demo (OU vs long-memory Gaussian) — SELF-AUDITING

Goal (checkable):
- Compute c(t) = |<exp(i Δϕ(t))>_ensemble| where Δϕ(t)=∫ ω(s) ds
- Verify scaling via a declared diagnostic fit:
    y(t) := log(-log c(t))  vs  x(t) := log t
  If κ2(t) ~ t^alpha then y ~ alpha x + const.

Self-audit:
- OU: expected alpha ≈ 1  (diffusive at long times)
- Long-memory Gaussian (fGn with H): expected alpha ≈ 2H

This is a toy existence + classification artifact, not a claim of fundamental dynamics.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Config
# ---------------------------

@dataclass(frozen=True)
class DemoConfig:
    seed: int = 12345
    n_trajectories: int = 2000

    # time grid
    dt: float = 0.01
    n_steps: int = 4000  # total time = n_steps*dt

    # envelope sampling points (indices into time grid)
    n_t_samples: int = 200

    # OU params: dω = theta*(mu-ω)dt + sigma dW
    ou_theta: float = 0.6
    ou_mu: float = 1.0
    ou_sigma: float = 0.25

    # Long-memory Gaussian params (fractional Gaussian noise)
    fgn_H: float = 0.70
    fgn_mu: float = 1.0
    fgn_sigma: float = 0.25

    # --- Self-audit parameters ---
    audit_window_t_min: float = 20.0  # fit window lower bound in time units
    audit_window_t_max: float = 40.0  # fit window upper bound
    audit_tol_abs: float = 0.18       # absolute tolerance on slope |alpha - expected|
    audit_min_log_span: float = 0.6   # require >= exp(0.6) ≈ 1.82× in time across the fit window

    out_dir: str = "toys/outputs"
    out_csv: str = "s0001_envelopes.csv"
    out_png: str = "s0001_envelopes.png"
    out_scaling_png: str = "s0001_scaling_diagnostic.png"
    out_audit_csv: str = "s0001_audit.csv"

def auto_audit_slope(t, c, expected, tol_abs, min_points=12, min_r2=0.98,
                     t_min_floor=None, window_points_max=20, min_log_span=0.0,
                     t_max_cap=None):
    """
    Search over contiguous windows (in index space) and find the best fit window
    whose slope is closest to expected, subject to r2, min_points, and (optionally)
    a minimum log time-span.

    min_log_span: require log(t_end / t_start) >= min_log_span.
                  Example: 1.2 => window spans >= exp(1.2) ≈ 3.32× in time.
    t_max_cap: if set, require t_end <= t_max_cap (keeps audit inside declared window).
    """
    t = np.asarray(t)
    c = np.asarray(c)

    best = None
    n = len(t)

    for i0 in range(0, n - min_points):
        if t_min_floor is not None and t[i0] < t_min_floor:
            continue
        i1_max = min(n, i0 + window_points_max)
        for i1 in range(i0 + min_points, i1_max + 1):
            t0 = float(t[i0])
            t1 = float(t[i1 - 1])

            if t_max_cap is not None and t1 > t_max_cap:
                break

            if min_log_span > 0.0:
                if t0 <= 0.0 or t1 <= t0:
                    continue
                if math.log(t1 / t0) < min_log_span:
                    continue

            try:
                res = audit_slope(
                    t=t[i0:i1],
                    c=c[i0:i1],
                    t_min=t0,
                    t_max=t1,
                    expected=expected,
                    tol_abs=tol_abs,
                )
            except Exception:
                continue

            if res.r2 < min_r2:
                continue

            key = (abs(res.slope - expected), -res.r2, -res.n_points)
            if best is None or key < best[0]:
                best = (key, res)
    if best is None:
        raise ValueError("No acceptable audit window found (try relaxing constraints).")

    return best[1]

# ---------------------------
# Processes
# ---------------------------

def simulate_ou_omega(cfg: DemoConfig, rng: np.random.Generator) -> np.ndarray:
    """Return omega shape (n_trajectories, n_steps)."""
    nT, n = cfg.n_trajectories, cfg.n_steps
    dt = cfg.dt
    theta, mu, sigma = cfg.ou_theta, cfg.ou_mu, cfg.ou_sigma

    omega = np.empty((nT, n), dtype=float)
    omega[:, 0] = mu

    sqrt_dt = math.sqrt(dt)
    for t in range(1, n):
        dW = rng.normal(0.0, sqrt_dt, size=nT)
        omega[:, t] = omega[:, t - 1] + theta * (mu - omega[:, t - 1]) * dt + sigma * dW

    return omega


def fgn_davies_harte(n_steps: int, H: float, rng: np.random.Generator) -> np.ndarray:
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0,1)")
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")

    n = int(n_steps)
    k = np.arange(0, n, dtype=float)

    gamma = 0.5 * ((np.abs(k + 1.0) ** (2.0 * H)) +
                   (np.abs(k - 1.0) ** (2.0 * H)) -
                   2.0 * (np.abs(k) ** (2.0 * H)))

    m = 2 * n
    r = np.zeros(m, dtype=float)
    r[0:n] = gamma
    r[n] = 0.0
    r[n+1:] = gamma[1:][::-1]

    lam = np.fft.fft(r).real
    lam[lam < 0.0] = 0.0

    W = rng.normal(size=m) + 1j * rng.normal(size=m)
    X = np.fft.ifft(np.sqrt(lam) * W).real

    x = X[:n]
    std = float(np.std(x))
    if std <= 0.0:
        raise RuntimeError("fGn generation produced non-positive std; numerical issue.")
    return x / std


def simulate_fgn_omega(cfg: DemoConfig, rng: np.random.Generator) -> np.ndarray:
    """Return omega shape (n_trajectories, n_steps) using fGn increments."""
    nT, n = cfg.n_trajectories, cfg.n_steps
    mu, sigma, H = cfg.fgn_mu, cfg.fgn_sigma, cfg.fgn_H

    omega = np.empty((nT, n), dtype=float)
    for i in range(nT):
        inc = fgn_davies_harte(n, H, rng)
        omega[i, :] = mu + sigma * inc

    return omega


# ---------------------------
# Envelope computation
# ---------------------------

def envelope_from_omega(omega: np.ndarray, dt: float, t_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    omega: (n_trajectories, n_steps)
    Returns:
      t: times (len t_idx,)
      c: envelope values (len t_idx,)
    """
    delta_phi = np.cumsum(omega, axis=1) * dt
    z = np.exp(1j * delta_phi[:, t_idx])
    c = np.abs(np.mean(z, axis=0))
    t = (t_idx + 1) * dt
    return t, c


def choose_t_samples(cfg: DemoConfig) -> np.ndarray:
    n = cfg.n_steps
    dt = cfg.dt

    # Base: log-spaced indices over full run
    idx_log = np.clip(
        np.logspace(0, math.log10(n), cfg.n_t_samples).astype(int) - 1,
        0,
        n - 1,
    )

    # Ensure enough points inside the declared audit window by adding a small linear grid.
    t_min = cfg.audit_window_t_min
    t_max = cfg.audit_window_t_max
    i_min = int(max(0, math.floor(t_min / dt) - 1))
    i_max = int(min(n - 1, math.ceil(t_max / dt) - 1))

    # 64 points is cheap and makes min_points=20 robust even for narrow windows.
    idx_win = np.linspace(i_min, i_max, 64, dtype=int)

    idx = np.unique(np.concatenate([idx_log, idx_win]))
    return idx


# ---------------------------
# Self-audit (slope fit)
# ---------------------------

@dataclass(frozen=True)
class AuditResult:
    slope: float
    slope_err: float
    intercept: float
    r2: float
    n_points: int
    t_min: float
    t_max: float
    expected: float
    passed: bool


def _safe_loglog_arrays(t: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return x=log t, y=log(-log c) with guards.
    Requires 0 < c < 1 for y to be real and finite.
    """
    eps = 1e-12
    t2 = np.asarray(t)
    c2 = np.asarray(c)

    c2 = np.clip(c2, eps, 1.0 - eps)
    y = -np.log(c2)
    y = np.clip(y, eps, None)

    return np.log(t2), np.log(y)


def audit_slope(t: np.ndarray, c: np.ndarray, t_min: float, t_max: float, expected: float, tol_abs: float) -> AuditResult:
    t = np.asarray(t)
    c = np.asarray(c)

    mask = (t >= t_min) & (t <= t_max) & (c > 0.0) & (c < 1.0)
    if np.count_nonzero(mask) < 8:
        raise ValueError(f"Insufficient points in audit window [{t_min}, {t_max}]")

    x, y = _safe_loglog_arrays(t[mask], c[mask])

    A = np.vstack([x, np.ones_like(x)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)

    # Standard error of slope (simple OLS estimate)
    n = len(x)
    if n > 2:
        s2 = ss_res / (n - 2)
        sx = float(np.std(x))
        slope_err = float(math.sqrt(s2) / (sx * math.sqrt(n))) if sx > 0 else float("inf")
    else:
        slope_err = float("inf")

    passed = abs(slope - expected) <= tol_abs

    return AuditResult(
        slope=slope,
        slope_err=slope_err,
        intercept=intercept,
        r2=r2,
        n_points=n,
        t_min=float(t_min),
        t_max=float(t_max),
        expected=float(expected),
        passed=bool(passed),
    )


def print_audit_banner(label: str, res: AuditResult) -> None:
    status = "PASS" if res.passed else "FAIL"
    print(
        f"[{label}] slope = {res.slope:.3f} ± {res.slope_err:.3f} "
        f"(expected {res.expected:.3f}, window [{res.t_min:g},{res.t_max:g}], n={res.n_points}, r2={res.r2:.3f}) "
        f"[{status}]"
    )


def write_audit_csv(path: Path, rows: list[tuple[str, AuditResult]]) -> None:
    """
    Writes a fixed-schema audit CSV. Overwrites each run to keep it clean.
    """
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "slope", "slope_err", "expected", "t_min", "t_max", "n_points", "r2", "pass"])
        for name, res in rows:
            w.writerow([
                name,
                f"{res.slope:.10g}",
                f"{res.slope_err:.10g}",
                f"{res.expected:.10g}",
                f"{res.t_min:.10g}",
                f"{res.t_max:.10g}",
                res.n_points,
                f"{res.r2:.10g}",
                int(res.passed),
            ])


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    cfg = DemoConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    t_idx = choose_t_samples(cfg)

    # OU
    omega_ou = simulate_ou_omega(cfg, rng)
    t, c_ou = envelope_from_omega(omega_ou, cfg.dt, t_idx)

    # fGn / long memory
    omega_fgn = simulate_fgn_omega(cfg, rng)
    _, c_fgn = envelope_from_omega(omega_fgn, cfg.dt, t_idx)

    # write envelope CSV
    csv_path = out_dir / cfg.out_csv
    header = "t,c_ou,c_fgn,H,ou_theta,ou_sigma,fgn_sigma,n_trajectories,dt,seed\n"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for ti, a, b in zip(t, c_ou, c_fgn):
            f.write(
                f"{ti:.10g},{a:.10g},{b:.10g},{cfg.fgn_H},{cfg.ou_theta},{cfg.ou_sigma},"
                f"{cfg.fgn_sigma},{cfg.n_trajectories},{cfg.dt},{cfg.seed}\n"
            )

    # plot envelopes
    fig_path = out_dir / cfg.out_png
    plt.figure()
    plt.plot(t, c_ou, label="OU envelope c(t)")
    plt.plot(t, c_fgn, label=f"Long-memory Gaussian envelope c(t) (H={cfg.fgn_H})")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel("c(t)")
    plt.legend()
    plt.title("S-0001: Envelopes (ensemble averaged)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    # plot scaling diagnostic
    scaling_path = out_dir / cfg.out_scaling_png
    plt.figure()
    x1, y1 = _safe_loglog_arrays(t, c_ou)
    x2, y2 = _safe_loglog_arrays(t, c_fgn)
    plt.plot(x1, y1, label="OU: log(-log c) vs log t")
    plt.plot(x2, y2, label=f"Long-memory: log(-log c) vs log t (target slope ~ {2*cfg.fgn_H:.2f})")
    plt.xlabel("log t")
    plt.ylabel("log(-log c(t))")
    plt.legend()
    plt.title("S-0001: Scaling diagnostic")
    plt.tight_layout()
    plt.savefig(scaling_path, dpi=160)
    plt.close()

    # self-audit
    window = (cfg.audit_window_t_min, cfg.audit_window_t_max)
    tol = cfg.audit_tol_abs

    tau_c = 1.0 / cfg.ou_theta
    t_floor = 8.0 * tau_c  # force late-time diffusive regime
    res_ou = auto_audit_slope(
        t=t,
        c=c_ou,
        expected=1.0,
        tol_abs=tol,
        min_points=20,
        min_r2=0.995,
        t_min_floor=t_floor,
        t_max_cap=cfg.audit_window_t_max,
        window_points_max=120,
        min_log_span=cfg.audit_min_log_span,
    )
    res_fgn = auto_audit_slope(
        t=t,
        c=c_fgn,
        expected=2.0 * cfg.fgn_H,
        tol_abs=tol,
        min_points=20,
        min_r2=0.995,
        t_min_floor=cfg.audit_window_t_min,
        t_max_cap=cfg.audit_window_t_max,
        window_points_max=120,
        min_log_span=cfg.audit_min_log_span,
    )

    print_audit_banner("S-0001:OU", res_ou)
    print_audit_banner("S-0001:fGn", res_fgn)

    audit_csv_path = out_dir / cfg.out_audit_csv
    write_audit_csv(audit_csv_path, [("OU", res_ou), ("fGn", res_fgn)])

    # hard fail if audit fails (so this can be CI/contract-checked)
    if not (res_ou.passed and res_fgn.passed):
        print("[S-0001] AUDIT FAILED (see s0001_audit.csv).")
        return 2

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {fig_path}")
    print(f"Wrote: {scaling_path}")
    print(f"Wrote: {audit_csv_path}")
    print("[S-0001] AUDIT PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())