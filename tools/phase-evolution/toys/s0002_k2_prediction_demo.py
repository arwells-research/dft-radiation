#!/usr/bin/env python3
"""
S-0002 κ2 predictive-closure demo (measured envelope vs κ2-predicted envelope) — SELF-AUDITING

Goal (checkable):
- Simulate ω(t) ensembles (OU, fGn).
- Compute measured envelope:
    c_meas(t) = | < exp(i Δϕ(t)) >_ensemble |
  with Δϕ(t) = dt * sum ω0 and ω0 centered by center_method.
- Compute predicted envelope from empirical autocovariance R(k):
    κ2[j] = dt^2 * [ R0*(j+1) + 2*sum_{k=1..kmax} (j+1-k) Rk ]
    c_pred(t) = exp(-κ2(t)/2)
- Self-audit compares c_meas vs c_pred over a declared time window with declared tolerances.

B1 split (implemented):
- OU: single case (expected to FAIL unless structural truncation is introduced).
- fGn_RAW: fGn with NO taper (intended "closure" variant).
- fGn_TAPER: fGn with Bartlett taper (intended "estimator-stability" variant; may drift).

Correctness requirements enforced:
- Measured and predicted paths use identical centering.
- κ2 computed via discrete sum (no continuum integral).
- fGn lag support covers the audit window maximum j (prevents artificial late-time truncation drift).
- Audit drift metric uses robust log-binned median error and regresses median(|Δlog c|) vs log(t).

Artifacts written:
- toys/outputs/s0002_envelope_compare.csv (includes OU, fGn_RAW, fGn_TAPER series)
- toys/outputs/s0002_envelope_compare.png
- toys/outputs/s0002_error_diagnostic.png
- toys/outputs/s0002_audit.csv  (rows: OU, fGn_RAW, fGn_TAPER)

This is a toy existence + closure artifact, not a claim of fundamental dynamics.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any
import datetime
import subprocess

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 12345

    # ensemble / grid
    n_trajectories: int = 512
    dt: float = 0.01
    n_steps: int = 4000  # total time = n_steps * dt

    # OU params: dω = θ(μ-ω)dt + σ dW
    ou_theta: float = 0.6
    ou_mu: float = 1.0
    ou_sigma: float = 0.25

    # fGn params (long-memory Gaussian increments)
    fgn_H: float = 0.70
    fgn_mu: float = 1.0
    fgn_sigma: float = 0.25

    # κ2 prediction controls
    max_lag_steps: int = 3500  # hard cap
    autocov_method: str = "fft"

    burn_in_steps: int = 0
    center_method: str = "per_traj_time"  # "none" | "global_time" | "per_traj_time"

    # OU lag truncation: integrate autocov to this many correlation times
    ou_lag_tau_mult: float = 10.0

    # fGn: preferred lag fraction (but will not truncate below audit support requirement)
    fgn_lag_fraction: float = 0.75

    # taper switch for fGn_TAPER subcase only
    use_bartlett_taper: bool = True

    # audit window (time units)
    audit_window_t_min: float = 20.0
    audit_window_t_max: float = 40.0

    # audit tolerances
    audit_log_err_median_tol: float = 0.12
    audit_log_err_p95_tol: float = 0.25
    audit_drift_slope_tol: float = 0.06  # units: (|Δlog c|) per log(t)

    # outputs
    out_dir: str = "toys/outputs"
    out_csv: str = "s0002_envelope_compare.csv"
    out_png: str = "s0002_envelope_compare.png"
    out_err_png: str = "s0002_error_diagnostic.png"
    out_audit_csv: str = "s0002_audit.csv"


def _run_utc_iso() -> str:
    # timezone-aware, future-proof
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,   # suppress fatal message
        ).strip()
    except Exception:
        return "UNKNOWN"

# ---------------------------
# Processes
# ---------------------------

def simulate_ou_omega(cfg: DemoConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Discrete-time Euler-Maruyama simulation of OU:
        dω = θ(μ-ω)dt + σ dW
    Stationary variance: σ^2/(2θ).
    """
    nT, n = cfg.n_trajectories, cfg.n_steps
    dt = cfg.dt
    theta, mu, sigma = cfg.ou_theta, cfg.ou_mu, cfg.ou_sigma

    omega = np.empty((nT, n), dtype=float)
    omega[:, 0] = mu + (sigma / math.sqrt(2.0 * theta)) * rng.normal(size=nT)

    sqrt_dt = math.sqrt(dt)
    for t in range(1, n):
        dW = rng.normal(0.0, sqrt_dt, size=nT)
        omega[:, t] = omega[:, t - 1] + theta * (mu - omega[:, t - 1]) * dt + sigma * dW

    return omega


def fgn_davies_harte(n_steps: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """
    Davies–Harte fGn generator.
    Returns a length-n_steps sequence with ~unit standard deviation.
    """
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0,1)")

    k = np.arange(0, n_steps)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * H) + np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H))

    r = np.concatenate([gamma, gamma[-2:0:-1]])
    lam = np.fft.rfft(r).real
    # Numerical guard: small negatives due to FFT roundoff
    lam[lam < 0] = 0.0

    m = lam.shape[0]
    z = rng.normal(size=m) + 1j * rng.normal(size=m)
    x = np.fft.irfft(np.sqrt(lam) * z, n=r.shape[0])[:n_steps]

    std = float(np.std(x))
    if std <= 0:
        raise RuntimeError("fGn generation produced non-positive std.")
    return x / std


def simulate_fgn_omega(cfg: DemoConfig, rng: np.random.Generator) -> np.ndarray:
    nT, n = cfg.n_trajectories, cfg.n_steps
    mu, sigma, H = cfg.fgn_mu, cfg.fgn_sigma, cfg.fgn_H

    omega = np.empty((nT, n), dtype=float)
    for i in range(nT):
        inc = fgn_davies_harte(n, H, rng)
        omega[i, :] = mu + sigma * inc

    return omega


# ---------------------------
# Centering (must be consistent for meas & pred)
# ---------------------------

def center_omega(omega: np.ndarray, mu: float, method: str) -> np.ndarray:
    """
    Return omega0 used consistently for BOTH measured envelope and autocovariance estimation.

    method:
      - "none": omega0 = omega - mu
      - "global_time": subtract global mean of (omega - mu)
      - "per_traj_time": subtract per-trajectory time mean of (omega - mu)
    """
    x = omega - float(mu)

    if method == "none":
        return x
    if method == "global_time":
        return x - float(np.mean(x))
    if method == "per_traj_time":
        return x - np.mean(x, axis=1, keepdims=True)

    raise ValueError(f"Unknown center_method: {method}")


# ---------------------------
# Envelope and κ2 prediction
# ---------------------------

def envelope_measured(omega0: np.ndarray, dt: float) -> np.ndarray:
    delta_phi = np.cumsum(omega0, axis=1) * dt
    z = np.exp(1j * delta_phi)
    return np.abs(np.mean(z, axis=0))


def autocov_fft_per_traj(omega0: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Per-trajectory FFT autocovariance estimator, averaged over trajectories.
    omega0 is assumed already centered upstream.
    Returns R[k] for k=0..max_lag where R[k] ≈ E[omega0[t]*omega0[t+k]].
    """
    nT, n = omega0.shape
    if max_lag >= n:
        raise ValueError("max_lag must be < n_steps")

    nfft = 1
    while nfft < 2 * n:
        nfft *= 2

    R = np.zeros(max_lag + 1, dtype=float)
    denom = (n - np.arange(0, max_lag + 1)).astype(float)
    denom[denom <= 0] = np.nan

    for i in range(nT):
        x = omega0[i, :]
        X = np.fft.rfft(x, n=nfft)
        S = X * np.conj(X)
        ac = np.fft.irfft(S, n=nfft)[:n]
        R += (ac[: max_lag + 1] / denom)

    R /= float(nT)
    return np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)


def bartlett_taper(R: np.ndarray) -> np.ndarray:
    """
    Bartlett (triangular) taper: w[k] = 1 - k/K for k=0..K.
    Suppresses noisy long-lag tail without changing R[0].
    """
    R = np.asarray(R, dtype=float)
    K = len(R) - 1
    if K <= 0:
        return R
    w = 1.0 - (np.arange(0, K + 1, dtype=float) / float(K))
    return R * w


def k2_from_autocov_discrete(R: np.ndarray, dt: float, n_steps: int, max_lag_steps: int | None = None) -> np.ndarray:
    """
    Discrete κ2(t) for Δφ(t) = dt * sum_{m=0}^{j} ω0[m], assuming stationary autocov R[k] = E[ω0[0] ω0[k]].

    κ2[j] = Var(Δφ_j) = dt^2 * [ R[0]*(j+1) + 2*sum_{k=1..j} (j+1-k)*R[k] ].
    """
    R = np.asarray(R, dtype=float)
    if max_lag_steps is None:
        max_lag_steps = len(R) - 1
    max_lag_steps = int(min(max_lag_steps, len(R) - 1))

    k2 = np.zeros(n_steps, dtype=float)
    for j in range(n_steps):
        kmax = min(j, max_lag_steps)
        acc = R[0] * (j + 1)
        if kmax >= 1:
            ks = np.arange(1, kmax + 1)
            acc += 2.0 * np.sum((j + 1 - ks) * R[ks])
        k2[j] = (dt * dt) * acc

    return k2


def envelope_predicted_from_kappa2(kappa2: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * kappa2)


# ---------------------------
# Audit (robust log-binned median drift)
# ---------------------------

def audit_predictive_closure(
    t: np.ndarray,
    c_meas: np.ndarray,
    c_pred: np.ndarray,
    t_min: float,
    t_max: float,
    med_tol: float,
    p95_tol: float,
    drift_slope_tol: float,
) -> Dict[str, Any]:
    t = np.asarray(t)
    c_meas = np.asarray(c_meas)
    c_pred = np.asarray(c_pred)

    eps = 1e-12
    c_meas2 = np.clip(c_meas, eps, 1.0 - eps)
    c_pred2 = np.clip(c_pred, eps, 1.0 - eps)

    mask = (t >= t_min) & (t <= t_max)
    if np.count_nonzero(mask) < 12:
        raise ValueError("Insufficient points in audit window.")

    tw = t[mask]
    lt = np.log(tw)

    # pointwise log-envelope error
    le = np.abs(np.log(c_meas2[mask]) - np.log(c_pred2[mask]))

    med = float(np.median(le))
    p95 = float(np.percentile(le, 95))

    # Robust drift: bin in log(t), take median(le) per bin, regress median(le) vs log(t)
    n_bins = 25
    lt_min = float(np.min(lt))
    lt_max = float(np.max(lt))
    if lt_max <= lt_min:
        raise ValueError("Invalid audit window (non-increasing log-time).")

    edges = np.linspace(lt_min, lt_max, n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    x = []
    y = []
    for i in range(n_bins):
        m = (lt >= edges[i]) & (lt < edges[i + 1] if i < n_bins - 1 else lt <= edges[i + 1])
        if np.count_nonzero(m) < 8:
            continue
        x.append(float(bin_centers[i]))
        y.append(float(np.median(le[m])))

    if len(y) < 6:
        raise ValueError("Insufficient populated bins for drift diagnostic.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    A = np.vstack([x, np.ones_like(x)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    passed = (med <= med_tol) and (p95 <= p95_tol) and (abs(slope) <= drift_slope_tol)

    return {
        "n_points": int(len(le)),
        "t_min": float(t_min),
        "t_max": float(t_max),
        "log_err_median": med,
        "log_err_p95": p95,
        "drift_slope": slope,          # units: (|Δlog c|) per log(t)
        "drift_intercept": intercept,
        "drift_bins_used": int(len(y)),
        "pass": bool(passed),
    }


def print_audit_banner(label: str, res: Dict[str, Any], cfg: DemoConfig) -> None:
    status = "PASS" if res["pass"] else "FAIL"
    bins = res.get("drift_bins_used", None)
    bins_str = f", bins={bins}" if bins is not None else ""
    print(
        f"[{label}] median|Δlog c|={res['log_err_median']:.3f} (tol {cfg.audit_log_err_median_tol:.3f}), "
        f"p95|Δlog c|={res['log_err_p95']:.3f} (tol {cfg.audit_log_err_p95_tol:.3f}), "
        f"drift_slope={res['drift_slope']:.3f} (tol {cfg.audit_drift_slope_tol:.3f}){bins_str}, "
        f"window [{res['t_min']:g},{res['t_max']:g}], n={res['n_points']} [{status}]"
    )


def write_audit_csv(path: Path, rows: list[tuple[str, Dict[str, Any]]]) -> None:
    """
    Backward-compatible audit CSV writer: writes optional columns only if present.
    """
    has_bins = any("drift_bins_used" in r for _, r in rows)

    header = [
        "case",
        "pass",
        "t_min",
        "t_max",
        "n_points",
        "log_err_median",
        "log_err_p95",
        "drift_slope",
        "drift_intercept",
    ]
    if has_bins:
        header.append("drift_bins_used")

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for name, r in rows:
            row = [
                name,
                int(bool(r.get("pass", False))),
                f"{r.get('t_min', float('nan')):.10g}",
                f"{r.get('t_max', float('nan')):.10g}",
                int(r.get("n_points", 0)),
                f"{r.get('log_err_median', float('nan')):.10g}",
                f"{r.get('log_err_p95', float('nan')):.10g}",
                f"{r.get('drift_slope', float('nan')):.10g}",
                f"{r.get('drift_intercept', float('nan')):.10g}",
            ]
            if has_bins:
                v = r.get("drift_bins_used", "")
                row.append(v if v == "" else int(v))
            w.writerow(row)


# ---------------------------
# Case runner (ensures fGn lag support covers audit window)
# ---------------------------

def _required_lag_for_audit(cfg: DemoConfig) -> int:
    # t = (j+1)dt -> j ≈ t/dt - 1
    j_max = int(math.ceil(float(cfg.audit_window_t_max) / float(cfg.dt))) - 1
    return max(1, min(j_max, cfg.n_steps - 1))


def _choose_max_lag_for_case(case_name: str, cfg: DemoConfig) -> int:
    hard_cap = int(min(cfg.max_lag_steps, cfg.n_steps - 1))
    req_audit = _required_lag_for_audit(cfg)

    if case_name.upper() == "OU":
        tau_c = 1.0 / float(cfg.ou_theta)
        max_lag_ou = int(round((float(cfg.ou_lag_tau_mult) * tau_c) / float(cfg.dt)))
        return min(hard_cap, max(1, max_lag_ou))

    # fGn: prefer fraction, but do not truncate below audit support
    preferred = int(float(cfg.fgn_lag_fraction) * float(cfg.n_steps - 1))
    preferred = max(1, min(preferred, cfg.n_steps - 1))
    return min(hard_cap, max(preferred, req_audit))


def run_case(case_name: str, omega: np.ndarray, mu: float, cfg: DemoConfig, *, taper: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Returns (t, c_meas, c_pred, k2, max_lag_used)
    taper applies only to fGn variants (ignored for OU).
    """
    omega0 = center_omega(omega, mu=mu, method=cfg.center_method)
    c_meas = envelope_measured(omega0, cfg.dt)

    max_lag = _choose_max_lag_for_case(case_name, cfg)

    R = autocov_fft_per_traj(omega0, max_lag=max_lag)

    if case_name.lower() == "fgn" and taper and bool(cfg.use_bartlett_taper):
        R = bartlett_taper(R)

    k2 = k2_from_autocov_discrete(R, dt=float(cfg.dt), n_steps=int(cfg.n_steps), max_lag_steps=int(max_lag))
    c_pred = envelope_predicted_from_kappa2(k2)

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)
    return t, c_meas, c_pred, k2, int(max_lag)


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    cfg = DemoConfig()
    run_utc = _run_utc_iso()
    git_commit = _git_commit_short()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    # OU
    omega_ou = simulate_ou_omega(cfg, rng)
    t, c_ou_meas, c_ou_pred, k2_ou, ou_max_lag = run_case("OU", omega_ou, cfg.ou_mu, cfg, taper=False)

    # fGn (one simulation; evaluate two prediction modes against the SAME measured envelope)
    omega_fgn = simulate_fgn_omega(cfg, rng)

    # RAW (no taper): intended closure
    _, c_fgn_meas_raw, c_fgn_pred_raw, k2_fgn_raw, fgn_max_lag_raw = run_case("fGn", omega_fgn, cfg.fgn_mu, cfg, taper=False)

    # TAPER (Bartlett): intended stability variant
    _, c_fgn_meas_tap, c_fgn_pred_tap, k2_fgn_tap, fgn_max_lag_tap = run_case("fGn", omega_fgn, cfg.fgn_mu, cfg, taper=True)

    # Sanity: measured envelope must be identical (same omega0 path)
    if not np.allclose(c_fgn_meas_raw, c_fgn_meas_tap, rtol=0, atol=0):
        raise RuntimeError("Internal error: fGn measured envelopes differ between RAW and TAPER paths.")

    c_fgn_meas = c_fgn_meas_raw

    # Write CSV
    csv_path = out_dir / cfg.out_csv
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "c_ou_meas",
            "c_ou_pred",
            "k2_ou",
            "c_fgn_meas",
            "c_fgn_pred_raw",
            "k2_fgn_raw",
            "c_fgn_pred_taper",
            "k2_fgn_taper",
            "dt",
            "n_steps",
            "n_trajectories",
            "seed",
            "center_method",
            "audit_t_min",
            "audit_t_max",
            "ou_theta",
            "ou_mu",
            "ou_sigma",
            "ou_max_lag_used",
            "fgn_H",
            "fgn_mu",
            "fgn_sigma",
            "fgn_max_lag_used_raw",
            "fgn_max_lag_used_taper",
            "max_lag_steps_cap",
            "fgn_lag_fraction",
            "use_bartlett_taper",
            "ou_lag_tau_mult",
            "run_utc",
            "git_commit",
        ])
        for i in range(len(t)):
            w.writerow([
                f"{t[i]:.10g}",
                f"{c_ou_meas[i]:.10g}",
                f"{c_ou_pred[i]:.10g}",
                f"{k2_ou[i]:.10g}",
                f"{c_fgn_meas[i]:.10g}",
                f"{c_fgn_pred_raw[i]:.10g}",
                f"{k2_fgn_raw[i]:.10g}",
                f"{c_fgn_pred_tap[i]:.10g}",
                f"{k2_fgn_tap[i]:.10g}",
                f"{cfg.dt:.10g}",
                cfg.n_steps,
                cfg.n_trajectories,
                cfg.seed,
                cfg.center_method,
                f"{cfg.audit_window_t_min:.10g}",
                f"{cfg.audit_window_t_max:.10g}",
                f"{cfg.ou_theta:.10g}",
                f"{cfg.ou_mu:.10g}",
                f"{cfg.ou_sigma:.10g}",
                ou_max_lag,
                f"{cfg.fgn_H:.10g}",
                f"{cfg.fgn_mu:.10g}",
                f"{cfg.fgn_sigma:.10g}",
                fgn_max_lag_raw,
                fgn_max_lag_tap,
                cfg.max_lag_steps,
                f"{cfg.fgn_lag_fraction:.10g}",
                int(bool(cfg.use_bartlett_taper)),
                f"{cfg.ou_lag_tau_mult:.10g}",
                run_utc,
                git_commit,
            ])

    # Plot envelopes
    fig_path = out_dir / cfg.out_png
    plt.figure()
    plt.plot(t, c_ou_meas, label="OU: c_meas(t)")
    plt.plot(t, c_ou_pred, "--", label="OU: c_pred(t) from κ2")
    plt.plot(t, c_fgn_meas, label=f"fGn(H={cfg.fgn_H}): c_meas(t)")
    plt.plot(t, c_fgn_pred_raw, "--", label="fGn_RAW: c_pred(t) from κ2 (no taper)")
    plt.plot(t, c_fgn_pred_tap, "--", label="fGn_TAPER: c_pred(t) from κ2 (Bartlett)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel("c(t)")
    plt.title("S-0002: Predictive closure (measured vs κ2-predicted envelope)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    # Plot error diagnostics
    err_png = out_dir / cfg.out_err_png
    eps = 1e-12
    ou_err = np.abs(np.log(np.clip(c_ou_meas, eps, 1.0 - eps)) - np.log(np.clip(c_ou_pred, eps, 1.0 - eps)))
    fgn_err_raw = np.abs(np.log(np.clip(c_fgn_meas, eps, 1.0 - eps)) - np.log(np.clip(c_fgn_pred_raw, eps, 1.0 - eps)))
    fgn_err_tap = np.abs(np.log(np.clip(c_fgn_meas, eps, 1.0 - eps)) - np.log(np.clip(c_fgn_pred_tap, eps, 1.0 - eps)))

    plt.figure()
    plt.plot(t, ou_err, label="OU: |Δ log c(t)|")
    plt.plot(t, fgn_err_raw, label="fGn_RAW: |Δ log c(t)|")
    plt.plot(t, fgn_err_tap, label="fGn_TAPER: |Δ log c(t)|")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel("|Δ log c(t)|")
    plt.title("S-0002: Error diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(err_png, dpi=160)
    plt.close()

    # Audit
    audit_ou = audit_predictive_closure(
        t=t,
        c_meas=c_ou_meas,
        c_pred=c_ou_pred,
        t_min=cfg.audit_window_t_min,
        t_max=cfg.audit_window_t_max,
        med_tol=cfg.audit_log_err_median_tol,
        p95_tol=cfg.audit_log_err_p95_tol,
        drift_slope_tol=cfg.audit_drift_slope_tol,
    )
    audit_fgn_raw = audit_predictive_closure(
        t=t,
        c_meas=c_fgn_meas,
        c_pred=c_fgn_pred_raw,
        t_min=cfg.audit_window_t_min,
        t_max=cfg.audit_window_t_max,
        med_tol=cfg.audit_log_err_median_tol,
        p95_tol=cfg.audit_log_err_p95_tol,
        drift_slope_tol=cfg.audit_drift_slope_tol,
    )
    audit_fgn_tap = audit_predictive_closure(
        t=t,
        c_meas=c_fgn_meas,
        c_pred=c_fgn_pred_tap,
        t_min=cfg.audit_window_t_min,
        t_max=cfg.audit_window_t_max,
        med_tol=cfg.audit_log_err_median_tol,
        p95_tol=cfg.audit_log_err_p95_tol,
        drift_slope_tol=cfg.audit_drift_slope_tol,
    )

    print_audit_banner("S-0002:OU", audit_ou, cfg)
    print_audit_banner("S-0002:fGn_RAW", audit_fgn_raw, cfg)
    print_audit_banner("S-0002:fGn_TAPER", audit_fgn_tap, cfg)

    audit_csv = out_dir / cfg.out_audit_csv
    write_audit_csv(audit_csv, [("OU", audit_ou), ("fGn_RAW", audit_fgn_raw), ("fGn_TAPER", audit_fgn_tap)])

    # PASS conditions for this combined artifact:
    # - OU is allowed (and expected) to fail: this is the structural diagnostic.
    # - fGn_RAW must pass for "closure" to be certified.
    # - fGn_TAPER is recorded; it may pass or fail depending on taper-induced bias/drift.
    if not audit_fgn_raw["pass"]:
        print("[S-0002] AUDIT FAILED (fGn_RAW did not pass). See s0002_audit.csv.")
        return 2

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {fig_path}")
    print(f"Wrote: {err_png}")
    print(f"Wrote: {audit_csv}")
    print("[S-0002] AUDIT PASSED (fGn_RAW). OU/fGn_TAPER status recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())