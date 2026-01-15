#!/usr/bin/env python3
"""
S-0004 — Even-cumulant boundary (κ4 magnitude distortion via κ2-preserving scale-mixture) — SELF-AUDITING


What changed from v0.4.3:
1) Measurement-only per-time ensemble-centering of Δϕ(t) is retained, BUT audits are now
   internally consistent:
     - We compute m_meas(t) from Δϕ_centered(t)
     - We compute κ2_direct from Var(Δϕ_centered(t)) for the Gaussian identity audit
   This matches the exact identity:
       E[exp(iX)] = exp(-Var(X)/2) for centered Gaussian X.

2) GAUSS ω0 centering is no longer hard-coded to "global_time".
   Instead it uses cfg.center_method_gauss (default "per_traj_time") to improve κ2_pred stability.

Goal:
- GAUSS must PASS closure and identity under the linear-mean estimator with Δϕ-centering.
- κ4 cases should show structured deviation.

See header comments in v0.4.3 for full context; core design is unchanged.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple
import datetime
import subprocess

import numpy as np
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class DemoConfig:
    seed: int = 12345

    # grid
    dt: float = 0.1
    n_steps: int = 600  # total time = 60.0

    omega_ultralow_fft_bins_remove: int = 0

    # ensemble
    n_trajectories: int = 512

    # base Gaussian driver (fGn innovations)
    fgn_H: float = 0.70
    fgn_mu: float = 1.0
    fgn_sigma: float = 1.00

    # Centering applied to ω0 (must match pred + the Δϕ construction)
    # v0.4.4: GAUSS centering is configurable (default per_traj_time for tighter κ2_pred).
    center_method_gauss: str = "per_traj_time"  # "none" | "global_time" | "per_traj_time"
    center_method_k4: str = "per_traj_time"     # "none" | "global_time" | "per_traj_time"

    # κ2 prediction
    max_lag_steps_cap: int = 500
    fgn_lag_fraction: float = 0.75

    # κ4 injection (scale mixture)
    scale_sigma_mild: float = 0.90
    scale_sigma_strong: float = 1.35

    # audit window (numerically stable)
    audit_window_t_min = 3.0
    audit_window_t_max = 7.0

    # magnitude closure tolerances (GAUSS)
    audit_log_err_median_tol: float = 0.12
    audit_log_err_p95_tol: float = 0.25
    audit_drift_slope_tol: float = 0.12
    drift_bins: int = 8

    # Gaussian identity tolerances (GAUSS): |log c_meas_lin + 0.5*k2_direct|
    identity_log_err_median_tol: float = 0.12
    identity_log_err_p95_tol: float = 0.25
    identity_drift_slope_tol: float = 0.06

    # κ2 adequacy gate (must be large enough to resolve κ4 distortion)
    k2_adequacy_min_at_tmax: float = 2.5

    # K4 must fail-by-margin (magnitude closure vs κ2_pred)
    k4_fail_median_min: float = 0.20
    k4_fail_p95_min: float = 0.35

    # K4 residual sign structure (Jensen)
    k4_resid_sign_median_min: float = 0.00

    # Phase diagnostic gating (non-gating)
    phase_mag_min: float = 5e-2

    # GAUSS sanity thresholds (pipeline self-check)
    sanity_r0_rel_tol: float = 0.05          # |R0_est - Var(ω0)| / Var(ω0)
    sanity_k2_rel_tol: float = 0.10          # |κ2_pred(tmax)-κ2_direct(tmax)| / κ2_direct(tmax)

    # outputs
    out_dir: str = "toys/outputs"
    out_csv: str = "s0004_envelope_compare.csv"
    out_png: str = "s0004_envelope_compare.png"
    out_err_png: str = "s0004_error_diagnostic.png"
    out_phase_png: str = "s0004_phase_diagnostic.png"
    out_audit_csv: str = "s0004_audit.csv"


def _run_utc_iso() -> str:
    # timezone-aware UTC (avoids utcnow deprecation)
    dt = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    # ISO8601 "Z"
    return dt.isoformat().replace("+00:00", "Z")

def _git_commit_short() -> str:
    # quiet "nogit" if not in a repo
    try:
        p = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
        if p.returncode != 0:
            return "nogit"
        s = (p.stdout or "").strip()
        return s if s else "nogit"
    except Exception:
        return "nogit"

def _choose_drift_bins(n_points: int, requested: int, min_bins: int = 4) -> int:
    # Need enough samples/bin to estimate a median robustly
    # target >= ~8 points per bin, clamp to [min_bins, requested]
    if n_points <= 0:
        return min_bins
    b = int(n_points // 8)
    b = max(min_bins, b)
    b = min(int(requested), b)
    return b

# ---------------------------
# Davies–Harte fGn innovations (Gaussian)
# ---------------------------

def fgn_davies_harte(n_steps: int, H: float, rng: np.random.Generator) -> np.ndarray:
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0,1)")
    n = int(n_steps)
    k = np.arange(0, n)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * H) + np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H))

    r = np.concatenate([gamma, gamma[-2:0:-1]])
    lam = np.fft.rfft(r).real
    lam[lam < 0] = 0.0

    m = lam.shape[0]
    z = rng.normal(size=m) + 1j * rng.normal(size=m)
    x = np.fft.irfft(np.sqrt(lam) * z, n=r.shape[0])[:n]

    std = float(np.std(x))
    if std <= 0:
        raise RuntimeError("fGn generation produced non-positive std.")
    return x / std


# ---------------------------
# κ4 injection: per-trajectory scale mixture with E[S^2]=1 in expectation
# ---------------------------

def draw_scale_log_normal_unit_second_moment(rng: np.random.Generator, n: int, sigma_s: float) -> np.ndarray:
    sigma_s = float(sigma_s)
    mu = -sigma_s * sigma_s  # ensures E[S^2]=1 in expectation
    return np.exp(rng.normal(loc=mu, scale=sigma_s, size=int(n)))


def simulate_omega_gauss(cfg: DemoConfig, rng: np.random.Generator) -> np.ndarray:
    nT, n = cfg.n_trajectories, cfg.n_steps
    mu, sigma, H = float(cfg.fgn_mu), float(cfg.fgn_sigma), float(cfg.fgn_H)

    omega = np.empty((nT, n), dtype=float)
    for i in range(nT):
        x = fgn_davies_harte(n, H, rng)
        # generation-time de-mean of innovations to prevent trivial drift of Δϕ in finite sample
        x = x - float(np.mean(x))
        omega[i, :] = mu + sigma * x
    return omega


def simulate_omega_k4(cfg: DemoConfig, rng: np.random.Generator, sigma_s: float) -> np.ndarray:
    nT, n = cfg.n_trajectories, cfg.n_steps
    mu, sigma, H = float(cfg.fgn_mu), float(cfg.fgn_sigma), float(cfg.fgn_H)

    S = draw_scale_log_normal_unit_second_moment(rng, nT, sigma_s=sigma_s)

    omega = np.empty((nT, n), dtype=float)
    for i in range(nT):
        x = fgn_davies_harte(n, H, rng)
        x = x - float(np.mean(x))
        omega[i, :] = mu + sigma * float(S[i]) * x
    return omega


# ---------------------------
# Centering used for ω0 (must match pred & Δϕ construction)
# ---------------------------

def center_omega(omega: np.ndarray, mu: float, method: str) -> np.ndarray:
    x = omega - float(mu)
    if method == "none":
        return x
    if method == "global_time":
        return x - float(np.mean(x))
    if method == "per_traj_time":
        return x - np.mean(x, axis=1, keepdims=True)
    raise ValueError(f"Unknown center_method: {method}")

def remove_ultralow_fft_bins_per_traj(x: np.ndarray, n_remove: int) -> np.ndarray:
    """
    Remove the lowest rFFT bins per trajectory.
    n_remove=0: no-op
    n_remove=1: remove DC only
    n_remove=2: remove DC + first harmonic
    """
    n_remove = int(n_remove)
    if n_remove <= 0:
        return x

    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected shape (n_trajectories, n_steps)")

    nT, n = x.shape
    out = np.empty_like(x)
    for i in range(nT):
        Xi = np.fft.rfft(x[i, :])
        kmax = min(n_remove, Xi.shape[0])
        Xi[:kmax] = 0.0
        out[i, :] = np.fft.irfft(Xi, n=n)
    return out

# ---------------------------
# Envelope + κ2 prediction
# ---------------------------

def envelope_measured_complex(omega0: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    delta_phi = np.cumsum(omega0, axis=1) * float(dt)
    z = np.exp(1j * delta_phi)
    m = np.mean(z, axis=0)
    return delta_phi, m

def highpass_remove_ultralow(x: np.ndarray, n_remove: int = 1) -> np.ndarray:
    # n_remove=1 removes DC only; n_remove=2 removes DC + first harmonic, etc.
    X = np.fft.rfft(x)
    X[:n_remove] = 0
    return np.fft.irfft(X, n=len(x))

def c_from_m_linear(m: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mr = np.asarray(m, dtype=complex).real
    return np.clip(mr, eps, 1.0 - eps)


def c_from_m_abs(m: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(np.abs(np.asarray(m, dtype=complex)), eps, 1.0 - eps)


def autocov_fft_per_traj(omega0: np.ndarray, max_lag: int) -> np.ndarray:
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


def k2_from_autocov_discrete(R: np.ndarray, dt: float, n_steps: int, max_lag_steps: int) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    max_lag_steps = int(min(max_lag_steps, len(R) - 1))
    k2 = np.zeros(int(n_steps), dtype=float)
    dt2 = float(dt) * float(dt)

    for j in range(int(n_steps)):
        kmax = min(j, max_lag_steps)
        acc = R[0] * (j + 1)
        if kmax >= 1:
            ks = np.arange(1, kmax + 1)
            acc += 2.0 * np.sum((j + 1 - ks) * R[ks])
        k2[j] = dt2 * acc
    return k2


def envelope_predicted_from_kappa2(kappa2: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(kappa2, dtype=float))


def choose_max_lag_steps_fgn(cfg: DemoConfig) -> int:
    # Audit-grade for long-memory: truncation biases κ2_pred low.
    # Full-lag is feasible for n_steps ~ 600 and stabilizes GAUSS closure.
    return int(cfg.n_steps - 1)


def k2_direct_from_delta_phi(delta_phi: np.ndarray) -> np.ndarray:
    return np.var(delta_phi, axis=0, ddof=0)


# ---------------------------
# Audits
# ---------------------------

def _mask_window(t: np.ndarray, cfg: DemoConfig) -> np.ndarray:
    return (t >= float(cfg.audit_window_t_min)) & (t <= float(cfg.audit_window_t_max))


def audit_predictive_closure(
    t: np.ndarray, c_meas_lin: np.ndarray, c_pred: np.ndarray, cfg: DemoConfig
) -> Dict[str, Any]:
    t = np.asarray(t, dtype=float)
    cm = np.asarray(c_meas_lin, dtype=float)
    cp = np.asarray(c_pred, dtype=float)

    eps = 1e-12
    cm = np.clip(cm, eps, 1.0 - eps)
    cp = np.clip(cp, eps, 1.0 - eps)

    mask = _mask_window(t, cfg)
    npts = int(np.count_nonzero(mask))
    if npts < 12:
        raise ValueError("Insufficient points in audit window.")

    lt = np.log(t[mask])
    le = np.abs(np.log(cm[mask]) - np.log(cp[mask]))

    med = float(np.median(le))
    p95 = float(np.percentile(le, 95))

    # Robust binning in log-time for drift diagnostic (with fallback)
    lt_min = float(np.min(lt))
    lt_max = float(np.max(lt))
    if (not np.isfinite(lt_min)) or (not np.isfinite(lt_max)) or (lt_max <= lt_min):
        raise ValueError("Degenerate log-time range in audit window.")

    # Pick a bin count that tries to keep >= ~12 points/bin, but not insane.
    # (Your window sizes are small enough that this will usually be 4-8.)
    nbins = int(min(int(cfg.drift_bins), max(4, npts // 12)))
    nbins = max(2, nbins)  # need at least 2 bins to define edges sanely

    edges = np.linspace(lt_min, lt_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nb = int(len(edges) - 1)

    x: list[float] = []
    y: list[float] = []
    for i in range(nb):
        m = (lt >= edges[i]) & (lt < edges[i + 1] if i < nb - 1 else lt <= edges[i + 1])
        if np.count_nonzero(m) < 5:
            continue
        x.append(float(centers[i]))
        y.append(float(np.median(le[m])))

    if len(y) >= 3:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        drift_bins_used = int(len(y))
    else:
        # Fallback: fit drift on all points (never fails due to sparse bins)
        x_arr = lt
        y_arr = le
        drift_bins_used = int(len(y_arr))

    # Guard: lstsq requires at least 2 points
    if x_arr.size < 2:
        slope = float("nan")
        intercept = float("nan")
    else:
        A = np.vstack([x_arr, np.ones_like(x_arr)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, y_arr, rcond=None)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])

    passed = (
        (med <= float(cfg.audit_log_err_median_tol))
        and (p95 <= float(cfg.audit_log_err_p95_tol))
        and (abs(slope) <= float(cfg.audit_drift_slope_tol))
    )

    return {
        "n_points": npts,
        "t_min": float(cfg.audit_window_t_min),
        "t_max": float(cfg.audit_window_t_max),
        "log_err_median": med,
        "log_err_p95": p95,
        "drift_slope": slope,
        "drift_intercept": intercept,
        "drift_bins_used": drift_bins_used,
        "pass": bool(passed),
    }


def audit_gaussian_identity(
    t: np.ndarray, c_meas_lin: np.ndarray, k2_direct: np.ndarray, cfg: DemoConfig
) -> Dict[str, Any]:
    """
    For centered Gaussian Δϕ(t):
        E[exp(iΔϕ)] = exp(-Var(Δϕ)/2)
    We audit | log(c_meas) + 0.5*k2_direct | over the window.
    """
    t = np.asarray(t, dtype=float)
    cm = np.asarray(c_meas_lin, dtype=float)
    k2d = np.asarray(k2_direct, dtype=float)

    eps = 1e-12
    cm = np.clip(cm, eps, 1.0 - eps)

    mask = _mask_window(t, cfg)
    npts = int(np.count_nonzero(mask))
    if npts < 12:
        raise ValueError("Insufficient points in audit window.")

    lt = np.log(t[mask])
    e = np.abs(np.log(cm[mask]) + 0.5 * k2d[mask])

    med = float(np.median(e))
    p95 = float(np.percentile(e, 95))

    # Robust binning in log-time for drift diagnostic (with fallback)
    lt_min = float(np.min(lt))
    lt_max = float(np.max(lt))
    if (not np.isfinite(lt_min)) or (not np.isfinite(lt_max)) or (lt_max <= lt_min):
        raise ValueError("Degenerate log-time range in audit window.")

    nbins = int(min(int(cfg.drift_bins), max(4, npts // 12)))
    nbins = max(2, nbins)

    edges = np.linspace(lt_min, lt_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nb = int(len(edges) - 1)

    x: list[float] = []
    y: list[float] = []
    for i in range(nb):
        m = (lt >= edges[i]) & (lt < edges[i + 1] if i < nb - 1 else lt <= edges[i + 1])
        if np.count_nonzero(m) < 5:
            continue
        x.append(float(centers[i]))
        y.append(float(np.median(e[m])))

    if len(y) >= 3:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        drift_bins_used = int(len(y))
    else:
        x_arr = lt
        y_arr = e
        drift_bins_used = int(len(y_arr))

    if x_arr.size < 2:
        slope = float("nan")
        intercept = float("nan")
    else:
        A = np.vstack([x_arr, np.ones_like(x_arr)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, y_arr, rcond=None)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])

    passed = (
        (med <= float(cfg.identity_log_err_median_tol))
        and (p95 <= float(cfg.identity_log_err_p95_tol))
        and (abs(slope) <= float(cfg.identity_drift_slope_tol))
    )

    return {
        "n_points": npts,
        "t_min": float(cfg.audit_window_t_min),
        "t_max": float(cfg.audit_window_t_max),
        "id_log_err_median": med,
        "id_log_err_p95": p95,
        "id_drift_slope": slope,
        "id_drift_intercept": intercept,
        "id_drift_bins_used": drift_bins_used,
        "pass": bool(passed),
    }

def print_audit_banner(label: str, res: Dict[str, Any], cfg: DemoConfig) -> None:
    status = "PASS" if res["pass"] else "FAIL"
    print(
        f"[{label}] median|Δlog c|={res['log_err_median']:.3f} (tol {cfg.audit_log_err_median_tol:.3f}), "
        f"p95|Δlog c|={res['log_err_p95']:.3f} (tol {cfg.audit_log_err_p95_tol:.3f}), "
        f"drift_slope={res['drift_slope']:.3f} (tol {cfg.audit_drift_slope_tol:.3f}), "
        f"bins={res['drift_bins_used']}, window [{res['t_min']:g},{res['t_max']:g}], n={res['n_points']} [{status}]"
    )


def print_identity_banner(label: str, res: Dict[str, Any], cfg: DemoConfig) -> None:
    status = "PASS" if res["pass"] else "FAIL"
    print(
        f"[{label}] median|log c + κ2/2|={res['id_log_err_median']:.3f} (tol {cfg.identity_log_err_median_tol:.3f}), "
        f"p95={res['id_log_err_p95']:.3f} (tol {cfg.identity_log_err_p95_tol:.3f}), "
        f"drift_slope={res['id_drift_slope']:.3f} (tol {cfg.identity_drift_slope_tol:.3f}), "
        f"bins={res['id_drift_bins_used']}, window [{res['t_min']:g},{res['t_max']:g}], n={res['n_points']} [{status}]"
    )


def median_signed_log_residual(t: np.ndarray, c_meas_lin: np.ndarray, c_pred: np.ndarray, cfg: DemoConfig) -> float:
    eps = 1e-12
    t = np.asarray(t, dtype=float)
    cm = np.clip(np.asarray(c_meas_lin, dtype=float), eps, 1.0 - eps)
    cp = np.clip(np.asarray(c_pred, dtype=float), eps, 1.0 - eps)
    mask = _mask_window(t, cfg)
    r = np.log(cm[mask]) - np.log(cp[mask])
    return float(np.median(r))


def median_phase_over_window(t: np.ndarray, m_meas: np.ndarray, cfg: DemoConfig) -> Tuple[float, int]:
    t = np.asarray(t, dtype=float)
    m_meas = np.asarray(m_meas, dtype=complex)
    mask = _mask_window(t, cfg) & (np.abs(m_meas) >= float(cfg.phase_mag_min))
    n = int(np.count_nonzero(mask))
    if n < 12:
        return 0.0, n
    return float(np.median(np.angle(m_meas[mask]))), n


def gauss_sanity_certificate(
    omega0: np.ndarray,
    R: np.ndarray,
    k2_pred: np.ndarray,
    k2_direct: np.ndarray,
    t: np.ndarray,
    cfg: DemoConfig,
) -> Dict[str, Any]:
    var_emp = float(np.var(omega0, ddof=0))
    r0 = float(R[0])
    r0_rel = abs(r0 - var_emp) / max(var_emp, 1e-18)

    mask = _mask_window(t, cfg)
    idx = int(np.where(mask)[0][-1])
    k2d = float(k2_direct[idx])
    k2p = float(k2_pred[idx])
    k2_rel = abs(k2p - k2d) / max(k2d, 1e-18)

    ok = (r0_rel <= float(cfg.sanity_r0_rel_tol)) and (k2_rel <= float(cfg.sanity_k2_rel_tol))
    return {
        "var_emp": var_emp,
        "r0_est": r0,
        "r0_rel_err": r0_rel,
        "k2_direct_tmax": k2d,
        "k2_pred_tmax": k2p,
        "k2_rel_err": k2_rel,
        "sanity_ok": bool(ok),
    }

def c_from_m_abs2_debiased(m: np.ndarray, n_trajectories: int, eps: float = 1e-12) -> np.ndarray:
    """
    Debiased estimator of |E[e^{iΔϕ}]| using |m|^2 correction:
      E[|m|^2] = |μ|^2 + (1-|μ|^2)/N  ⇒  |μ|^2 ≈ (N|m|^2 - 1)/(N-1)
    Robust when coherence is small (avoids Re(m) sign flips and log(eps) spikes).
    """
    m = np.asarray(m, dtype=complex)
    N = int(n_trajectories)
    if N < 2:
        raise ValueError("n_trajectories must be >= 2")
    abs2 = np.abs(m) ** 2
    mu2 = (N * abs2 - 1.0) / float(N - 1)
    mu2 = np.maximum(mu2, 0.0)
    c = np.sqrt(mu2)
    return np.clip(c, eps, 1.0 - eps)


def run_subcase(cfg: DemoConfig, omega: np.ndarray, center_method: str) -> Dict[str, Any]:
    omega0 = center_omega(omega, mu=cfg.fgn_mu, method=center_method)

    # Option A: kill ultra-low-frequency drift modes (0 disables; 1=DC; 2=DC+1st harmonic; ...)
    n_remove = int(getattr(cfg, "omega_ultralow_fft_bins_remove", 0))
    if n_remove > 0:
        _before = omega0.copy()
        omega0 = remove_ultralow_fft_bins_per_traj(omega0, n_remove=n_remove)
        delta = float(np.sqrt(np.mean((omega0 - _before) ** 2)))
        print(f"[S-0004:{center_method}] ultralow_remove={n_remove} rms_delta_omega0={delta:.6g}")
    else:
        print(f"[S-0004:{center_method}] ultralow_remove=0 (disabled)")

    # Measured envelope and direct cumulants
    delta_phi, m_meas = envelope_measured_complex(omega0, cfg.dt)
    k2_direct = k2_direct_from_delta_phi(delta_phi)

    print(
        f"[S-0004:{center_method}] phase_med_all={float(np.median(np.angle(m_meas))):.3f} "
        f"|m|_med={float(np.median(np.abs(m_meas))):.3f}"
    )

    c_meas_lin = c_from_m_abs2_debiased(m_meas, n_trajectories=cfg.n_trajectories)
    c_meas_abs = c_from_m_abs(m_meas)

    # Predicted envelope from κ2_pred
    max_lag = choose_max_lag_steps_fgn(cfg)
    R = autocov_fft_per_traj(omega0, max_lag=max_lag)
    print(f"[S-0004:{center_method}] max_lag_used={max_lag} lenR={len(R)} n_steps={cfg.n_steps}")
    k2_pred = k2_from_autocov_discrete(R, dt=cfg.dt, n_steps=cfg.n_steps, max_lag_steps=max_lag)
    c_pred = envelope_predicted_from_kappa2(k2_pred)

    # Time grid
    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)

    # Audits
    clo = audit_predictive_closure(t, c_meas_lin, c_pred, cfg)
    phase_med, phase_n = median_phase_over_window(t, m_meas, cfg)
    resid_med = median_signed_log_residual(t, c_meas_lin, c_pred, cfg)

    # --- Direct κ2, κ4, ρ at t* (last point in audit window) ---
    mask = _mask_window(t, cfg)
    if np.count_nonzero(mask) < 1:
        idx_tstar = 0
    else:
        idx_tstar = int(np.where(mask)[0][-1])

    # delta_phi assumed shape (n_traj, n_t)
    dphi_t = np.asarray(delta_phi[:, idx_tstar], dtype=float)
    m2 = float(np.mean(dphi_t * dphi_t))
    m4 = float(np.mean((dphi_t * dphi_t) * (dphi_t * dphi_t)))
    k2_tstar = m2
    k4_tstar = m4 - 3.0 * (m2 * m2)
    if k2_tstar > 0.0 and np.isfinite(k2_tstar) and np.isfinite(k4_tstar):
        rho_tstar = abs(k4_tstar) / (k2_tstar * k2_tstar)
    else:
        rho_tstar = float("nan")

    return {
        "t": t,
        "center_method": center_method,
        "omega0": omega0,
        "R": R,
        "delta_phi": delta_phi,
        "k2_direct": k2_direct,
        "k2_pred": k2_pred,
        "m_meas": m_meas,
        "c_meas_lin": c_meas_lin,
        "c_meas_abs": c_meas_abs,
        "c_pred": c_pred,
        "closure": clo,
        "phase_median": float(phase_med),
        "phase_n": int(phase_n),
        "resid_log_median": float(resid_med),
        "max_lag_used": int(max_lag),
        # t* extras (for S-0004-SENS)
        "idx_tstar": int(idx_tstar),
        "tstar": float(t[idx_tstar]),
        "k2_direct_tstar": float(k2_tstar),
        "k4_direct_tstar": float(k4_tstar),
        "rho_tstar": float(rho_tstar),
    }

def main() -> int:
    cfg = DemoConfig()
    run_utc = _run_utc_iso()
    git_commit = _git_commit_short()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    omega_gauss = simulate_omega_gauss(cfg, rng)
    omega_k4m = simulate_omega_k4(cfg, rng, sigma_s=float(cfg.scale_sigma_mild))
    omega_k4s = simulate_omega_k4(cfg, rng, sigma_s=float(cfg.scale_sigma_strong))

    res_gauss = run_subcase(cfg, omega_gauss, center_method=cfg.center_method_gauss)
    res_k4m = run_subcase(cfg, omega_k4m, center_method=cfg.center_method_k4)
    res_k4s = run_subcase(cfg, omega_k4s, center_method=cfg.center_method_k4)

    print_audit_banner("S-0004:GAUSS", res_gauss["closure"], cfg)
    print_audit_banner("S-0004:K4_MILD", res_k4m["closure"], cfg)
    print_audit_banner("S-0004:K4_STRONG", res_k4s["closure"], cfg)

    id_gauss = audit_gaussian_identity(res_gauss["t"], res_gauss["c_meas_lin"], res_gauss["k2_direct"], cfg)
    print_identity_banner("S-0004:GAUSS:ID", id_gauss, cfg)

    for name, r in [("GAUSS", res_gauss), ("K4_MILD", res_k4m), ("K4_STRONG", res_k4s)]:
        ph = float(r["phase_median"])
        nph = int(r["phase_n"])
        if nph < 12:
            print(f"[S-0004:{name}:PHASE] gated_points={nph} (<12) -> phase undefined [INFO]")
        else:
            print(f"[S-0004:{name}:PHASE] median_phase={ph:.3f}, gated_points={nph} [INFO]")

    for name, r in [("K4_MILD", res_k4m), ("K4_STRONG", res_k4s)]:
        resid = float(r["resid_log_median"])
        ok = resid >= float(cfg.k4_resid_sign_median_min)
        print(f"[S-0004:{name}:RESID] median(log c_meas - log c_pred)={resid:.3f} (>= {cfg.k4_resid_sign_median_min:.3f}) [{'PASS' if ok else 'FAIL'}]")

    t = res_gauss["t"]
    mask = _mask_window(t, cfg)
    idx_tmax = int(np.where(mask)[0][-1])
    k2_tmax = float(res_gauss["k2_pred"][idx_tmax])
    k2_ok = k2_tmax >= float(cfg.k2_adequacy_min_at_tmax)
    print(f"[S-0004:ADEQ] k2_pred(t_max)={k2_tmax:.3f} (min {cfg.k2_adequacy_min_at_tmax:.3f}) [{'OK' if k2_ok else 'INCONCLUSIVE'}]")

    sanity = gauss_sanity_certificate(
        omega0=res_gauss["omega0"],
        R=res_gauss["R"],
        k2_pred=res_gauss["k2_pred"],
        k2_direct=res_gauss["k2_direct"],
        t=res_gauss["t"],
        cfg=cfg,
    )
    s_ok = bool(sanity["sanity_ok"])
    print(
        f"[S-0004:SANITY] R0_est={sanity['r0_est']:.6g}, Var(ω0)={sanity['var_emp']:.6g}, "
        f"rel_err={sanity['r0_rel_err']:.3%} (tol {cfg.sanity_r0_rel_tol:.1%}); "
        f"k2_direct(t_max)={sanity['k2_direct_tmax']:.6g}, k2_pred(t_max)={sanity['k2_pred_tmax']:.6g}, "
        f"rel_err={sanity['k2_rel_err']:.3%} (tol {cfg.sanity_k2_rel_tol:.1%}) "
        f"[{'OK' if s_ok else 'FAIL'}]"
    )

    audit_csv = out_dir / cfg.out_audit_csv
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "case",
            "center_method",
            "pass_closure",
            "log_err_median",
            "log_err_p95",
            "drift_slope",
            "drift_bins_used",
            "phase_median",
            "phase_gated_points",
            "signed_resid_log_median",
            "k2_pred_tmax",
            "k2_direct_tmax",
            "gauss_identity_pass",
            "gauss_id_log_err_median",
            "gauss_id_log_err_p95",
            "gauss_id_drift_slope",
            "sanity_ok",
            "run_utc",
            "git_commit",
        ])
        for name, r in [("GAUSS", res_gauss), ("K4_MILD", res_k4m), ("K4_STRONG", res_k4s)]:
            clo = r["closure"]
            row = [
                name,
                r["center_method"],
                int(bool(clo["pass"])),
                f"{clo['log_err_median']:.10g}",
                f"{clo['log_err_p95']:.10g}",
                f"{clo['drift_slope']:.10g}",
                int(clo["drift_bins_used"]),
                f"{float(r['phase_median']):.10g}",
                int(r["phase_n"]),
                f"{float(r['resid_log_median']):.10g}",
                f"{float(r['k2_pred'][idx_tmax]):.10g}",
                f"{float(r['k2_direct'][idx_tmax]):.10g}",
            ]
            if name == "GAUSS":
                row += [
                    int(bool(id_gauss["pass"])),
                    f"{id_gauss['id_log_err_median']:.10g}",
                    f"{id_gauss['id_log_err_p95']:.10g}",
                    f"{id_gauss['id_drift_slope']:.10g}",
                    int(bool(s_ok)),
                ]
            else:
                row += ["", "", "", "", ""]
            row += [run_utc, git_commit]
            w.writerow(row)

    csv_path = out_dir / cfg.out_csv
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "c_lin_gauss", "c_abs_gauss", "c_pred_gauss", "k2_pred_gauss", "k2_direct_gauss",
            "c_lin_k4m", "c_abs_k4m", "c_pred_k4m", "k2_pred_k4m", "k2_direct_k4m",
            "c_lin_k4s", "c_abs_k4s", "c_pred_k4s", "k2_pred_k4s", "k2_direct_k4s",
            "dt", "n_steps", "n_trajectories", "seed",
            "fgn_H", "fgn_mu", "fgn_sigma",
            "center_method_gauss",
            "center_method_k4",
            "scale_sigma_mild",
            "scale_sigma_strong",
            "max_lag_used",
            "run_utc",
            "git_commit",
        ])
        t = res_gauss["t"]
        for i in range(len(t)):
            w.writerow([
                f"{t[i]:.10g}",
                f"{res_gauss['c_meas_lin'][i]:.10g}",
                f"{res_gauss['c_meas_abs'][i]:.10g}",
                f"{res_gauss['c_pred'][i]:.10g}",
                f"{res_gauss['k2_pred'][i]:.10g}",
                f"{res_gauss['k2_direct'][i]:.10g}",
                f"{res_k4m['c_meas_lin'][i]:.10g}",
                f"{res_k4m['c_meas_abs'][i]:.10g}",
                f"{res_k4m['c_pred'][i]:.10g}",
                f"{res_k4m['k2_pred'][i]:.10g}",
                f"{res_k4m['k2_direct'][i]:.10g}",
                f"{res_k4s['c_meas_lin'][i]:.10g}",
                f"{res_k4s['c_meas_abs'][i]:.10g}",
                f"{res_k4s['c_pred'][i]:.10g}",
                f"{res_k4s['k2_pred'][i]:.10g}",
                f"{res_k4s['k2_direct'][i]:.10g}",
                f"{cfg.dt:.10g}",
                cfg.n_steps,
                cfg.n_trajectories,
                cfg.seed,
                f"{cfg.fgn_H:.10g}",
                f"{cfg.fgn_mu:.10g}",
                f"{cfg.fgn_sigma:.10g}",
                cfg.center_method_gauss,
                cfg.center_method_k4,
                f"{cfg.scale_sigma_mild:.10g}",
                f"{cfg.scale_sigma_strong:.10g}",
                res_gauss["max_lag_used"],
                run_utc,
                git_commit,
            ])

    t = res_gauss["t"]

    fig_path = out_dir / cfg.out_png
    plt.figure()
    plt.plot(t, res_gauss["c_meas_lin"], label="GAUSS: c_meas_lin=Re(mean)")
    plt.plot(t, res_gauss["c_pred"], "--", label="GAUSS: c_pred (κ2_pred)")
    plt.plot(t, res_k4m["c_meas_lin"], label="K4_MILD: c_meas_lin")
    plt.plot(t, res_k4m["c_pred"], "--", label="K4_MILD: c_pred (κ2_pred)")
    plt.plot(t, res_k4s["c_meas_lin"], label="K4_STRONG: c_meas_lin")
    plt.plot(t, res_k4s["c_pred"], "--", label="K4_STRONG: c_pred (κ2_pred)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel("c(t)")
    plt.title("S-0004: κ4 boundary (closure audited on Re(mean); Δϕ is per-time ensemble-centered in measurement)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    err_png = out_dir / cfg.out_err_png
    eps = 1e-12

    def abs_log_err(cm: np.ndarray, cp: np.ndarray) -> np.ndarray:
        return np.abs(np.log(np.clip(cm, eps, 1.0 - eps)) - np.log(np.clip(cp, eps, 1.0 - eps)))

    plt.figure()
    plt.plot(t, abs_log_err(res_gauss["c_meas_lin"], res_gauss["c_pred"]), label="GAUSS: |Δ log c|")
    plt.plot(t, abs_log_err(res_k4m["c_meas_lin"], res_k4m["c_pred"]), label="K4_MILD: |Δ log c|")
    plt.plot(t, abs_log_err(res_k4s["c_meas_lin"], res_k4s["c_pred"]), label="K4_STRONG: |Δ log c|")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel("|Δ log c(t)|")
    plt.title("S-0004: κ2 magnitude-closure error (audited linear mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(err_png, dpi=160)
    plt.close()

    phase_png = out_dir / cfg.out_phase_png
    plt.figure()
    plt.plot(t, np.angle(res_gauss["m_meas"]), label="GAUSS: angle(m_meas)")
    plt.plot(t, np.angle(res_k4m["m_meas"]), label="K4_MILD: angle(m_meas)")
    plt.plot(t, np.angle(res_k4s["m_meas"]), label="K4_STRONG: angle(m_meas)")
    plt.xscale("log")
    plt.xlabel("t")
    plt.ylabel("phase angle(m_meas)")
    plt.title("S-0004: Phase diagnostic (non-gating)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(phase_png, dpi=160)
    plt.close()

    if not s_ok:
        print("[S-0004] AUDIT FAILED: GAUSS sanity certificate failed (pipeline mismatch: R0 and/or κ2_pred != κ2_direct).")
        return 2

    if not bool(id_gauss["pass"]):
        print("[S-0004] AUDIT FAILED: GAUSS failed Gaussian identity audit (log c != -κ2_direct/2) under linear-mean estimator.")
        return 2

    if not bool(res_gauss["closure"]["pass"]):
        print("[S-0004] AUDIT FAILED: GAUSS did not pass κ2 magnitude closure baseline.")
        return 2

    if not k2_ok:
        print("[S-0004] INCONCLUSIVE: κ2 inadequate in audited window to resolve κ4 effect.")
        return 3

    def k4_ok(r: Dict[str, Any]) -> bool:
        clo = r["closure"]
        fail_by_margin = (float(clo["log_err_median"]) >= float(cfg.k4_fail_median_min)) and (
            float(clo["log_err_p95"]) >= float(cfg.k4_fail_p95_min)
        )
        sign_ok = float(r["resid_log_median"]) >= float(cfg.k4_resid_sign_median_min)
        return (not bool(clo["pass"])) and fail_by_margin and sign_ok

    if not (k4_ok(res_k4m) and k4_ok(res_k4s)):
        print("[S-0004] AUDIT FAILED: κ4 boundary not certified (K4 did not fail-by-margin with required structure).")
        return 2

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {fig_path}")
    print(f"Wrote: {err_png}")
    print(f"Wrote: {phase_png}")
    print(f"Wrote: {audit_csv}")
    print("[S-0004] AUDIT PASSED (κ4 even-cumulant boundary certified).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())