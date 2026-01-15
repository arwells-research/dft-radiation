#!/usr/bin/env python3
"""
S-0003 — Boundary of Gaussian sufficiency (κ3 phase-bias diagnostic) — SELF-AUDITING

Updated (LOCKED INTENT):
- GAUSS / K3+ / K3− are all expected to PASS κ2 magnitude-closure for:
      c_meas(t) = | < exp(i Δϕ(t)) > |
      c_pred(t) = exp(-κ2(t)/2)
  where κ2 is computed from the empirical autocovariance via the discrete formula.

- The diagnostic boundary of Gaussian sufficiency is *phase*, not magnitude:
      m_meas(t) = < exp(i Δϕ(t)) >   (complex)
      phase(t) = arg(m_meas(t))
  K3+ and K3− must show sign-sensitive, coherent phase bias:
      median_sign(K3+) = -median_sign(K3−), both nonzero
      sign_fraction(K3±) >= residual_sign_bin_frac

Non-Gaussianity is injected minimally at the innovation level (ε with Var=1 and κ3,ε != 0),
while preserving κ2 by construction (linear Davies–Harte; optional κ2-preserving mixture).

No claim is made that κ3 is fundamental; this is a diagnostic perturbation only.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List
import datetime
import subprocess

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Config
# ---------------------------

@dataclass(frozen=True)
class DemoConfig:
    seed: int = 12345

    # grid (coarser dt reduces κ3 washout; still compatible with κ2 closure)
    dt: float = 0.1
    n_steps: int = 600  # total time = 60.0

    # ensemble
    n_trajectories: int = 512

    # fGn base params
    fgn_H: float = 0.70
    fgn_mu: float = 1.0
    fgn_sigma: float = 0.25

    # Centering (must be identical for meas & pred)
    center_method: str = "per_traj_time"  # "none" | "global_time" | "per_traj_time"

    # κ2 prediction controls
    max_lag_steps_cap: int = 500
    fgn_lag_fraction: float = 0.75

    # κ3 injection knob (innovation-level third moment)
    # NOTE: sign encoded by subcase; magnitude set here.
    k3_eps_mag: float = 2.0

    # κ2-preserving mixture strength λ for K3± (0..1)
    # GAUSS uses λ=0.0; K3± uses this value.
    # Set to 1.0 to maximize skew content; keep <=1.0 to preserve κ2 via a^2+b^2=1.
    k3_mix_lambda: float = 1.0

    # audit window (time units)
    audit_window_t_min: float = 20.0
    audit_window_t_max: float = 40.0

    # magnitude-closure tolerances
    audit_log_err_median_tol: float = 0.12
    audit_log_err_p95_tol: float = 0.25
    audit_drift_slope_tol: float = 0.06

    # phase signature criteria
    residual_bins: int = 20
    residual_sign_bin_frac: float = 0.70

    # GAUSS phase should be near zero (soft threshold; prevents accidental bias)
    gauss_phase_abs_median_max: float = 0.20  # radians

    # outputs
    out_dir: str = "toys/outputs"
    out_csv: str = "s0003_envelope_compare.csv"
    out_png: str = "s0003_envelope_compare.png"
    out_err_png: str = "s0003_error_diagnostic.png"
    out_phase_png: str = "s0003_phase_bias.png"
    out_audit_csv: str = "s0003_audit.csv"


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
# Innovations: κ3 injection with exact mean=0, var=1
# ---------------------------

def innovations_k3_unitvar(rng: np.random.Generator, size: int, k3_eps: float) -> np.ndarray:
    """
    i.i.d innovations ε with:
      E[ε]=0, Var(ε)=1, E[ε^3]=k3_eps

    Construction:
      - For k3_eps=0: ε ~ N(0,1)
      - For k3_eps!=0: standardized Gamma(shape=k,scale=1):
            ε = sign(k3_eps) * (G-k)/sqrt(k),  G ~ Gamma(k,1)
        Var=1 and E[ε^3]=2/sqrt(k)=|k3_eps|.
    """
    k3 = float(k3_eps)
    if abs(k3) < 1e-15:
        return rng.normal(size=size)

    k_shape = (2.0 / abs(k3)) ** 2
    g = rng.gamma(shape=k_shape, scale=1.0, size=size)
    eps = (g - k_shape) / math.sqrt(k_shape)
    eps *= math.copysign(1.0, k3)
    return eps


# ---------------------------
# Davies–Harte with customizable innovations (linear → κ2 preserved if Var=1)
# ---------------------------

def fgn_davies_harte_with_innovations(n_steps: int, H: float, rng: np.random.Generator, *, k3_eps: float) -> np.ndarray:
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0,1)")

    n = int(n_steps)
    k = np.arange(0, n)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * H) + np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H))

    r = np.concatenate([gamma, gamma[-2:0:-1]])
    lam = np.fft.rfft(r).real
    lam[lam < 0] = 0.0

    m = lam.shape[0]

    # i.i.d. real sequences with mean 0, var 1
    u = innovations_k3_unitvar(rng, m, k3_eps=k3_eps)
    v = innovations_k3_unitvar(rng, m, k3_eps=k3_eps)
    z = (u + 1j * v) / math.sqrt(2.0)

    x = np.fft.irfft(np.sqrt(lam) * z, n=r.shape[0])[:n]

    std = float(np.std(x))
    if std <= 0:
        raise RuntimeError("fGn generation produced non-positive std.")
    return x / std


def simulate_fgn_omega(cfg: DemoConfig, rng: np.random.Generator, *, k3_eps: float, mix_lambda: float) -> np.ndarray:
    """
    ω(t)=μ+σ*x(t), where x(t) is a κ2-preserving mixture:

        x = sqrt(1-λ^2) * xG + λ * xS

    xG: Gaussian fGn (k3_eps=0)
    xS: skew-innovation fGn (k3_eps!=0)

    If xG and xS share the same target covariance and are independent, Cov(x) matches the target
    (up to Monte Carlo error) provided a^2+b^2=1.
    """
    lam = float(mix_lambda)
    if not (0.0 <= lam <= 1.0):
        raise ValueError("mix_lambda must be in [0,1]")

    a = math.sqrt(max(0.0, 1.0 - lam * lam))
    b = lam

    nT, n = cfg.n_trajectories, cfg.n_steps
    mu, sigma, H = float(cfg.fgn_mu), float(cfg.fgn_sigma), float(cfg.fgn_H)

    omega = np.empty((nT, n), dtype=float)
    for i in range(nT):
        xG = fgn_davies_harte_with_innovations(n, H, rng, k3_eps=0.0)
        if abs(k3_eps) < 1e-15 or b == 0.0:
            x = xG
        else:
            xS = fgn_davies_harte_with_innovations(n, H, rng, k3_eps=k3_eps)
            x = a * xG + b * xS
            # normalize mixture to unit std (keeps ω scale stable)
            x = x / float(np.std(x))
        omega[i, :] = mu + sigma * x

    return omega


# ---------------------------
# Centering
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


# ---------------------------
# Envelope, autocov, κ2 prediction
# ---------------------------

def envelope_measured_complex(omega0: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    delta_phi = np.cumsum(omega0, axis=1) * float(dt)
    z = np.exp(1j * delta_phi)
    m = np.mean(z, axis=0)
    c = np.abs(m)
    return m, c


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


# ---------------------------
# Audit: κ2 magnitude closure
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
    *,
    n_bins: int,
) -> Dict[str, Any]:
    t = np.asarray(t, dtype=float)
    c_meas = np.asarray(c_meas, dtype=float)
    c_pred = np.asarray(c_pred, dtype=float)

    eps = 1e-12
    cm = np.clip(c_meas, eps, 1.0 - eps)
    cp = np.clip(c_pred, eps, 1.0 - eps)

    mask = (t >= float(t_min)) & (t <= float(t_max))
    if np.count_nonzero(mask) < 12:
        raise ValueError("Insufficient points in audit window.")

    tw = t[mask]
    lt = np.log(tw)
    le = np.abs(np.log(cm[mask]) - np.log(cp[mask]))

    med = float(np.median(le))
    p95 = float(np.percentile(le, 95))

    # robust drift: bin in log(t), median(le) per bin, regress median(le) vs log(t)
    edges = np.linspace(float(np.min(lt)), float(np.max(lt)), int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    x = []
    y = []
    for i in range(int(n_bins)):
        m = (lt >= edges[i]) & (lt < edges[i + 1] if i < int(n_bins) - 1 else lt <= edges[i + 1])
        if np.count_nonzero(m) < 5:
            continue
        x.append(float(centers[i]))
        y.append(float(np.median(le[m])))

    if len(y) < 6:
        raise ValueError("Insufficient populated bins for drift diagnostic.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    passed = (med <= float(med_tol)) and (p95 <= float(p95_tol)) and (abs(slope) <= float(drift_slope_tol))

    return {
        "n_points": int(len(le)),
        "t_min": float(t_min),
        "t_max": float(t_max),
        "log_err_median": med,
        "log_err_p95": p95,
        "drift_slope": slope,
        "drift_intercept": intercept,
        "drift_bins_used": int(len(y)),
        "pass": bool(passed),
    }


# ---------------------------
# Phase signature (odd-cumulant sign sensitivity)
# ---------------------------

def binned_median_phase_sign_fraction(
    t: np.ndarray,
    m_meas: np.ndarray,
    t_min: float,
    t_max: float,
    *,
    n_bins: int,
) -> Dict[str, Any]:
    t = np.asarray(t, dtype=float)
    m_meas = np.asarray(m_meas, dtype=complex)

    mask = (t >= float(t_min)) & (t <= float(t_max))
    if np.count_nonzero(mask) < 12:
        raise ValueError("Insufficient points for phase signature.")

    tw = t[mask]
    ph = np.angle(m_meas[mask])
    lt = np.log(tw)

    edges = np.linspace(float(np.min(lt)), float(np.max(lt)), int(n_bins) + 1)

    med_bins: List[float] = []
    for i in range(int(n_bins)):
        m = (lt >= edges[i]) & (lt < edges[i + 1] if i < int(n_bins) - 1 else lt <= edges[i + 1])
        if np.count_nonzero(m) < 5:
            continue
        med_bins.append(float(np.median(ph[m])))

    if len(med_bins) < 6:
        raise ValueError("Insufficient populated bins for phase signature.")

    med_bins_arr = np.asarray(med_bins, dtype=float)
    m0 = float(np.median(med_bins_arr))
    s0 = 1 if m0 > 0 else (-1 if m0 < 0 else 0)

    signs = np.sign(med_bins_arr)
    nonzero = signs != 0
    frac = float(np.mean(signs[nonzero] == float(s0))) if np.count_nonzero(nonzero) else 0.0

    return {
        "bins_used": int(len(med_bins_arr)),
        "median_phase": m0,
        "median_sign": int(s0),
        "sign_fraction": frac,
    }


def structured_phase_check(cfg: DemoConfig, plus_sig: Dict[str, Any], minus_sig: Dict[str, Any]) -> Dict[str, Any]:
    s_plus = int(plus_sig["median_sign"])
    s_minus = int(minus_sig["median_sign"])
    sign_sensitive = (s_plus != 0) and (s_minus != 0) and (s_plus == -s_minus)

    frac_ok = (float(plus_sig["sign_fraction"]) >= float(cfg.residual_sign_bin_frac)) and (
        float(minus_sig["sign_fraction"]) >= float(cfg.residual_sign_bin_frac)
    )

    return {
        "pass": bool(sign_sensitive and frac_ok),
        "sign_sensitive": bool(sign_sensitive),
        "frac_ok": bool(frac_ok),
        "plus_sign_fraction": float(plus_sig["sign_fraction"]),
        "minus_sign_fraction": float(minus_sig["sign_fraction"]),
        "m_plus": float(plus_sig["median_phase"]),
        "m_minus": float(minus_sig["median_phase"]),
    }


# ---------------------------
# Utilities
# ---------------------------

def choose_max_lag_steps_fgn(cfg: DemoConfig) -> int:
    k = int(float(cfg.fgn_lag_fraction) * float(cfg.n_steps - 1))
    return max(1, min(k, cfg.n_steps - 1, cfg.max_lag_steps_cap))


def print_audit_banner(label: str, res: Dict[str, Any], cfg: DemoConfig) -> None:
    status = "PASS" if res["pass"] else "FAIL"
    print(
        f"[{label}] median|Δlog c|={res['log_err_median']:.3f} (tol {cfg.audit_log_err_median_tol:.3f}), "
        f"p95|Δlog c|={res['log_err_p95']:.3f} (tol {cfg.audit_log_err_p95_tol:.3f}), "
        f"drift_slope={res['drift_slope']:.3f} (tol {cfg.audit_drift_slope_tol:.3f}), "
        f"bins={res['drift_bins_used']}, window [{res['t_min']:g},{res['t_max']:g}], "
        f"n={res['n_points']} [{status}]"
    )


def write_audit_csv(
    path: Path,
    closure_rows: List[Tuple[str, Dict[str, Any]]],
    phase_rows: Dict[str, Dict[str, Any]],
    phase_check: Dict[str, Any],
    run_utc: str,
    git_commit: str,
) -> None:
    header = [
        "case",
        "pass_closure",
        "t_min",
        "t_max",
        "n_points",
        "log_err_median",
        "log_err_p95",
        "drift_slope",
        "drift_intercept",
        "drift_bins_used",
        "median_phase",
        "median_sign",
        "phase_sign_fraction",
        "phase_bins_used",
        "run_utc",
        "git_commit",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for name, r in closure_rows:
            ph = phase_rows[name]
            w.writerow([
                name,
                int(bool(r.get("pass", False))),
                f"{r.get('t_min', float('nan')):.10g}",
                f"{r.get('t_max', float('nan')):.10g}",
                int(r.get("n_points", 0)),
                f"{r.get('log_err_median', float('nan')):.10g}",
                f"{r.get('log_err_p95', float('nan')):.10g}",
                f"{r.get('drift_slope', float('nan')):.10g}",
                f"{r.get('drift_intercept', float('nan')):.10g}",
                int(r.get("drift_bins_used", 0)),
                f"{ph.get('median_phase', float('nan')):.10g}",
                int(ph.get("median_sign", 0)),
                f"{ph.get('sign_fraction', float('nan')):.10g}",
                int(ph.get("bins_used", 0)),
                run_utc,
                git_commit,
            ])

        w.writerow([])
        w.writerow(["PHASE_CHECK", int(phase_check["pass"]), "", "", "", "", "", "", "", "",
                    f"{phase_check['m_plus']:.10g}", "", "", "",
                    run_utc, git_commit])
        w.writerow(["PHASE_CHECK_DETAILS",
                    "",
                    "sign_sensitive", int(phase_check["sign_sensitive"]),
                    "frac_ok", int(phase_check["frac_ok"]),
                    "sign_frac_plus", f"{phase_check['plus_sign_fraction']:.10g}",
                    "sign_frac_minus", f"{phase_check['minus_sign_fraction']:.10g}",
                    "m_minus", f"{phase_check['m_minus']:.10g}",
                    "", "", "", "",
                    run_utc, git_commit])


# ---------------------------
# Subcase runner
# ---------------------------

def run_subcase(cfg: DemoConfig, rng: np.random.Generator, *, label: str, k3_eps: float, mix_lambda: float) -> Dict[str, Any]:
    omega = simulate_fgn_omega(cfg, rng, k3_eps=k3_eps, mix_lambda=mix_lambda)
    omega0 = center_omega(omega, mu=cfg.fgn_mu, method=cfg.center_method)

    m_meas, c_meas = envelope_measured_complex(omega0, cfg.dt)

    max_lag = choose_max_lag_steps_fgn(cfg)
    R = autocov_fft_per_traj(omega0, max_lag=max_lag)
    k2 = k2_from_autocov_discrete(R, dt=cfg.dt, n_steps=cfg.n_steps, max_lag_steps=max_lag)
    c_pred = envelope_predicted_from_kappa2(k2)

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)

    closure = audit_predictive_closure(
        t=t,
        c_meas=c_meas,
        c_pred=c_pred,
        t_min=cfg.audit_window_t_min,
        t_max=cfg.audit_window_t_max,
        med_tol=cfg.audit_log_err_median_tol,
        p95_tol=cfg.audit_log_err_p95_tol,
        drift_slope_tol=cfg.audit_drift_slope_tol,
        n_bins=cfg.residual_bins,
    )

    phase_sig = binned_median_phase_sign_fraction(
        t=t,
        m_meas=m_meas,
        t_min=cfg.audit_window_t_min,
        t_max=cfg.audit_window_t_max,
        n_bins=cfg.residual_bins,
    )

    return {
        "label": label,
        "k3_eps": float(k3_eps),
        "mix_lambda": float(mix_lambda),
        "t": t,
        "m_meas": m_meas,
        "c_meas": c_meas,
        "c_pred": c_pred,
        "k2": k2,
        "closure": closure,
        "phase_sig": phase_sig,
        "max_lag_used": int(max_lag),
    }


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

    k3 = float(cfg.k3_eps_mag)
    lam = float(cfg.k3_mix_lambda)

    # Declared subcases (GAUSS has no κ3 and no mixture)
    cases = [
        ("GAUSS", 0.0, 0.0),
        ("K3+", +k3, lam),
        ("K3-", -k3, lam),
    ]

    results: Dict[str, Dict[str, Any]] = {}
    for name, k3_eps, mix_lambda in cases:
        results[name] = run_subcase(cfg, rng, label=name, k3_eps=k3_eps, mix_lambda=mix_lambda)

    # Print magnitude closure audits (expected PASS for all)
    print_audit_banner("S-0003:GAUSS", results["GAUSS"]["closure"], cfg)
    print_audit_banner("S-0003:K3+", results["K3+"]["closure"], cfg)
    print_audit_banner("S-0003:K3-", results["K3-"]["closure"], cfg)

    # Phase structured residual (required PASS)
    phase_check = structured_phase_check(cfg, results["K3+"]["phase_sig"], results["K3-"]["phase_sig"])
    print(
        f"[S-0003:PHASE] sign_sensitive={int(phase_check['sign_sensitive'])}, "
        f"sign_frac(+)= {phase_check['plus_sign_fraction']:.3f}, "
        f"sign_frac(-)= {phase_check['minus_sign_fraction']:.3f}, "
        f"frac_ok={int(phase_check['frac_ok'])} "
        f"[{'PASS' if phase_check['pass'] else 'FAIL'}]"
    )

    # GAUSS phase should be ~0 (soft gate, but auditable)
    gauss_phase_med = float(results["GAUSS"]["phase_sig"]["median_phase"])
    gauss_phase_ok = (abs(gauss_phase_med) <= float(cfg.gauss_phase_abs_median_max))
    print(
        f"[S-0003:GAUSS_PHASE] median_phase={gauss_phase_med:.3f} "
        f"(abs<= {cfg.gauss_phase_abs_median_max:.3f}) "
        f"[{'PASS' if gauss_phase_ok else 'FAIL'}]"
    )

    # Write series CSV
    t = results["GAUSS"]["t"]
    csv_path = out_dir / cfg.out_csv
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "c_meas_gauss", "c_pred_gauss", "k2_gauss", "phase_gauss",
            "c_meas_k3p",   "c_pred_k3p",   "k2_k3p",   "phase_k3p",
            "c_meas_k3m",   "c_pred_k3m",   "k2_k3m",   "phase_k3m",
            "dt", "n_steps", "n_trajectories", "seed",
            "fgn_H", "fgn_mu", "fgn_sigma",
            "center_method",
            "k3_eps_mag",
            "k3_mix_lambda",
            "max_lag_used",
            "audit_t_min", "audit_t_max",
            "run_utc",
            "git_commit",
        ])
        for i in range(len(t)):
            w.writerow([
                f"{t[i]:.10g}",
                f"{results['GAUSS']['c_meas'][i]:.10g}",
                f"{results['GAUSS']['c_pred'][i]:.10g}",
                f"{results['GAUSS']['k2'][i]:.10g}",
                f"{float(np.angle(results['GAUSS']['m_meas'][i])):.10g}",
                f"{results['K3+']['c_meas'][i]:.10g}",
                f"{results['K3+']['c_pred'][i]:.10g}",
                f"{results['K3+']['k2'][i]:.10g}",
                f"{float(np.angle(results['K3+']['m_meas'][i])):.10g}",
                f"{results['K3-']['c_meas'][i]:.10g}",
                f"{results['K3-']['c_pred'][i]:.10g}",
                f"{results['K3-']['k2'][i]:.10g}",
                f"{float(np.angle(results['K3-']['m_meas'][i])):.10g}",
                f"{cfg.dt:.10g}",
                cfg.n_steps,
                cfg.n_trajectories,
                cfg.seed,
                f"{cfg.fgn_H:.10g}",
                f"{cfg.fgn_mu:.10g}",
                f"{cfg.fgn_sigma:.10g}",
                cfg.center_method,
                f"{cfg.k3_eps_mag:.10g}",
                f"{cfg.k3_mix_lambda:.10g}",
                results["GAUSS"]["max_lag_used"],
                f"{cfg.audit_window_t_min:.10g}",
                f"{cfg.audit_window_t_max:.10g}",
                run_utc,
                git_commit,
            ])

    # Plot magnitude envelopes
    fig_path = out_dir / cfg.out_png
    plt.figure()
    plt.plot(t, results["GAUSS"]["c_meas"], label="GAUSS: c_meas")
    plt.plot(t, results["GAUSS"]["c_pred"], "--", label="GAUSS: c_pred (κ2)")
    plt.plot(t, results["K3+"]["c_meas"], label="K3+: c_meas")
    plt.plot(t, results["K3+"]["c_pred"], "--", label="K3+: c_pred (κ2)")
    plt.plot(t, results["K3-"]["c_meas"], label="K3-: c_meas")
    plt.plot(t, results["K3-"]["c_pred"], "--", label="K3-: c_pred (κ2)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel("c(t)")
    plt.title("S-0003: Gaussian sufficiency boundary (κ3 phase-bias; κ2 magnitude closure)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    # Magnitude error diagnostic
    err_png = out_dir / cfg.out_err_png
    eps = 1e-12

    def abs_log_err(c_meas: np.ndarray, c_pred: np.ndarray) -> np.ndarray:
        return np.abs(np.log(np.clip(c_meas, eps, 1.0 - eps)) - np.log(np.clip(c_pred, eps, 1.0 - eps)))

    plt.figure()
    plt.plot(t, abs_log_err(results["GAUSS"]["c_meas"], results["GAUSS"]["c_pred"]), label="GAUSS: |Δ log c|")
    plt.plot(t, abs_log_err(results["K3+"]["c_meas"], results["K3+"]["c_pred"]), label="K3+: |Δ log c|")
    plt.plot(t, abs_log_err(results["K3-"]["c_meas"], results["K3-"]["c_pred"]), label="K3-: |Δ log c|")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel("|Δ log c(t)|")
    plt.title("S-0003: Magnitude κ2-closure error diagnostic (expected small)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(err_png, dpi=160)
    plt.close()

    # Phase bias diagnostic
    phase_png = out_dir / cfg.out_phase_png
    plt.figure()
    plt.plot(t, np.angle(results["GAUSS"]["m_meas"]), label="GAUSS: angle(m_meas)")
    plt.plot(t, np.angle(results["K3+"]["m_meas"]), label="K3+: angle(m_meas)")
    plt.plot(t, np.angle(results["K3-"]["m_meas"]), label="K3-: angle(m_meas)")
    plt.xscale("log")
    plt.xlabel("t")
    plt.ylabel("phase angle(m_meas)")
    plt.title("S-0003: Phase bias diagnostic (odd-cumulant signature)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(phase_png, dpi=160)
    plt.close()

    # Write audit CSV (closure + phase signature + global phase-check)
    audit_csv = out_dir / cfg.out_audit_csv
    closure_rows = [
        ("GAUSS", results["GAUSS"]["closure"]),
        ("K3+", results["K3+"]["closure"]),
        ("K3-", results["K3-"]["closure"]),
    ]
    phase_rows = {
        "GAUSS": results["GAUSS"]["phase_sig"],
        "K3+": results["K3+"]["phase_sig"],
        "K3-": results["K3-"]["phase_sig"],
    }
    write_audit_csv(audit_csv, closure_rows, phase_rows, phase_check, run_utc=run_utc, git_commit=git_commit)

    # PASS/FAIL semantics (updated):
    # - All subcases must pass κ2 magnitude closure (GAUSS/K3+/K3-)
    # - GAUSS median phase must be near 0 (soft gate)
    # - Phase structured residual must pass (K3± sign-sensitive and coherent)
    all_closure_pass = bool(results["GAUSS"]["closure"]["pass"] and results["K3+"]["closure"]["pass"] and results["K3-"]["closure"]["pass"])
    if not all_closure_pass:
        print("[S-0003] AUDIT FAILED: κ2 magnitude-closure did not pass for all subcases.")
        return 2
    if not gauss_phase_ok:
        print("[S-0003] AUDIT FAILED: GAUSS phase bias is not near zero (unexpected baseline bias).")
        return 2
    if not bool(phase_check["pass"]):
        print("[S-0003] AUDIT FAILED: κ3 phase-bias structured residual did not pass.")
        return 2

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {fig_path}")
    print(f"Wrote: {err_png}")
    print(f"Wrote: {phase_png}")
    print(f"Wrote: {audit_csv}")
    print("[S-0003] AUDIT PASSED (κ3 phase-bias boundary certified; κ2 magnitude closure retained).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())