#!/usr/bin/env python3
"""
S-0026 — Real-stream admissibility harness for κ₂-slope + Σ₂ guard bundle (C5 deployment gate)

STATUS: toy skeleton (contract-aligned; no new semantics beyond declared construction)

Goal
----
Certify a *deployment harness* that runs the existing κ₂-slope classifier and Σ₂ guard bundle
on a single observed stream ω̂(t), while enforcing strict “no false OK_*” logic under C5
(irregular sampling, missingness/gaps, dt mismatch, resampling artifacts).

S-0026 does NOT claim new physics. It certifies the minimal I/O + refusal semantics needed
to apply the already-certified ladder (S-0001..S-0025) to a measurement-like input.

Inputs
------
Optional external input:
  --input_csv <path>
Expected columns (header row required):
  time, omega
Units are arbitrary but must be consistent (time monotone increasing).

If --input_csv is omitted, the toy runs fixed internal subcases:
  - C1_OU_BASE (uniform dt, OU ω)
  - C2_LM_TRUE (uniform dt, fGn ω with fixed H)
  - C5_IRREGULAR_GAPS (measurement-like: irregular sampling + gaps; must refuse)

Core computations (fixed)
------------------------
- Centering: global DC removal only (one constant mean over ensemble×time), per case.
- Phase accumulation: Δϕ(t) = ∫ ω dt (discrete cumulative sum with fixed dt on the *analysis grid*).
- κ₂(t) = Var_i[Δϕ_i(t)] for synthetic ensemble cases.
- For real-stream case (single trajectory), we use a *pseudo-ensemble bootstrap*:
    create n_boot "surrogates" by block-resampling increments (fixed block size, fixed seed),
    solely to compute κ₂(t) as a diagnostic variance curve.
  This is an epistemic (C5) device, not an ontological claim.

Windows / persistence (fixed)
-----------------------------
Window A: [20, 40]
Window B: [40, 80]
Continuation admissibility: |α_A - α_B| <= alpha_consistency_max.

Classifier bands (fixed)
------------------------
OU band: [0.82, 1.18]
LM band: [1.30, 1.70] with fGn baseline using H=0.75 (so expected α≈1.5).

Admissibility gates (fixed)
---------------------------
Fit admissibility (per window): r2>=min_r2, k2_end>=min_k2_end, n_points>=min_points.

Σ₂ guard bundle (window-restricted; any violation => BOUNDARY):
- coupling (r_mean): Var_t(mean_i ω_i(t)) / mean_i Var_t(ω_i(t))   [synthetic ensembles only]
- curvature (curv_z): |c|/SE(c) from quadratic fit to ω̄(t) in window
- variance drift (drift_z): |log(v_end/v_start)|*sqrt(n_w) where v(t)=Var_i ω_i(t)
- aperture (eta_max): max_t sqrt(κ₂(t))/L

C5 harness refusal gates (deployment-focused; any triggers => BOUNDARY):
- dt irregularity above declared tolerance after snapping to analysis grid
- gap fraction above declared threshold
- resampling factor above declared maximum (if inferred from timestamps)
- non-monotone time or malformed CSV => INCONCLUSIVE

Exit codes
----------
  0 PASS
  2 FAIL
  3 INCONCLUSIVE

Outputs (only with --write_outputs)
----------------------------------
  toys/outputs/s0026_cases.csv
  toys/outputs/s0026_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class S0026Config:
    seed: int = 26026

    # Analysis grid (uniform dt) used by the classifier/guards
    dt: float = 0.2
    n_steps: int = 1600  # covers up to t=320 with dt=0.2
    n_trajectories: int = 1024  # for synthetic ensemble cases

    # Bootstrap pseudo-ensemble for single-stream input
    n_boot: int = 512
    boot_block: int = 32  # fixed; no tuning

    # Windows (fixed)
    tA_min: float = 20.0
    tA_max: float = 40.0
    tB_min: float = 40.0
    tB_max: float = 80.0

    min_points: int = 24
    min_r2: float = 0.985
    min_k2_end: float = 5.0e-3
    eps: float = 1e-12

    # Bands
    ou_alpha_min: float = 0.82
    ou_alpha_max: float = 1.18
    fgn_H: float = 0.75
    lm_alpha_min: float = 1.30
    lm_alpha_max: float = 1.70

    alpha_consistency_max: float = 0.18

    # Σ₂ guard thresholds (match S-0024/S-0025 defaults)
    coupling_rmean_max: float = 0.02
    curv_z_max: float = 12.0
    var_drift_z_max: float = 2.5

    # Aperture
    L: float = 20.0

    # C5 harness refusal thresholds (deployment)
    # Fraction of missing samples after snapping to analysis grid.
    gap_frac_max: float = 0.08
    # Relative dt jitter tolerance before we refuse.
    dt_jitter_rel_max: float = 0.05
    # Max inferred resample factor allowed (if timestamps are much finer than dt)
    resample_factor_max: int = 4

    # Internal “measurement-like” construction parameters (fixed)
    irregular_jitter_frac: float = 0.20  # timestamps jittered relative to dt
    dropout_prob: float = 0.06          # drop samples (gaps)

    # OU parameters (fixed)
    ou_theta: float = 0.60
    ou_mu: float = 0.0
    ou_sigma: float = 1.0

    out_dir: str = "toys/outputs"
    out_cases_csv: str = "s0026_cases.csv"
    out_audit_csv: str = "s0026_audit.csv"


# -----------------------------
# Provenance / output utilities
# -----------------------------

def _run_utc_iso() -> str:
    dt = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def _git_commit_short() -> str:
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


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_csv_with_provenance_header(path: Path, header_kv: Dict[str, str], rows: List[List[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        for k, v in header_kv.items():
            f.write(f"# {k}={v}\n")
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


# -----------------------------
# Core math helpers
# -----------------------------

def _audit_mask(t: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.isfinite(t) & (t >= float(t_min)) & (t <= float(t_max))


def delta_phi_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(np.asarray(omega, dtype=float), axis=1) * float(dt)


def k2_of_delta_phi(delta_phi: np.ndarray) -> np.ndarray:
    x = np.asarray(delta_phi, dtype=float)
    mu = np.mean(x, axis=0)
    xc = x - mu[None, :]
    return np.mean(xc**2, axis=0)


def _linear_fit_loglog(t: np.ndarray, k2: np.ndarray, mask: np.ndarray, eps: float) -> Tuple[float, float]:
    tt = np.asarray(t, dtype=float)[mask]
    yy = np.asarray(k2, dtype=float)[mask]
    yy = np.maximum(yy, eps)

    x = np.log(tt)
    y = np.log(yy)

    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept = float(coef[0])
    slope = float(coef[1])
    yhat = intercept + slope * x

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    return slope, r2


def classify_alpha(cfg: S0026Config, alpha: float) -> str:
    if cfg.ou_alpha_min <= alpha <= cfg.ou_alpha_max:
        return "OU"
    if cfg.lm_alpha_min <= alpha <= cfg.lm_alpha_max:
        return "LM"
    return "OTHER"


# -----------------------------
# Σ₂ guard bundle (window-restricted)
# -----------------------------

def guard_coupling_rmean(cfg: S0026Config, omega: np.ndarray, mask: np.ndarray) -> float:
    """
    r_mean = Var_t(mean_i ω_i(t)) / mean_i Var_t(ω_i(t))
    Window-restricted.
    """
    x = np.asarray(omega, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return float("nan")
    xm = x[:, mask]
    mean_t = np.mean(xm, axis=0)            # mean across i
    var_mean = float(np.var(mean_t, ddof=0))
    var_i = np.var(xm, axis=1, ddof=0)      # variance over t per trajectory
    denom = float(np.mean(var_i))
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(var_mean / denom)


def guard_curvature_z(cfg: S0026Config, omega: np.ndarray, t: np.ndarray, mask: np.ndarray) -> float:
    """
    Fit window-restricted ensemble mean ω̄(t) ≈ a + b t + c t^2.
    curv_z = |c| / SE(c) via simple OLS proxy.
    """
    x = np.asarray(omega, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return float("nan")
    tt = np.asarray(t, dtype=float)[mask]
    mu = np.mean(x, axis=0)[mask]

    X = np.vstack([np.ones_like(tt), tt, tt**2]).T
    coef, *_ = np.linalg.lstsq(X, mu, rcond=None)
    resid = mu - X @ coef
    n = int(tt.size)
    if n < 6:
        return float("nan")

    # Residual variance estimate
    s2 = float(np.sum(resid**2) / max(n - 3, 1))
    XtX = X.T @ X
    try:
        cov = s2 * np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return float("nan")

    c = float(coef[2])
    se_c = float(np.sqrt(max(cov[2, 2], 0.0)))
    if se_c <= 0.0 or not np.isfinite(se_c):
        return float("nan")
    return float(abs(c) / se_c)


def guard_variance_drift_z(cfg: S0026Config, omega: np.ndarray, mask: np.ndarray) -> float:
    """
    v(t)=Var_i(ω_i(t)) in-window.
    drift_z = |log(v_end/v_start)| * sqrt(n_w)
    """
    x = np.asarray(omega, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return float("nan")
    xm = x[:, mask]
    v = np.var(xm, axis=0, ddof=0)
    if v.size < 2:
        return float("nan")
    v0 = float(v[0])
    v1 = float(v[-1])
    if v0 <= 0.0 or v1 <= 0.0 or (not np.isfinite(v0)) or (not np.isfinite(v1)):
        return float("nan")
    n_w = int(v.size)
    return float(abs(np.log(v1 / v0)) * np.sqrt(float(n_w)))


def guard_aperture_eta_max(cfg: S0026Config, k2: np.ndarray, mask: np.ndarray) -> float:
    """
    eta(t)=sqrt(k2(t))/L. Return max_{t in window} eta(t).
    """
    yy = np.asarray(k2, dtype=float)[mask]
    yy = np.maximum(yy, 0.0)
    eta = np.sqrt(yy) / float(cfg.L)
    return float(np.max(eta)) if eta.size else float("nan")


# -----------------------------
# Generators (diagnostic; reused from S-0020 style)
# -----------------------------

def simulate_ou(cfg: S0026Config, rng: np.random.Generator) -> np.ndarray:
    nT = int(cfg.n_trajectories)
    nS = int(cfg.n_steps)
    dt = float(cfg.dt)

    theta = float(cfg.ou_theta)
    mu = float(cfg.ou_mu)
    sigma = float(cfg.ou_sigma)

    omega = np.empty((nT, nS), dtype=float)
    omega[:, 0] = mu

    sqrt_dt = float(np.sqrt(dt))
    for k in range(1, nS):
        dW = rng.standard_normal(size=nT).astype(float) * sqrt_dt
        omega[:, k] = omega[:, k - 1] + theta * (mu - omega[:, k - 1]) * dt + sigma * dW

    return omega


def fgn_davies_harte(n_steps: int, H: float, rng: np.random.Generator) -> np.ndarray:
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0,1)")
    n = int(n_steps)
    if n < 2:
        raise ValueError("n_steps must be >= 2")

    k = np.arange(0, n, dtype=float)
    gamma = 0.5 * (
        (np.abs(k + 1.0) ** (2.0 * H))
        + (np.abs(k - 1.0) ** (2.0 * H))
        - 2.0 * (np.abs(k) ** (2.0 * H))
    )

    m = 2 * n
    r = np.zeros(m, dtype=float)
    r[0:n] = gamma
    r[n] = 0.0
    r[n + 1 :] = gamma[1:][::-1]

    lam = np.fft.fft(r).real
    lam[lam < 0.0] = 0.0

    W = rng.standard_normal(size=m) + 1j * rng.standard_normal(size=m)
    X = np.fft.ifft(np.sqrt(lam) * W).real
    x = X[:n]

    std = float(np.std(x))
    if std <= 0.0 or not np.isfinite(std):
        raise RuntimeError("fGn generation produced invalid std.")
    return (x / std).astype(float)


def simulate_fgn(cfg: S0026Config, rng: np.random.Generator) -> np.ndarray:
    nT = int(cfg.n_trajectories)
    nS = int(cfg.n_steps)
    H = float(cfg.fgn_H)

    omega = np.empty((nT, nS), dtype=float)
    for i in range(nT):
        inc = fgn_davies_harte(nS, H, rng=rng)
        omega[i, :] = inc
    return omega


# -----------------------------
# C5 harness: CSV ingestion + snapping + pseudo-ensemble
# -----------------------------

def _read_time_omega_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if (r.fieldnames is None) or ("time" not in r.fieldnames) or ("omega" not in r.fieldnames):
                return np.array([]), np.array([]), "missing_required_columns(time,omega)"
            tt: List[float] = []
            oo: List[float] = []
            for row in r:
                tt.append(float(row["time"]))
                oo.append(float(row["omega"]))
        t = np.asarray(tt, dtype=float)
        o = np.asarray(oo, dtype=float)
        if t.size < 4:
            return np.array([]), np.array([]), "too_few_rows"
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(o)):
            return np.array([]), np.array([]), "non_finite_values"
        if not np.all(np.diff(t) > 0.0):
            return np.array([]), np.array([]), "time_not_strictly_increasing"
        return t, o, None
    except Exception as e:
        return np.array([]), np.array([]), f"csv_read_error:{type(e).__name__}"


def _snap_to_grid(cfg: S0026Config, t: np.ndarray, omega: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Snap irregular samples to the uniform analysis grid by nearest-neighbor binning.
    Missing bins are left as NaN (gap).
    Returns omega_grid (length n_steps) and diagnostics dict.

    Refusal logic is handled upstream by inspecting diagnostics (gap_frac, dt_jitter_rel, etc).
    """
    t0 = float(t[0])
    dt = float(cfg.dt)
    nS = int(cfg.n_steps)

    grid_t = t0 + np.arange(nS, dtype=float) * dt
    omega_grid = np.full(nS, np.nan, dtype=float)

    # Nearest bin index
    idx = np.rint((t - t0) / dt).astype(int)
    good = (idx >= 0) & (idx < nS)
    idx = idx[good]
    o = omega[good]

    # If multiple land in same bin, average (fixed rule)
    if idx.size:
        # group by bin
        for k in np.unique(idx):
            omega_grid[k] = float(np.mean(o[idx == k]))

    # Diagnostics
    gaps = np.isnan(omega_grid)
    gap_frac = float(np.mean(gaps))

    # dt jitter estimate from raw diffs relative to median dt
    diffs = np.diff(t)
    med = float(np.median(diffs))
    dt_jitter_rel = float(np.median(np.abs(diffs - med)) / max(med, 1e-12))

    # inferred resample factor: how much finer raw sampling is vs analysis dt
    resample_factor = int(max(1, round(float(cfg.dt) / max(med, 1e-12))))

    return omega_grid, {
        "t0": t0,
        "gap_frac": gap_frac,
        "dt_raw_med": med,
        "dt_jitter_rel": dt_jitter_rel,
        "resample_factor": resample_factor,
        "n_raw": int(t.size),
        "n_grid": int(nS),
    }


def _bootstrap_surrogates(cfg: S0026Config, omega_grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Construct a pseudo-ensemble from a single ω̂(t) grid by block-resampling increments.
    This is strictly an epistemic C5 device to compute κ₂(t) for the ladder.

    Fixed:
      - difference increments dω[k] = ω[k]-ω[k-1]
      - block length boot_block
      - circular block resampling
    """
    x = np.asarray(omega_grid, dtype=float)

    # Fill gaps by linear interpolation for the purpose of forming increments
    # (gaps are still refusal-gated by gap_frac upstream).
    n = x.size
    idx = np.arange(n)
    good = np.isfinite(x)
    if np.count_nonzero(good) < 4:
        return np.empty((0, 0), dtype=float)

    x_filled = np.interp(idx.astype(float), idx[good].astype(float), x[good]).astype(float)

    d = np.diff(x_filled, prepend=x_filled[0])
    B = int(cfg.boot_block)
    n_boot = int(cfg.n_boot)

    # circular blocks
    starts = rng.integers(0, max(1, n - B), size=n_boot)
    sur = np.empty((n_boot, n), dtype=float)
    for i in range(n_boot):
        s = int(starts[i])
        # stitch blocks to length n
        out = np.empty(n, dtype=float)
        pos = 0
        while pos < n:
            blk = d[s : s + B]
            m = min(B, n - pos)
            out[pos : pos + m] = blk[:m]
            pos += m
            s = int((s + B) % max(1, n - B))
        # integrate increments to reconstruct ω surrogate
        sur[i, :] = np.cumsum(out)
    # Normalize to unit variance (fixed stabilization)
    v = np.var(sur, axis=1, keepdims=True)
    sur = sur / np.sqrt(np.maximum(v, 1e-12))
    return sur


# -----------------------------
# Case audit
# -----------------------------

def _fit_window(cfg: S0026Config, t: np.ndarray, k2: np.ndarray, t_min: float, t_max: float) -> Dict[str, object]:
    mask = _audit_mask(t, t_min, t_max)
    n_w = int(np.count_nonzero(mask))
    if n_w < int(cfg.min_points):
        return {"ok": False, "reason": "too_few_points", "n_w": n_w}

    k2_end = float(k2[np.where(mask)[0][-1]])
    alpha, r2 = _linear_fit_loglog(t, k2, mask, eps=float(cfg.eps))
    if (not np.isfinite(alpha)) or (not np.isfinite(r2)) or (r2 < float(cfg.min_r2)) or (k2_end < float(cfg.min_k2_end)):
        return {
            "ok": False,
            "reason": "inadmissible_fit",
            "n_w": n_w,
            "alpha": float(alpha),
            "r2": float(r2),
            "k2_end": float(k2_end),
        }
    band = classify_alpha(cfg, float(alpha))
    return {"ok": True, "n_w": n_w, "alpha": float(alpha), "r2": float(r2), "k2_end": float(k2_end), "band": band}


def audit_ensemble_case(cfg: S0026Config, name: str, omega: np.ndarray, t: np.ndarray) -> Dict[str, object]:
    x = np.asarray(omega, dtype=float)
    x = x - float(np.mean(x))  # global DC removal only

    dphi = delta_phi_from_omega(x, cfg.dt)
    k2 = k2_of_delta_phi(dphi)

    A = _fit_window(cfg, t, k2, cfg.tA_min, cfg.tA_max)
    B = _fit_window(cfg, t, k2, cfg.tB_min, cfg.tB_max)
    if (not A.get("ok")) or (not B.get("ok")):
        return {"case": name, "verdict": "INCONCLUSIVE", "reason": "fit_inadmissible", "A": A, "B": B}

    # Continuation consistency
    alphaA = float(A["alpha"])
    alphaB = float(B["alpha"])
    alpha_cons_ok = (abs(alphaA - alphaB) <= float(cfg.alpha_consistency_max))

    # Guards (each window)
    maskA = _audit_mask(t, cfg.tA_min, cfg.tA_max)
    maskB = _audit_mask(t, cfg.tB_min, cfg.tB_max)

    rA = guard_coupling_rmean(cfg, x, maskA)
    rB = guard_coupling_rmean(cfg, x, maskB)
    curvA = guard_curvature_z(cfg, x, t, maskA)
    curvB = guard_curvature_z(cfg, x, t, maskB)
    driftA = guard_variance_drift_z(cfg, x, maskA)
    driftB = guard_variance_drift_z(cfg, x, maskB)
    etaA = guard_aperture_eta_max(cfg, k2, maskA)
    etaB = guard_aperture_eta_max(cfg, k2, maskB)

    viol_cpl = int((np.isfinite(rA) and rA > cfg.coupling_rmean_max) or (np.isfinite(rB) and rB > cfg.coupling_rmean_max))
    viol_curv = int((np.isfinite(curvA) and curvA > cfg.curv_z_max) or (np.isfinite(curvB) and curvB > cfg.curv_z_max))
    viol_vardrift = int((np.isfinite(driftA) and driftA > cfg.var_drift_z_max) or (np.isfinite(driftB) and driftB > cfg.var_drift_z_max))
    viol_ap = int((np.isfinite(etaA) and etaA > 1.0) or (np.isfinite(etaB) and etaB > 1.0))

    guard_violation = int(viol_cpl or viol_curv or viol_vardrift or viol_ap)

    # Tag logic
    bandA = str(A["band"])
    bandB = str(B["band"])
    if guard_violation or (not alpha_cons_ok) or (bandA != bandB):
        tag = "BOUNDARY"
        notes = "guard_or_continuation_violation"
    else:
        if bandA == "OU":
            tag = "OK_OU"
            notes = "ok"
        elif bandA == "LM":
            tag = "OK_LM"
            notes = "ok"
        else:
            tag = "BOUNDARY"
            notes = "band_other"

    return {
        "case": name,
        "verdict": "OK",
        "A_alpha": alphaA,
        "A_r2": float(A["r2"]),
        "B_alpha": alphaB,
        "B_r2": float(B["r2"]),
        "bandA": bandA,
        "bandB": bandB,
        "alpha_cons_ok": int(alpha_cons_ok),
        "r_mean_max": float(np.nanmax([rA, rB])),
        "curv_z_max": float(np.nanmax([curvA, curvB])),
        "drift_z_max": float(np.nanmax([driftA, driftB])),
        "eta_max": float(np.nanmax([etaA, etaB])),
        "viol_cpl": viol_cpl,
        "viol_curv": viol_curv,
        "viol_vardrift": viol_vardrift,
        "viol_ap": viol_ap,
        "tag": tag,
        "notes": notes,
    }


def audit_stream_case(cfg: S0026Config, name: str, t_raw: np.ndarray, omega_raw: np.ndarray) -> Dict[str, object]:
    """
    Single-stream harness:
      - snap to uniform analysis grid
      - refuse on dt jitter / gaps / resample factor
      - bootstrap pseudo-ensemble to compute κ₂(t)
      - run the same windows + continuation + guards (except coupling r_mean, which is N/A)
    """
    omega_grid, diag = _snap_to_grid(cfg, t_raw, omega_raw)

    gap_frac = float(diag["gap_frac"])
    dt_jitter_rel = float(diag["dt_jitter_rel"])
    resample_factor = int(diag["resample_factor"])

    c5_violation = int(
        (gap_frac > cfg.gap_frac_max)
        or (dt_jitter_rel > cfg.dt_jitter_rel_max)
        or (resample_factor > cfg.resample_factor_max)
    )

    # Analysis grid time
    t0 = float(diag["t0"])
    t = t0 + np.arange(int(cfg.n_steps), dtype=float) * float(cfg.dt)

    # If malformed enough, return INCONCLUSIVE rather than BOUNDARY
    if omega_grid.size != int(cfg.n_steps):
        return {"case": name, "verdict": "INCONCLUSIVE", "reason": "grid_shape_mismatch", "c5_violation": c5_violation, **diag}

    rng = np.random.default_rng(cfg.seed + 99)
    omega_sur = _bootstrap_surrogates(cfg, omega_grid, rng=rng)
    if omega_sur.size == 0:
        return {"case": name, "verdict": "INCONCLUSIVE", "reason": "bootstrap_failed", "c5_violation": c5_violation, **diag}

    # Center global DC across pseudo-ensemble×time
    omega_sur = omega_sur - float(np.mean(omega_sur))

    dphi = delta_phi_from_omega(omega_sur, cfg.dt)
    k2 = k2_of_delta_phi(dphi)

    A = _fit_window(cfg, t, k2, cfg.tA_min, cfg.tA_max)
    B = _fit_window(cfg, t, k2, cfg.tB_min, cfg.tB_max)
    if (not A.get("ok")) or (not B.get("ok")):
        return {"case": name, "verdict": "INCONCLUSIVE", "reason": "fit_inadmissible", "c5_violation": c5_violation, "A": A, "B": B, **diag}

    alphaA = float(A["alpha"])
    alphaB = float(B["alpha"])
    alpha_cons_ok = (abs(alphaA - alphaB) <= float(cfg.alpha_consistency_max))

    maskA = _audit_mask(t, cfg.tA_min, cfg.tA_max)
    maskB = _audit_mask(t, cfg.tB_min, cfg.tB_max)

    curvA = guard_curvature_z(cfg, omega_sur, t, maskA)
    curvB = guard_curvature_z(cfg, omega_sur, t, maskB)
    driftA = guard_variance_drift_z(cfg, omega_sur, maskA)
    driftB = guard_variance_drift_z(cfg, omega_sur, maskB)
    etaA = guard_aperture_eta_max(cfg, k2, maskA)
    etaB = guard_aperture_eta_max(cfg, k2, maskB)

    viol_curv = int((np.isfinite(curvA) and curvA > cfg.curv_z_max) or (np.isfinite(curvB) and curvB > cfg.curv_z_max))
    viol_vardrift = int((np.isfinite(driftA) and driftA > cfg.var_drift_z_max) or (np.isfinite(driftB) and driftB > cfg.var_drift_z_max))
    viol_ap = int((np.isfinite(etaA) and etaA > 1.0) or (np.isfinite(etaB) and etaB > 1.0))

    guard_violation = int(c5_violation or viol_curv or viol_vardrift or viol_ap)

    bandA = str(A["band"])
    bandB = str(B["band"])

    # For a stream case, any C5 violation forces BOUNDARY (deployment refusal).
    if guard_violation or (not alpha_cons_ok) or (bandA != bandB):
        tag = "BOUNDARY"
        notes = "c5_or_guard_or_continuation_violation"
    else:
        if bandA == "OU":
            tag = "OK_OU"
            notes = "ok"
        elif bandA == "LM":
            tag = "OK_LM"
            notes = "ok"
        else:
            tag = "BOUNDARY"
            notes = "band_other"

    return {
        "case": name,
        "verdict": "OK",
        "A_alpha": alphaA,
        "A_r2": float(A["r2"]),
        "B_alpha": alphaB,
        "B_r2": float(B["r2"]),
        "bandA": bandA,
        "bandB": bandB,
        "alpha_cons_ok": int(alpha_cons_ok),
        "gap_frac": gap_frac,
        "dt_jitter_rel": dt_jitter_rel,
        "resample_factor": resample_factor,
        "c5_violation": int(c5_violation),
        "curv_z_max": float(np.nanmax([curvA, curvB])),
        "drift_z_max": float(np.nanmax([driftA, driftB])),
        "eta_max": float(np.nanmax([etaA, etaB])),
        "viol_curv": viol_curv,
        "viol_vardrift": viol_vardrift,
        "viol_ap": viol_ap,
        "tag": tag,
        "notes": notes,
    }


# -----------------------------
# Internal measurement-like construction (fixed)
# -----------------------------

def make_irregular_gappy_stream(cfg: S0026Config, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic “measurement-like” stream:
      - underlying ω generated from LM_TRUE (single trajectory)
      - timestamps jittered around uniform dt with fixed jitter fraction
      - random dropout creates gaps
    This case is expected to trigger C5 refusal (BOUNDARY) under declared thresholds.
    """
    n = int(cfg.n_steps)
    dt = float(cfg.dt)

    # Underlying omega (single trajectory) from fGn increments
    om = fgn_davies_harte(n, float(cfg.fgn_H), rng=rng).astype(float)

    # Jitter timestamps
    base_t = np.arange(n, dtype=float) * dt
    jitter = (rng.standard_normal(size=n).astype(float) * float(cfg.irregular_jitter_frac) * dt)
    t = base_t + jitter

    # Force monotone increasing by sorting with stable pairing (C5 realistic ordering)
    order = np.argsort(t)
    t = t[order]
    om = om[order]

    # Dropout
    keep = (rng.random(size=n) >= float(cfg.dropout_prob))
    t = t[keep]
    om = om[keep]

    return t, om


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="S-0026: Real-stream admissibility harness (C5 deployment gate).")
    p.add_argument("--seed", type=int, default=S0026Config.seed)
    p.add_argument("--input_csv", type=str, default="")
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0026Config(seed=int(args.seed))
    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)

    maskA = _audit_mask(t, cfg.tA_min, cfg.tA_max)
    maskB = _audit_mask(t, cfg.tB_min, cfg.tB_max)
    nA = int(np.count_nonzero(maskA))
    nB = int(np.count_nonzero(maskB))

    print(
        f"[S-0026] winA=[{cfg.tA_min:.1f},{cfg.tA_max:.1f}] nA={nA} "
        f"winB=[{cfg.tB_min:.1f},{cfg.tB_max:.1f}] nB={nB} dt={cfg.dt:.3g} "
        f"admiss(min_r2={cfg.min_r2:.4f}, min_k2_end={cfg.min_k2_end:g}, min_points={cfg.min_points}) "
        f"OU_band=[{cfg.ou_alpha_min:.2f},{cfg.ou_alpha_max:.2f}] "
        f"LM_band=[{cfg.lm_alpha_min:.2f},{cfg.lm_alpha_max:.2f}] (H={cfg.fgn_H:g}) "
        f"alpha_consistency_max={cfg.alpha_consistency_max:g} "
        f"C5(gap_frac_max={cfg.gap_frac_max:g}, dt_jitter_rel_max={cfg.dt_jitter_rel_max:g}, resample_factor_max={cfg.resample_factor_max}) "
        f"L={cfg.L:g} nT={cfg.n_trajectories} n_boot={cfg.n_boot}"
    )

    if (nA < int(cfg.min_points)) or (nB < int(cfg.min_points)):
        print("[S-0026] INCONCLUSIVE: audit windows have too few points on analysis grid.")
        return 3

    results: Dict[str, Dict[str, object]] = {}
    any_inconclusive = False

    # Baselines (always run)
    rng_ou = np.random.default_rng(cfg.seed + 1)
    rng_lm = np.random.default_rng(cfg.seed + 2)

    omega_ou = simulate_ou(cfg, rng=rng_ou)
    omega_lm = simulate_fgn(cfg, rng=rng_lm)

    for name, om in [("C1_OU_BASE", omega_ou), ("C2_LM_TRUE", omega_lm)]:
        r = audit_ensemble_case(cfg, name=name, omega=om, t=t)
        results[name] = r

    # Stream case: either external CSV or internal measurement-like construction
    if args.input_csv:
        path = Path(args.input_csv)
        tt, oo, err = _read_time_omega_csv(path)
        if err is not None:
            results["C5_INPUT_STREAM"] = {"case": "C5_INPUT_STREAM", "verdict": "INCONCLUSIVE", "reason": err}
        else:
            r = audit_stream_case(cfg, name="C5_INPUT_STREAM", t_raw=tt, omega_raw=oo)
            results["C5_INPUT_STREAM"] = r
    else:
        rng_stream = np.random.default_rng(cfg.seed + 3)
        tt, oo = make_irregular_gappy_stream(cfg, rng=rng_stream)
        r = audit_stream_case(cfg, name="C5_IRREGULAR_GAPS", t_raw=tt, omega_raw=oo)
        results["C5_IRREGULAR_GAPS"] = r

    # Print results
    for key in list(results.keys()):
        r = results[key]
        verdict = str(r.get("verdict", "INCONCLUSIVE"))
        if verdict != "OK":
            any_inconclusive = True
            print(f"[S-0026:{key}] verdict=INCONCLUSIVE reason={str(r.get('reason',''))}")
            continue

        tag = str(r.get("tag", ""))
        if key.startswith("C5_"):
            print(
                f"[S-0026:{key}] A(alpha={float(r.get('A_alpha', float('nan'))):.4g}, r2={float(r.get('A_r2', float('nan'))):.4g}, band={str(r.get('bandA',''))}) "
                f"B(alpha={float(r.get('B_alpha', float('nan'))):.4g}, r2={float(r.get('B_r2', float('nan'))):.4g}, band={str(r.get('bandB',''))}) "
                f"c5_violation={int(r.get('c5_violation', 0))} gap_frac={float(r.get('gap_frac', float('nan'))):.4g} "
                f"dt_jitter_rel={float(r.get('dt_jitter_rel', float('nan'))):.4g} resample_factor={int(r.get('resample_factor', 0))} "
                f"eta_max={float(r.get('eta_max', float('nan'))):.4g} tag={tag} notes={str(r.get('notes',''))}"
            )
        else:
            print(
                f"[S-0026:{key}] A(alpha={float(r.get('A_alpha', float('nan'))):.4g}, r2={float(r.get('A_r2', float('nan'))):.4g}, band={str(r.get('bandA',''))}, viol={int(r.get('viol_cpl',0))}{int(r.get('viol_curv',0))}{int(r.get('viol_vardrift',0))}{int(r.get('viol_ap',0))}) "
                f"B(alpha={float(r.get('B_alpha', float('nan'))):.4g}, r2={float(r.get('B_r2', float('nan'))):.4g}, band={str(r.get('bandB',''))}, viol={int(r.get('viol_cpl',0))}{int(r.get('viol_curv',0))}{int(r.get('viol_vardrift',0))}{int(r.get('viol_ap',0))}) "
                f"tag={tag} notes={str(r.get('notes',''))}"
            )

    # Decide audit pass/fail
    if any_inconclusive:
        print("[S-0026] INCONCLUSIVE")
        exit_code = 3
    else:
        ou_ok = (str(results["C1_OU_BASE"].get("tag")) == "OK_OU")
        lm_ok = (str(results["C2_LM_TRUE"].get("tag")) == "OK_LM")

        # Stream case must never produce OK_* if any C5/refusal triggers; we accept BOUNDARY as correct behavior.
        stream_key = "C5_INPUT_STREAM" if args.input_csv else "C5_IRREGULAR_GAPS"
        stream_tag = str(results[stream_key].get("tag", ""))
        forbidden_ok = stream_tag in ("OK_OU", "OK_LM")

        if forbidden_ok:
            print("[S-0026] AUDIT FAILED: stream harness produced forbidden OK_* under deployment conditions.")
            exit_code = 2
        elif ou_ok and lm_ok:
            print("[S-0026] AUDIT PASSED")
            exit_code = 0
        else:
            print(f"[S-0026] AUDIT FAILED: ou_ok={int(ou_ok)} lm_ok={int(lm_ok)} stream_tag={stream_tag}")
            exit_code = 2

    # Optional outputs
    if args.write_outputs:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        header_kv = {
            "git_commit": _git_commit_short(),
            "run_utc": _run_utc_iso(),
            "toy_file": Path(__file__).name,
            "source_sha256": _sha256_of_file(Path(__file__)),
        }

        rows = [[
            "case", "tag", "verdict",
            "A_alpha", "A_r2", "B_alpha", "B_r2",
            "bandA", "bandB", "alpha_cons_ok",
            "r_mean_max", "curv_z_max", "drift_z_max", "eta_max",
            "c5_violation", "gap_frac", "dt_jitter_rel", "resample_factor",
            "notes", "reason"
        ]]

        for nm, r in results.items():
            rows.append([
                nm,
                str(r.get("tag", "")),
                str(r.get("verdict", "")),
                r.get("A_alpha", ""),
                r.get("A_r2", ""),
                r.get("B_alpha", ""),
                r.get("B_r2", ""),
                str(r.get("bandA", "")),
                str(r.get("bandB", "")),
                r.get("alpha_cons_ok", ""),
                r.get("r_mean_max", ""),
                r.get("curv_z_max", ""),
                r.get("drift_z_max", ""),
                r.get("eta_max", ""),
                r.get("c5_violation", ""),
                r.get("gap_frac", ""),
                r.get("dt_jitter_rel", ""),
                r.get("resample_factor", ""),
                str(r.get("notes", "")),
                str(r.get("reason", "")),
            ])

        _write_csv_with_provenance_header(out_dir / cfg.out_cases_csv, header_kv, rows)
        _write_csv_with_provenance_header(
            out_dir / cfg.out_audit_csv,
            header_kv,
            [
                ["field", "value"],
                ["exit_code", exit_code],
                ["seed", int(cfg.seed)],
                ["dt", float(cfg.dt)],
                ["n_steps", int(cfg.n_steps)],
                ["n_trajectories", int(cfg.n_trajectories)],
                ["n_boot", int(cfg.n_boot)],
                ["boot_block", int(cfg.boot_block)],
                ["tA_min", float(cfg.tA_min)],
                ["tA_max", float(cfg.tA_max)],
                ["tB_min", float(cfg.tB_min)],
                ["tB_max", float(cfg.tB_max)],
                ["min_r2", float(cfg.min_r2)],
                ["min_k2_end", float(cfg.min_k2_end)],
                ["alpha_consistency_max", float(cfg.alpha_consistency_max)],
                ["gap_frac_max", float(cfg.gap_frac_max)],
                ["dt_jitter_rel_max", float(cfg.dt_jitter_rel_max)],
                ["resample_factor_max", int(cfg.resample_factor_max)],
            ],
        )

        print(f"Wrote (untracked): {out_dir / cfg.out_cases_csv}")
        print(f"Wrote (untracked): {out_dir / cfg.out_audit_csv}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
