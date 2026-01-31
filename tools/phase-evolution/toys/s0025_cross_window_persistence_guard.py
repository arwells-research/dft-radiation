#!/usr/bin/env python3
"""
S-0025 — Cross-window persistence of κ₂-slope regime admissibility (Σ₂ continuation gate)

Goal
----
Certify that κ₂-slope regime attribution is admissible only when it persists across
adjacent late-time continuation windows *and* all declared Σ₂ guards remain quiet in
both windows.

This is a continuation-level integrity test that closes the remaining hole:
a regime can appear admissible in one window but fail under continuation.

Core definitions
----------------
Phase accumulation:
    Δϕ_i[k] := sum_{j<=k} ω_i[j] * dt

Second cumulant:
    κ₂[k] := Var_i[Δϕ_i[k]]

κ₂-slope exponent per window:
    α := slope of log κ₂ vs log t over a declared window.

Bands (fixed):
    OU band: [ou_alpha_min, ou_alpha_max]
    LM band: [lm_alpha_min, lm_alpha_max] (centered near 2H for fixed H baseline)

Continuation windows (fixed):
    Window A: [tA_min, tA_max]
    Window B: [tB_min, tB_max]

Σ₂ guard bundle (fixed; window-restricted):
    1) Coupling (C4-style shared component):
         r_mean := Var_t( mean_i ω_i(t) ) / mean_i Var_t( ω_i(t) )
         coupling_violation := (r_mean > coupling_rmean_max)

    2) Temporal curvature (C1 nonstationary mean structure):
         fit ω̄(t) ≈ a + b t + c t² in-window; compute curv_z := |c| / SE(c)
         curvature_violation := (curv_z > curv_z_max)

    3) Variance drift (Σ₂ / C5 temporal nonstationarity of variance budget):
         v̂(t) := Var_i(ω_i(t))
         drift_z := |log(v̂(t_max)/v̂(t_min))| * sqrt(n_w)
         vardrift_violation := (drift_z > var_drift_z_max)

    4) Coherence aperture budget (DFT admissibility link, S-0021-style):
         σϕ(t) := sqrt(κ₂(t)), η(t) := σϕ(t)/L
         aperture_violation := (max_{t in window} η(t) > 1)

Certification rule (fixed)
--------------------------
A case is eligible for an OK_* tag only if:
  - both windows are admissible fits, AND
  - both windows classify into the same band (OU or LM), AND
  - |α_A - α_B| <= alpha_consistency_max, AND
  - no Σ₂ guard fires in either window.

If any Σ₂ guard fires in either window, tag is BOUNDARY (never OK_*).
If either window is inadmissible (fit/points/floor), verdict is INCONCLUSIVE (no claim).

Fixed subcases
--------------
- C1_OU_BASE:
    OU ω_i(t) independent across trajectories. Expected: OK_OU.

- C2_LM_TRUE:
    fGn ω_i(t) (Davies–Harte) with fixed H. Expected: OK_LM.

- C4_COUPLED:
    OU plus shared common component across trajectories. Expected: BOUNDARY (coupling).

- C1_CURVED:
    OU plus deterministic quadratic mean term. Expected: BOUNDARY (curvature).

- C5_APERTURE_EXCESS:
    LM_TRUE scaled by aperture_scale to force η>1 while preserving κ₂ band. Expected: BOUNDARY (aperture).

No scanning. No tuning. No post-hoc threshold adjustment.

Exit codes
----------
  0 PASS
  2 FAIL
  3 INCONCLUSIVE

Outputs (only with --write_outputs)
----------------------------------
  toys/outputs/s0025_cases.csv
  toys/outputs/s0025_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class S0025Config:
    seed: int = 25025

    dt: float = 0.20
    n_steps: int = 512
    n_trajectories: int = 1024

    # Continuation windows (absolute times)
    tA_min: float = 20.0
    tA_max: float = 40.0
    tB_min: float = 40.0
    tB_max: float = 80.0
    min_points: int = 24

    # Admissibility for κ2 fit (per window)
    min_r2: float = 0.985
    min_k2_end: float = 5.0e-3

    # Classification bands
    ou_alpha_min: float = 0.82
    ou_alpha_max: float = 1.18

    fgn_H: float = 0.75
    lm_alpha_min: float = 1.30
    lm_alpha_max: float = 1.70

    # Continuation consistency (fixed)
    alpha_consistency_max: float = 0.18

    # OU generator params (fixed)
    ou_theta: float = 0.60
    ou_mu: float = 0.0
    ou_sigma: float = 1.0

    # Guard thresholds (fixed)
    coupling_rmean_max: float = 0.02
    curv_z_max: float = 12.0
    var_drift_z_max: float = 2.5

    # Aperture L (declared, fixed)
    aperture_L: float = 20.0

    # Constructions (fixed; no tuning)
    coupled_amp: float = 0.35
    curved_c: float = 0.00035  # coefficient multiplying (t - t_mid)^2 added to ω
    aperture_scale: float = 2.0

    eps: float = 1e-12

    out_dir: str = "toys/outputs"
    out_cases_csv: str = "s0025_cases.csv"
    out_audit_csv: str = "s0025_audit.csv"


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


def _window_mask(t: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
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


def classify_alpha(cfg: S0025Config, alpha: float) -> str:
    if cfg.ou_alpha_min <= alpha <= cfg.ou_alpha_max:
        return "OU"
    if cfg.lm_alpha_min <= alpha <= cfg.lm_alpha_max:
        return "LM"
    return "BOUNDARY"


def simulate_ou(cfg: S0025Config, rng: np.random.Generator) -> np.ndarray:
    """
    Euler–Maruyama OU. Initialize from an approximate stationary distribution to avoid
    degenerate early-time variance (important for window-restricted guards).
    """
    nT = int(cfg.n_trajectories)
    nS = int(cfg.n_steps)
    dt = float(cfg.dt)

    theta = float(cfg.ou_theta)
    mu = float(cfg.ou_mu)
    sigma = float(cfg.ou_sigma)

    omega = np.empty((nT, nS), dtype=float)

    # Approx stationary variance for continuous OU: Var = sigma^2 / (2 theta)
    # This is diagnostic only; no claim of exact discrete stationarity is made.
    if theta > 0.0:
        std0 = float(sigma / np.sqrt(2.0 * theta))
    else:
        std0 = float(sigma)
    omega[:, 0] = mu + rng.standard_normal(size=nT).astype(float) * std0

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


def simulate_fgn(cfg: S0025Config, rng: np.random.Generator) -> np.ndarray:
    nT = int(cfg.n_trajectories)
    nS = int(cfg.n_steps)
    H = float(cfg.fgn_H)

    omega = np.empty((nT, nS), dtype=float)
    for i in range(nT):
        inc = fgn_davies_harte(nS, H, rng=rng)
        omega[i, :] = inc
    return omega


def apply_coupling(cfg: S0025Config, omega: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Add a shared common component across trajectories (fixed amplitude; no scanning).
    """
    x = np.asarray(omega, dtype=float).copy()
    common = rng.standard_normal(size=x.shape[1]).astype(float)
    common = common / float(np.std(common) + cfg.eps)
    x = x + float(cfg.coupled_amp) * common[None, :]
    return x


def apply_curvature(cfg: S0025Config, omega: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Add a deterministic quadratic mean term to all trajectories to trigger curvature guard.
    Construction is fixed: +c*(t - t_mid)^2.
    """
    x = np.asarray(omega, dtype=float).copy()
    t0 = 0.5 * (float(cfg.tA_min) + float(cfg.tB_max))
    quad = (np.asarray(t, dtype=float) - t0) ** 2
    x = x + float(cfg.curved_c) * quad[None, :]
    return x


def scale_for_aperture(cfg: S0025Config, omega: np.ndarray) -> np.ndarray:
    return np.asarray(omega, dtype=float) * float(cfg.aperture_scale)


def guard_coupling_rmean(cfg: S0025Config, omega: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
    x = np.asarray(omega, dtype=float)[:, mask]
    if x.size == 0:
        return float("nan"), 0
    mean_i = np.mean(x, axis=0)
    var_mean = float(np.var(mean_i, ddof=0))
    var_i = np.var(x, axis=1, ddof=0)
    denom = float(np.mean(var_i)) + float(cfg.eps)
    r_mean = var_mean / denom
    return float(r_mean), int(r_mean > float(cfg.coupling_rmean_max))


def guard_curvature_z(cfg: S0025Config, omega: np.ndarray, t: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
    x = np.asarray(omega, dtype=float)
    tt = np.asarray(t, dtype=float)[mask]
    if tt.size < 5:
        return float("nan"), 0
    mu = np.mean(x[:, mask], axis=0)  # ω̄(t) over window

    # Quadratic fit mu ≈ a + b t + c t^2
    A = np.vstack([np.ones_like(tt), tt, tt**2]).T
    coef, *_ = np.linalg.lstsq(A, mu, rcond=None)
    resid = mu - (A @ coef)

    n = int(tt.size)
    p = 3
    s2 = float(np.sum(resid**2) / max(n - p, 1))
    cov = s2 * np.linalg.inv(A.T @ A)
    se_c = float(np.sqrt(max(cov[2, 2], 0.0)))
    c = float(coef[2])

    curv_z = abs(c) / (se_c + float(cfg.eps))
    return float(curv_z), int(curv_z > float(cfg.curv_z_max))


def guard_variance_drift(cfg: S0025Config, omega: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
    x = np.asarray(omega, dtype=float)[:, mask]
    n_w = int(np.count_nonzero(mask))
    if n_w < 2 or x.size == 0:
        return float("nan"), 0
    v = np.var(x, axis=0, ddof=0)
    v0 = float(max(v[0], float(cfg.eps)))
    v1 = float(max(v[-1], float(cfg.eps)))
    drift_z = abs(float(np.log(v1 / v0))) * float(np.sqrt(n_w))
    return float(drift_z), int(drift_z > float(cfg.var_drift_z_max))


def guard_aperture(cfg: S0025Config, k2: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
    kk = np.asarray(k2, dtype=float)[mask]
    if kk.size == 0:
        return float("nan"), 0
    sigma_phi = np.sqrt(np.maximum(kk, 0.0))
    eta = sigma_phi / float(cfg.aperture_L)
    eta_max = float(np.max(eta))
    return float(eta_max), int(eta_max > 1.0)


def audit_window(
    cfg: S0025Config, omega: np.ndarray, t: np.ndarray, mask: np.ndarray
) -> Dict[str, object]:
    # Global DC removal only (per case)
    x = np.asarray(omega, dtype=float)
    x = x - float(np.mean(x))

    dphi = delta_phi_from_omega(x, cfg.dt)
    k2 = k2_of_delta_phi(dphi)

    if (not np.any(mask)) or (int(np.count_nonzero(mask)) < int(cfg.min_points)):
        return {"verdict": "INCONCLUSIVE", "reason": "too_few_points"}

    idx = np.where(mask)[0]
    k2_end = float(k2[int(idx[-1])])
    alpha, r2 = _linear_fit_loglog(t, k2, mask, eps=float(cfg.eps))

    if (not np.isfinite(alpha)) or (not np.isfinite(r2)) or (r2 < float(cfg.min_r2)) or (k2_end < float(cfg.min_k2_end)):
        return {
            "verdict": "INCONCLUSIVE",
            "reason": "inadmissible_fit",
            "alpha": float(alpha),
            "r2": float(r2),
            "k2_end": float(k2_end),
        }

    band = classify_alpha(cfg, float(alpha))

    r_mean, cpl = guard_coupling_rmean(cfg, x, mask)
    curv_z, curv = guard_curvature_z(cfg, x, t, mask)
    drift_z, vdr = guard_variance_drift(cfg, x, mask)
    eta_max, apv = guard_aperture(cfg, k2, mask)

    return {
        "verdict": "OK",
        "alpha": float(alpha),
        "r2": float(r2),
        "k2_end": float(k2_end),
        "band": str(band),
        "r_mean": float(r_mean),
        "coupling_violation": int(cpl),
        "curv_z": float(curv_z),
        "curvature_violation": int(curv),
        "drift_z": float(drift_z),
        "vardrift_violation": int(vdr),
        "eta_max": float(eta_max),
        "aperture_violation": int(apv),
        "any_guard_violation": int(bool(cpl or curv or vdr or apv)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="S-0025: cross-window persistence of κ2-slope regime admissibility.")
    p.add_argument("--seed", type=int, default=S0025Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0025Config(seed=int(args.seed))

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)

    maskA = _window_mask(t, cfg.tA_min, cfg.tA_max)
    maskB = _window_mask(t, cfg.tB_min, cfg.tB_max)

    nA = int(np.count_nonzero(maskA))
    nB = int(np.count_nonzero(maskB))

    print(
        f"[S-0025] winA=[{cfg.tA_min:.1f},{cfg.tA_max:.1f}] nA={nA} "
        f"winB=[{cfg.tB_min:.1f},{cfg.tB_max:.1f}] nB={nB} dt={cfg.dt:.3g} "
        f"admiss(min_r2={cfg.min_r2:.4f}, min_k2_end={cfg.min_k2_end:g}, min_points={cfg.min_points}) "
        f"OU_band=[{cfg.ou_alpha_min:.2f},{cfg.ou_alpha_max:.2f}] "
        f"LM_band=[{cfg.lm_alpha_min:.2f},{cfg.lm_alpha_max:.2f}] (H={cfg.fgn_H:g}) "
        f"alpha_consistency_max={cfg.alpha_consistency_max:.3g} "
        f"guards(rmean_max={cfg.coupling_rmean_max:g}, curv_z_max={cfg.curv_z_max:g}, "
        f"vardrift_z_max={cfg.var_drift_z_max:g}, L={cfg.aperture_L:g}) "
        f"construct(coupled_amp={cfg.coupled_amp:g}, curved_c={cfg.curved_c:g}, aperture_scale={cfg.aperture_scale:g}) "
        f"nT={cfg.n_trajectories}"
    )

    if (nA < int(cfg.min_points)) or (nB < int(cfg.min_points)):
        print("[S-0025] INCONCLUSIVE: continuation windows have too few points.")
        return 3

    rng_ou = np.random.default_rng(cfg.seed + 1)
    rng_lm = np.random.default_rng(cfg.seed + 2)
    rng_cpl = np.random.default_rng(cfg.seed + 3)

    omega_ou = simulate_ou(cfg, rng=rng_ou)
    omega_lm = simulate_fgn(cfg, rng=rng_lm)

    omega_cpl = apply_coupling(cfg, omega_ou, rng=rng_cpl)
    omega_curv = apply_curvature(cfg, omega_ou, t=t)
    omega_apx = scale_for_aperture(cfg, omega_lm)

    cases = [
        ("C1_OU_BASE", omega_ou),
        ("C2_LM_TRUE", omega_lm),
        ("C4_COUPLED", omega_cpl),
        ("C1_CURVED", omega_curv),
        ("C5_APERTURE_EXCESS", omega_apx),
    ]

    results: Dict[str, Dict[str, object]] = {}
    any_inconclusive = False

    def _tag_from_windows(rA: Dict[str, object], rB: Dict[str, object]) -> Tuple[str, str]:
        if (str(rA.get("verdict")) != "OK") or (str(rB.get("verdict")) != "OK"):
            return "INCONCLUSIVE", "inadmissible_fit"
        if int(rA.get("any_guard_violation", 0)) or int(rB.get("any_guard_violation", 0)):
            return "BOUNDARY", "guard_violation"

        bandA = str(rA.get("band", "BOUNDARY"))
        bandB = str(rB.get("band", "BOUNDARY"))
        aA = float(rA.get("alpha", float("nan")))
        aB = float(rB.get("alpha", float("nan")))

        if (bandA in ("OU", "LM")) and (bandA == bandB) and (abs(aA - aB) <= float(cfg.alpha_consistency_max)):
            return ("OK_OU" if bandA == "OU" else "OK_LM"), "ok"
        return "BOUNDARY", "continuation_mismatch"

    for name, om in cases:
        rA = audit_window(cfg, om, t=t, mask=maskA)
        rB = audit_window(cfg, om, t=t, mask=maskB)

        tag, notes = _tag_from_windows(rA, rB)

        results[name] = {
            "case": name,
            "tag": tag,
            "notes": notes,
            "A_verdict": rA.get("verdict"),
            "A_alpha": rA.get("alpha"),
            "A_r2": rA.get("r2"),
            "A_k2_end": rA.get("k2_end"),
            "A_band": rA.get("band"),
            "A_r_mean": rA.get("r_mean"),
            "A_curv_z": rA.get("curv_z"),
            "A_drift_z": rA.get("drift_z"),
            "A_eta_max": rA.get("eta_max"),
            "A_viol_cpl": rA.get("coupling_violation", 0),
            "A_viol_curv": rA.get("curvature_violation", 0),
            "A_viol_vdr": rA.get("vardrift_violation", 0),
            "A_viol_ap": rA.get("aperture_violation", 0),
            "B_verdict": rB.get("verdict"),
            "B_alpha": rB.get("alpha"),
            "B_r2": rB.get("r2"),
            "B_k2_end": rB.get("k2_end"),
            "B_band": rB.get("band"),
            "B_r_mean": rB.get("r_mean"),
            "B_curv_z": rB.get("curv_z"),
            "B_drift_z": rB.get("drift_z"),
            "B_eta_max": rB.get("eta_max"),
            "B_viol_cpl": rB.get("coupling_violation", 0),
            "B_viol_curv": rB.get("curvature_violation", 0),
            "B_viol_vdr": rB.get("vardrift_violation", 0),
            "B_viol_ap": rB.get("aperture_violation", 0),
        }

        if tag == "INCONCLUSIVE":
            any_inconclusive = True
            print(f"[S-0025:{name}] verdict=INCONCLUSIVE notes={notes}")
            continue

        print(
            f"[S-0025:{name}] "
            f"A(alpha={float(results[name]['A_alpha']):.4g}, r2={float(results[name]['A_r2']):.4g}, band={results[name]['A_band']}, "
            f"viol={int(results[name]['A_viol_cpl'])}{int(results[name]['A_viol_curv'])}{int(results[name]['A_viol_vdr'])}{int(results[name]['A_viol_ap'])}) "
            f"B(alpha={float(results[name]['B_alpha']):.4g}, r2={float(results[name]['B_r2']):.4g}, band={results[name]['B_band']}, "
            f"viol={int(results[name]['B_viol_cpl'])}{int(results[name]['B_viol_curv'])}{int(results[name]['B_viol_vdr'])}{int(results[name]['B_viol_ap'])}) "
            f"tag={tag} notes={notes}"
        )

    if any_inconclusive:
        print("[S-0025] INCONCLUSIVE")
        exit_code = 3
    else:
        ou_ok = str(results["C1_OU_BASE"]["tag"]) == "OK_OU"
        lm_ok = str(results["C2_LM_TRUE"]["tag"]) == "OK_LM"

        cpl_ok = str(results["C4_COUPLED"]["tag"]) == "BOUNDARY"
        curv_ok = str(results["C1_CURVED"]["tag"]) == "BOUNDARY"
        apx_ok = str(results["C5_APERTURE_EXCESS"]["tag"]) == "BOUNDARY"

        # Forbidden: any constructed violation yields OK_*
        forbidden = False
        for nm in ["C4_COUPLED", "C1_CURVED", "C5_APERTURE_EXCESS"]:
            if str(results[nm]["tag"]) in ("OK_OU", "OK_LM"):
                forbidden = True

        if forbidden:
            print("[S-0025] AUDIT FAILED: forbidden OK_* under guard violations / continuation mismatch.")
            exit_code = 2
        elif ou_ok and lm_ok and cpl_ok and curv_ok and apx_ok:
            print("[S-0025] AUDIT PASSED")
            exit_code = 0
        else:
            print(
                "[S-0025] AUDIT FAILED: "
                f"ou_ok={int(ou_ok)} lm_ok={int(lm_ok)} "
                f"coupled_ok={int(cpl_ok)} curved_ok={int(curv_ok)} ap_ok={int(apx_ok)}"
            )
            exit_code = 2

    if args.write_outputs:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        header_kv = {
            "git_commit": _git_commit_short(),
            "run_utc": _run_utc_iso(),
            "toy_file": Path(__file__).name,
            "source_sha256": _sha256_of_file(Path(__file__)),
        }

        rows = [
            [
                "case",
                "tag",
                "notes",
                "A_alpha",
                "A_r2",
                "A_band",
                "A_r_mean",
                "A_curv_z",
                "A_drift_z",
                "A_eta_max",
                "A_viol_cpl",
                "A_viol_curv",
                "A_viol_vdr",
                "A_viol_ap",
                "B_alpha",
                "B_r2",
                "B_band",
                "B_r_mean",
                "B_curv_z",
                "B_drift_z",
                "B_eta_max",
                "B_viol_cpl",
                "B_viol_curv",
                "B_viol_vdr",
                "B_viol_ap",
            ]
        ]

        for nm in ["C1_OU_BASE", "C2_LM_TRUE", "C4_COUPLED", "C1_CURVED", "C5_APERTURE_EXCESS"]:
            r = results[nm]
            rows.append(
                [
                    nm,
                    str(r.get("tag", "")),
                    str(r.get("notes", "")),
                    float(r.get("A_alpha", float("nan"))) if r.get("A_alpha") is not None else "",
                    float(r.get("A_r2", float("nan"))) if r.get("A_r2") is not None else "",
                    str(r.get("A_band", "")),
                    float(r.get("A_r_mean", float("nan"))) if r.get("A_r_mean") is not None else "",
                    float(r.get("A_curv_z", float("nan"))) if r.get("A_curv_z") is not None else "",
                    float(r.get("A_drift_z", float("nan"))) if r.get("A_drift_z") is not None else "",
                    float(r.get("A_eta_max", float("nan"))) if r.get("A_eta_max") is not None else "",
                    int(r.get("A_viol_cpl", 0)),
                    int(r.get("A_viol_curv", 0)),
                    int(r.get("A_viol_vdr", 0)),
                    int(r.get("A_viol_ap", 0)),
                    float(r.get("B_alpha", float("nan"))) if r.get("B_alpha") is not None else "",
                    float(r.get("B_r2", float("nan"))) if r.get("B_r2") is not None else "",
                    str(r.get("B_band", "")),
                    float(r.get("B_r_mean", float("nan"))) if r.get("B_r_mean") is not None else "",
                    float(r.get("B_curv_z", float("nan"))) if r.get("B_curv_z") is not None else "",
                    float(r.get("B_drift_z", float("nan"))) if r.get("B_drift_z") is not None else "",
                    float(r.get("B_eta_max", float("nan"))) if r.get("B_eta_max") is not None else "",
                    int(r.get("B_viol_cpl", 0)),
                    int(r.get("B_viol_curv", 0)),
                    int(r.get("B_viol_vdr", 0)),
                    int(r.get("B_viol_ap", 0)),
                ]
            )

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
                ["tA_min", float(cfg.tA_min)],
                ["tA_max", float(cfg.tA_max)],
                ["tB_min", float(cfg.tB_min)],
                ["tB_max", float(cfg.tB_max)],
                ["min_r2", float(cfg.min_r2)],
                ["min_k2_end", float(cfg.min_k2_end)],
                ["alpha_consistency_max", float(cfg.alpha_consistency_max)],
                ["coupling_rmean_max", float(cfg.coupling_rmean_max)],
                ["curv_z_max", float(cfg.curv_z_max)],
                ["var_drift_z_max", float(cfg.var_drift_z_max)],
                ["aperture_L", float(cfg.aperture_L)],
                ["coupled_amp", float(cfg.coupled_amp)],
                ["curved_c", float(cfg.curved_c)],
                ["aperture_scale", float(cfg.aperture_scale)],
            ],
        )

        print(f"Wrote (untracked): {out_dir / cfg.out_cases_csv}")
        print(f"Wrote (untracked): {out_dir / cfg.out_audit_csv}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
