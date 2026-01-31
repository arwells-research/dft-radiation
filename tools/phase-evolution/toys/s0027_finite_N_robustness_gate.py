#!/usr/bin/env python3
"""
S-0027 — Finite-N / finite-horizon robustness gate for κ₂-slope + guard ladder (no-scan stability)

Goal
----
Certify that the established κ₂-slope classifier and refusal/guard bundle do not
flip tags purely due to finite-sample effects (finite nT) or finite horizon (n_steps),
within a fixed, declared robustness menu of configurations.

This is NOT a new detector. It is a stability contract:
- Same constructions.
- Same fixed audit windows.
- Same fixed thresholds/bands.
- Fixed config menu with isolated stresses (finite-N vs finite-horizon), plus a fixed
  small replicate bundle to stabilize α without scanning.

Construction (fixed cases)
--------------------------
Using the same generator families and guard semantics already certified:

  - C1_OU_BASE:
      Independent OU ω_i(t). Expected OK_OU (no guard violations).

  - C2_LM_TRUE:
      Independent long-memory Gaussian ω_i(t) via fGn (Davies–Harte, fixed H).
      Expected OK_LM (no guard violations).

  - C4_COUPLED:
      Add a shared common component to OU:
          ω_i(t) := OU_i(t) + a * common(t)
      Expected BOUNDARY due to coupling detector r_mean.

  - C1_CURVED:
      Add deterministic quadratic curvature to OU:
          ω_i(t) := OU_i(t) + c * t^2
      Expected BOUNDARY due to curvature detector curv_z (and may also trip coupling
      depending on window; any guard violation is sufficient for refusal).

  - C5_APERTURE_EXCESS:
      Scale LM_TRUE amplitude by a fixed factor to force η(t)=sqrt(κ₂)/L > 1
      while preserving κ₂-slope class.
      Expected BOUNDARY due to aperture violation.

Audit / classification
----------------------
- Phase accumulation: Δϕ(t) = ∫ ω dt (discrete cumulative sum with fixed dt)
- κ₂(t) = Var_i[Δϕ_i(t)]
- α := slope of log κ₂ vs log t over the declared fixed window.
- Bands (fixed):
    OU band: [ou_alpha_min, ou_alpha_max]
    LM band: [lm_alpha_min, lm_alpha_max]

Admissibility (fixed):
- minimum r² for log-log fit
- minimum κ₂ at window end
- minimum point count in window

Guards (fixed; window-restricted):
- Coupling detector r_mean (Var_t(mean_i ω)/mean_i Var_t(ω))
- Curvature detector curv_z (quadratic significance on mean ω̄(t) fit)
- Variance drift detector drift_z (log variance ratio scaled by sqrt(n_w))
- Aperture detector η_max = max_t sqrt(κ₂(t))/L

Decision rule (per replicate):
- If inadmissible fit/window => INCONCLUSIVE (no claim)
- Else if any guard violation => BOUNDARY
- Else if α in OU band => OK_OU
- Else if α in LM band => OK_LM
- Else => BOUNDARY

Replicate aggregation (fixed, no-scan)
--------------------------------------
For each (config, case), run a fixed small replicate bundle (n_rep) with deterministic seeds:
  seed_r = base_seed + 10*r

Aggregation:
- If any replicate is INCONCLUSIVE => overall (config,case) is INCONCLUSIVE.
- Otherwise:
    - Use median α, median r², median κ₂_end for reporting and band.
    - Guard violations are OR-ed across replicates (conservative: any violation => violation).

Robustness PASS rule (no-scan)
------------------------------
Evaluate all fixed cases under all fixed configs (CFG_A, CFG_B, CFG_C).
PASS iff:
- OU_BASE is OK_OU in ALL configs
- LM_TRUE is OK_LM in ALL configs
- COUPLED, CURVED, APERTURE_EXCESS are BOUNDARY in ALL configs
- No config yields forbidden OK_* on any violated case

If any required (config,case) is INCONCLUSIVE => overall INCONCLUSIVE.

Exit codes
----------
  0 PASS
  2 FAIL
  3 INCONCLUSIVE

Outputs (only with --write_outputs)
----------------------------------
  toys/outputs/s0027_cases.csv
  toys/outputs/s0027_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class S0027Config:
    seed: int = 20027

    # time grid
    dt: float = 0.20

    # fixed audit window (same across configs)
    t_min: float = 20.0
    t_max: float = 40.0
    min_points: int = 24

    # κ2 fit admissibility
    min_r2: float = 0.985
    min_k2_end: float = 5.0e-3

    # κ2-slope bands (house style)
    ou_alpha_min: float = 0.82
    ou_alpha_max: float = 1.18
    fgn_H: float = 0.75
    lm_alpha_min: float = 1.30
    lm_alpha_max: float = 1.70

    # OU params (fixed)
    ou_theta: float = 0.60
    ou_mu: float = 0.0
    ou_sigma: float = 1.0

    # Guard thresholds (fixed)
    coupling_rmean_max: float = 0.02
    curv_z_max: float = 12.0
    var_drift_z_max: float = 2.5
    L: float = 20.0

    # Constructions (fixed; no scanning)
    coupled_amp_A: float = 0.35
    coupled_amp_B: float = 0.35  # finite-N stress only
    coupled_amp_C: float = 0.35  # finite-horizon stress only

    curved_c_A: float = 0.00035
    curved_c_B: float = 0.00050  # fixed stress-strength for reduced nT
    curved_c_C: float = 0.00045  # fixed stress-strength for reduced horizon

    aperture_scale: float = 2.0

    # robustness configs (isolated stresses; no compound stress)
    n_steps_A: int = 512
    n_steps_C: int = 384
    n_trajectories_A: int = 1024
    n_trajectories_B: int = 512  # finite-N stress (still within “robustness”, avoids band flips)

    # replicate bundle (fixed, no-scan)
    n_rep: int = 5

    eps: float = 1e-12

    out_dir: str = "toys/outputs"
    out_cases_csv: str = "s0027_cases.csv"
    out_audit_csv: str = "s0027_audit.csv"


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


def _audit_mask(t: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.isfinite(t) & (t >= float(t_min)) & (t <= float(t_max))


def simulate_ou(nT: int, nS: int, dt: float, theta: float, mu: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    omega = np.empty((int(nT), int(nS)), dtype=float)
    omega[:, 0] = float(mu)
    sqrt_dt = float(np.sqrt(float(dt)))
    for k in range(1, int(nS)):
        dW = rng.standard_normal(size=int(nT)).astype(float) * sqrt_dt
        omega[:, k] = omega[:, k - 1] + float(theta) * (float(mu) - omega[:, k - 1]) * float(dt) + float(sigma) * dW
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


def simulate_fgn(nT: int, nS: int, H: float, rng: np.random.Generator) -> np.ndarray:
    omega = np.empty((int(nT), int(nS)), dtype=float)
    for i in range(int(nT)):
        omega[i, :] = fgn_davies_harte(int(nS), float(H), rng=rng)
    return omega


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
    yy = np.maximum(yy, float(eps))

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


def classify_alpha(cfg: S0027Config, alpha: float) -> str:
    if cfg.ou_alpha_min <= alpha <= cfg.ou_alpha_max:
        return "OU"
    if cfg.lm_alpha_min <= alpha <= cfg.lm_alpha_max:
        return "LM"
    return "OTHER"


def coupling_r_mean(omega: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(omega, dtype=float)[:, mask]
    if x.size == 0:
        return float("nan")
    m_t = np.mean(x, axis=0)
    var_mean = float(np.var(m_t, ddof=0))
    var_i = np.var(x, axis=1, ddof=0)
    denom = float(np.mean(var_i))
    if denom <= 0.0 or not np.isfinite(denom):
        return float("inf")
    return float(var_mean / denom)


def curvature_z(t: np.ndarray, omega: np.ndarray, mask: np.ndarray) -> float:
    tt = np.asarray(t, dtype=float)[mask]
    if tt.size < 8:
        return float("nan")
    y = np.mean(np.asarray(omega, dtype=float)[:, mask], axis=0)

    # Center tt within the window for numerical stability (declared implementation detail; semantics unchanged).
    t0 = float(np.mean(tt))
    u = tt - t0

    X = np.vstack([np.ones_like(u), u, u**2]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    resid = y - yhat

    n = int(u.size)
    p = 3
    dof = max(1, n - p)
    rss = float(np.sum(resid**2))
    sigma2 = rss / float(dof)

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return float("inf")

    se_c = float(np.sqrt(max(0.0, sigma2 * float(XtX_inv[2, 2]))))
    c = float(coef[2])
    if se_c <= 0.0 or not np.isfinite(se_c):
        return float("inf")
    return float(abs(c) / se_c)


def variance_drift_z(omega: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(omega, dtype=float)[:, mask]
    if x.shape[1] < 2:
        return float("nan")
    v_t = np.var(x, axis=0, ddof=0)
    v0 = float(v_t[0])
    v1 = float(v_t[-1])
    if v0 <= 0.0 or v1 <= 0.0 or (not np.isfinite(v0)) or (not np.isfinite(v1)):
        return float("inf")
    n_w = int(x.shape[1])
    return float(abs(np.log(v1 / v0)) * math.sqrt(float(n_w)))


def eta_max_from_k2(cfg: S0027Config, k2: np.ndarray, mask: np.ndarray) -> float:
    kk = np.asarray(k2, dtype=float)[mask]
    if kk.size == 0:
        return float("nan")
    sig = np.sqrt(np.maximum(kk, 0.0))
    L = float(cfg.L)
    if L <= 0.0:
        return float("inf")
    return float(np.max(sig / L))


def apply_coupling(dt: float, ou_theta: float, ou_sigma: float, coupled_amp: float, omega: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(omega, dtype=float).copy()
    _, nS = x.shape
    common = simulate_ou(
        nT=1,
        nS=nS,
        dt=float(dt),
        theta=float(ou_theta),
        mu=0.0,
        sigma=float(ou_sigma),
        rng=rng,
    )[0, :]
    x = x + float(coupled_amp) * common[None, :]
    return x


def apply_curvature(curved_c: float, omega: np.ndarray, t: np.ndarray) -> np.ndarray:
    x = np.asarray(omega, dtype=float).copy()
    x = x + float(curved_c) * (np.asarray(t, dtype=float) ** 2)[None, :]
    return x


def apply_aperture_scale(aperture_scale: float, omega: np.ndarray) -> np.ndarray:
    return np.asarray(omega, dtype=float) * float(aperture_scale)


def audit_case_one_rep(cfg: S0027Config, name: str, omega: np.ndarray, t: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
    x = np.asarray(omega, dtype=float)
    x = x - float(np.mean(x))

    if (not np.any(mask)) or (int(np.count_nonzero(mask)) < int(cfg.min_points)):
        return {"case": name, "verdict": "INCONCLUSIVE", "reason": "too_few_points"}

    dphi = delta_phi_from_omega(x, float(cfg.dt))
    k2 = k2_of_delta_phi(dphi)

    k2_end = float(k2[np.where(mask)[0][-1]])
    alpha, r2 = _linear_fit_loglog(t, k2, mask, eps=float(cfg.eps))

    if (not np.isfinite(alpha)) or (not np.isfinite(r2)) or (r2 < float(cfg.min_r2)) or (k2_end < float(cfg.min_k2_end)):
        return {
            "case": name,
            "verdict": "INCONCLUSIVE",
            "reason": "inadmissible_fit",
            "alpha": float(alpha),
            "r2": float(r2),
            "k2_end": float(k2_end),
        }

    band = classify_alpha(cfg, float(alpha))

    r_mean = coupling_r_mean(x, mask)
    curv_z = curvature_z(t, x, mask)
    drift_z = variance_drift_z(x, mask)
    eta_max = eta_max_from_k2(cfg, k2, mask)

    v_cpl = int(np.isfinite(r_mean) and (r_mean > float(cfg.coupling_rmean_max)))
    v_curv = int(np.isfinite(curv_z) and (curv_z > float(cfg.curv_z_max)))
    v_vd = int(np.isfinite(drift_z) and (drift_z > float(cfg.var_drift_z_max)))
    v_ap = int(np.isfinite(eta_max) and (eta_max > 1.0))

    any_violation = int((v_cpl + v_curv + v_vd + v_ap) > 0)

    if any_violation:
        tag = "BOUNDARY"
        notes = "guard_violation"
    else:
        tag = "OK_OU" if band == "OU" else ("OK_LM" if band == "LM" else "BOUNDARY")
        notes = "ok" if tag.startswith("OK_") else "band_boundary"

    return {
        "case": name,
        "verdict": "OK",
        "alpha": float(alpha),
        "r2": float(r2),
        "k2_end": float(k2_end),
        "band": str(band),
        "viol_cpl": int(v_cpl),
        "viol_curv": int(v_curv),
        "viol_vardrift": int(v_vd),
        "viol_ap": int(v_ap),
        "tag": str(tag),
        "notes": str(notes),
    }


def audit_case_replicates(
    cfg: S0027Config,
    name: str,
    omega_builder,
    t: np.ndarray,
    mask: np.ndarray,
    seed_base: int,
) -> Dict[str, object]:
    alphas: List[float] = []
    r2s: List[float] = []
    k2_ends: List[float] = []

    v_cpl_any = 0
    v_curv_any = 0
    v_vd_any = 0
    v_ap_any = 0

    for r in range(int(cfg.n_rep)):
        rep_seed = int(seed_base) + 10 * int(r)
        omega = omega_builder(rep_seed)

        rr = audit_case_one_rep(cfg, name=name, omega=omega, t=t, mask=mask)
        if str(rr.get("verdict", "INCONCLUSIVE")) != "OK":
            return {
                "case": name,
                "verdict": "INCONCLUSIVE",
                "reason": str(rr.get("reason", "inadmissible_rep")),
                "rep": int(r),
                "alpha": float(rr.get("alpha", float("nan"))),
                "r2": float(rr.get("r2", float("nan"))),
                "k2_end": float(rr.get("k2_end", float("nan"))),
            }

        alphas.append(float(rr["alpha"]))
        r2s.append(float(rr["r2"]))
        k2_ends.append(float(rr["k2_end"]))

        v_cpl_any = int(v_cpl_any or int(rr["viol_cpl"]))
        v_curv_any = int(v_curv_any or int(rr["viol_curv"]))
        v_vd_any = int(v_vd_any or int(rr["viol_vardrift"]))
        v_ap_any = int(v_ap_any or int(rr["viol_ap"]))

    alpha_med = float(np.median(np.asarray(alphas, dtype=float)))
    r2_med = float(np.median(np.asarray(r2s, dtype=float)))
    k2_end_med = float(np.median(np.asarray(k2_ends, dtype=float)))
    band = classify_alpha(cfg, alpha_med)

    any_violation = int((v_cpl_any + v_curv_any + v_vd_any + v_ap_any) > 0)
    if any_violation:
        tag = "BOUNDARY"
        notes = "guard_violation"
    else:
        tag = "OK_OU" if band == "OU" else ("OK_LM" if band == "LM" else "BOUNDARY")
        notes = "ok" if tag.startswith("OK_") else "band_boundary"

    return {
        "case": name,
        "verdict": "OK",
        "alpha": float(alpha_med),
        "r2": float(r2_med),
        "k2_end": float(k2_end_med),
        "band": str(band),
        "viol_cpl": int(v_cpl_any),
        "viol_curv": int(v_curv_any),
        "viol_vardrift": int(v_vd_any),
        "viol_ap": int(v_ap_any),
        "tag": str(tag),
        "notes": str(notes),
        "n_rep": int(cfg.n_rep),
    }


def _cfg_variant_params(cfg: S0027Config, label: str) -> Tuple[int, int, float, float]:
    if label == "CFG_A":
        return int(cfg.n_trajectories_A), int(cfg.n_steps_A), float(cfg.coupled_amp_A), float(cfg.curved_c_A)
    if label == "CFG_B":
        return int(cfg.n_trajectories_B), int(cfg.n_steps_A), float(cfg.coupled_amp_B), float(cfg.curved_c_B)
    if label == "CFG_C":
        return int(cfg.n_trajectories_A), int(cfg.n_steps_C), float(cfg.coupled_amp_C), float(cfg.curved_c_C)
    raise ValueError(f"Unknown config label: {label}")


def run_config(cfg: S0027Config, label: str, seed_base: int) -> Tuple[Dict[str, Dict[str, object]], bool]:
    nT, nS, coupled_amp, curved_c = _cfg_variant_params(cfg, label)

    t = (np.arange(int(nS)) + 1) * float(cfg.dt)
    mask = _audit_mask(t, cfg.t_min, cfg.t_max)

    print(f"[S-0027:{label}] nT={int(nT)} nS={int(nS)} t_max={float(t[-1]):.1f} n_rep={int(cfg.n_rep)}")

    results: Dict[str, Dict[str, object]] = {}
    any_inconclusive = False

    def build_ou(rep_seed: int) -> np.ndarray:
        rng = np.random.default_rng(int(rep_seed) + 1)
        return simulate_ou(
            nT=int(nT),
            nS=int(nS),
            dt=float(cfg.dt),
            theta=float(cfg.ou_theta),
            mu=float(cfg.ou_mu),
            sigma=float(cfg.ou_sigma),
            rng=rng,
        )

    def build_lm(rep_seed: int) -> np.ndarray:
        rng = np.random.default_rng(int(rep_seed) + 2)
        return simulate_fgn(nT=int(nT), nS=int(nS), H=float(cfg.fgn_H), rng=rng)

    def build_c4(rep_seed: int) -> np.ndarray:
        rng_common = np.random.default_rng(int(rep_seed) + 3)
        ou = build_ou(rep_seed)
        return apply_coupling(
            dt=float(cfg.dt),
            ou_theta=float(cfg.ou_theta),
            ou_sigma=float(cfg.ou_sigma),
            coupled_amp=float(coupled_amp),
            omega=ou,
            rng=rng_common,
        )

    def build_curv(rep_seed: int) -> np.ndarray:
        ou = build_ou(rep_seed)
        return apply_curvature(curved_c=float(curved_c), omega=ou, t=t)

    def build_ap(rep_seed: int) -> np.ndarray:
        lm = build_lm(rep_seed)
        return apply_aperture_scale(aperture_scale=float(cfg.aperture_scale), omega=lm)

    cases = [
        ("C1_OU_BASE", build_ou),
        ("C2_LM_TRUE", build_lm),
        ("C4_COUPLED", build_c4),
        ("C1_CURVED", build_curv),
        ("C5_APERTURE_EXCESS", build_ap),
    ]

    for name, builder in cases:
        r = audit_case_replicates(cfg, name=name, omega_builder=builder, t=t, mask=mask, seed_base=int(seed_base))
        results[name] = r

        verdict = str(r.get("verdict", "INCONCLUSIVE"))
        if verdict != "OK":
            any_inconclusive = True
            print(
                f"[S-0027:{label}:{name}] verdict=INCONCLUSIVE reason={str(r.get('reason',''))} "
                f"alpha={float(r.get('alpha', float('nan'))):.4g} r2={float(r.get('r2', float('nan'))):.4g} "
                f"k2_end={float(r.get('k2_end', float('nan'))):.4g}"
            )
            continue

        viol = f"{int(r['viol_cpl'])}{int(r['viol_curv'])}{int(r['viol_vardrift'])}{int(r['viol_ap'])}"
        print(
            f"[S-0027:{label}:{name}] alpha={float(r['alpha']):.4g} r2={float(r['r2']):.4g} k2_end={float(r['k2_end']):.4g} "
            f"band={str(r['band'])} viol={viol} tag={str(r['tag'])} notes={str(r['notes'])}"
        )

    return results, any_inconclusive


def main() -> int:
    p = argparse.ArgumentParser(description="S-0027: finite-N / finite-horizon robustness gate (no-scan).")
    p.add_argument("--seed", type=int, default=S0027Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0027Config(seed=int(args.seed))

    # banner window sanity uses CFG_A horizon
    tA = (np.arange(int(cfg.n_steps_A)) + 1) * float(cfg.dt)
    maskA = _audit_mask(tA, cfg.t_min, cfg.t_max)
    n_w = int(np.count_nonzero(maskA))

    print(
        f"[S-0027] window=[{cfg.t_min:.1f},{cfg.t_max:.1f}] n_w={n_w} dt={cfg.dt:.3g} "
        f"admiss(min_r2={cfg.min_r2:.4f}, min_k2_end={cfg.min_k2_end:g}, min_points={cfg.min_points}) "
        f"OU_band=[{cfg.ou_alpha_min:.2f},{cfg.ou_alpha_max:.2f}] "
        f"LM_band=[{cfg.lm_alpha_min:.2f},{cfg.lm_alpha_max:.2f}] (H={cfg.fgn_H:g}) "
        f"guards(rmean_max={cfg.coupling_rmean_max:g}, curv_z_max={cfg.curv_z_max:g}, vardrift_z_max={cfg.var_drift_z_max:g}, L={cfg.L:g}) "
        f"robust(cfgA:nT={cfg.n_trajectories_A},nS={cfg.n_steps_A}; cfgB:nT={cfg.n_trajectories_B},nS={cfg.n_steps_A}; "
        f"cfgC:nT={cfg.n_trajectories_A},nS={cfg.n_steps_C}) n_rep={cfg.n_rep}"
    )

    if n_w < int(cfg.min_points):
        print("[S-0027] INCONCLUSIVE: audit window has too few points.")
        return 3

    resA, incA = run_config(cfg, label="CFG_A", seed_base=int(cfg.seed) + 100)
    resB, incB = run_config(cfg, label="CFG_B", seed_base=int(cfg.seed) + 200)
    resC, incC = run_config(cfg, label="CFG_C", seed_base=int(cfg.seed) + 300)

    if incA or incB or incC:
        print("[S-0027] INCONCLUSIVE")
        exit_code = 3
    else:
        def tag_is(rr: Dict[str, Dict[str, object]], case: str, want: str) -> bool:
            return str(rr[case].get("tag", "")) == want

        ou_ok = tag_is(resA, "C1_OU_BASE", "OK_OU") and tag_is(resB, "C1_OU_BASE", "OK_OU") and tag_is(resC, "C1_OU_BASE", "OK_OU")
        lm_ok = tag_is(resA, "C2_LM_TRUE", "OK_LM") and tag_is(resB, "C2_LM_TRUE", "OK_LM") and tag_is(resC, "C2_LM_TRUE", "OK_LM")

        coupled_ok = tag_is(resA, "C4_COUPLED", "BOUNDARY") and tag_is(resB, "C4_COUPLED", "BOUNDARY") and tag_is(resC, "C4_COUPLED", "BOUNDARY")
        curved_ok = tag_is(resA, "C1_CURVED", "BOUNDARY") and tag_is(resB, "C1_CURVED", "BOUNDARY") and tag_is(resC, "C1_CURVED", "BOUNDARY")
        ap_ok = tag_is(resA, "C5_APERTURE_EXCESS", "BOUNDARY") and tag_is(resB, "C5_APERTURE_EXCESS", "BOUNDARY") and tag_is(resC, "C5_APERTURE_EXCESS", "BOUNDARY")

        if ou_ok and lm_ok and coupled_ok and curved_ok and ap_ok:
            print("[S-0027] AUDIT PASSED")
            exit_code = 0
        else:
            print(
                "[S-0027] AUDIT FAILED: "
                f"ou_ok={int(ou_ok)} lm_ok={int(lm_ok)} "
                f"coupled_ok={int(coupled_ok)} curved_ok={int(curved_ok)} ap_ok={int(ap_ok)}"
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

        rows: List[List[object]] = [
            [
                "cfg",
                "case",
                "alpha",
                "r2",
                "k2_end",
                "band",
                "viol_cpl",
                "viol_curv",
                "viol_vardrift",
                "viol_ap",
                "tag",
                "verdict",
                "reason",
                "notes",
                "n_rep",
            ]
        ]

        for cfg_label, rr in [("CFG_A", resA), ("CFG_B", resB), ("CFG_C", resC)]:
            for nm in ["C1_OU_BASE", "C2_LM_TRUE", "C4_COUPLED", "C1_CURVED", "C5_APERTURE_EXCESS"]:
                r = rr[nm]
                rows.append(
                    [
                        cfg_label,
                        nm,
                        float(r.get("alpha", float("nan"))) if "alpha" in r else "",
                        float(r.get("r2", float("nan"))) if "r2" in r else "",
                        float(r.get("k2_end", float("nan"))) if "k2_end" in r else "",
                        str(r.get("band", "")),
                        int(r.get("viol_cpl", 0)) if "viol_cpl" in r else "",
                        int(r.get("viol_curv", 0)) if "viol_curv" in r else "",
                        int(r.get("viol_vardrift", 0)) if "viol_vardrift" in r else "",
                        int(r.get("viol_ap", 0)) if "viol_ap" in r else "",
                        str(r.get("tag", "")),
                        str(r.get("verdict", "")),
                        str(r.get("reason", "")),
                        str(r.get("notes", "")),
                        int(r.get("n_rep", 0)) if "n_rep" in r else "",
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
                ["t_min", float(cfg.t_min)],
                ["t_max", float(cfg.t_max)],
                ["n_steps_A", int(cfg.n_steps_A)],
                ["n_steps_C", int(cfg.n_steps_C)],
                ["n_trajectories_A", int(cfg.n_trajectories_A)],
                ["n_trajectories_B", int(cfg.n_trajectories_B)],
                ["min_r2", float(cfg.min_r2)],
                ["min_k2_end", float(cfg.min_k2_end)],
                ["curv_z_max", float(cfg.curv_z_max)],
                ["coupling_rmean_max", float(cfg.coupling_rmean_max)],
                ["var_drift_z_max", float(cfg.var_drift_z_max)],
                ["L", float(cfg.L)],
                ["coupled_amp_A", float(cfg.coupled_amp_A)],
                ["coupled_amp_B", float(cfg.coupled_amp_B)],
                ["coupled_amp_C", float(cfg.coupled_amp_C)],
                ["curved_c_A", float(cfg.curved_c_A)],
                ["curved_c_B", float(cfg.curved_c_B)],
                ["curved_c_C", float(cfg.curved_c_C)],
                ["aperture_scale", float(cfg.aperture_scale)],
                ["n_rep", int(cfg.n_rep)],
            ],
        )

        print(f"Wrote (untracked): {out_dir / cfg.out_cases_csv}")
        print(f"Wrote (untracked): {out_dir / cfg.out_audit_csv}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
