#!/usr/bin/env python3
"""
S-0016 — Non-stationary temporal curvature boundary for κ2-slope admissibility (C1 guard)

Goal
----
Certify that κ2-slope classification is admissible only in locally stationary regimes.
A deterministic / slowly varying temporal curvature in ω(t) can mimic long-memory-like κ2 scaling
within a finite audit window; therefore κ2-slope alone is insufficient for C2 attribution unless
a C1 curvature guard is applied.

Construction
------------
We generate three fixed cases over a declared late-time window:

  - C1_OU_BASE: independent OU ω_i(t) (short-memory); expected κ2 slope α ≈ 1 in window.
  - C2_LM_TRUE: uncoupled long-memory Gaussian ω_i(t) (fractional Gaussian noise with fixed H);
               expected κ2 slope α ≈ 2H in window.
  - C1_CURVED:  OU noise + per-trajectory quadratic curvature term:
                  ω_i(t) := ω_i(t) + γ_i * t^2
               where γ_i has a nonzero mean component so that ensemble-mean ω_bar(t)
               exhibits detectable curvature in the window.

Centering
---------
Global DC removal only (one constant mean over ensemble×time), per case. No adaptive centering.

Diagnostics
-----------
1) κ2-slope classifier on κ2(t)=Var_i[Δϕ_i(t)] with Δϕ=∫ω dt (discrete cumsum):
     α := slope of log κ2 vs log t over the declared window.
   Bands (fixed):
     - OU band: [ou_alpha_min, ou_alpha_max]
     - LM band: [lm_alpha_min, lm_alpha_max]

2) Curvature detector (declared, fixed; window-restricted):
   Fit ω_bar(t) := mean_i ω_i(t) over the audit window to:
      ω_bar(t) ≈ a + b t + c t^2
   Let SE(c) be the OLS standard error of the quadratic coefficient in that fit.
   Define:
      curvature_z := |c| / max(SE(c), eps)
      curvature_detected := (curvature_z > curvature_z_bound)

Certification rule
------------------
- Baselines must classify correctly with curvature_detected=0:
    OU_BASE -> OK_OU
    LM_TRUE -> OK_LM
- Any case with curvature_detected=1 is forced to BOUNDARY (never OK_*),
  forbidding false OK_LM under C1 curvature.

Exit codes
----------
  0 PASS
  2 FAIL
  3 INCONCLUSIVE

Outputs (only with --write_outputs)
----------------------------------
  toys/outputs/s0016_cases.csv
  toys/outputs/s0016_audit.csv
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
class S0016Config:
    seed: int = 16016

    dt: float = 0.20
    n_steps: int = 512
    n_trajectories: int = 1024

    # Fixed audit window (absolute times)
    t_min: float = 20.0
    t_max: float = 40.0
    min_points: int = 24

    # Admissibility for κ2-slope fit
    min_r2: float = 0.9850
    min_k2_end: float = 5.0e-3

    # κ2-slope class bands (fixed, no scanning)
    ou_alpha_min: float = 0.82
    ou_alpha_max: float = 1.18

    fgn_H: float = 0.75
    lm_alpha_min: float = 1.30
    lm_alpha_max: float = 1.70

    # OU generator (ω innovations)
    ou_theta: float = 0.50
    ou_mu: float = 0.0
    ou_sigma: float = 1.00

    # Long-memory generator (fGn innovations, ω = mu + sigma * fgn)
    fgn_mu: float = 0.0
    fgn_sigma: float = 1.00

    # Curvature amplitude (nonzero mean ensures ω_bar(t) has quadratic curvature)
    gamma_mean: float = 1.5e-3
    gamma_sigma: float = 1.0e-4

    # Scale down OU noise for the curved case only (keeps κ2 fit admissible)
    curved_noise_scale: float = 0.60

    # Quadratic-coefficient significance gate (z-score / t-stat proxy)
    # Declared, fixed: "resolvable quadratic curvature" in ω_bar(t) within the audit window.
    curvature_z_bound: float = 8.0
    curvature_eps: float = 1.0e-12

    out_dir: str = "toys/outputs"
    out_cases_csv: str = "s0016_cases.csv"
    out_audit_csv: str = "s0016_audit.csv"


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


def _ols_slope_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 4:
        return float("nan"), float("nan")
    xv = x[good]
    yv = y[good]
    A = np.vstack([np.ones_like(xv), xv]).T
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    a0 = float(coef[0])
    a1 = float(coef[1])
    yhat = a0 + a1 * xv
    ss_res = float(np.sum((yv - yhat) ** 2))
    ss_tot = float(np.sum((yv - float(np.mean(yv))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    return a1, r2


def _classify_alpha(cfg: S0016Config, alpha: float) -> str:
    if not np.isfinite(alpha):
        return "BOUNDARY"
    if float(cfg.ou_alpha_min) <= float(alpha) <= float(cfg.ou_alpha_max):
        return "OK_OU"
    if float(cfg.lm_alpha_min) <= float(alpha) <= float(cfg.lm_alpha_max):
        return "OK_LM"
    return "BOUNDARY"


def _center_global_dc(omega: np.ndarray) -> np.ndarray:
    x = np.asarray(omega, dtype=float)
    m = float(np.mean(x))
    return x - m


def simulate_ou_omega(cfg: S0016Config, rng: np.random.Generator) -> np.ndarray:
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

    W = rng.normal(size=m) + 1j * rng.normal(size=m)
    X = np.fft.ifft(np.sqrt(lam) * W).real
    x = X[:n]

    s = float(np.std(x))
    if s <= 0.0 or not np.isfinite(s):
        raise RuntimeError("fGn generation produced non-positive std; numerical issue.")
    return x / s


def simulate_fgn_omega(cfg: S0016Config, rng: np.random.Generator) -> np.ndarray:
    nT = int(cfg.n_trajectories)
    nS = int(cfg.n_steps)

    H = float(cfg.fgn_H)
    mu = float(cfg.fgn_mu)
    sig = float(cfg.fgn_sigma)

    omega = np.empty((nT, nS), dtype=float)
    for i in range(nT):
        inc = fgn_davies_harte(nS, H, rng)
        omega[i, :] = mu + sig * inc
    return omega


def apply_quadratic_curvature(
    *,
    omega: np.ndarray,
    t: np.ndarray,
    rng: np.random.Generator,
    gamma_mean: float,
    gamma_sigma: float,
) -> np.ndarray:
    """
    Add per-trajectory ω curvature γ_i t^2 with γ_i ~ Normal(gamma_mean, gamma_sigma).
    The nonzero mean component ensures ω_bar(t) exhibits curvature in the window.
    """
    x = np.asarray(omega, dtype=float)
    t2 = np.asarray(t, dtype=float) ** 2
    nT = x.shape[0]
    gam = rng.normal(loc=float(gamma_mean), scale=float(gamma_sigma), size=nT).astype(float)
    return x + gam[:, None] * t2[None, :]


def delta_phi_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(np.asarray(omega, dtype=float), axis=1) * float(dt)


def k2_of_delta_phi(delta_phi: np.ndarray) -> np.ndarray:
    x = np.asarray(delta_phi, dtype=float)
    mu = np.mean(x, axis=0)
    xc = x - mu[None, :]
    return np.mean(xc**2, axis=0)


def curvature_detector(
    *,
    omega: np.ndarray,
    t: np.ndarray,
    cfg: S0016Config,
) -> Tuple[float, int]:
    """
    Fit ω_bar(t) ≈ a + b t + c t^2 over audit window and compute:
      curvature_z := |c| / SE(c)
    where SE(c) is the OLS standard error of the quadratic coefficient computed
    from residual variance and (X'X)^(-1). This correctly shrinks under large nT
    because ω_bar(t) noise shrinks ~1/sqrt(nT), reducing residual variance.
    """
    t = np.asarray(t, dtype=float)
    mask = _audit_mask(t, cfg.t_min, cfg.t_max)
    if np.count_nonzero(mask) < int(cfg.min_points):
        return float("nan"), 0

    om = np.asarray(omega, dtype=float)
    tw = t[mask]
    wbar = np.mean(om, axis=0)
    yw = wbar[mask]

    # Quadratic OLS fit on ω_bar(t)
    X = np.vstack([np.ones_like(tw), tw, tw**2]).T  # (m,3)
    coef, *_ = np.linalg.lstsq(X, yw, rcond=None)
    c = float(coef[2])

    # Residuals and residual variance estimate
    yhat = X @ coef
    resid = yw - yhat
    m = int(tw.size)
    dof = m - 3
    if dof <= 0:
        return float("nan"), 0

    s2 = float(np.sum(resid * resid)) / float(dof)
    if not np.isfinite(s2) or s2 <= 0.0:
        return float("nan"), 0

    # Covariance of coefficients: s2 * (X'X)^(-1)
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return float("nan"), 0

    se_c = float(np.sqrt(s2 * XtX_inv[2, 2]))
    denom = max(se_c, float(cfg.curvature_eps))
    z = abs(c) / denom
    detected = int(bool(np.isfinite(z) and (z > float(cfg.curvature_z_bound))))
    return float(z), detected


def audit_case(
    *,
    name: str,
    omega: np.ndarray,
    t: np.ndarray,
    cfg: S0016Config,
) -> Dict[str, float | int | str]:
    """
    Compute κ2-slope alpha and curvature detector, then tag with refusal logic.
    """
    # global DC removal only
    omega_c = _center_global_dc(omega)

    dphi = delta_phi_from_omega(omega_c, cfg.dt)
    k2 = k2_of_delta_phi(dphi)

    mask = _audit_mask(t, cfg.t_min, cfg.t_max)
    if np.count_nonzero(mask) < int(cfg.min_points):
        return {"case": name, "verdict": "INCONCLUSIVE", "reason": "too_few_points"}

    k2_end = float(k2[np.where(mask)[0][-1]])
    if (not np.isfinite(k2_end)) or (k2_end < float(cfg.min_k2_end)):
        return {"case": name, "verdict": "INCONCLUSIVE", "reason": "k2_end_too_small", "k2_end": k2_end}

    tw = t[mask]
    k2w = k2[mask]
    good = np.isfinite(tw) & np.isfinite(k2w) & (tw > 0.0) & (k2w > 0.0)
    if np.count_nonzero(good) < int(cfg.min_points):
        return {"case": name, "verdict": "INCONCLUSIVE", "reason": "bad_points"}

    alpha, r2 = _ols_slope_r2(np.log(tw[good]), np.log(k2w[good]))
    admissible = bool(np.isfinite(alpha) and np.isfinite(r2) and (r2 >= float(cfg.min_r2)))

    curv_strength, curv_detected = curvature_detector(omega=omega_c, t=t, cfg=cfg)

    if not admissible:
        return {
            "case": name,
            "verdict": "INCONCLUSIVE",
            "reason": "inadmissible_fit",
            "alpha": float(alpha),
            "r2": float(r2),
            "k2_end": k2_end,
            "curvature_strength": float(curv_strength),
            "curvature_detected": int(curv_detected),
        }

    tag = _classify_alpha(cfg, float(alpha))

    # C1 refusal rule: detected curvature forces BOUNDARY, never OK_*
    if int(curv_detected) == 1:
        tag = "BOUNDARY"

    return {
        "case": name,
        "verdict": "OK",
        "alpha": float(alpha),
        "r2": float(r2),
        "k2_end": k2_end,
        "curvature_strength": float(curv_strength),
        "curvature_detected": int(curv_detected),
        "tag": tag,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="S-0016: curvature boundary for κ2-slope admissibility (C1 guard).")
    p.add_argument("--seed", type=int, default=S0016Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0016Config(seed=int(args.seed))

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)
    mask = _audit_mask(t, cfg.t_min, cfg.t_max)
    n_w = int(np.count_nonzero(mask))

    print(
        f"[S-0016] window=[{cfg.t_min:.1f},{cfg.t_max:.1f}] n_w={n_w} dt={cfg.dt:.3g} "
        f"admiss(min_r2={cfg.min_r2:.4f}, min_k2_end={cfg.min_k2_end:g}, min_points={cfg.min_points}) "
        f"OU_band=[{cfg.ou_alpha_min:g},{cfg.ou_alpha_max:g}] "
        f"LM_band=[{cfg.lm_alpha_min:g},{cfg.lm_alpha_max:g}] (H={cfg.fgn_H:g}) "
        f"curvature_bound={cfg.curvature_z_bound:g} eps={cfg.curvature_eps:g} "
        f"nT={cfg.n_trajectories}"
    )

    if n_w < int(cfg.min_points):
        print("[S-0016] INCONCLUSIVE: audit window has too few points.")
        return 3

    rng_ou = np.random.default_rng(cfg.seed + 1)
    rng_fgn = np.random.default_rng(cfg.seed + 2)
    rng_curv = np.random.default_rng(cfg.seed + 3)

    # Generate cases
    omega_ou = simulate_ou_omega(cfg, rng=rng_ou)
    omega_lm = simulate_fgn_omega(cfg, rng=rng_fgn)

    # Curved: OU + quadratic curvature term
    omega_curv = simulate_ou_omega(cfg, rng=np.random.default_rng(cfg.seed + 4))
    omega_curv = omega_curv * float(cfg.curved_noise_scale)
    omega_curv = apply_quadratic_curvature(
        omega=omega_curv,
        t=t,
        rng=rng_curv,
        gamma_mean=cfg.gamma_mean,
        gamma_sigma=cfg.gamma_sigma,
    )

    audits: List[Dict[str, float | int | str]] = []
    audits.append(audit_case(name="C1_OU_BASE", omega=omega_ou, t=t, cfg=cfg))
    audits.append(audit_case(name="C2_LM_TRUE", omega=omega_lm, t=t, cfg=cfg))
    audits.append(audit_case(name="C1_CURVED", omega=omega_curv, t=t, cfg=cfg))

    # Print summaries
    for r in audits:
        case = str(r.get("case", ""))
        verdict = str(r.get("verdict", "INCONCLUSIVE"))
        if verdict != "OK":
            reason = str(r.get("reason", ""))
            alpha = r.get("alpha", float("nan"))
            r2 = r.get("r2", float("nan"))
            k2_end = r.get("k2_end", float("nan"))
            cz = r.get("curvature_z", r.get("curvature_strength", float("nan")))
            cd = r.get("curvature_detected", 0)
            print(
                f"[S-0016:{case}] verdict=INCONCLUSIVE reason={reason} "
                f"alpha={float(alpha):.4g} r2={float(r2):.4g} k2_end={float(k2_end):.4g} "
                f"curv_z={float(cz):.4g} curv_detected={int(cd)}"
            )
            continue

        cz = r.get("curvature_z", r.get("curvature_strength", float("nan")))
        print(
            f"[S-0016:{case}] "
            f"alpha={float(r['alpha']):.4g} r2={float(r['r2']):.4g} k2_end={float(r['k2_end']):.4g} "
            f"curv_z={float(cz):.4g} curv_detected={int(r['curvature_detected'])} "
            f"tag={str(r['tag'])}"
        )

    # Certification logic
    by_case = {str(r.get("case", "")): r for r in audits}

    def _tag(case: str) -> str:
        r = by_case.get(case, {})
        if str(r.get("verdict", "")) != "OK":
            return "INCONCLUSIVE"
        return str(r.get("tag", "BOUNDARY"))

    def _curv(case: str) -> int:
        r = by_case.get(case, {})
        if str(r.get("verdict", "")) != "OK":
            return 0
        return int(r.get("curvature_detected", 0))

    any_inconclusive = any(str(r.get("verdict", "")) != "OK" for r in audits)

    ou_tag = _tag("C1_OU_BASE")
    lm_tag = _tag("C2_LM_TRUE")
    curv_tag = _tag("C1_CURVED")

    ou_ok = (ou_tag == "OK_OU") and (_curv("C1_OU_BASE") == 0)
    lm_ok = (lm_tag == "OK_LM") and (_curv("C2_LM_TRUE") == 0)

    # Forbidden condition: curvature detected but tagged OK_LM (should be forced to BOUNDARY)
    curv_forbidden_ok_lm = (str(curv_tag) == "OK_LM") and (_curv("C1_CURVED") == 1)

    if curv_forbidden_ok_lm:
        print("[S-0016] AUDIT FAILED: forbidden OK_LM under detected curvature (false LM certification permitted).")
        exit_code = 2
    elif any_inconclusive:
        print("[S-0016] INCONCLUSIVE")
        exit_code = 3
    elif ou_ok and lm_ok and (curv_tag == "BOUNDARY"):
        print("[S-0016] AUDIT PASSED")
        exit_code = 0
    else:
        print(f"[S-0016] AUDIT FAILED: ou={ou_tag} lm={lm_tag} curv={curv_tag}")
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
                "case",
                "alpha",
                "r2",
                "k2_end",
                "curvature_strength",
                "curvature_detected",
                "tag_or_reason",
                "verdict",
            ]
        ]
        for r in audits:
            if str(r.get("verdict", "")) == "OK":
                rows.append(
                    [
                        str(r.get("case", "")),
                        float(r.get("alpha", float("nan"))),
                        float(r.get("r2", float("nan"))),
                        float(r.get("k2_end", float("nan"))),
                        float(r.get("curvature_strength", float("nan"))),
                        int(r.get("curvature_detected", 0)),
                        str(r.get("tag", "")),
                        "OK",
                    ]
                )
            else:
                rows.append(
                    [
                        str(r.get("case", "")),
                        float(r.get("alpha", float("nan"))),
                        float(r.get("r2", float("nan"))),
                        float(r.get("k2_end", float("nan"))),
                        float(r.get("curvature_strength", float("nan"))),
                        int(r.get("curvature_detected", 0)),
                        str(r.get("reason", "")),
                        "INCONCLUSIVE",
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
                ["t_min", float(cfg.t_min)],
                ["t_max", float(cfg.t_max)],
                ["min_r2", float(cfg.min_r2)],
                ["min_k2_end", float(cfg.min_k2_end)],
                ["ou_alpha_min", float(cfg.ou_alpha_min)],
                ["ou_alpha_max", float(cfg.ou_alpha_max)],
                ["fgn_H", float(cfg.fgn_H)],
                ["lm_alpha_min", float(cfg.lm_alpha_min)],
                ["lm_alpha_max", float(cfg.lm_alpha_max)],
                ["curvature_bound", float(cfg.curvature_bound)],
                ["gamma_mean", float(cfg.gamma_mean)],
                ["gamma_sigma", float(cfg.gamma_sigma)],
                ["ou_tag", ou_tag],
                ["lm_tag", lm_tag],
                ["curv_tag", curv_tag],
            ],
        )

        print(f"Wrote (untracked): {out_dir / cfg.out_cases_csv}")
        print(f"Wrote (untracked): {out_dir / cfg.out_audit_csv}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
