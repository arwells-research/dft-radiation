#!/usr/bin/env python3
"""
S-0018 — Non-Gaussian innovation masquerade boundary for κ2-slope classification integrity (C2 ⟷ distribution guard)

Goal
----
Certify that κ2-slope classification (OU-like α≈1 vs long-memory α≈2H) is NOT sufficient to
guarantee Gaussian admissibility at the innovation level. A process can match κ2(t) scaling
while being strongly non-Gaussian in ω, and therefore must be refused (BOUNDARY) rather than
certified as OK_OU / OK_LM.

Construction
------------
Three fixed cases over one declared late-time audit window:

  - OU_BASE:
      Gaussian OU ω_i(t) (short-memory). Expected κ2-slope α in OU band and kurtosis not detected.

  - LM_TRUE:
      Uncoupled long-memory Gaussian ω_i(t) (fractional Gaussian noise with fixed H).
      Expected κ2-slope α in LM band and kurtosis not detected.

  - NON_GAUSS:
      OU recursion driven by symmetric heavy-tail innovations (mixture spikes) normalized to Var≈1.
      κ2-slope remains OU-like (α≈1) but ω exhibits large excess kurtosis; must be refused (BOUNDARY).

Centering
---------
Global DC removal only (one constant mean over ensemble×time), per case. No adaptive detrending.

Diagnostics
-----------
1) κ2-slope classifier on κ2(t)=Var_i[Δϕ_i(t)] with Δϕ=∫ω dt (discrete cumsum):
     α := slope of log κ2 vs log t over the declared window.
   Bands (fixed):
     - OU band: [ou_alpha_min, ou_alpha_max]
     - LM band: [lm_alpha_min, lm_alpha_max]  (H fixed; no scanning)

2) Innovation kurtosis guard (declared, fixed; window-restricted):
   Over the audit window, compute ensemble excess kurtosis of ω:
      g2 := μ4 / μ2^2 - 3
   Define:
      kurtosis_detected := (|g2| > kurtosis_bound)
   Any kurtosis_detected=1 forces BOUNDARY (never OK_*).

Certification rule
------------------
- Baselines must classify correctly with kurtosis_detected=0:
    OU_BASE -> OK_OU
    LM_TRUE -> OK_LM
- NON_GAUSS must be refused:
    NON_GAUSS -> BOUNDARY (never OK_OU/OK_LM), with kurtosis_detected=1.

Exit codes
----------
  0 PASS
  2 FAIL
  3 INCONCLUSIVE

Outputs (only with --write_outputs)
----------------------------------
  toys/outputs/s0018_cases.csv
  toys/outputs/s0018_audit.csv
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
class S0018Config:
    seed: int = 18018

    dt: float = 0.20
    n_steps: int = 1024
    n_trajectories: int = 1024

    # Fixed late-time audit window (absolute times)
    t_min: float = 20.0
    t_max: float = 40.0
    min_points: int = 24

    # Admissibility gates for κ2 fit (fixed)
    min_r2: float = 0.985
    min_k2_end: float = 5.0e-3

    # Classification bands (fixed)
    ou_alpha_min: float = 0.82
    ou_alpha_max: float = 1.18

    # Long-memory baseline (fixed)
    fgn_H: float = 0.75
    lm_alpha_min: float = 1.30
    lm_alpha_max: float = 1.70

    # Kurtosis gate (declared, fixed)
    kurtosis_bound: float = 0.50  # |g2| > bound => kurtosis_detected

    # OU parameters (fixed)
    ou_theta: float = 0.60
    ou_sigma: float = 0.25
    ou_mu: float = 0.0

    # NON_GAUSS heavy-tail innovation parameters (fixed)
    # Mixture spikes: with prob 2q, add ±b; otherwise 0. Normalized to Var≈1.
    ng_b: float = 8.0
    ng_q: float = 2.0e-3

    out_dir: str = "toys/outputs"
    out_cases_csv: str = "s0018_cases.csv"
    out_audit_csv: str = "s0018_audit.csv"


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


def _linear_fit_slope_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
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
    return a1, float(r2)


def _global_dc_remove(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x - float(np.mean(x))


def _excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return float("nan")
    mu = float(np.mean(x))
    xc = x - mu
    m2 = float(np.mean(xc**2))
    if m2 <= 0.0 or not np.isfinite(m2):
        return float("nan")
    m4 = float(np.mean(xc**4))
    g2 = (m4 / (m2 * m2)) - 3.0
    return float(g2)


def _k2_slope(delta_phi: np.ndarray, t: np.ndarray, cfg: S0018Config) -> Tuple[float, float, float]:
    """
    Returns (alpha, r2, k2_end) computed on the fixed audit window.
    """
    x = np.asarray(delta_phi, dtype=float)
    t = np.asarray(t, dtype=float)

    k2 = np.var(x, axis=0, ddof=0)
    mask = _audit_mask(t, cfg.t_min, cfg.t_max)
    if np.count_nonzero(mask) < int(cfg.min_points):
        return float("nan"), float("nan"), float("nan")

    tt = t[mask]
    k2w = k2[mask]
    k2_end = float(k2w[-1])

    eps = 1e-12
    xx = np.log(np.maximum(tt, eps))
    yy = np.log(np.maximum(k2w, eps))

    alpha, r2 = _linear_fit_slope_r2(xx, yy)
    return float(alpha), float(r2), float(k2_end)


def _tag_from_alpha(alpha: float, cfg: S0018Config) -> str:
    if not np.isfinite(alpha):
        return "INCONCLUSIVE"
    if cfg.ou_alpha_min <= alpha <= cfg.ou_alpha_max:
        return "OK_OU"
    if cfg.lm_alpha_min <= alpha <= cfg.lm_alpha_max:
        return "OK_LM"
    return "BOUNDARY"


def simulate_ou(cfg: S0018Config, rng: np.random.Generator) -> np.ndarray:
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


def _fgn_davies_harte(n_steps: int, H: float, rng: np.random.Generator) -> np.ndarray:
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

    s = float(np.std(x))
    if s <= 0.0 or not np.isfinite(s):
        raise RuntimeError("fGn generation produced invalid std.")
    return x / s


def simulate_fgn(cfg: S0018Config, rng: np.random.Generator) -> np.ndarray:
    nT = int(cfg.n_trajectories)
    nS = int(cfg.n_steps)
    H = float(cfg.fgn_H)

    omega = np.empty((nT, nS), dtype=float)
    for i in range(nT):
        omega[i, :] = _fgn_davies_harte(nS, H, rng)
    return omega


def _heavy_tail_innovations(n: Tuple[int, int], b: float, q: float, rng: np.random.Generator) -> np.ndarray:
    """
    Symmetric spike mixture:
      with prob q: +b
      with prob q: -b
      else: 0
    Normalized later (after recursion) by global var.
    """
    nT, nS = int(n[0]), int(n[1])
    u = rng.random(size=(nT, nS)).astype(float)
    signs = rng.choice(np.array([-1.0, +1.0]), size=(nT, nS), replace=True)
    spikes = (u < (2.0 * q)).astype(float)
    amp = spikes * signs * float(b)
    return amp.astype(float)


def simulate_ou_nongauss(cfg: S0018Config, rng: np.random.Generator) -> np.ndarray:
    """
    OU recursion driven by heavy-tail innovations (non-Gaussian), then globally normalized.

    This preserves OU-like correlation time while yielding large excess kurtosis in ω.
    """
    nT = int(cfg.n_trajectories)
    nS = int(cfg.n_steps)
    dt = float(cfg.dt)

    theta = float(cfg.ou_theta)
    mu = float(cfg.ou_mu)

    b = float(cfg.ng_b)
    q = float(cfg.ng_q)

    # Base OU state
    omega = np.empty((nT, nS), dtype=float)
    omega[:, 0] = mu

    # Innovations: spikes + small Gaussian background (keeps dynamics from freezing)
    # Note: background is fixed and not tuned; spike mixture drives kurtosis.
    bg_sigma = 0.15
    sqrt_dt = float(np.sqrt(dt))

    spikes = _heavy_tail_innovations((nT, nS), b=b, q=q, rng=rng)

    for k in range(1, nS):
        eps = (bg_sigma * rng.standard_normal(size=nT).astype(float) + spikes[:, k]) * sqrt_dt
        omega[:, k] = omega[:, k - 1] + theta * (mu - omega[:, k - 1]) * dt + eps

    # Global DC remove and normalize to Var=1 (keeps κ2 comparable across cases)
    omega = omega - float(np.mean(omega))
    v = float(np.var(omega))
    if v <= 0.0 or not np.isfinite(v):
        raise ValueError("simulate_ou_nongauss: variance invalid.")
    omega = omega / float(np.sqrt(v))
    return omega


def delta_phi_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(np.asarray(omega, dtype=float), axis=1) * float(dt)


def audit_case(*, case: str, omega: np.ndarray, t: np.ndarray, cfg: S0018Config) -> Dict[str, object]:
    """
    Compute κ2-slope metrics and kurtosis guard (both window-restricted), then tag.
    Enforces refusal: kurtosis_detected => BOUNDARY.
    """
    omega = np.asarray(omega, dtype=float)
    t = np.asarray(t, dtype=float)
    mask = _audit_mask(t, cfg.t_min, cfg.t_max)
    if np.count_nonzero(mask) < int(cfg.min_points):
        return {"case": case, "verdict": "INCONCLUSIVE", "reason": "window_too_small"}

    # Global DC removal only (per-case)
    omega_c = _global_dc_remove(omega)

    # κ2 slope on Δϕ
    dphi = delta_phi_from_omega(omega_c, cfg.dt)
    alpha, r2, k2_end = _k2_slope(dphi, t, cfg)

    # Kurtosis on ω in window (flattened)
    w_flat = omega_c[:, mask].reshape(-1)
    g2 = _excess_kurtosis(w_flat)
    kurt_detected = int(bool(np.isfinite(g2) and (abs(float(g2)) > float(cfg.kurtosis_bound))))

    # Admissibility gates for the κ2 fit
    admissible = bool(
        np.isfinite(alpha)
        and np.isfinite(r2)
        and np.isfinite(k2_end)
        and (float(r2) >= float(cfg.min_r2))
        and (float(k2_end) >= float(cfg.min_k2_end))
    )
    if not admissible:
        return {
            "case": case,
            "verdict": "INCONCLUSIVE",
            "reason": "inadmissible_fit",
            "alpha": float(alpha),
            "r2": float(r2),
            "k2_end": float(k2_end),
            "g2": float(g2),
            "kurtosis_detected": int(kurt_detected),
            "tag": "INCONCLUSIVE",
        }

    tag = _tag_from_alpha(float(alpha), cfg)

    # Refusal rule: any detected kurtosis forces BOUNDARY (never OK_*)
    if kurt_detected == 1:
        tag = "BOUNDARY"

    return {
        "case": case,
        "verdict": "OK",
        "alpha": float(alpha),
        "r2": float(r2),
        "k2_end": float(k2_end),
        "g2": float(g2),
        "kurtosis_detected": int(kurt_detected),
        "tag": str(tag),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="S-0018: κ2-slope masquerade boundary via kurtosis refusal gate.")
    p.add_argument("--seed", type=int, default=S0018Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0018Config(seed=int(args.seed))

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)
    mask = _audit_mask(t, cfg.t_min, cfg.t_max)
    n_w = int(np.count_nonzero(mask))

    print(
        f"[S-0018] window=[{cfg.t_min:.1f},{cfg.t_max:.1f}] n_w={n_w} dt={cfg.dt:.3g} "
        f"admiss(min_r2={cfg.min_r2:.4f}, min_k2_end={cfg.min_k2_end:g}, min_points={cfg.min_points}) "
        f"OU_band=[{cfg.ou_alpha_min:.2f},{cfg.ou_alpha_max:.2f}] "
        f"LM_band=[{cfg.lm_alpha_min:.2f},{cfg.lm_alpha_max:.2f}] (H={cfg.fgn_H:.2f}) "
        f"kurtosis_bound={cfg.kurtosis_bound:g} nT={cfg.n_trajectories}"
    )

    if n_w < int(cfg.min_points):
        print("[S-0018] INCONCLUSIVE")
        return 3

    rng_ou = np.random.default_rng(cfg.seed + 1)
    rng_lm = np.random.default_rng(cfg.seed + 2)
    rng_ng = np.random.default_rng(cfg.seed + 3)

    omega_ou = simulate_ou(cfg, rng=rng_ou)
    omega_lm = simulate_fgn(cfg, rng=rng_lm)
    omega_ng = simulate_ou_nongauss(cfg, rng=rng_ng)

    results: List[Dict[str, object]] = []
    results.append(audit_case(case="C1_OU_BASE", omega=omega_ou, t=t, cfg=cfg))
    results.append(audit_case(case="C2_LM_TRUE", omega=omega_lm, t=t, cfg=cfg))
    results.append(audit_case(case="NON_GAUSS", omega=omega_ng, t=t, cfg=cfg))

    # Print summaries (house style)
    for r in results:
        case = str(r.get("case", ""))
        verdict = str(r.get("verdict", "INCONCLUSIVE"))
        if verdict != "OK":
            reason = str(r.get("reason", ""))
            alpha = r.get("alpha", float("nan"))
            r2 = r.get("r2", float("nan"))
            k2_end = r.get("k2_end", float("nan"))
            g2 = r.get("g2", float("nan"))
            kd = r.get("kurtosis_detected", 0)
            print(
                f"[S-0018:{case}] verdict=INCONCLUSIVE reason={reason} "
                f"alpha={float(alpha):.4g} r2={float(r2):.4g} k2_end={float(k2_end):.4g} "
                f"g2={float(g2):.4g} kurt_detected={int(kd)}"
            )
            continue

        print(
            f"[S-0018:{case}] "
            f"alpha={float(r['alpha']):.4g} r2={float(r['r2']):.4g} k2_end={float(r['k2_end']):.4g} "
            f"g2={float(r['g2']):.4g} kurt_detected={int(r['kurtosis_detected'])} "
            f"tag={str(r['tag'])}"
        )

    by_case = {str(r.get("case", "")): r for r in results}
    any_inconclusive = any(str(r.get("verdict")) != "OK" for r in results)

    ou_tag = str(by_case.get("C1_OU_BASE", {}).get("tag", "INCONCLUSIVE"))
    lm_tag = str(by_case.get("C2_LM_TRUE", {}).get("tag", "INCONCLUSIVE"))
    ng_tag = str(by_case.get("NON_GAUSS", {}).get("tag", "INCONCLUSIVE"))

    # PASS conditions:
    #  - OU_BASE -> OK_OU
    #  - LM_TRUE -> OK_LM
    #  - NON_GAUSS -> BOUNDARY
    # Forbidden:
    #  - NON_GAUSS -> OK_OU or OK_LM (false certification)
    forbidden = (ng_tag in ("OK_OU", "OK_LM"))

    if forbidden:
        print("[S-0018] AUDIT FAILED: forbidden OK_* under non-Gaussian innovation (masquerade permitted).")
        exit_code = 2
    elif any_inconclusive:
        print("[S-0018] INCONCLUSIVE")
        exit_code = 3
    elif (ou_tag == "OK_OU") and (lm_tag == "OK_LM") and (ng_tag == "BOUNDARY"):
        print("[S-0018] AUDIT PASSED")
        exit_code = 0
    else:
        print(f"[S-0018] AUDIT FAILED: ou={ou_tag} lm={lm_tag} non_gauss={ng_tag}")
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
            ["case", "alpha", "r2", "k2_end", "g2", "kurtosis_detected", "tag", "verdict", "reason"],
        ]
        for r in results:
            rows.append(
                [
                    str(r.get("case", "")),
                    float(r.get("alpha", float("nan"))) if "alpha" in r else "",
                    float(r.get("r2", float("nan"))) if "r2" in r else "",
                    float(r.get("k2_end", float("nan"))) if "k2_end" in r else "",
                    float(r.get("g2", float("nan"))) if "g2" in r else "",
                    int(r.get("kurtosis_detected", 0)) if "kurtosis_detected" in r else "",
                    str(r.get("tag", "")),
                    str(r.get("verdict", "")),
                    str(r.get("reason", "")) if "reason" in r else "",
                ]
            )

        _write_csv_with_provenance_header(out_dir / cfg.out_cases_csv, header_kv, rows)
        _write_csv_with_provenance_header(
            out_dir / cfg.out_audit_csv,
            header_kv,
            [
                ["field", "value"],
                ["exit_code", int(exit_code)],
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
                ["lm_alpha_min", float(cfg.lm_alpha_min)],
                ["lm_alpha_max", float(cfg.lm_alpha_max)],
                ["fgn_H", float(cfg.fgn_H)],
                ["kurtosis_bound", float(cfg.kurtosis_bound)],
                ["ou_tag", ou_tag],
                ["lm_tag", lm_tag],
                ["non_gauss_tag", ng_tag],
            ],
        )

        print(f"Wrote (untracked): {out_dir / cfg.out_cases_csv}")
        print(f"Wrote (untracked): {out_dir / cfg.out_audit_csv}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
