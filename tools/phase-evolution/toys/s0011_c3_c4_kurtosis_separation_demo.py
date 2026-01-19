#!/usr/bin/env python3
"""
S-0011 — C3 vs C4 separability via ω non-Gaussianity (κ₄ / kurtosis) under κ₂-slope invariance

This toy follows S-0010’s certified identifiability boundary: κ₂-slope (α) alone is not
sufficient to discriminate C3 vs a declared class of stationary C4 effects. Here we add a
second, window-audited statistic on ω to restore separability in a declared family:

  - Baseline ω is Gaussian OU.
  - C3 is deterministic linear filtering (per-trajectory FIR smoothing).
    For a Gaussian baseline, linear filtering preserves Gaussianity => excess kurtosis ~ 0.
  - C4 is stationary shot-like injection into ω, which produces heavy tails => excess kurtosis > 0.

Two-stage audit (fixed window; no scanning):
  1) κ₂-slope invariance gate (relative to OU_BASE): |Δα| and |Δr²| bounded.
     If this fails or is inadmissible, we do not proceed to any C3/C4 attribution.
  2) Kurtosis separation on ω samples in the same window:
        g2 := E[ω⁴]/(E[ω²]²) − 3
     Requirements:
       - OU_BASE and C3 cases: |g2| <= max_abs_excess_kurtosis_gauss
       - C4_SHOT: g2 >= min_excess_kurtosis_shot

Exit codes:
  0 — PASS: κ₂ invariance gate passes for all required cases and kurtosis separation passes.
  2 — FAIL: Any required audit condition fails on admissible cases.
  3 — INCONCLUSIVE: Any required case is inadmissible in-window (or variance floor violated).

Conventions:
  - Deterministic seeds only.
  - Fixed audit window; no scanning / optimization.
  - Global DC removal on ω (one constant across ensemble×time), declared and enforced.
  - Outputs written only under toys/outputs/ and only when --write_outputs is passed.
  - CSV outputs include provenance header lines (# key=value).
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
class S0011Config:
    seed: int = 11011

    dt: float = 0.02
    n_steps: int = 4096
    n_trajectories: int = 512

    # OU parameters (Gaussian baseline)
    ou_theta: float = 0.35
    ou_sigma: float = 1.00

    # Fixed audit window (absolute times)
    t_min: float = 40.0
    t_max: float = 70.0

    # κ2-slope admissibility
    min_r2: float = 0.985
    k2_min_at_tmax: float = 0.50
    min_points: int = 18

    # κ2-slope invariance bounds (relative to OU_BASE)
    max_abs_delta_alpha: float = 0.02
    max_abs_delta_r2: float = 0.01

    # C3 transport models (FIR smoothing window lengths)
    c3_m_smooth: int = 8
    c3_m_disp: int = 16

    # C4 shot injection (stationary; heavy tails)
    shot_rate_per_time: float = 0.02
    shot_amp_mean: float = 4.00
    shot_tau_samples: int = 2
    shot_sign_symmetric: bool = True

    # Kurtosis separation thresholds (window-restricted ω samples)
    max_abs_excess_kurtosis_gauss: float = 0.10
    min_excess_kurtosis_shot: float = 0.30

    # Variance floor guard for kurtosis computation
    omega_var_floor: float = 1.0e-6

    out_dir: str = "toys/outputs"
    out_cases_csv: str = "s0011_cases.csv"
    out_audit_csv: str = "s0011_audit.csv"


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


def simulate_ou(cfg: S0011Config, rng: np.random.Generator) -> np.ndarray:
    nT, n = int(cfg.n_trajectories), int(cfg.n_steps)
    dt = float(cfg.dt)
    theta = float(cfg.ou_theta)
    sigma = float(cfg.ou_sigma)

    omega = np.empty((nT, n), dtype=float)
    sdt = float(np.sqrt(dt))
    # initialize from stationary variance
    x0_std = sigma / np.sqrt(2.0 * theta)
    for i in range(nT):
        x = float(rng.normal(scale=x0_std))
        for k in range(n):
            x = x + (-theta * x) * dt + sigma * sdt * float(rng.standard_normal())
            omega[i, k] = x
    return omega


def center_global_dc(omega: np.ndarray) -> np.ndarray:
    return np.asarray(omega, dtype=float) - float(np.mean(omega))


def fir_smooth_per_traj(x: np.ndarray, m: int) -> np.ndarray:
    m = int(m)
    if m <= 1:
        return np.asarray(x, dtype=float).copy()

    k = np.ones(m, dtype=float) / float(m)
    pad = m // 2

    x = np.asarray(x, dtype=float)
    nT, n = x.shape
    out = np.empty_like(x)
    for i in range(nT):
        xp = np.pad(x[i, :], (pad, pad), mode="reflect")
        y = np.convolve(xp, k, mode="valid")
        out[i, :] = y[:n]
    return out


def add_shot_in_omega(omega: np.ndarray, cfg: S0011Config, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(omega, dtype=float).copy()
    nT, n = y.shape

    lam = float(cfg.shot_rate_per_time)
    amp_mean = float(cfg.shot_amp_mean)
    dt = float(cfg.dt)
    tau = int(cfg.shot_tau_samples)
    if tau <= 0 or lam <= 0:
        return y

    mean_events = lam * (n * dt)
    L = int(max(1, np.ceil(tau * 14.0)))
    kernel = np.exp(-np.arange(L, dtype=float) / float(tau))

    for i in range(nT):
        n_events = int(rng.poisson(mean_events))
        if n_events <= 0:
            continue

        t_idx = rng.integers(low=0, high=n, size=n_events)
        amps = rng.exponential(scale=amp_mean, size=n_events)
        if bool(cfg.shot_sign_symmetric):
            signs = rng.choice([-1.0, 1.0], size=n_events)
            amps = amps * signs

        for j in range(n_events):
            t0 = int(t_idx[j])
            A = float(amps[j])
            t1 = min(n, t0 + L)
            y[i, t0:t1] += A * kernel[: (t1 - t0)]

    return y


def delta_phi_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(np.asarray(omega, dtype=float), axis=1) * float(dt)


def k2_direct(delta_phi: np.ndarray) -> np.ndarray:
    return np.var(np.asarray(delta_phi, dtype=float), axis=0, ddof=0)


def _fit_loglog(tw: np.ndarray, k2w: np.ndarray) -> Tuple[float, float, int]:
    tw = np.asarray(tw, dtype=float)
    k2w = np.asarray(k2w, dtype=float)

    eps = 1e-18
    good = np.isfinite(tw) & (tw > 0) & np.isfinite(k2w) & (k2w > eps)
    npts = int(np.count_nonzero(good))
    if npts < 4:
        return float("nan"), float("nan"), npts

    x = np.log(tw[good])
    y = np.log(k2w[good])

    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    alpha = float(coef[1])
    return alpha, float(r2), npts


def fit_k2_slope_in_window(t: np.ndarray, k2: np.ndarray, cfg: S0011Config) -> Dict[str, float]:
    t = np.asarray(t, dtype=float)
    k2 = np.asarray(k2, dtype=float)

    mask = _audit_mask(t, cfg.t_min, cfg.t_max) & np.isfinite(k2)
    tw = t[mask]
    k2w = k2[mask]

    alpha, r2, npts = _fit_loglog(tw, k2w)
    k2_tmax = float(k2w[-1]) if k2w.size else float("nan")

    admissible = bool(
        np.isfinite(alpha)
        and np.isfinite(r2)
        and (npts >= int(cfg.min_points))
        and (r2 >= float(cfg.min_r2))
        and (k2_tmax >= float(cfg.k2_min_at_tmax))
    )
    return {
        "alpha": float(alpha),
        "r2": float(r2),
        "n_points": float(npts),
        "k2_tmax": float(k2_tmax),
        "admissible": float(int(admissible)),
    }


def excess_kurtosis_window(omega: np.ndarray, t: np.ndarray, cfg: S0011Config) -> Dict[str, float]:
    omega = np.asarray(omega, dtype=float)
    t = np.asarray(t, dtype=float)
    mask = _audit_mask(t, cfg.t_min, cfg.t_max)

    if not np.any(mask):
        return {"g2": float("nan"), "m2": float("nan"), "ok_var": float(0)}

    # pool samples within window across all trajectories
    w = omega[:, mask].reshape(-1)
    w = w[np.isfinite(w)]
    if w.size < 64:
        return {"g2": float("nan"), "m2": float("nan"), "ok_var": float(0)}

    m2 = float(np.mean(w**2))
    if not np.isfinite(m2) or (m2 < float(cfg.omega_var_floor)):
        return {"g2": float("nan"), "m2": float(m2), "ok_var": float(0)}

    m4 = float(np.mean(w**4))
    g2 = float(m4 / (m2 * m2) - 3.0)
    return {"g2": float(g2), "m2": float(m2), "ok_var": float(1)}


def main() -> int:
    p = argparse.ArgumentParser(description="S-0011: C3 vs C4 separability via ω kurtosis under κ2-slope invariance.")
    p.add_argument("--seed", type=int, default=S0011Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0011Config(seed=int(args.seed))

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)

    print(
        f"[S-0011] window=[{cfg.t_min:.1f},{cfg.t_max:.1f}] "
        f"min_r2={cfg.min_r2:.3f} k2_min@tmax={cfg.k2_min_at_tmax:.3g} min_points={cfg.min_points} "
        f"Δα_max={cfg.max_abs_delta_alpha:.3g} Δr2_max={cfg.max_abs_delta_r2:.3g} "
        f"g2_gauss_max={cfg.max_abs_excess_kurtosis_gauss:.3g} g2_shot_min={cfg.min_excess_kurtosis_shot:.3g}"
    )

    omega_base = simulate_ou(cfg, rng=np.random.default_rng(cfg.seed + 1))
    omega_base = center_global_dc(omega_base)

    # Deterministic derived cases (C3 / C4)
    omega_c3_smooth = center_global_dc(fir_smooth_per_traj(omega_base, cfg.c3_m_smooth))
    omega_c3_disp = center_global_dc(fir_smooth_per_traj(omega_base, cfg.c3_m_disp))
    omega_c4_shot = center_global_dc(add_shot_in_omega(omega_base, cfg, rng=np.random.default_rng(cfg.seed + 2)))

    cases: List[Tuple[str, np.ndarray, str]] = [
        ("OU_BASE", omega_base, "GAUSS"),
        ("OU_C3_SMOOTH", omega_c3_smooth, "GAUSS"),
        ("OU_C3_DISP", omega_c3_disp, "GAUSS"),
        ("OU_C4_SHOT", omega_c4_shot, "SHOT"),
    ]

    # κ2-slope fits
    dphi_base = delta_phi_from_omega(omega_base, cfg.dt)
    k2_base = k2_direct(dphi_base)
    fit0 = fit_k2_slope_in_window(t, k2_base, cfg)

    alpha0 = float(fit0["alpha"])
    r20 = float(fit0["r2"])
    base_adm = bool(int(fit0["admissible"]) == 1)

    rows: List[List[object]] = []
    any_fail = False
    any_inconclusive = False

    for name, omega, kind in cases:
        dphi = delta_phi_from_omega(omega, cfg.dt)
        k2 = k2_direct(dphi)
        fit = fit_k2_slope_in_window(t, k2, cfg)

        alpha = float(fit["alpha"])
        r2 = float(fit["r2"])
        npts = int(fit["n_points"])
        k2_tmax = float(fit["k2_tmax"])
        admissible = bool(int(fit["admissible"]) == 1)

        if not base_adm:
            verdict = "INCONCLUSIVE"
            any_inconclusive = True
            delta_alpha = float("nan")
            delta_r2 = float("nan")
            inv_ok = False
        else:
            delta_alpha = float(alpha - alpha0) if np.isfinite(alpha) else float("nan")
            delta_r2 = float(r2 - r20) if np.isfinite(r2) else float("nan")
            inv_ok = bool(
                admissible
                and np.isfinite(delta_alpha)
                and np.isfinite(delta_r2)
                and (abs(delta_alpha) <= float(cfg.max_abs_delta_alpha))
                and (abs(delta_r2) <= float(cfg.max_abs_delta_r2))
            )
            if not admissible:
                verdict = "INCONCLUSIVE"
                any_inconclusive = True
            elif not inv_ok:
                verdict = "FAIL"
                any_fail = True
            else:
                verdict = "OK"

        # Kurtosis audit (only meaningful if invariance gate ok for required cases)
        kurt = excess_kurtosis_window(omega, t, cfg)
        g2 = float(kurt["g2"])
        ok_var = bool(int(kurt["ok_var"]) == 1)

        if verdict == "OK":
            if not ok_var:
                verdict = "INCONCLUSIVE"
                any_inconclusive = True
            else:
                if kind == "GAUSS":
                    k_ok = bool(np.isfinite(g2) and (abs(g2) <= float(cfg.max_abs_excess_kurtosis_gauss)))
                else:
                    k_ok = bool(np.isfinite(g2) and (g2 >= float(cfg.min_excess_kurtosis_shot)))
                if not k_ok:
                    verdict = "FAIL"
                    any_fail = True
        else:
            # not evaluated for kurtosis decision if already FAIL/INCONCLUSIVE
            k_ok = False

        print(
            f"[S-0011:{name}] "
            f"alpha={alpha:.3f} r2={r2:.3f} n={npts} k2_tmax={k2_tmax:.3g} adm={int(admissible)} "
            f"base_alpha={alpha0:.3f} d_alpha={(delta_alpha if np.isfinite(delta_alpha) else float('nan')):.3g} "
            f"d_r2={(delta_r2 if np.isfinite(delta_r2) else float('nan')):.3g} inv_ok={int(inv_ok)} "
            f"g2={(g2 if np.isfinite(g2) else float('nan')):.3g} verdict={verdict}"
        )

        rows.append(
            [
                name,
                kind,
                float(alpha),
                float(r2),
                int(npts),
                float(k2_tmax),
                int(admissible),
                float(alpha0),
                float(delta_alpha) if np.isfinite(delta_alpha) else "",
                float(delta_r2) if np.isfinite(delta_r2) else "",
                int(inv_ok),
                float(g2) if np.isfinite(g2) else "",
                int(ok_var),
                int(k_ok),
                verdict,
            ]
        )

    if any_fail:
        print("[S-0011] AUDIT FAILED")
        exit_code = 2
    elif any_inconclusive:
        print("[S-0011] INCONCLUSIVE")
        exit_code = 3
    else:
        print("[S-0011] AUDIT PASSED")
        exit_code = 0

    if args.write_outputs:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        header_kv = {
            "git_commit": _git_commit_short(),
            "run_utc": _run_utc_iso(),
            "toy_file": Path(__file__).name,
            "source_sha256": _sha256_of_file(Path(__file__)),
        }

        cases_path = out_dir / cfg.out_cases_csv
        out_rows: List[List[object]] = [[
            "case",
            "kind",
            "alpha",
            "r2",
            "n_points",
            "k2_tmax",
            "admissible",
            "base_alpha",
            "delta_alpha",
            "delta_r2",
            "k2_invariance_ok",
            "omega_excess_kurtosis_g2",
            "omega_var_ok",
            "kurtosis_ok",
            "verdict",
        ]]
        out_rows.extend(rows)
        _write_csv_with_provenance_header(cases_path, header_kv, out_rows)
        print(f"Wrote (untracked): {cases_path}")

        audit_path = out_dir / cfg.out_audit_csv
        audit_rows: List[List[object]] = [
            ["field", "value"],
            ["exit_code", exit_code],
            ["seed", int(cfg.seed)],
            ["dt", float(cfg.dt)],
            ["n_steps", int(cfg.n_steps)],
            ["n_trajectories", int(cfg.n_trajectories)],
            ["t_min", float(cfg.t_min)],
            ["t_max", float(cfg.t_max)],
            ["min_r2", float(cfg.min_r2)],
            ["k2_min_at_tmax", float(cfg.k2_min_at_tmax)],
            ["min_points", int(cfg.min_points)],
            ["max_abs_delta_alpha", float(cfg.max_abs_delta_alpha)],
            ["max_abs_delta_r2", float(cfg.max_abs_delta_r2)],
            ["max_abs_excess_kurtosis_gauss", float(cfg.max_abs_excess_kurtosis_gauss)],
            ["min_excess_kurtosis_shot", float(cfg.min_excess_kurtosis_shot)],
            ["omega_var_floor", float(cfg.omega_var_floor)],
        ]
        _write_csv_with_provenance_header(audit_path, header_kv, audit_rows)
        print(f"Wrote (untracked): {audit_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
