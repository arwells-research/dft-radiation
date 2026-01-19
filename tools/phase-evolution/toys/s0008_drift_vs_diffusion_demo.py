#!/usr/bin/env python3
"""
S-0008 — Drift-limited vs diffusion-limited coherence (C2 ⟷ C5 boundary)

This toy certifies that a κ₂-slope classifier (log κ₂ vs log t) can:
  - correctly tag a diffusion-limited baseline as OK (OU-like α≈1),
  - refuse (BOUNDARY) a slow-drift construction that should not be treated as diffusion,
  - and NEVER make a wrong claim on admissible data.

The drift construction is C5-style: a slow trend in ω(t) that is not shared as a
common-mode deterministic term across the ensemble. If the drift were identical
for all trajectories it would cancel in κ₂(t)=Var[Δϕ(t)] and be invisible to this
diagnostic. Therefore drift is modeled as an unknown per-trajectory slope:
  β_i ~ N(0, drift_beta_sigma^2),
  ω_i(t) := ω_OU,i(t) + β_i * t,
followed by the same global DC removal as the baseline.

Exit codes:
  0 — PASS: DIFFUSION is OK; DRIFT is BOUNDARY; no wrong claims on admissible data.
  2 — FAIL: any admissible case makes a forbidden claim OR DIFFUSION baseline is not OK.
  3 — INCONCLUSIVE: no admissible cases (classifier outside certified window for this run).

Conventions:
  - Global DC removal only on ω (one constant across ensemble×time).
  - Window-local re-origining for fits (on κ₂, not on ω):
        tw  := (t - t0) + dt
        k2w := k2 - k2(t0)
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
class S0008Config:
    seed: int = 8008

    dt: float = 0.02
    n_steps: int = 4096
    n_trajectories: int = 512

    # OU parameters (baseline diffusion-like)
    ou_theta: float = 0.35
    ou_sigma: float = 1.00

    # Drift term: per-trajectory slope beta_i ~ N(0, drift_beta_sigma^2)
    drift_beta_sigma: float = 5.0e-2  # omega_drift_i(t) = beta_i * t

    # Fixed audit window (late-time). These are absolute times in the simulation.
    t_min: float = 40.0
    t_max: float = 70.0

    # Admissibility thresholds
    min_r2: float = 0.985
    k2_min_at_tmax: float = 0.50
    min_points: int = 18

    # Diffusion expectation band (OU-like α≈1)
    ou_alpha_target: float = 1.00
    ou_alpha_tol: float = 0.15

    out_dir: str = "toys/outputs"
    out_audit_csv: str = "s0008_audit.csv"
    out_cases_csv: str = "s0008_cases.csv"


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


def center_global_dc(omega: np.ndarray) -> np.ndarray:
    return omega - float(np.mean(omega))


def simulate_ou(cfg: S0008Config, rng: np.random.Generator) -> np.ndarray:
    """
    Euler-Maruyama OU:
      dX = -theta X dt + sigma dW
    """
    nT, n = int(cfg.n_trajectories), int(cfg.n_steps)
    dt = float(cfg.dt)
    theta = float(cfg.ou_theta)
    sigma = float(cfg.ou_sigma)

    omega = np.empty((nT, n), dtype=float)
    sdt = np.sqrt(dt)
    for i in range(nT):
        # Start at stationary std to reduce transient bias.
        x = float(rng.normal(scale=(sigma / np.sqrt(2.0 * theta))))
        for k in range(n):
            x = x + (-theta * x) * dt + sigma * sdt * float(rng.standard_normal())
            omega[i, k] = x
    return omega


def add_per_traj_linear_drift(omega: np.ndarray, cfg: S0008Config, rng: np.random.Generator) -> np.ndarray:
    """
    Per-trajectory slow drift:
      beta_i ~ N(0, drift_beta_sigma^2)
      omega_i(t) += beta_i * t
    """
    nT, n = omega.shape
    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)
    betas = rng.normal(loc=0.0, scale=float(cfg.drift_beta_sigma), size=nT).astype(float)
    return omega + betas[:, None] * t[None, :]


def delta_phi_from_omega(omega0: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(omega0, axis=1) * float(dt)


def k2_direct(delta_phi: np.ndarray) -> np.ndarray:
    return np.var(delta_phi, axis=0, ddof=0)


def _fit_loglog(
    t: np.ndarray,
    k2: np.ndarray,
    *,
    min_r2: float,
    k2_min_at_tmax: float,
    min_points: int,
) -> Dict[str, float]:
    t = np.asarray(t, dtype=float)
    k2 = np.asarray(k2, dtype=float)

    eps = 1e-18
    good = np.isfinite(t) & (t > 0) & np.isfinite(k2) & (k2 > eps)

    npts = int(np.count_nonzero(good))
    if npts < int(min_points):
        idx_last = int(np.where(good)[0][-1]) if npts > 0 else -1
        k2_tmax = float(k2[idx_last]) if idx_last >= 0 else float("nan")
        return {"alpha": float("nan"), "r2": float("nan"), "n_points": npts, "k2_tmax": k2_tmax, "pass": False}

    x = np.log(t[good])
    y = np.log(k2[good])

    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    alpha = float(coef[1])

    idx_last = int(np.where(good)[0][-1])
    k2_tmax = float(k2[idx_last])

    passed = bool(
        np.isfinite(alpha)
        and np.isfinite(r2)
        and (r2 >= float(min_r2))
        and (k2_tmax >= float(k2_min_at_tmax))
    )
    return {"alpha": alpha, "r2": float(r2), "n_points": npts, "k2_tmax": k2_tmax, "pass": passed}


def _window_local_series(
    *,
    t_abs: np.ndarray,
    k2_abs: np.ndarray,
    dt: float,
    t_min: float,
    t_max: float,
    min_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    t_abs = np.asarray(t_abs, dtype=float)
    k2_abs = np.asarray(k2_abs, dtype=float)

    mask = (t_abs >= float(t_min)) & (t_abs <= float(t_max)) & np.isfinite(k2_abs) & np.isfinite(t_abs)
    tw = t_abs[mask]
    k2w = k2_abs[mask].astype(float, copy=True)

    if int(tw.size) < int(min_points):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    t0 = float(tw[0])
    k20 = float(k2w[0])
    tw = (tw - t0) + float(dt)
    k2w = (k2w - k20)
    return tw, k2w


def fit_window_shifted(
    *,
    t_abs: np.ndarray,
    k2_abs: np.ndarray,
    dt: float,
    t_min: float,
    t_max: float,
    min_r2: float,
    k2_min_at_tmax: float,
    min_points: int,
) -> Dict[str, float]:
    tw, k2w = _window_local_series(
        t_abs=t_abs,
        k2_abs=k2_abs,
        dt=dt,
        t_min=t_min,
        t_max=t_max,
        min_points=min_points,
    )
    if tw.size == 0:
        return {"alpha": float("nan"), "r2": float("nan"), "n_points": 0, "k2_tmax": float("nan"), "pass": False}
    return _fit_loglog(tw, k2w, min_r2=min_r2, k2_min_at_tmax=k2_min_at_tmax, min_points=min_points)


def in_range(alpha: float, lo: float, hi: float) -> bool:
    return bool(np.isfinite(alpha) and (alpha >= lo) and (alpha <= hi))


def main() -> int:
    p = argparse.ArgumentParser(description="S-0008: drift vs diffusion audit with fixed late-time window.")
    p.add_argument("--seed", type=int, default=S0008Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    p.add_argument("--min_r2", type=float, default=S0008Config.min_r2)
    p.add_argument("--k2_min_at_tmax", type=float, default=S0008Config.k2_min_at_tmax)
    args = p.parse_args()

    cfg = S0008Config(
        seed=int(args.seed),
        min_r2=float(args.min_r2),
        k2_min_at_tmax=float(args.k2_min_at_tmax),
    )

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)

    ou_lo = float(cfg.ou_alpha_target - cfg.ou_alpha_tol)
    ou_hi = float(cfg.ou_alpha_target + cfg.ou_alpha_tol)

    print(
        f"[S-0008] window=[{cfg.t_min:.1f},{cfg.t_max:.1f}] min_r2={cfg.min_r2:.3f} k2_min@tmax={cfg.k2_min_at_tmax:.3g} "
        f"OU alpha~{cfg.ou_alpha_target:.2f}±{cfg.ou_alpha_tol:.2f}"
    )

    rng_ou = np.random.default_rng(cfg.seed + 1)
    rng_drift = np.random.default_rng(cfg.seed + 2)

    omega_diff = simulate_ou(cfg, rng=rng_ou)
    omega_diff = center_global_dc(omega_diff)

    omega_drift = add_per_traj_linear_drift(omega_diff, cfg, rng=rng_drift)
    omega_drift = center_global_dc(omega_drift)

    def eval_case(case: str, omega: np.ndarray) -> Dict[str, object]:
        dphi = delta_phi_from_omega(omega, cfg.dt)
        k2 = k2_direct(dphi)

        fit = fit_window_shifted(
            t_abs=t,
            k2_abs=k2,
            dt=cfg.dt,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            min_r2=cfg.min_r2,
            k2_min_at_tmax=cfg.k2_min_at_tmax,
            min_points=cfg.min_points,
        )
        admissible = bool(fit["pass"])
        alpha = float(fit["alpha"])
        r2 = float(fit["r2"])
        k2_tmax = float(fit["k2_tmax"])
        n_points = int(fit["n_points"])

        if case == "DIFFUSION":
            if not admissible:
                tag = "BOUNDARY"
            else:
                tag = "OK" if in_range(alpha, ou_lo, ou_hi) else "BOUNDARY"
        elif case == "DRIFT":
            if not admissible:
                tag = "BOUNDARY"
            else:
                tag = "FAIL" if in_range(alpha, ou_lo, ou_hi) else "BOUNDARY"
        else:
            raise ValueError("unknown case")

        return {
            "case": case,
            "alpha": alpha,
            "r2": r2,
            "k2_tmax": k2_tmax,
            "n_points": n_points,
            "admissible": int(admissible),
            "tag": tag,
        }

    diff = eval_case("DIFFUSION", omega_diff)
    drift = eval_case("DRIFT", omega_drift)

    print(
        f"[S-0008:{diff['case']}] alpha={diff['alpha']:.3f} r2={diff['r2']:.3f} "
        f"k2_tmax={diff['k2_tmax']:.3g} n={diff['n_points']} admissible={diff['admissible']} tag={diff['tag']}"
    )
    print(
        f"[S-0008:{drift['case']}]     alpha={drift['alpha']:.3f} r2={drift['r2']:.3f} "
        f"k2_tmax={drift['k2_tmax']:.3g} n={drift['n_points']} admissible={drift['admissible']} tag={drift['tag']}"
    )

    any_admissible = bool(diff["admissible"]) or bool(drift["admissible"])

    if not any_admissible:
        print("[S-0008] INCONCLUSIVE: no admissible cases in declared window.")
        exit_code = 3
    else:
        if (diff["tag"] == "OK") and (drift["tag"] == "BOUNDARY"):
            print("[S-0008] AUDIT PASSED")
            exit_code = 0
        else:
            print("[S-0008] AUDIT FAILED")
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

        cases_path = out_dir / cfg.out_cases_csv
        rows: List[List[object]] = [
            ["case", "alpha", "r2", "k2_tmax", "n_points", "admissible", "tag"],
            [diff["case"], diff["alpha"], diff["r2"], diff["k2_tmax"], diff["n_points"], diff["admissible"], diff["tag"]],
            [drift["case"], drift["alpha"], drift["r2"], drift["k2_tmax"], drift["n_points"], drift["admissible"], drift["tag"]],
        ]
        _write_csv_with_provenance_header(cases_path, header_kv, rows)
        print(f"Wrote (untracked): {cases_path}")

        audit_path = out_dir / cfg.out_audit_csv
        audit_rows: List[List[object]] = [
            ["field", "value"],
            ["exit_code", exit_code],
            ["seed", int(cfg.seed)],
            ["dt", float(cfg.dt)],
            ["n_steps", int(cfg.n_steps)],
            ["n_trajectories", int(cfg.n_trajectories)],
            ["ou_theta", float(cfg.ou_theta)],
            ["ou_sigma", float(cfg.ou_sigma)],
            ["drift_beta_sigma", float(cfg.drift_beta_sigma)],
            ["t_min", float(cfg.t_min)],
            ["t_max", float(cfg.t_max)],
            ["min_r2", float(cfg.min_r2)],
            ["k2_min_at_tmax", float(cfg.k2_min_at_tmax)],
            ["min_points", int(cfg.min_points)],
            ["ou_alpha_target", float(cfg.ou_alpha_target)],
            ["ou_alpha_tol", float(cfg.ou_alpha_tol)],
        ]
        _write_csv_with_provenance_header(audit_path, header_kv, audit_rows)
        print(f"Wrote (untracked): {audit_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
