#!/usr/bin/env python3
"""
S-0010 — κ₂ transport vs interface discrimination (C3 ⟷ C4 identifiability)

This toy certifies that κ₂(t)=Var(Δϕ(t)) scaling remains invariant under
declared geometric transport (C3), but changes detectably under interface
coupling or injection (C4).

Exit codes:
  0 — PASS
  2 — FAIL
  3 — INCONCLUSIVE
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
class S0010Config:
    seed: int = 10010

    dt: float = 0.02
    n_steps: int = 4096
    n_trajectories: int = 512

    ou_theta: float = 0.35
    ou_sigma: float = 1.0

    t_min: float = 40.0
    t_max: float = 70.0

    min_r2: float = 0.985
    k2_min_at_tmax: float = 0.50
    min_points: int = 18

    ou_alpha_target: float = 1.0
    ou_alpha_tol: float = 0.15

    # Invariance criterion: compare alpha to OU_BASE (declared reference)
    max_abs_delta_alpha: float = 0.03
    max_abs_delta_r2: float = 0.01

    c3_m: int = 8

    shot_rate_per_time: float = 0.05
    shot_amp_mean: float = 0.75
    shot_tau_samples: int = 16

    out_dir: str = "toys/outputs"
    out_cases_csv: str = "s0010_cases.csv"
    out_audit_csv: str = "s0010_audit.csv"


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


def simulate_ou(cfg: S0010Config, rng: np.random.Generator) -> np.ndarray:
    nT, n = cfg.n_trajectories, cfg.n_steps
    dt = cfg.dt
    theta = cfg.ou_theta
    sigma = cfg.ou_sigma

    omega = np.empty((nT, n))
    sdt = np.sqrt(dt)
    for i in range(nT):
        x = rng.normal(scale=(sigma / np.sqrt(2 * theta)))
        for k in range(n):
            x += (-theta * x) * dt + sigma * sdt * rng.standard_normal()
            omega[i, k] = x
    return omega


def fir_smooth_per_traj(x: np.ndarray, m: int) -> np.ndarray:
    k = np.ones(m) / float(m)
    pad = m // 2
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        xp = np.pad(x[i], (pad, pad), mode="reflect")
        out[i] = np.convolve(xp, k, mode="valid")[: x.shape[1]]
    return out


def add_shot_in_omega(omega: np.ndarray, cfg: S0010Config, rng: np.random.Generator) -> np.ndarray:
    y = omega.copy()
    lam = cfg.shot_rate_per_time
    tau = cfg.shot_tau_samples
    mean_events = lam * (cfg.n_steps * cfg.dt)

    kernel = np.exp(-np.arange(tau * 8) / float(tau))
    for i in range(y.shape[0]):
        n_events = rng.poisson(mean_events)
        if n_events == 0:
            continue
        t_idx = rng.integers(0, cfg.n_steps, size=n_events)
        amps = rng.exponential(scale=cfg.shot_amp_mean, size=n_events)
        for t0, A in zip(t_idx, amps):
            t1 = min(cfg.n_steps, t0 + kernel.size)
            y[i, t0:t1] += A * kernel[: (t1 - t0)]
    return y


def delta_phi_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(omega, axis=1) * dt


def k2_direct(delta_phi: np.ndarray) -> np.ndarray:
    return np.var(delta_phi, axis=0)


def fit_loglog(t: np.ndarray, k2: np.ndarray, cfg: S0010Config) -> Tuple[float, float, int]:
    eps = 1e-18
    good = (t > 0) & (k2 > eps)

    if np.count_nonzero(good) < cfg.min_points:
        return float("nan"), float("nan"), 0

    x = np.log(t[good])
    y = np.log(k2[good])
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(coef[1]), r2, int(np.count_nonzero(good))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0010Config()

    t = (np.arange(cfg.n_steps) + 1) * cfg.dt
    mask = (t >= cfg.t_min) & (t <= cfg.t_max)

    omega_base = simulate_ou(cfg, rng=np.random.default_rng(cfg.seed + 1))

    cases = {
        "OU_BASE": omega_base,
        "OU_C3_SMOOTH": fir_smooth_per_traj(omega_base, cfg.c3_m),
        "OU_C3_DISP": fir_smooth_per_traj(omega_base, cfg.c3_m * 2),
        "OU_C4_SHOT": add_shot_in_omega(omega_base, cfg, rng=np.random.default_rng(cfg.seed + 3)),
        "OU_C4_GAINLOSS": add_shot_in_omega(omega_base * 0.5, cfg, rng=np.random.default_rng(cfg.seed + 4)),
    }

    # First pass: compute metrics for all cases.
    metrics: Dict[str, Dict[str, float | int | bool]] = {}
    for name, omega in cases.items():
        dphi = delta_phi_from_omega(omega, cfg.dt)
        k2 = k2_direct(dphi)

        alpha, r2, npts = fit_loglog(t[mask], k2[mask], cfg)

        admissible = bool(
            np.isfinite(alpha)
            and np.isfinite(r2)
            and (r2 >= float(cfg.min_r2))
            and (k2[mask][-1] >= float(cfg.k2_min_at_tmax))
            and (int(npts) >= int(cfg.min_points))
        )

        metrics[name] = {
            "alpha": float(alpha),
            "r2": float(r2),
            "npts": int(npts),
            "admissible": bool(admissible),
        }

    # Reference: OU_BASE.
    base = metrics.get("OU_BASE", None)
    if base is None:
        print("[S-0010] AUDIT FAILED: missing OU_BASE")
        return 2

    base_adm = bool(base["admissible"])
    alpha0 = float(base["alpha"])
    r20 = float(base["r2"])

    rows: List[List[object]] = []
    any_fail = False
    any_inconclusive = False

    # Second pass: compare each case to OU_BASE (invariance test).
    for name in sorted(metrics.keys()):
        m = metrics[name]
        alpha = float(m["alpha"])
        r2 = float(m["r2"])
        npts = int(m["npts"])
        admissible = bool(m["admissible"])

        if (not base_adm) or (not admissible):
            tag = "INCONCLUSIVE"
            any_inconclusive = True
            d_alpha = float("nan")
            d_r2 = float("nan")
        else:
            d_alpha = float(abs(alpha - alpha0))
            d_r2 = float(abs(r2 - r20))

            ok = bool(
                (d_alpha <= float(cfg.max_abs_delta_alpha))
                and (d_r2 <= float(cfg.max_abs_delta_r2))
            )
            tag = "OK" if ok else "FAIL"
            if tag == "FAIL":
                any_fail = True

        print(
            f"[S-0010:{name}] "
            f"alpha={alpha:.3f} r2={r2:.3f} n={npts} "
            f"base_alpha={alpha0:.3f} d_alpha={d_alpha:.3g} d_r2={d_r2:.3g} "
            f"admissible={int(admissible)} tag={tag}"
        )

        rows.append([name, alpha, r2, npts, d_alpha, d_r2, int(admissible), tag])

    if any_fail:
        print("[S-0010] AUDIT FAILED")
        exit_code = 2
    elif any_inconclusive:
        print("[S-0010] INCONCLUSIVE")
        exit_code = 3
    else:
        print("[S-0010] AUDIT PASSED")
        exit_code = 0

    if args.write_outputs:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        header = {
            "git_commit": _git_commit_short(),
            "run_utc": _run_utc_iso(),
            "toy_file": Path(__file__).name,
            "source_sha256": _sha256_of_file(Path(__file__)),
        }

        _write_csv_with_provenance_header(
            out_dir / cfg.out_cases_csv,
            header,
            [
                ["case", "alpha", "r2", "n_points", "abs_delta_alpha_vs_base", "abs_delta_r2_vs_base", "admissible", "tag"],
                *rows,
            ],
        )
        _write_csv_with_provenance_header(
            out_dir / cfg.out_audit_csv,
            header,
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
                ["k2_min_at_tmax", float(cfg.k2_min_at_tmax)],
                ["min_points", int(cfg.min_points)],
                ["max_abs_delta_alpha", float(cfg.max_abs_delta_alpha)],
                ["max_abs_delta_r2", float(cfg.max_abs_delta_r2)],
            ],
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
