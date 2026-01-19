#!/usr/bin/env python3
"""
S-0012 — κ₆ even-cumulant magnitude sufficiency boundary

Tests whether higher-order even cumulants (κ₆) produce a magnitude-envelope
deviation beyond κ₂+κ₄ closure.

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
from typing import Dict, List
import numpy as np


@dataclass(frozen=True)
class S0012Config:
    seed: int = 12012
    dt: float = 0.02
    n_steps: int = 4096
    n_trajectories: int = 1024

    t_min: float = 40.0
    t_max: float = 70.0

    min_r2: float = 0.985
    k2_min_at_tmax: float = 0.50
    min_points: int = 18

    max_median_abs_dlogc: float = 0.08
    max_p95_abs_dlogc: float = 0.20
    max_abs_drift_slope: float = 0.06

    coherence_floor: float = 1e-3

    out_dir: str = "toys/outputs"
    out_cases_csv: str = "s0012_cases.csv"
    out_audit_csv: str = "s0012_audit.csv"


def _run_utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def delta_phi_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(omega, axis=1) * dt


def coherence(delta_phi: np.ndarray) -> np.ndarray:
    return np.mean(np.exp(1j*delta_phi), axis=0)


def cumulants_2_4_6(delta_phi: np.ndarray):
    mu = np.mean(delta_phi, axis=0)
    xc = delta_phi - mu
    m2 = np.mean(xc**2, axis=0)
    m4 = np.mean(xc**4, axis=0)
    m6 = np.mean(xc**6, axis=0)
    k2 = m2
    k4 = m4 - 3*m2**2
    k6 = m6 - 15*m4*m2 + 30*m2**3
    return k2, k4, k6


def predict_mag(k2: np.ndarray, k4: np.ndarray) -> np.ndarray:
    return np.exp(-0.5*k2 + k4/24.0)


def simulate_gauss(cfg: S0012Config, rng: np.random.Generator):
    return rng.standard_normal((cfg.n_trajectories, cfg.n_steps))


def simulate_k6(cfg: S0012Config, rng: np.random.Generator, sign: float):
    base = rng.standard_normal((cfg.n_trajectories, cfg.n_steps))
    # inject rare symmetric large increments to modify κ6 only
    mask = rng.random(base.shape) < 0.01
    base[mask] += sign * rng.choice([-3,3], size=np.count_nonzero(mask))
    return base


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0012Config()
    rng = np.random.default_rng(cfg.seed)

    t = (np.arange(cfg.n_steps)+1)*cfg.dt
    win = (t>=cfg.t_min)&(t<=cfg.t_max)

    cases = {
        "GAUSS": simulate_gauss(cfg, rng),
        "K6+": simulate_k6(cfg, rng, +1.0),
        "K6-": simulate_k6(cfg, rng, -1.0),
    }

    rows: List[List[object]] = []
    any_fail=False
    any_inconclusive=False

    for name, omega in cases.items():
        dphi = delta_phi_from_omega(omega, cfg.dt)
        z = coherence(dphi)
        c = np.abs(z)

        k2,k4,k6 = cumulants_2_4_6(dphi)
        c_pred = predict_mag(k2,k4)

        if np.min(c[win]) < cfg.coherence_floor:
            print(f"[S-0012:{name}] INCONCLUSIVE (low coherence)")
            any_inconclusive=True
            tag="INCONCLUSIVE"
        else:
            dlog = np.log(c[win]+1e-18) - np.log(c_pred[win]+1e-18)
            med=np.median(np.abs(dlog))
            p95=np.quantile(np.abs(dlog),0.95)
            drift=np.polyfit(t[win], dlog,1)[0]
            ok=(med<=cfg.max_median_abs_dlogc and p95<=cfg.max_p95_abs_dlogc and abs(drift)<=cfg.max_abs_drift_slope)
            tag="OK" if ok else "FAIL"
            any_fail |= (tag=="FAIL")
            print(f"[S-0012:{name}] med|dlogc|={med:.3g} p95={p95:.3g} drift={drift:.3g} κ6(t*)={k6[win][-1]:.3g} tag={tag}")

        rows.append([name,tag])

    exit_code = 2 if any_fail else (3 if any_inconclusive else 0)
    print("[S-0012] AUDIT PASSED" if exit_code==0 else "[S-0012] AUDIT FAILED")

    if args.write_outputs:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        header={"git_commit":_git_commit_short(),"run_utc":_run_utc_iso(),"toy_file":Path(__file__).name,"sha256":_sha256_of_file(Path(__file__))}
        with (out/cfg.out_cases_csv).open("w",newline="") as f:
            for k,v in header.items(): f.write(f"# {k}={v}\n")
            w=csv.writer(f); w.writerow(["case","tag"]); w.writerows(rows)
        with (out/cfg.out_audit_csv).open("w",newline="") as f:
            for k,v in header.items(): f.write(f"# {k}={v}\n")
            w=csv.writer(f); w.writerow(["exit_code",exit_code])

    return exit_code


if __name__=="__main__":
    raise SystemExit(main())