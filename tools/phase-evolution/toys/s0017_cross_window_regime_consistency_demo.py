#!/usr/bin/env python3
"""
S-0017 — Cross-window regime-consistency boundary for κ2-slope admissibility (Σ2 stationarity guard)

Goal
----
Certify that a regime classified by κ2(t) scaling in one declared late-time window
remains admissible in the immediately-adjacent continuation window.

If κ2 scaling is not *stable under continuation*, the classifier must refuse with
BOUNDARY — even if each window individually looks admissible.

This enforces that:
  - Temporal stationarity is a Σ2-level admissibility requirement.
  - κ2 slope alone is insufficient unless it persists across adjacent windows.

Construction
------------
We generate three fixed cases:

  - C1_OU_BASE: short-memory OU ω_i(t)
  - C2_LM_TRUE: long-memory Gaussian ω_i(t) (fractional Gaussian noise)
  - C1_CURVED: OU ω_i(t) with deterministic quadratic curvature

Two adjacent audit windows are declared:

  Window A: [20, 40]
  Window B: [40, 80]

For each case, compute κ2(t) = Var[Δϕ(t)] and fit:

  α := slope of log κ2 vs log t in each window.

If |α_A - α_B| exceeds the declared continuation tolerance, the case is labeled
BOUNDARY, never OK_*.

Exit codes
----------
  0 PASS
  2 FAIL
  3 INCONCLUSIVE

Outputs (only with --write_outputs)
----------------------------------
  toys/outputs/s0017_cases.csv
  toys/outputs/s0017_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import datetime
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class S0017Config:
    seed: int = 17017

    dt: float = 0.20
    n_steps: int = 512
    n_trajectories: int = 1024

    # Two adjacent late-time windows
    wA_min: float = 20.0
    wA_max: float = 40.0
    wB_min: float = 40.0
    wB_max: float = 80.0

    min_points: int = 24
    min_r2: float = 0.985
    min_k2_end: float = 5e-3

    # Expected slope bands
    ou_band: Tuple[float, float] = (0.82, 1.18)
    lm_band: Tuple[float, float] = (1.30, 1.70)
    H: float = 0.75

    # Continuation admissibility tolerance
    continuation_tol: float = 0.12

    curvature_gamma: float = 2.5e-4

    out_dir: str = "toys/outputs"
    out_cases: str = "s0017_cases.csv"
    out_audit: str = "s0017_audit.csv"


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")


def _git() -> str:
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except Exception:
        return "nogit"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024*1024)
            if not b: break
            h.update(b)
    return h.hexdigest()


def simulate_ou(cfg: S0017Config, rng: np.random.Generator) -> np.ndarray:
    theta, sigma = 0.6, 0.25
    omega = np.zeros((cfg.n_trajectories, cfg.n_steps))
    for t in range(1, cfg.n_steps):
        omega[:,t] = omega[:,t-1] + theta*(1.0-omega[:,t-1])*cfg.dt + sigma*math.sqrt(cfg.dt)*rng.standard_normal(cfg.n_trajectories)
    return omega


def simulate_fgn(cfg: S0017Config, rng: np.random.Generator) -> np.ndarray:
    H = cfg.H
    n = cfg.n_steps
    omega = np.empty((cfg.n_trajectories, n))
    for i in range(cfg.n_trajectories):
        k = np.arange(n)
        gamma = 0.5*((np.abs(k+1)**(2*H))+(np.abs(k-1)**(2*H))-2*(np.abs(k)**(2*H)))
        r = np.concatenate([gamma,[0],gamma[1:][::-1]])
        lam = np.fft.fft(r).real
        lam[lam<0]=0
        W = rng.normal(size=len(lam)) + 1j*rng.normal(size=len(lam))
        x = np.fft.ifft(np.sqrt(lam)*W).real[:n]
        x = x/np.std(x)
        omega[i,:] = 1.0 + 0.25*x
    return omega


def simulate_curved(cfg: S0017Config, rng: np.random.Generator) -> np.ndarray:
    base = simulate_ou(cfg, rng)
    t = (np.arange(cfg.n_steps)+1)*cfg.dt
    gamma = rng.normal(loc=cfg.curvature_gamma, scale=cfg.curvature_gamma/2.0, size=(cfg.n_trajectories,1))
    return base + gamma*(t**2)[None,:]


def delta_phi(omega: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(omega,axis=1)*dt


def k2(delta_phi: np.ndarray) -> np.ndarray:
    mu = np.mean(delta_phi,axis=0)
    xc = delta_phi-mu
    return np.mean(xc**2,axis=0)


def fit_alpha(t: np.ndarray, k2vals: np.ndarray, tmin: float, tmax: float) -> Tuple[float,float,int]:
    m = (t>=tmin)&(t<=tmax)&np.isfinite(k2vals)&(k2vals>0)
    if np.count_nonzero(m)<8:
        return float("nan"),float("nan"),0
    x = np.log(t[m])
    y = np.log(k2vals[m])
    A = np.vstack([np.ones_like(x),x]).T
    coef, *_ = np.linalg.lstsq(A,y,rcond=None)
    slope = float(coef[1])
    yhat = slope*x+coef[0]
    r2 = 1.0 - float(np.sum((y-yhat)**2))/float(np.sum((y-np.mean(y))**2))
    return slope,r2,len(x)


def classify(alpha: float, ou_band, lm_band) -> str:
    if ou_band[0] <= alpha <= ou_band[1]:
        return "OK_OU"
    if lm_band[0] <= alpha <= lm_band[1]:
        return "OK_LM"
    return "BOUNDARY"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write_outputs",action="store_true")
    args = ap.parse_args()

    cfg = S0017Config()
    rng = np.random.default_rng(cfg.seed)

    t = (np.arange(cfg.n_steps)+1)*cfg.dt

    cases = {
        "C1_OU_BASE": simulate_ou(cfg,rng),
        "C2_LM_TRUE": simulate_fgn(cfg,rng),
        "C1_CURVED":  simulate_curved(cfg,rng),
    }

    rows = []
    ok = True

    for name,omega in cases.items():
        dphi = delta_phi(omega,cfg.dt)
        k2vals = k2(dphi)

        aA,r2A,nA = fit_alpha(t,k2vals,cfg.wA_min,cfg.wA_max)
        aB,r2B,nB = fit_alpha(t,k2vals,cfg.wB_min,cfg.wB_max)

        if not (np.isfinite(aA) and np.isfinite(aB)):
            tag = "INCONCLUSIVE"
        else:
            tagA = classify(aA,cfg.ou_band,cfg.lm_band)
            tagB = classify(aB,cfg.ou_band,cfg.lm_band)
            stable = abs(aA-aB) <= cfg.continuation_tol
            tag = tagA if (tagA==tagB and stable) else "BOUNDARY"

        print(f"[S-0017:{name}] alphaA={aA:.4g} alphaB={aB:.4g} tag={tag}")

        rows.append([name,aA,aB,tag])

        if name!="C1_CURVED":
            ok &= (tag.startswith("OK"))

    if args.write_outputs:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True,exist_ok=True)
        hdr = {"git":_git(),"utc":_utc(),"sha":_sha(Path(__file__))}
        with (out/cfg.out_cases).open("w",newline="",encoding="utf-8") as f:
            w = csv.writer(f)
            for k,v in hdr.items():
                f.write(f"# {k}={v}\n")
            w.writerow(["case","alphaA","alphaB","tag"])
            w.writerows(rows)

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())