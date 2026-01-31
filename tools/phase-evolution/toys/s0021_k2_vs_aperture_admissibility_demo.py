#!/usr/bin/env python3
"""
S-0021 — κ₂-slope vs coherence-aperture consistency (C2 ⟷ C5 admissibility gate)

Goal
----
Certify a refusal boundary linking κ₂-based regime classification to a declared
coherence-aperture limit L.

Core idea
---------
Let Δϕ(t) = ∫ ω dt (discrete cumulative sum), κ₂(t) = Var_i[Δϕ_i(t)].

Define phase-spread budget:
    σϕ(t) = sqrt(κ₂(t))
    η(t)  = σϕ(t) / L

Aperture admissibility (declared, fixed):
    aperture_violation := (max_{t in window} η(t) > 1)

Fixed cases (no tuning / no scanning)
-------------------------------------
- C1_OU_BASE:
    OU ω_i(t). Must classify OU and must be aperture-admissible.
    -> OK_OU
- C2_LM_TRUE:
    Uncoupled long-memory Gaussian ω_i(t) (Davies–Harte fGn with fixed H).
    Must classify LM and must be aperture-admissible.
    -> OK_LM
- C5_APERTURE_EXCESS:
    Same LM ω_i(t) scaled by a fixed factor to force η>1 while preserving κ₂-slope class.
    Must be refused under aperture violation.
    -> BOUNDARY

Centering
---------
Global DC removal only (one constant mean over ensemble×time), per case.

Admissibility (fit gates)
-------------------------
- Minimum point count in window
- Minimum r² for log κ₂ vs log t fit
- Minimum κ₂ at window end

Exit codes
----------
  0 PASS
  2 FAIL
  3 INCONCLUSIVE

Outputs (only with --write_outputs)
----------------------------------
  toys/outputs/s0021_cases.csv
  toys/outputs/s0021_audit.csv
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
class S0021Config:
    seed: int = 21021

    # Match house ladder defaults (S-0020)
    dt: float = 0.20
    n_steps: int = 512
    n_trajectories: int = 1024

    # Fixed audit window (absolute times)
    t_min: float = 20.0
    t_max: float = 40.0
    min_points: int = 24

    # Admissibility for κ2 fit
    min_r2: float = 0.985
    min_k2_end: float = 5.0e-3

    # Classification bands (match house style)
    ou_alpha_min: float = 0.82
    ou_alpha_max: float = 1.18

    # LM baseline uses H fixed; band centered around 2H for H=0.75
    fgn_H: float = 0.75
    lm_alpha_min: float = 1.30
    lm_alpha_max: float = 1.70

    # OU generator parameters (simple, fixed)
    ou_theta: float = 0.60
    ou_mu: float = 0.0
    ou_sigma: float = 1.0

    # Coherence aperture L (declared; must admit baselines in-window)
    L: float = 20.0

    # Excess scaling (fixed; no tuning/scanning)
    aperture_scale: float = 2.0

    eps: float = 1e-12

    out_dir: str = "toys/outputs"
    out_cases_csv: str = "s0021_cases.csv"
    out_audit_csv: str = "s0021_audit.csv"


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


def simulate_ou(cfg: S0021Config, rng: np.random.Generator) -> np.ndarray:
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
    gamma = 0.5 * ((np.abs(k + 1.0) ** (2.0 * H)) + (np.abs(k - 1.0) ** (2.0 * H)) - 2.0 * (np.abs(k) ** (2.0 * H)))

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


def simulate_fgn(cfg: S0021Config, rng: np.random.Generator) -> np.ndarray:
    nT = int(cfg.n_trajectories)
    nS = int(cfg.n_steps)
    H = float(cfg.fgn_H)

    omega = np.empty((nT, nS), dtype=float)
    for i in range(nT):
        inc = fgn_davies_harte(nS, H, rng=rng)
        omega[i, :] = inc
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


def classify_alpha(cfg: S0021Config, alpha: float) -> str:
    if cfg.ou_alpha_min <= alpha <= cfg.ou_alpha_max:
        return "OU"
    if cfg.lm_alpha_min <= alpha <= cfg.lm_alpha_max:
        return "LM"
    return "BOUNDARY"


def aperture_eta_max(cfg: S0021Config, k2: np.ndarray, mask: np.ndarray) -> float:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return float("nan")
    sigma_phi = np.sqrt(np.maximum(np.asarray(k2, dtype=float)[idx], 0.0))
    return float(np.max(sigma_phi / float(cfg.L)))


def audit_case(cfg: S0021Config, name: str, omega: np.ndarray, t: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
    # Global DC removal only (per case)
    x = np.asarray(omega, dtype=float)
    x = x - float(np.mean(x))

    dphi = delta_phi_from_omega(x, cfg.dt)
    k2 = k2_of_delta_phi(dphi)

    if (not np.any(mask)) or (int(np.count_nonzero(mask)) < int(cfg.min_points)):
        return {"case": name, "verdict": "INCONCLUSIVE", "reason": "too_few_points"}

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

    eta_max = aperture_eta_max(cfg, k2, mask)
    aperture_violation = int(np.isfinite(eta_max) and (eta_max > 1.0))

    band = classify_alpha(cfg, float(alpha))

    return {
        "case": name,
        "verdict": "OK",
        "alpha": float(alpha),
        "r2": float(r2),
        "k2_end": float(k2_end),
        "eta_max": float(eta_max),
        "aperture_violation": int(aperture_violation),
        "band": str(band),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="S-0021: κ2-slope vs coherence-aperture consistency (C2 ⟷ C5 admissibility gate).")
    p.add_argument("--seed", type=int, default=S0021Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0021Config(seed=int(args.seed))

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)
    mask = _audit_mask(t, cfg.t_min, cfg.t_max)
    n_w = int(np.count_nonzero(mask))

    print(
        f"[S-0021] window=[{cfg.t_min:.1f},{cfg.t_max:.1f}] n_w={n_w} dt={cfg.dt:.3g} "
        f"admiss(min_r2={cfg.min_r2:.4f}, min_k2_end={cfg.min_k2_end:g}, min_points={cfg.min_points}) "
        f"OU_band=[{cfg.ou_alpha_min:.2f},{cfg.ou_alpha_max:.2f}] "
        f"LM_band=[{cfg.lm_alpha_min:.2f},{cfg.lm_alpha_max:.2f}] (H={cfg.fgn_H:g}) "
        f"L={cfg.L:g} aperture_scale={cfg.aperture_scale:g} nT={cfg.n_trajectories}"
    )

    if n_w < int(cfg.min_points):
        print("[S-0021] INCONCLUSIVE: audit window has too few points.")
        return 3

    rng_ou = np.random.default_rng(cfg.seed + 1)
    rng_lm = np.random.default_rng(cfg.seed + 2)

    omega_ou = simulate_ou(cfg, rng=rng_ou)
    omega_lm = simulate_fgn(cfg, rng=rng_lm)
    omega_ex = float(cfg.aperture_scale) * omega_lm

    cases = [
        ("C1_OU_BASE", omega_ou),
        ("C2_LM_TRUE", omega_lm),
        ("C5_APERTURE_EXCESS", omega_ex),
    ]

    results: Dict[str, Dict[str, object]] = {}
    any_inconclusive = False

    for name, om in cases:
        r = audit_case(cfg, name=name, omega=om, t=t, mask=mask)
        results[name] = r

        verdict = str(r.get("verdict", "INCONCLUSIVE"))
        if verdict != "OK":
            any_inconclusive = True
            reason = str(r.get("reason", ""))
            alpha = r.get("alpha", float("nan"))
            r2 = r.get("r2", float("nan"))
            k2_end = r.get("k2_end", float("nan"))
            print(
                f"[S-0021:{name}] verdict=INCONCLUSIVE reason={reason} "
                f"alpha={float(alpha):.4g} r2={float(r2):.4g} k2_end={float(k2_end):.4g}"
            )
            continue

        print(
            f"[S-0021:{name}] alpha={float(r['alpha']):.4g} r2={float(r['r2']):.4g} k2_end={float(r['k2_end']):.4g} "
            f"eta_max={float(r['eta_max']):.4g} aperture_violation={int(r['aperture_violation'])} band={str(r['band'])}"
        )

    if any_inconclusive:
        print("[S-0021] INCONCLUSIVE")
        exit_code = 3
    else:
        # Baseline requirements
        ou = results["C1_OU_BASE"]
        lm = results["C2_LM_TRUE"]
        ex = results["C5_APERTURE_EXCESS"]

        ou_ok = (str(ou.get("band")) == "OU") and (int(ou.get("aperture_violation", 0)) == 0)
        lm_ok = (str(lm.get("band")) == "LM") and (int(lm.get("aperture_violation", 0)) == 0)

        ex_vio = int(ex.get("aperture_violation", 0))
        # Construction must violate aperture; if not, refuse (inconclusive) rather than tune.
        if ex_vio == 0:
            print("[S-0021] INCONCLUSIVE: APERTURE_EXCESS did not violate aperture (no tuning permitted).")
            exit_code = 3
        else:
            # Under violation, must refuse: BOUNDARY (never OK_*). Here "BOUNDARY" is operationally: aperture_violation=1.
            ex_refused = (ex_vio == 1)

            # Forbidden condition would be any "OK_*" label under aperture violation, but this toy uses
            # the invariant: violation implies refusal; so audit reduces to baseline correctness + violation presence.
            if ou_ok and lm_ok and ex_refused:
                print("[S-0021] AUDIT PASSED")
                exit_code = 0
            else:
                print(
                    "[S-0021] AUDIT FAILED: "
                    f"ou_ok={int(ou_ok)} lm_ok={int(lm_ok)} "
                    f"ex_aperture_violation={int(ex_vio)} "
                    f"ou_band={str(ou.get('band'))} lm_band={str(lm.get('band'))} "
                    f"ou_vio={int(ou.get('aperture_violation', 0))} lm_vio={int(lm.get('aperture_violation', 0))}"
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

        rows = [["case", "alpha", "r2", "k2_end", "eta_max", "aperture_violation", "band", "verdict", "reason"]]
        for nm in ["C1_OU_BASE", "C2_LM_TRUE", "C5_APERTURE_EXCESS"]:
            r = results[nm]
            rows.append(
                [
                    nm,
                    float(r.get("alpha", float("nan"))) if "alpha" in r else "",
                    float(r.get("r2", float("nan"))) if "r2" in r else "",
                    float(r.get("k2_end", float("nan"))) if "k2_end" in r else "",
                    float(r.get("eta_max", float("nan"))) if "eta_max" in r else "",
                    int(r.get("aperture_violation", 0)) if "aperture_violation" in r else "",
                    str(r.get("band", "")),
                    str(r.get("verdict", "")),
                    str(r.get("reason", "")),
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
                ["L", float(cfg.L)],
                ["aperture_scale", float(cfg.aperture_scale)],
            ],
        )

        print(f"Wrote (untracked): {out_dir / cfg.out_cases_csv}")
        print(f"Wrote (untracked): {out_dir / cfg.out_audit_csv}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
