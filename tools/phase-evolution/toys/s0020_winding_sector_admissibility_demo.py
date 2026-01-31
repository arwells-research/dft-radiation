#!/usr/bin/env python3
"""
S-0020 — Winding-sector admissibility guard for κ2-slope regime claims (T-frame C1 gate)

Goal
----
Connect κ2(t) scaling admissibility to a concrete winding-sector constraint in the T-frame.

A process can exhibit an apparently admissible κ2-slope α (even within an LM-like band)
while violating a stepwise winding-sector condition: the per-step phase advance exceeds ±π.
Such "sector hops" represent an inadmissible regime for κ2-based attribution over that window.

S-0020 certifies that κ2-slope classification is admissible only when winding-sector
violations are absent in the declared audit window.

Construction
------------
Fixed cases over a declared late-time window:

  - C1_OU_BASE:
      Independent OU ω_i(t) (short-memory); expected α in OU band; no sector hops.

  - C2_LM_TRUE:
      Uncoupled long-memory Gaussian ω_i(t) (fractional Gaussian noise with fixed H);
      expected α in LM band; no sector hops.

  - C1_WIND_HOP:
      OU-like ω with rare deterministic spike injections that can push |Δϕ_step| > π:
        ω_i(t) := OU_i(t) + spike_i(t)
      Spikes are fixed-probability, fixed-amplitude, symmetric (no tuning / scanning).
      κ2-slope may appear misleadingly "fast" (sometimes LM-like), but must be refused
      when any winding-sector hop is detected in-window.

Centering
---------
Global DC removal only (one constant mean over ensemble×time), per case.

Diagnostics
-----------
1) κ2-slope classifier on κ2(t)=Var_i[Δϕ_i(t)], Δϕ=∫ω dt:
     α := slope of log κ2 vs log t over the declared window.
   Bands (fixed):
     - OU band: [ou_alpha_min, ou_alpha_max]
     - LM band: [lm_alpha_min, lm_alpha_max]

2) Winding-sector hop detector (declared, fixed; window-restricted):
   Per-step phase increment is:
       dϕ_i[k] := ω_i[k] * dt
   A "sector hop" occurs if |dϕ_i[k]| > π (stepwise phase advance crosses a winding boundary).

   We compute, restricted to the audit window steps:
       n_hops := count_{i,k in window_steps} 1[ |dϕ_i[k]| > π ]
       hop_detected := (n_hops > 0)

Certification rule
------------------
- Baselines must classify correctly with hop_detected=0:
    OU_BASE -> OK_OU
    LM_TRUE -> OK_LM
- Any case with hop_detected=1 is forced to BOUNDARY (never OK_*),
  forbidding false regime certification under winding-sector violation.

Exit codes
----------
  0 PASS
  2 FAIL
  3 INCONCLUSIVE

Outputs (only with --write_outputs)
----------------------------------
  toys/outputs/s0020_cases.csv
  toys/outputs/s0020_audit.csv
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
class S0020Config:
    seed: int = 20020

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

    # Spike injection (fixed; no tuning/scanning)
    # Probability per step per trajectory; amplitude in ω-units (so phase step is amp*dt)
    spike_prob: float = 0.008
    spike_amp: float = 40.0  # with dt=0.2 => phase step ~ 8 rad > π -> hop

    eps: float = 1e-12

    out_dir: str = "toys/outputs"
    out_cases_csv: str = "s0020_cases.csv"
    out_audit_csv: str = "s0020_audit.csv"


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


def simulate_ou(cfg: S0020Config, rng: np.random.Generator) -> np.ndarray:
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


def simulate_fgn(cfg: S0020Config, rng: np.random.Generator) -> np.ndarray:
    nT = int(cfg.n_trajectories)
    nS = int(cfg.n_steps)
    H = float(cfg.fgn_H)

    omega = np.empty((nT, nS), dtype=float)
    for i in range(nT):
        inc = fgn_davies_harte(nS, H, rng=rng)
        omega[i, :] = inc
    return omega


def apply_spike_injection(cfg: S0020Config, omega: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Rare symmetric spikes added to ω. Fixed probability, fixed amplitude.
    This is a *winding-sector violation* construction, not a tuned scaling design.
    """
    x = np.asarray(omega, dtype=float).copy()
    nT, nS = x.shape

    p = float(cfg.spike_prob)
    A = float(cfg.spike_amp)

    # Avoid k=0 init step; inject across 1..nS-1
    mask = (rng.random(size=(nT, nS)) < p)
    mask[:, 0] = False

    signs = np.where(rng.standard_normal(size=(nT, nS)) >= 0.0, 1.0, -1.0)
    x = x + mask.astype(float) * signs.astype(float) * A
    return x


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


def classify_alpha(cfg: S0020Config, alpha: float) -> str:
    if cfg.ou_alpha_min <= alpha <= cfg.ou_alpha_max:
        return "OU"
    if cfg.lm_alpha_min <= alpha <= cfg.lm_alpha_max:
        return "LM"
    return "BOUNDARY"


def winding_hops_count(cfg: S0020Config, omega: np.ndarray, mask: np.ndarray) -> int:
    """
    Count window-restricted winding-sector hops:
        hop if |dϕ_step| = |ω * dt| > π
    """
    x = np.asarray(omega, dtype=float)
    dt = float(cfg.dt)

    # window mask is on t[k] points; hops are between steps.
    idx = np.where(mask)[0]
    if idx.size < 2:
        return 0

    k0 = int(idx[0])
    k1 = int(idx[-1])

    dphi = x[:, k0:k1] * dt  # steps spanning window interior
    hops = np.abs(dphi) > np.pi
    return int(np.count_nonzero(hops))


def audit_case(cfg: S0020Config, name: str, omega: np.ndarray, t: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
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

    n_hops = winding_hops_count(cfg, x, mask)
    hop_detected = int(n_hops > 0)

    band = classify_alpha(cfg, float(alpha))
    if hop_detected:
        tag = "BOUNDARY"
    else:
        tag = "OK_OU" if band == "OU" else ("OK_LM" if band == "LM" else "BOUNDARY")

    return {
        "case": name,
        "verdict": "OK",
        "alpha": float(alpha),
        "r2": float(r2),
        "k2_end": float(k2_end),
        "n_hops": int(n_hops),
        "hop_detected": int(hop_detected),
        "tag": str(tag),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="S-0020: winding-sector admissibility guard for κ2-slope regime claims.")
    p.add_argument("--seed", type=int, default=S0020Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0020Config(seed=int(args.seed))

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)
    mask = _audit_mask(t, cfg.t_min, cfg.t_max)
    n_w = int(np.count_nonzero(mask))

    print(
        f"[S-0020] window=[{cfg.t_min:.1f},{cfg.t_max:.1f}] n_w={n_w} dt={cfg.dt:.3g} "
        f"admiss(min_r2={cfg.min_r2:.4f}, min_k2_end={cfg.min_k2_end:g}, min_points={cfg.min_points}) "
        f"OU_band=[{cfg.ou_alpha_min:.2f},{cfg.ou_alpha_max:.2f}] "
        f"LM_band=[{cfg.lm_alpha_min:.2f},{cfg.lm_alpha_max:.2f}] (H={cfg.fgn_H:g}) "
        f"winding_step_bound=pi spike_prob={cfg.spike_prob:g} spike_amp={cfg.spike_amp:g} nT={cfg.n_trajectories}"
    )

    if n_w < int(cfg.min_points):
        print("[S-0020] INCONCLUSIVE: audit window has too few points.")
        return 3

    rng_ou = np.random.default_rng(cfg.seed + 1)
    rng_lm = np.random.default_rng(cfg.seed + 2)
    rng_wh = np.random.default_rng(cfg.seed + 3)
    rng_sp = np.random.default_rng(cfg.seed + 4)

    omega_ou = simulate_ou(cfg, rng=rng_ou)
    omega_lm = simulate_fgn(cfg, rng=rng_lm)

    omega_wh = simulate_ou(cfg, rng=rng_wh)
    omega_wh = apply_spike_injection(cfg, omega_wh, rng=rng_sp)

    cases = [
        ("C1_OU_BASE", omega_ou),
        ("C2_LM_TRUE", omega_lm),
        ("C1_WIND_HOP", omega_wh),
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
            n_hops = r.get("n_hops", 0)
            hop = r.get("hop_detected", 0)
            print(
                f"[S-0020:{name}] verdict=INCONCLUSIVE reason={reason} "
                f"alpha={float(alpha):.4g} r2={float(r2):.4g} k2_end={float(k2_end):.4g} "
                f"n_hops={int(n_hops)} hop_detected={int(hop)}"
            )
            continue

        print(
            f"[S-0020:{name}] alpha={float(r['alpha']):.4g} r2={float(r['r2']):.4g} k2_end={float(r['k2_end']):.4g} "
            f"n_hops={int(r['n_hops'])} hop_detected={int(r['hop_detected'])} tag={str(r['tag'])}"
        )

    if any_inconclusive:
        print("[S-0020] INCONCLUSIVE")
        exit_code = 3
    else:
        ou_ok = (str(results["C1_OU_BASE"].get("tag")) == "OK_OU") and (int(results["C1_OU_BASE"].get("hop_detected", 0)) == 0)
        lm_ok = (str(results["C2_LM_TRUE"].get("tag")) == "OK_LM") and (int(results["C2_LM_TRUE"].get("hop_detected", 0)) == 0)

        wh_tag = str(results["C1_WIND_HOP"].get("tag"))
        wh_hop = int(results["C1_WIND_HOP"].get("hop_detected", 0))

        forbidden_ok = (wh_tag in ("OK_OU", "OK_LM")) and (wh_hop == 1)
        if forbidden_ok:
            print("[S-0020] AUDIT FAILED: forbidden OK_* under winding-sector hops (false-pass permitted).")
            exit_code = 2
        elif ou_ok and lm_ok and (wh_tag == "BOUNDARY") and (wh_hop == 1):
            print("[S-0020] AUDIT PASSED")
            exit_code = 0
        else:
            print(
                "[S-0020] AUDIT FAILED: "
                f"ou_ok={int(ou_ok)} lm_ok={int(lm_ok)} "
                f"wind_tag={wh_tag} hop_detected={wh_hop}"
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

        rows = [["case", "alpha", "r2", "k2_end", "n_hops", "hop_detected", "tag", "verdict", "reason"]]
        for nm in ["C1_OU_BASE", "C2_LM_TRUE", "C1_WIND_HOP"]:
            r = results[nm]
            rows.append(
                [
                    nm,
                    float(r.get("alpha", float("nan"))) if "alpha" in r else "",
                    float(r.get("r2", float("nan"))) if "r2" in r else "",
                    float(r.get("k2_end", float("nan"))) if "k2_end" in r else "",
                    int(r.get("n_hops", 0)) if "n_hops" in r else "",
                    int(r.get("hop_detected", 0)) if "hop_detected" in r else "",
                    str(r.get("tag", "")),
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
                ["spike_prob", float(cfg.spike_prob)],
                ["spike_amp", float(cfg.spike_amp)],
            ],
        )

        print(f"Wrote (untracked): {out_dir / cfg.out_cases_csv}")
        print(f"Wrote (untracked): {out_dir / cfg.out_audit_csv}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
