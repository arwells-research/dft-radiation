#!/usr/bin/env python3
"""
S-0022 — Estimator-manufactured κ2 scaling guard (C5 integrity gate)

Goal
----
Certify that κ2-slope regime classification (OU vs long-memory) is admissible only
when the inference pipeline does not contain *declared estimator pathologies* that
can manufacture or invalidate scaling.

S-0022 is explicitly a C5 refusal gate: it does NOT attempt to “fix” estimator bias.
It certifies that certain inference choices must force BOUNDARY (never OK_*),
even if κ2-slope appears to fall in-band.

Core objects
------------
Δϕ_i(t) = ∫ ω_i dt  (discrete cumulative sum with fixed dt used by the inference)
κ2(t)   = Var_i[Δϕ_i(t)]
α       = slope of log κ2 vs log t over a fixed audit window

Estimator integrity flags (declared, fixed)
-------------------------------------------
We certify three concrete C5 confounds:

1) OVERLAP_REUSE:
   Heavy moving-average smoothing with stride=1 reuses samples W times (reuse_factor=W).
   Rule: if reuse_factor > reuse_factor_max -> force BOUNDARY.

2) RESAMPLE_DT_MISMATCH:
   Downsampling by factor m changes the effective time step to dt_true = m*dt.
   If inference mistakenly continues to integrate using dt_used=dt, it can fabricate α.
   Rule: if dt_mismatch is present -> force BOUNDARY (fit may still be computed).

3) DIFFERENCING:
   Replacing ω with Δω changes the modeled observable (units + spectrum).
   Without an explicit model binding, κ2-slope attribution on Δω must be refused.
   Rule: if differencing is applied -> force BOUNDARY immediately (no fit required).

Baselines
---------
- C1_OU_BASE: independent OU ω_i(t) -> α in OU band -> OK_OU
- C2_LM_TRUE: uncoupled fGn ω_i(t) with fixed H -> α in LM band -> OK_LM

Certification rule
------------------
- Baselines must classify correctly (OK_OU / OK_LM) with no integrity flags.
- Any case carrying a declared estimator-integrity violation is forced to BOUNDARY
  (never OK_*), regardless of κ2-slope band.

Inadmissible fits (r²/point-count/κ2 floor) => INCONCLUSIVE (no claim),
except for DIFFERENCING which is refused by construction.

No scanning. No tuning. No post-hoc threshold adjustment.

Exit codes
----------
  0 PASS
  2 FAIL
  3 INCONCLUSIVE

Outputs (only with --write_outputs)
----------------------------------
  toys/outputs/s0022_cases.csv
  toys/outputs/s0022_audit.csv
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
class S0022Config:
    seed: int = 22022

    dt: float = 0.20
    n_steps: int = 1024
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

    # Estimator confound parameters (fixed; no tuning/scanning)
    ma_window: int = 25            # overlap reuse factor with stride=1
    reuse_factor_max: int = 4      # declared: >4x reuse forces refusal
    resample_factor: int = 4       # downsample by m (true dt becomes m*dt)

    eps: float = 1e-12

    out_dir: str = "toys/outputs"
    out_cases_csv: str = "s0022_cases.csv"
    out_audit_csv: str = "s0022_audit.csv"


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


def delta_phi_from_omega(omega: np.ndarray, dt_used: float) -> np.ndarray:
    return np.cumsum(np.asarray(omega, dtype=float), axis=1) * float(dt_used)


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


def classify_alpha(cfg: S0022Config, alpha: float) -> str:
    if cfg.ou_alpha_min <= alpha <= cfg.ou_alpha_max:
        return "OU"
    if cfg.lm_alpha_min <= alpha <= cfg.lm_alpha_max:
        return "LM"
    return "BOUNDARY"


def simulate_ou(cfg: S0022Config, rng: np.random.Generator) -> np.ndarray:
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


def simulate_fgn(cfg: S0022Config, rng: np.random.Generator) -> np.ndarray:
    nT = int(cfg.n_trajectories)
    nS = int(cfg.n_steps)
    H = float(cfg.fgn_H)

    omega = np.empty((nT, nS), dtype=float)
    for i in range(nT):
        inc = fgn_davies_harte(nS, H, rng=rng)
        omega[i, :] = inc
    return omega


def moving_average_overlap(cfg: S0022Config, omega: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    x = np.asarray(omega, dtype=float)
    w = int(cfg.ma_window)
    if w < 1:
        return x.copy(), {"reuse_factor": 1, "reuse_violation": 0}

    kernel = np.ones(w, dtype=float) / float(w)
    y = np.empty_like(x)
    for i in range(x.shape[0]):
        y[i, :] = np.convolve(x[i, :], kernel, mode="same")

    reuse_factor = w
    reuse_violation = int(reuse_factor > int(cfg.reuse_factor_max))
    return y, {"reuse_factor": reuse_factor, "reuse_violation": reuse_violation}


def resample_dt_mismatch(cfg: S0022Config, omega: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    x = np.asarray(omega, dtype=float)
    m = int(cfg.resample_factor)
    if m <= 1:
        return x.copy(), {"resample_factor": 1, "dt_mismatch": 0}

    y = x[:, ::m].copy()
    return y, {"resample_factor": m, "dt_mismatch": 1, "dt_true": float(cfg.dt) * float(m), "dt_used": float(cfg.dt)}


def differencing_transform(cfg: S0022Config, omega: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    x = np.asarray(omega, dtype=float)
    y = np.diff(x, axis=1)
    return y, {"diff_applied": 1}


def audit_case(
    cfg: S0022Config,
    name: str,
    omega: np.ndarray,
    t: np.ndarray,
    mask: np.ndarray,
    meta: Dict[str, object],
) -> Dict[str, object]:
    # Global DC removal only (per case)
    x = np.asarray(omega, dtype=float)
    x = x - float(np.mean(x))

    # Immediate C5 refusal for differencing (declared: changes observable)
    diff_applied = int(meta.get("diff_applied", 0))
    if diff_applied == 1:
        return {
            "case": name,
            "verdict": "OK",
            "alpha": "",
            "r2": "",
            "k2_end": "",
            "band": "",
            "tag": "BOUNDARY",
            "notes": "c5_diff_refusal",
            "reuse_factor": "",
            "reuse_violation": "",
            "resample_factor": "",
            "dt_mismatch": "",
            "diff_applied": 1,
        }

    if (not np.any(mask)) or (int(np.count_nonzero(mask)) < int(cfg.min_points)):
        return {"case": name, "verdict": "INCONCLUSIVE", "reason": "too_few_points", **meta}

    dphi = delta_phi_from_omega(x, dt_used=float(cfg.dt))
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
            **meta,
        }

    band = classify_alpha(cfg, float(alpha))

    reuse_violation = int(meta.get("reuse_violation", 0))
    dt_mismatch = int(meta.get("dt_mismatch", 0))

    integrity_violation = int((reuse_violation == 1) or (dt_mismatch == 1))
    if integrity_violation:
        tag = "BOUNDARY"
        notes = "c5_estimator_violation"
    else:
        tag = "OK_OU" if band == "OU" else ("OK_LM" if band == "LM" else "BOUNDARY")
        notes = "ok" if tag.startswith("OK_") else "out_of_band"

    return {
        "case": name,
        "verdict": "OK",
        "alpha": float(alpha),
        "r2": float(r2),
        "k2_end": float(k2_end),
        "band": str(band),
        "tag": str(tag),
        "notes": str(notes),
        "reuse_factor": int(meta.get("reuse_factor", 0)) if "reuse_factor" in meta else "",
        "reuse_violation": int(meta.get("reuse_violation", 0)) if "reuse_violation" in meta else "",
        "resample_factor": int(meta.get("resample_factor", 0)) if "resample_factor" in meta else "",
        "dt_mismatch": int(meta.get("dt_mismatch", 0)) if "dt_mismatch" in meta else "",
        "diff_applied": int(meta.get("diff_applied", 0)) if "diff_applied" in meta else "",
    }


def main() -> int:
    p = argparse.ArgumentParser(description="S-0022: estimator-manufactured κ2 scaling guard (C5 integrity gate).")
    p.add_argument("--seed", type=int, default=S0022Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0022Config(seed=int(args.seed))

    t_full = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)
    mask_full = _audit_mask(t_full, cfg.t_min, cfg.t_max)
    n_w = int(np.count_nonzero(mask_full))

    print(
        f"[S-0022] window=[{cfg.t_min:.1f},{cfg.t_max:.1f}] n_w={n_w} dt={cfg.dt:.3g} "
        f"admiss(min_r2={cfg.min_r2:.4f}, min_k2_end={cfg.min_k2_end:g}, min_points={cfg.min_points}) "
        f"OU_band=[{cfg.ou_alpha_min:.2f},{cfg.ou_alpha_max:.2f}] "
        f"LM_band=[{cfg.lm_alpha_min:.2f},{cfg.lm_alpha_max:.2f}] (H={cfg.fgn_H:g}) "
        f"ma_window={cfg.ma_window} reuse_factor_max={cfg.reuse_factor_max} resample_factor={cfg.resample_factor} nT={cfg.n_trajectories}"
    )

    if n_w < int(cfg.min_points):
        print("[S-0022] INCONCLUSIVE: audit window has too few points.")
        return 3

    rng_ou = np.random.default_rng(cfg.seed + 1)
    rng_lm = np.random.default_rng(cfg.seed + 2)

    omega_ou = simulate_ou(cfg, rng=rng_ou)
    omega_lm = simulate_fgn(cfg, rng=rng_lm)

    omega_ma, meta_ma = moving_average_overlap(cfg, omega_ou)
    omega_rs, meta_rs = resample_dt_mismatch(cfg, omega_ou)
    omega_df, meta_df = differencing_transform(cfg, omega_ou)

    cases: List[Tuple[str, np.ndarray, Dict[str, object]]] = [
        ("C1_OU_BASE", omega_ou, {}),
        ("C2_LM_TRUE", omega_lm, {}),
        ("C5_OVERLAP_REUSE", omega_ma, meta_ma),
        ("C5_RESAMPLE_DT_MISMATCH", omega_rs, meta_rs),
        ("C5_DIFFERENCED", omega_df, meta_df),
    ]

    results: Dict[str, Dict[str, object]] = {}
    any_inconclusive = False

    for name, om, meta in cases:
        t = (np.arange(int(om.shape[1])) + 1) * float(cfg.dt)
        mask = _audit_mask(t, cfg.t_min, cfg.t_max)

        r = audit_case(cfg, name=name, omega=om, t=t, mask=mask, meta=meta)
        results[name] = r

        verdict = str(r.get("verdict", "INCONCLUSIVE"))
        if verdict != "OK":
            any_inconclusive = True
            reason = str(r.get("reason", ""))
            alpha = r.get("alpha", float("nan"))
            r2 = r.get("r2", float("nan"))
            k2_end = r.get("k2_end", float("nan"))
            print(f"[S-0022:{name}] verdict=INCONCLUSIVE reason={reason} alpha={float(alpha):.4g} r2={float(r2):.4g} k2_end={float(k2_end):.4g}")
            continue

        alpha_str = r.get("alpha", "")
        r2_str = r.get("r2", "")
        k2_end_str = r.get("k2_end", "")

        if alpha_str == "":
            print(f"[S-0022:{name}] tag={str(r['tag'])} notes={str(r['notes'])} diff_applied={int(r.get('diff_applied', 0))}")
            continue

        reuse_factor = r.get("reuse_factor", "")
        reuse_violation = r.get("reuse_violation", "")
        resample_factor = r.get("resample_factor", "")
        dt_mismatch = r.get("dt_mismatch", "")
        diff_applied = r.get("diff_applied", "")

        print(
            f"[S-0022:{name}] alpha={float(alpha_str):.4g} r2={float(r2_str):.4g} k2_end={float(k2_end_str):.4g} "
            f"band={str(r.get('band', ''))} tag={str(r['tag'])} notes={str(r['notes'])} "
            f"reuse_factor={reuse_factor} reuse_violation={reuse_violation} "
            f"resample_factor={resample_factor} dt_mismatch={dt_mismatch} diff_applied={diff_applied}"
        )

    if any_inconclusive:
        print("[S-0022] INCONCLUSIVE")
        exit_code = 3
    else:
        ou_ok = (str(results["C1_OU_BASE"].get("tag")) == "OK_OU")
        lm_ok = (str(results["C2_LM_TRUE"].get("tag")) == "OK_LM")

        forbidden_ok = False
        for nm in ["C5_OVERLAP_REUSE", "C5_RESAMPLE_DT_MISMATCH", "C5_DIFFERENCED"]:
            tag = str(results[nm].get("tag", ""))
            if tag in ("OK_OU", "OK_LM"):
                forbidden_ok = True

        if not (ou_ok and lm_ok):
            print(f"[S-0022] AUDIT FAILED: baselines incorrect (ou_tag={results['C1_OU_BASE'].get('tag')}, lm_tag={results['C2_LM_TRUE'].get('tag')}).")
            exit_code = 2
        elif forbidden_ok:
            print("[S-0022] AUDIT FAILED: estimator produced forbidden OK_* under declared C5 integrity violation.")
            exit_code = 2
        else:
            ok_refusals = all(str(results[nm].get("tag")) == "BOUNDARY" for nm in ["C5_OVERLAP_REUSE", "C5_RESAMPLE_DT_MISMATCH", "C5_DIFFERENCED"])
            if ok_refusals:
                print("[S-0022] AUDIT PASSED")
                exit_code = 0
            else:
                print("[S-0022] AUDIT FAILED: expected BOUNDARY refusal did not hold for all C5 cases.")
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
            [
                "case",
                "alpha",
                "r2",
                "k2_end",
                "band",
                "tag",
                "verdict",
                "reason",
                "notes",
                "reuse_factor",
                "reuse_violation",
                "resample_factor",
                "dt_mismatch",
                "diff_applied",
            ]
        ]
        for nm, _, _meta in cases:
            r = results[nm]
            rows.append(
                [
                    nm,
                    r.get("alpha", ""),
                    r.get("r2", ""),
                    r.get("k2_end", ""),
                    str(r.get("band", "")),
                    str(r.get("tag", "")),
                    str(r.get("verdict", "")),
                    str(r.get("reason", "")),
                    str(r.get("notes", "")),
                    r.get("reuse_factor", ""),
                    r.get("reuse_violation", ""),
                    r.get("resample_factor", ""),
                    r.get("dt_mismatch", ""),
                    r.get("diff_applied", ""),
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
                ["ma_window", int(cfg.ma_window)],
                ["reuse_factor_max", int(cfg.reuse_factor_max)],
                ["resample_factor", int(cfg.resample_factor)],
            ],
        )

        print(f"Wrote (untracked): {out_dir / cfg.out_cases_csv}")
        print(f"Wrote (untracked): {out_dir / cfg.out_audit_csv}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
