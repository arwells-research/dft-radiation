#!/usr/bin/env python3
"""
S-0023 — Transport-manufactured κ2 scaling guard (C3 integrity gate)

Goal
----
Certify that κ2-slope regime classification (OU vs long-memory) is not admissible when
a C3 geometric-transport binding can manufacture long-memory-like scaling signatures
in the observed ω(t) stream.

Core risk
---------
A fixed linear transport operator (dispersion / mode-mixing style LTI binding) applied
to an underlying short-memory ω can introduce long-range correlation structure in the
observed stream ω̂ that can shift κ2(t)=Var[Δϕ(t)] scaling within a finite audit window.

S-0023 enforces: when a *transport signature* is detected, κ2-slope regime attribution
is refused (BOUNDARY), even if α falls within OU/LM bands.

Definitions
-----------
Phase accumulation (discrete):
    Δϕ_i[k] = sum_{j<=k} ω_i[j] * dt

Second cumulant:
    κ2[k] = Var_i[Δϕ_i[k]]

κ2 slope (admissible only if fit is admissible):
    α := slope of log κ2 vs log t over declared window

C3 transport binding (declared, fixed)
--------------------------------------
We construct a dispersive long-tail FIR kernel:
    h[m] ∝ (m+1)^(-p) * cos(2π m / period),  m=0..(Lh-1)
normalized by sum(|h|)=1 (purely diagnostic; no tuning).

Observed ω̂ is the convolution (per-trajectory, identical kernel):
    ω̂_i = h * ω_i

Transport signature detector (declared, fixed)
----------------------------------------------
Compute a window-restricted ensemble autocovariance of ω̂ over lags 1..max_lag.
Transport is flagged if BOTH:
  1) sign-change count in autocovariance exceeds a minimum (oscillatory/ringing signature)
  2) negative-mass ratio exceeds a minimum (non-monotone transport imprint)

This is designed to distinguish dispersive transport-induced structure from
a true long-memory baseline (fGn with H>0.5 tends to have non-oscillatory positive ACF).

Certification rule (fixed subcases)
-----------------------------------
- C1_OU_BASE:
    - κ2-slope α in OU band
    - transport_detected = 0
    -> OK_OU
- C2_LM_TRUE:
    - κ2-slope α in LM band
    - transport_detected = 0
    -> OK_LM
- C3_DISPERSION_FILTERED:
    - underlying ω is OU, but observed ω̂ includes declared C3 transport kernel
    - regardless of κ2 slope band, if transport_detected = 1 -> BOUNDARY
    -> BOUNDARY

Inadmissible κ2 fit (r²/point-count/κ2 floor) => INCONCLUSIVE (no claim).

No scanning. No tuning. No post-hoc threshold adjustment.

Exit codes
----------
  0 PASS
  2 FAIL
  3 INCONCLUSIVE

Outputs (only with --write_outputs)
----------------------------------
  toys/outputs/s0023_cases.csv
  toys/outputs/s0023_audit.csv
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
class S0023Config:
    seed: int = 23023

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

    # C3 transport kernel (fixed; no scanning)
    c3_h_len: int = 64
    c3_power_p: float = 0.25
    c3_period: int = 8

    # Transport signature detector (fixed)
    max_lag: int = 40
    min_sign_changes: int = 2
    min_neg_mass_ratio: float = 0.10

    eps: float = 1e-12

    out_dir: str = "toys/outputs"
    out_cases_csv: str = "s0023_cases.csv"
    out_audit_csv: str = "s0023_audit.csv"


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


def simulate_ou(cfg: S0023Config, rng: np.random.Generator) -> np.ndarray:
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


def simulate_fgn(cfg: S0023Config, rng: np.random.Generator) -> np.ndarray:
    nT = int(cfg.n_trajectories)
    nS = int(cfg.n_steps)
    H = float(cfg.fgn_H)

    omega = np.empty((nT, nS), dtype=float)
    for i in range(nT):
        inc = fgn_davies_harte(nS, H, rng=rng)
        omega[i, :] = inc
    return omega


def _c3_kernel(cfg: S0023Config) -> np.ndarray:
    L = int(cfg.c3_h_len)
    p = float(cfg.c3_power_p)
    per = int(cfg.c3_period)

    m = np.arange(L, dtype=float)
    h = (m + 1.0) ** (-p) * np.cos(2.0 * np.pi * m / float(per))
    # Normalize by L1 to keep scale comparable; avoid dividing by 0.
    s = float(np.sum(np.abs(h)))
    if s <= 0.0 or not np.isfinite(s):
        raise RuntimeError("C3 kernel normalization failed.")
    return (h / s).astype(float)


def apply_c3_transport(cfg: S0023Config, omega: np.ndarray) -> np.ndarray:
    """
    Apply a fixed dispersive transport kernel per-trajectory (LTI convolution).
    """
    x = np.asarray(omega, dtype=float)
    h = _c3_kernel(cfg)
    nT, nS = x.shape

    y = np.empty_like(x)
    for i in range(nT):
        # "same" length via centered convolution
        y[i, :] = np.convolve(x[i, :], h, mode="same")
    return y


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


def classify_alpha(cfg: S0023Config, alpha: float) -> str:
    if cfg.ou_alpha_min <= alpha <= cfg.ou_alpha_max:
        return "OU"
    if cfg.lm_alpha_min <= alpha <= cfg.lm_alpha_max:
        return "LM"
    return "BOUNDARY"


def _autocov_lags_windowed(omega: np.ndarray, mask: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Window-restricted autocovariance for lags 0..max_lag using the masked time indices.
    Uses ensemble+time averaging:
        R[lag] = mean_{i} mean_{k in window, k+lag in window} ω_i[k] ω_i[k+lag]
    """
    x = np.asarray(omega, dtype=float)
    idx = np.where(mask)[0]
    if idx.size < (max_lag + 2):
        max_lag = max(0, int(idx.size) - 2)

    if max_lag <= 0:
        return np.zeros(1, dtype=float)

    k0 = int(idx[0])
    k1 = int(idx[-1])

    R = np.zeros(max_lag + 1, dtype=float)
    # Use only pairs entirely inside [k0,k1]
    for lag in range(0, max_lag + 1):
        a = x[:, k0 : (k1 + 1 - lag)]
        b = x[:, (k0 + lag) : (k1 + 1)]
        if a.size == 0 or b.size == 0:
            R[lag] = 0.0
        else:
            R[lag] = float(np.mean(a * b))
    return R


def transport_signature(cfg: S0023Config, omega: np.ndarray, mask: np.ndarray) -> Tuple[int, float, int]:
    """
    Detect dispersive transport signature via oscillatory autocovariance:
      - count sign changes in R[1..max_lag]
      - compute negative mass ratio over |R| for lags 1..max_lag

    Returns: (transport_detected, neg_mass_ratio, sign_changes)
    """
    R = _autocov_lags_windowed(omega, mask, max_lag=int(cfg.max_lag))
    if R.size <= 2:
        return 0, 0.0, 0

    r = np.asarray(R[1:], dtype=float)
    # Ignore tiny values when counting sign changes (noise floor)
    floor = 1e-10 * float(np.max(np.abs(r)) if np.max(np.abs(r)) > 0 else 1.0)
    sgn = np.sign(np.where(np.abs(r) < floor, 0.0, r))

    # Count sign changes among nonzero entries
    nz = sgn[sgn != 0.0]
    sign_changes = 0
    if nz.size >= 2:
        sign_changes = int(np.count_nonzero(nz[1:] * nz[:-1] < 0.0))

    neg_mass = float(np.sum(np.abs(r[r < 0.0])))
    tot_mass = float(np.sum(np.abs(r))) + float(cfg.eps)
    neg_ratio = neg_mass / tot_mass

    detected = int((sign_changes >= int(cfg.min_sign_changes)) and (neg_ratio >= float(cfg.min_neg_mass_ratio)))
    return detected, float(neg_ratio), int(sign_changes)


def audit_case(cfg: S0023Config, name: str, omega: np.ndarray, t: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
    # Global DC removal only (per case)
    x = np.asarray(omega, dtype=float)
    x = x - float(np.mean(x))

    if (not np.any(mask)) or (int(np.count_nonzero(mask)) < int(cfg.min_points)):
        return {"case": name, "verdict": "INCONCLUSIVE", "reason": "too_few_points"}

    dphi = delta_phi_from_omega(x, cfg.dt)
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
        }

    band = classify_alpha(cfg, float(alpha))

    td, neg_ratio, sign_changes = transport_signature(cfg, x, mask)
    if td:
        tag = "BOUNDARY"
        notes = "c3_transport_detected"
    else:
        tag = "OK_OU" if band == "OU" else ("OK_LM" if band == "LM" else "BOUNDARY")
        notes = "ok" if tag != "BOUNDARY" else "slope_out_of_band"

    return {
        "case": name,
        "verdict": "OK",
        "alpha": float(alpha),
        "r2": float(r2),
        "k2_end": float(k2_end),
        "band": str(band),
        "transport_detected": int(td),
        "transport_neg_ratio": float(neg_ratio),
        "transport_sign_changes": int(sign_changes),
        "tag": str(tag),
        "notes": str(notes),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="S-0023: transport-manufactured κ2 scaling guard (C3 integrity gate).")
    p.add_argument("--seed", type=int, default=S0023Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0023Config(seed=int(args.seed))

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)
    mask = _audit_mask(t, cfg.t_min, cfg.t_max)
    n_w = int(np.count_nonzero(mask))

    print(
        f"[S-0023] window=[{cfg.t_min:.1f},{cfg.t_max:.1f}] n_w={n_w} dt={cfg.dt:.3g} "
        f"admiss(min_r2={cfg.min_r2:.4f}, min_k2_end={cfg.min_k2_end:g}, min_points={cfg.min_points}) "
        f"OU_band=[{cfg.ou_alpha_min:.2f},{cfg.ou_alpha_max:.2f}] "
        f"LM_band=[{cfg.lm_alpha_min:.2f},{cfg.lm_alpha_max:.2f}] (H={cfg.fgn_H:g}) "
        f"C3_kernel(L={cfg.c3_h_len}, p={cfg.c3_power_p:g}, period={cfg.c3_period}) "
        f"transport_sig(max_lag={cfg.max_lag}, min_sign_changes={cfg.min_sign_changes}, min_neg_ratio={cfg.min_neg_mass_ratio:g}) "
        f"nT={cfg.n_trajectories}"
    )

    if n_w < int(cfg.min_points):
        print("[S-0023] INCONCLUSIVE: audit window has too few points.")
        return 3

    rng_ou = np.random.default_rng(cfg.seed + 1)
    rng_lm = np.random.default_rng(cfg.seed + 2)
    rng_c3 = np.random.default_rng(cfg.seed + 3)

    omega_ou = simulate_ou(cfg, rng=rng_ou)
    omega_lm = simulate_fgn(cfg, rng=rng_lm)

    omega_c3 = simulate_ou(cfg, rng=rng_c3)
    omega_c3 = apply_c3_transport(cfg, omega_c3)

    cases = [
        ("C1_OU_BASE", omega_ou),
        ("C2_LM_TRUE", omega_lm),
        ("C3_DISPERSION_FILTERED", omega_c3),
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
                f"[S-0023:{name}] verdict=INCONCLUSIVE reason={reason} "
                f"alpha={float(alpha):.4g} r2={float(r2):.4g} k2_end={float(k2_end):.4g}"
            )
            continue

        print(
            f"[S-0023:{name}] alpha={float(r['alpha']):.4g} r2={float(r['r2']):.4g} k2_end={float(r['k2_end']):.4g} "
            f"band={str(r['band'])} transport_detected={int(r['transport_detected'])} "
            f"neg_ratio={float(r['transport_neg_ratio']):.4g} sign_changes={int(r['transport_sign_changes'])} "
            f"tag={str(r['tag'])} notes={str(r['notes'])}"
        )

    if any_inconclusive:
        print("[S-0023] INCONCLUSIVE")
        exit_code = 3
    else:
        ou_ok = (str(results["C1_OU_BASE"].get("tag")) == "OK_OU") and (int(results["C1_OU_BASE"].get("transport_detected", 0)) == 0)
        lm_ok = (str(results["C2_LM_TRUE"].get("tag")) == "OK_LM") and (int(results["C2_LM_TRUE"].get("transport_detected", 0)) == 0)

        c3_tag = str(results["C3_DISPERSION_FILTERED"].get("tag"))
        c3_td = int(results["C3_DISPERSION_FILTERED"].get("transport_detected", 0))

        forbidden_ok = (c3_tag in ("OK_OU", "OK_LM")) and (c3_td == 1)
        if forbidden_ok:
            print("[S-0023] AUDIT FAILED: forbidden OK_* under transport_detected=1 (false-pass permitted).")
            exit_code = 2
        elif ou_ok and lm_ok and (c3_tag == "BOUNDARY") and (c3_td == 1):
            print("[S-0023] AUDIT PASSED")
            exit_code = 0
        else:
            print(
                "[S-0023] AUDIT FAILED: "
                f"ou_ok={int(ou_ok)} lm_ok={int(lm_ok)} "
                f"c3_tag={c3_tag} transport_detected={c3_td}"
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

        rows = [
            [
                "case",
                "alpha",
                "r2",
                "k2_end",
                "band",
                "transport_detected",
                "transport_neg_ratio",
                "transport_sign_changes",
                "tag",
                "verdict",
                "reason",
                "notes",
            ]
        ]
        for nm in ["C1_OU_BASE", "C2_LM_TRUE", "C3_DISPERSION_FILTERED"]:
            r = results[nm]
            rows.append(
                [
                    nm,
                    float(r.get("alpha", float("nan"))) if "alpha" in r else "",
                    float(r.get("r2", float("nan"))) if "r2" in r else "",
                    float(r.get("k2_end", float("nan"))) if "k2_end" in r else "",
                    str(r.get("band", "")),
                    int(r.get("transport_detected", 0)) if "transport_detected" in r else "",
                    float(r.get("transport_neg_ratio", float("nan"))) if "transport_neg_ratio" in r else "",
                    int(r.get("transport_sign_changes", 0)) if "transport_sign_changes" in r else "",
                    str(r.get("tag", "")),
                    str(r.get("verdict", "")),
                    str(r.get("reason", "")),
                    str(r.get("notes", "")),
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
                ["c3_h_len", int(cfg.c3_h_len)],
                ["c3_power_p", float(cfg.c3_power_p)],
                ["c3_period", int(cfg.c3_period)],
                ["max_lag", int(cfg.max_lag)],
                ["min_sign_changes", int(cfg.min_sign_changes)],
                ["min_neg_mass_ratio", float(cfg.min_neg_mass_ratio)],
            ],
        )

        print(f"Wrote (untracked): {out_dir / cfg.out_cases_csv}")
        print(f"Wrote (untracked): {out_dir / cfg.out_audit_csv}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
