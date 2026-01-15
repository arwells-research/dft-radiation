#!/usr/bin/env python3
"""
S-0005 — Cross-Constraint Closure (C1 ⟷ C2 ⟷ C5): measurement kernel cannot fake regime transition (OU vs long-memory)

This toy is SELF-AUDITING and is intended to be a tight, mechanical C0005 deliverable.

Goal (auditable, non-ad-hoc):
  Demonstrate that a C5 measurement/inference kernel applied to the phase-driving signal ω(t)
  (windowing + finite detector response + additive measurement noise) can degrade inferred
  quantities, but cannot fake a regime transition (OU ↔ long-memory) when the classification
  discriminant is applied in an admissible scaling window.

Constraint tie-in:
  - C1 (temporal): ω(t) process class controls κ₂(t)=Var(Δϕ(t)) scaling (OU ~ t, long-memory ~ t^{2H}).
  - C2 (statistical): coherence envelope existence is tied to phase accumulation; κ₂(t) is the second-order driver.
  - C5 (epistemic): measurement kernel K acts on ω(t) and can distort estimators (smoothing, bandwidth limits, SNR).

Core discriminant (audit-grade, reusing the S-0001 ladder idea):
  Compute Δϕ(t)=∫ω(s)ds for an ensemble. Fit slope α in:
      κ₂(t) = Var(Δϕ(t))  and  log κ₂ ~ const + α log t
  in a declared late-time window where the relation is approximately linear (high r²).

Design rules:
  - deterministic seed
  - no post-hoc tuning: kernel sweeps are declared
  - explicit admissibility checks (r², κ₂ adequacy)
  - explicit failure interpretation (C5 dominance vs true misclassification)
  - outputs may be written locally under toys/outputs/ but must remain untracked (per .gitignore)

Run:
  python toys/s0005_c1_c2_c5_windowing_demo.py
  python toys/s0005_c1_c2_c5_windowing_demo.py --write_outputs

Exit code:
  0 on AUDIT PASS
  2 on AUDIT FAIL
  3 on INCONCLUSIVE (no admissible cases to certify invariance; C5 dominance everywhere)

Notes:
  - This toy is intentionally self-contained to avoid creating a new shared framework.
  - If you later refactor common helpers from S-0001..S-0004 into a shared module,
    replace local functions with imports without changing semantics.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class S0005Config:
    # deterministic
    seed: int = 12345

    # grid (match S-0001 style: long run, fit late window)
    dt: float = 0.1
    n_steps: int = 2048     # total time = 60.0

    # ensemble
    n_trajectories: int = 512

    # Regime A: OU (short memory)
    ou_theta: float = 0.12   # mean reversion (per unit time)
    ou_sigma: float = 1.00   # diffusion scale

    # Regime B: fGn long-memory baseline (Davies–Harte)
    fgn_H: float = 0.70
    fgn_sigma: float = 1.00

    # ω centering (kept simple and consistent with S-0001 ladder)
    # For both regimes, remove per-trajectory mean to prevent trivial drift of Δϕ.
    center_method: str = "per_traj_time"  # "per_traj_time" only in this toy

    # C5 measurement kernel sweep (acts on ω)
    # win_m: moving-average width (samples)
    # tau_det: causal exponential LPF time constant (samples; 0 disables)
    win_m_list: Tuple[int, ...] = (1, 4, 16, 64)
    tau_det_list: Tuple[int, ...] = (0, 4, 16, 64)
    meas_sigma_list: Tuple[float, ...] = (0.0, 0.03, 0.10)  # additive noise on ω (std dev)

    # audit window in time units (same spirit as S-0001: late-time scaling window)
    audit_window_t_min: float = 70.0
    audit_window_t_max: float = 170.0

    # admissibility + separation gate
    min_r2: float = 0.985
    min_alpha_gap: float = 0.20  # require α_LM - α_OU >= this on admissible cases

    # κ2 adequacy: avoid classifying near numerical/noise floor
    k2_min_at_tmax: float = 0.50

    # require at least this many admissible cases in the sweep to certify invariance
    min_admissible_cases: int = 6

    # outputs (untracked)
    out_dir: str = "toys/outputs"
    out_audit_csv: str = "s0005_audit.csv"
    out_cases_csv: str = "s0005_cases.csv"


# -----------------------------
# Provenance helpers (CSV comment header convention)
# -----------------------------

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


# -----------------------------
# Generators
# -----------------------------

def simulate_ou(cfg: S0005Config, rng: np.random.Generator) -> np.ndarray:
    """
    Euler–Maruyama OU per trajectory:
        dω = -θ ω dt + σ dW
    """
    nT, n = int(cfg.n_trajectories), int(cfg.n_steps)
    dt = float(cfg.dt)
    theta = float(cfg.ou_theta)
    sigma = float(cfg.ou_sigma)

    omega = np.empty((nT, n), dtype=float)
    sdt = np.sqrt(dt)
    for i in range(nT):
        x = 0.0
        for k in range(n):
            x = x + (-theta * x) * dt + sigma * sdt * float(rng.standard_normal())
            omega[i, k] = x
    return omega


def fgn_davies_harte(n_steps: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """
    Davies–Harte fGn innovations (Gaussian).
    Returns unit-std series (approximately), length n_steps.
    """
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0,1)")
    n = int(n_steps)

    k = np.arange(0, n)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * H) + np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H))
    r = np.concatenate([gamma, gamma[-2:0:-1]])

    lam = np.fft.rfft(r).real
    lam[lam < 0] = 0.0

    m = lam.shape[0]
    z = rng.normal(size=m) + 1j * rng.normal(size=m)
    x = np.fft.irfft(np.sqrt(lam) * z, n=r.shape[0])[:n]

    std = float(np.std(x))
    if std <= 0:
        raise RuntimeError("fGn generation produced non-positive std.")
    return x / std


def simulate_fgn(cfg: S0005Config, rng: np.random.Generator) -> np.ndarray:
    nT, n = int(cfg.n_trajectories), int(cfg.n_steps)
    H = float(cfg.fgn_H)
    sigma = float(cfg.fgn_sigma)

    omega = np.empty((nT, n), dtype=float)
    for i in range(nT):
        x = fgn_davies_harte(n, H, rng)
        omega[i, :] = sigma * x
    return omega


def center_global(omega: np.ndarray) -> np.ndarray:
    return omega - float(np.mean(omega))


# -----------------------------
# C5 measurement kernel on ω
# -----------------------------

def moving_average_per_traj(x: np.ndarray, m: int) -> np.ndarray:
    """
    Symmetric moving average (reflect padding) applied per trajectory.
    """
    m = int(m)
    if m <= 1:
        return x.copy()

    k = np.ones(m, dtype=float) / float(m)
    pad = m // 2

    nT, n = x.shape
    out = np.empty_like(x)
    for i in range(nT):
        xp = np.pad(x[i, :], (pad, pad), mode="reflect")
        y = np.convolve(xp, k, mode="valid")
        out[i, :] = y[:n]
    return out


def exp_lowpass_per_traj(x: np.ndarray, tau_det: int) -> np.ndarray:
    """
    Causal exponential low-pass (RC-like) per trajectory:
        y[k] = y[k-1] + a (x[k] - y[k-1]),  a = 1/tau_det
    """
    tau_det = int(tau_det)
    if tau_det <= 0:
        return x.copy()

    a = 1.0 / float(tau_det)
    nT, n = x.shape
    out = np.empty_like(x)
    for i in range(nT):
        acc = 0.0
        for k in range(n):
            acc = acc + a * (float(x[i, k]) - acc)
            out[i, k] = acc
    return out


def apply_c5_kernel(
    omega_true: np.ndarray,
    win_m: int,
    tau_det: int,
    meas_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    ω_true -> windowing -> detector LPF -> additive noise  (all in ω-space)

    This keeps the observable aligned with the ω→Δϕ ladder and avoids
    the inference pathologies of trying to reconstruct ω from noisy exp(iϕ).
    """
    y = moving_average_per_traj(omega_true, m=win_m)
    y = exp_lowpass_per_traj(y, tau_det=tau_det)

    s = float(meas_sigma)
    if s > 0:
        y = y + rng.normal(loc=0.0, scale=s, size=y.shape)

    return y


# -----------------------------
# κ2(t) and scaling fit
# -----------------------------

def delta_phi_from_omega(omega0: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(omega0, axis=1) * float(dt)


def k2_direct(delta_phi: np.ndarray) -> np.ndarray:
    return np.var(delta_phi, axis=0, ddof=0)


def _mask_window(t: np.ndarray, cfg: S0005Config) -> np.ndarray:
    return (t >= float(cfg.audit_window_t_min)) & (t <= float(cfg.audit_window_t_max))


def fit_alpha_loglog(t: np.ndarray, k2: np.ndarray, cfg: S0005Config) -> Dict[str, float]:
    """
    Fit log k2 = b + alpha log t over the audit window.
    Returns alpha, r2, n_points, k2_at_tmax (for adequacy), plus pass flag.
    """
    t = np.asarray(t, dtype=float)
    k2 = np.asarray(k2, dtype=float)

    eps = 1e-18
    mask = _mask_window(t, cfg) & np.isfinite(k2) & (k2 > eps)

    npts = int(np.count_nonzero(mask))
    if npts < 12:
        return {"alpha": float("nan"), "r2": float("nan"), "n_points": npts, "k2_tmax": float("nan"), "pass": False}

    x = np.log(t[mask])
    y = np.log(k2[mask])

    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    alpha = float(coef[1])

    # adequacy at tmax (last point inside window)
    idx_tmax = int(np.where(mask)[0][-1])
    k2_tmax = float(k2[idx_tmax])

    passed = bool(np.isfinite(alpha) and np.isfinite(r2) and (r2 >= float(cfg.min_r2)) and (k2_tmax >= float(cfg.k2_min_at_tmax)))
    return {"alpha": alpha, "r2": float(r2), "n_points": npts, "k2_tmax": k2_tmax, "pass": passed}


# -----------------------------
# Audit logic
# -----------------------------

def classify_case(ou_fit: Dict[str, float], lm_fit: Dict[str, float], cfg: S0005Config) -> Tuple[str, float]:
    """
    Returns (case_status, alpha_gap).
      - ADMISSIBLE if both fits pass admissibility (r2 + k2 adequacy)
      - C5_DOMINATED if either fit fails admissibility
      - MISCLASSIFIED if admissible but ordering/gap fails
      - OK if admissible and gap/order pass
    """
    alpha_gap = float(lm_fit["alpha"] - ou_fit["alpha"])
    if not (bool(ou_fit["pass"]) and bool(lm_fit["pass"])):
        return "C5_DOMINATED", alpha_gap

    # ordering expected: long-memory exponent should exceed OU exponent
    if not np.isfinite(alpha_gap):
        return "MISCLASSIFIED", alpha_gap

    if alpha_gap < float(cfg.min_alpha_gap):
        return "MISCLASSIFIED", alpha_gap

    return "OK", alpha_gap


def main() -> int:
    p = argparse.ArgumentParser(description="S-0005: C1/C2/C5 closure via κ2 scaling under measurement kernel.")
    p.add_argument("--seed", type=int, default=S0005Config.seed)
    p.add_argument("--write_outputs", action="store_true", help="Write untracked CSV outputs to toys/outputs/.")

    # allow overriding only the key audit knobs (keep defaults stable)
    p.add_argument("--min_r2", type=float, default=S0005Config.min_r2)
    p.add_argument("--min_alpha_gap", type=float, default=S0005Config.min_alpha_gap)
    p.add_argument("--k2_min_at_tmax", type=float, default=S0005Config.k2_min_at_tmax)
    p.add_argument("--min_admissible_cases", type=int, default=S0005Config.min_admissible_cases)

    args = p.parse_args()

    cfg = S0005Config(
        seed=int(args.seed),
        min_r2=float(args.min_r2),
        min_alpha_gap=float(args.min_alpha_gap),
        k2_min_at_tmax=float(args.k2_min_at_tmax),
        min_admissible_cases=int(args.min_admissible_cases),
    )

    rng = np.random.default_rng(cfg.seed)
    rng_lm = np.random.default_rng(cfg.seed + 777)  # independent stream

    # Generate underlying regimes (truth-side ω)
    omega_ou_true = simulate_ou(cfg, rng)
    omega_lm_true = simulate_fgn(cfg, rng_lm)

    # Centering (trajectory-mean removal)
    omega_ou_true = center_global(omega_ou_true)
    omega_lm_true = center_global(omega_lm_true)

    # Time grid
    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)

    # Baseline (no C5) fits
    dphi_ou = delta_phi_from_omega(omega_ou_true, cfg.dt)
    dphi_lm = delta_phi_from_omega(omega_lm_true, cfg.dt)
    k2_ou = k2_direct(dphi_ou)
    k2_lm = k2_direct(dphi_lm)

    fit_ou_base = fit_alpha_loglog(t, k2_ou, cfg)
    fit_lm_base = fit_alpha_loglog(t, k2_lm, cfg)

    print(
        f"[S-0005:BASE] OU  alpha={fit_ou_base['alpha']:.3f} r2={fit_ou_base['r2']:.3f} k2_tmax={fit_ou_base['k2_tmax']:.3g} "
        f"({'PASS' if fit_ou_base['pass'] else 'FAIL'})"
    )
    print(
        f"[S-0005:BASE] LM  alpha={fit_lm_base['alpha']:.3f} r2={fit_lm_base['r2']:.3f} k2_tmax={fit_lm_base['k2_tmax']:.3g} "
        f"({'PASS' if fit_lm_base['pass'] else 'FAIL'})"
    )
    base_gap = float(fit_lm_base["alpha"] - fit_ou_base["alpha"])
    print(f"[S-0005:BASE] alpha_gap = {base_gap:.3f} (min {cfg.min_alpha_gap:.3f})")

    # Baseline must be admissible; otherwise the toy is not interpretable
    if not (bool(fit_ou_base["pass"]) and bool(fit_lm_base["pass"])):
        print("[S-0005] AUDIT FAILED: baseline fits are not admissible (no audited scaling window / inadequate κ2 / low r²).")
        return 2
    if base_gap < float(cfg.min_alpha_gap):
        print("[S-0005] AUDIT FAILED: baseline regime separation is not strong enough under declared gap threshold.")
        return 2

    # Sweep C5 kernel strengths
    cases_rows: List[List[object]] = []
    admissible = 0
    misclassified = 0

    case_id = 0
    for win_m in cfg.win_m_list:
        for tau_det in cfg.tau_det_list:
            for meas_sigma in cfg.meas_sigma_list:
                case_rng = np.random.default_rng(cfg.seed + 10000 + case_id)

                omega_ou_meas = apply_c5_kernel(omega_ou_true, win_m=win_m, tau_det=tau_det, meas_sigma=meas_sigma, rng=case_rng)
                omega_lm_meas = apply_c5_kernel(omega_lm_true, win_m=win_m, tau_det=tau_det, meas_sigma=meas_sigma, rng=case_rng)

                # keep centering consistent post-measurement
                omega_ou_meas = center_global(omega_ou_meas)
                omega_lm_meas = center_global(omega_lm_meas)

                k2_ou_m = k2_direct(delta_phi_from_omega(omega_ou_meas, cfg.dt))
                k2_lm_m = k2_direct(delta_phi_from_omega(omega_lm_meas, cfg.dt))

                fit_ou = fit_alpha_loglog(t, k2_ou_m, cfg)
                fit_lm = fit_alpha_loglog(t, k2_lm_m, cfg)

                status, gap = classify_case(fit_ou, fit_lm, cfg)

                if status in ("OK", "MISCLASSIFIED"):
                    admissible += 1
                if status == "MISCLASSIFIED":
                    misclassified += 1

                print(
                    f"[S-0005:CASE#{case_id:02d}] win_m={win_m:>3d} tau_det={tau_det:>3d} meas_sigma={meas_sigma:.3f} | "
                    f"OU alpha={fit_ou['alpha']:.3f} r2={fit_ou['r2']:.3f} k2t={fit_ou['k2_tmax']:.3g} "
                    f"|| LM alpha={fit_lm['alpha']:.3f} r2={fit_lm['r2']:.3f} k2t={fit_lm['k2_tmax']:.3g} "
                    f"|| gap={gap:.3f} [{status}]"
                )

                cases_rows.append(
                    [
                        case_id,
                        win_m,
                        tau_det,
                        float(meas_sigma),
                        float(fit_ou["alpha"]),
                        float(fit_ou["r2"]),
                        float(fit_ou["k2_tmax"]),
                        int(bool(fit_ou["pass"])),
                        float(fit_lm["alpha"]),
                        float(fit_lm["r2"]),
                        float(fit_lm["k2_tmax"]),
                        int(bool(fit_lm["pass"])),
                        float(gap),
                        status,
                    ]
                )
                case_id += 1

    # Gate logic:
    #  - We do NOT fail the toy because C5 dominates (inadmissible fits); that is a boundary condition.
    #  - We DO fail if there exists any admissible case that misclassifies (i.e., fakes a regime transition).
    #  - We also require enough admissible cases to claim invariance meaningfully.
    if admissible < int(cfg.min_admissible_cases):
        print(
            f"[S-0005] INCONCLUSIVE: only {admissible} admissible cases (min {cfg.min_admissible_cases}). "
            f"C5 dominance prevents certification under this sweep."
        )
        # still write outputs if requested
        exit_code = 3
    elif misclassified > 0:
        print(f"[S-0005] AUDIT FAILED: found {misclassified} admissible cases that violate regime ordering / gap.")
        exit_code = 2
    else:
        print(
            f"[S-0005] AUDIT PASSED: {admissible} admissible cases; "
            f"no admissible case faked a regime transition under the declared C5 kernel sweep."
        )
        exit_code = 0

    if args.write_outputs:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        run_utc = _run_utc_iso()
        git_commit = _git_commit_short()
        toy_file = Path(__file__).name
        source_sha256 = _sha256_of_file(Path(__file__))

        header_kv = {
            "git_commit": git_commit,
            "run_utc": run_utc,
            "toy_file": toy_file,
            "source_sha256": source_sha256,
        }

        # cases csv
        cases_path = out_dir / cfg.out_cases_csv
        rows = [
            [
                "case_id",
                "win_m",
                "tau_det",
                "meas_sigma",
                "alpha_ou",
                "r2_ou",
                "k2t_ou",
                "admissible_ou",
                "alpha_lm",
                "r2_lm",
                "k2t_lm",
                "admissible_lm",
                "alpha_gap",
                "status",
            ]
        ]
        rows.extend(cases_rows)
        _write_csv_with_provenance_header(cases_path, header_kv, rows)
        print(f"Wrote (untracked): {cases_path}")

        # audit summary csv
        audit_path = out_dir / cfg.out_audit_csv
        audit_rows = [
            ["field", "value"],
            ["exit_code", exit_code],
            ["admissible_cases", admissible],
            ["misclassified_cases", misclassified],
            ["min_admissible_cases", int(cfg.min_admissible_cases)],
            ["min_r2", float(cfg.min_r2)],
            ["min_alpha_gap", float(cfg.min_alpha_gap)],
            ["k2_min_at_tmax", float(cfg.k2_min_at_tmax)],
            ["audit_window_t_min", float(cfg.audit_window_t_min)],
            ["audit_window_t_max", float(cfg.audit_window_t_max)],
            ["base_alpha_ou", float(fit_ou_base["alpha"])],
            ["base_r2_ou", float(fit_ou_base["r2"])],
            ["base_k2t_ou", float(fit_ou_base["k2_tmax"])],
            ["base_alpha_lm", float(fit_lm_base["alpha"])],
            ["base_r2_lm", float(fit_lm_base["r2"])],
            ["base_k2t_lm", float(fit_lm_base["k2_tmax"])],
            ["base_alpha_gap", float(base_gap)],
            ["fgn_H", float(cfg.fgn_H)],
            ["ou_theta", float(cfg.ou_theta)],
            ["ou_sigma", float(cfg.ou_sigma)],
            ["dt", float(cfg.dt)],
            ["n_steps", int(cfg.n_steps)],
            ["n_trajectories", int(cfg.n_trajectories)],
            ["seed", int(cfg.seed)],
        ]
        _write_csv_with_provenance_header(audit_path, header_kv, audit_rows)
        print(f"Wrote (untracked): {audit_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())