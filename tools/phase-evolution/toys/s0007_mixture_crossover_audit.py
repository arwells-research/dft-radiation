#!/usr/bin/env python3
"""
S-0007 — Mixture + crossover “no-claim” certification (OU ⊕ fGn; early/late windows)

This toy certifies classifier restraint under mixtures:
  - It must NOT make wrong regime claims when a window is admissible.
  - It may return BOUNDARY (no-claim) in mixed/crossover or inadmissible windows.

Important: This does NOT guarantee that a linear mixture in ω will necessarily
produce an fGn-dominant late window for small eps over a finite horizon.
Therefore, for eps>0 the late window is NOT a required "must be fGn" slot.
Instead, the certification is:
  - no forbidden claims, and
  - if/when late becomes admissible and fGn-like, it should tag OK(fGn).

Exit codes:
  0 — PASS: no wrong claims in any admissible window; OU baseline OK in both windows.
  2 — FAIL: any admissible window makes a forbidden claim OR eps=0 baseline not OK in both windows.

Conventions:
  - Global DC removal only on ω (one constant across ensemble×time).
  - Window-local re-origining for fits (on κ₂, not on ω):
        tw  := (t - t0) + dt
        k2w := k2 - k2(t0)
  - Optional drift gate is applied ONLY for eps>0 (mixture restraint), not for eps=0 baselines.
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
class S0007Config:
    seed: int = 35791

    dt: float = 0.1
    n_steps: int = 2048
    n_trajectories: int = 512

    ou_theta: float = 0.30
    ou_sigma: float = 1.00

    fgn_H: float = 0.70
    fgn_sigma: float = 1.00

    eps_list: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)

    early_t_min: float = 10.0
    early_t_max: float = 40.0
    late_t_min: float = 120.0
    late_t_max: float = 190.0

    min_r2: float = 0.985
    k2_min_at_tmax: float = 0.50
    min_points: int = 12

    # drift gate (mixture-only); set <=0 to disable
    alpha_drift_max: float = 0.20

    ou_alpha_target: float = 1.00
    ou_alpha_tol: float = 0.20
    fgn_alpha_tol: float = 0.20

    c3_m: int = 8

    shot_rate_per_time: float = 0.03
    shot_amp_mean: float = 0.50
    shot_tau_samples: int = 12
    shot_sign_symmetric: bool = True

    out_dir: str = "toys/outputs"
    out_audit_csv: str = "s0007_audit.csv"
    out_cases_csv: str = "s0007_cases.csv"


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


def simulate_ou(cfg: S0007Config, rng: np.random.Generator) -> np.ndarray:
    nT, n = int(cfg.n_trajectories), int(cfg.n_steps)
    dt = float(cfg.dt)
    theta = float(cfg.ou_theta)
    sigma = float(cfg.ou_sigma)

    omega = np.empty((nT, n), dtype=float)
    sdt = np.sqrt(dt)
    for i in range(nT):
        x = float(rng.normal(scale=(sigma / np.sqrt(2.0 * theta))))
        for k in range(n):
            x = x + (-theta * x) * dt + sigma * sdt * float(rng.standard_normal())
            omega[i, k] = x
    return omega


def fgn_davies_harte(n_steps: int, H: float, rng: np.random.Generator) -> np.ndarray:
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


def simulate_fgn(cfg: S0007Config, rng: np.random.Generator) -> np.ndarray:
    nT, n = int(cfg.n_trajectories), int(cfg.n_steps)
    H = float(cfg.fgn_H)
    sigma = float(cfg.fgn_sigma)

    omega = np.empty((nT, n), dtype=float)
    for i in range(nT):
        x = fgn_davies_harte(n, H, rng)
        omega[i, :] = sigma * x
    return omega


def center_global_dc(omega: np.ndarray) -> np.ndarray:
    return omega - float(np.mean(omega))


def fir_smooth_per_traj(x: np.ndarray, m: int) -> np.ndarray:
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


def add_shot_in_omega(omega: np.ndarray, cfg: S0007Config, rng: np.random.Generator) -> np.ndarray:
    y = omega.copy()
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


def drift_in_window(
    tw: np.ndarray,
    k2w: np.ndarray,
    *,
    min_points: int,
) -> float:
    if tw.size < max(8, min_points):
        return float("nan")
    mid = int(tw.size // 2)
    tw1, k21 = tw[:mid], k2w[:mid]
    tw2, k22 = tw[mid:], k2w[mid:]
    if tw1.size < 4 or tw2.size < 4:
        return float("nan")
    f1 = _fit_loglog(tw1, k21, min_r2=-1.0, k2_min_at_tmax=-np.inf, min_points=4)
    f2 = _fit_loglog(tw2, k22, min_r2=-1.0, k2_min_at_tmax=-np.inf, min_points=4)
    a1 = float(f1["alpha"])
    a2 = float(f2["alpha"])
    if not (np.isfinite(a1) and np.isfinite(a2)):
        return float("nan")
    return float(abs(a1 - a2))


def expected_ranges(cfg: S0007Config) -> Dict[str, Tuple[float, float]]:
    ou_lo = float(cfg.ou_alpha_target - cfg.ou_alpha_tol)
    ou_hi = float(cfg.ou_alpha_target + cfg.ou_alpha_tol)

    fgn_target = float(2.0 * cfg.fgn_H)
    fgn_lo = float(fgn_target - cfg.fgn_alpha_tol)
    fgn_hi = float(fgn_target + cfg.fgn_alpha_tol)

    return {"OU": (ou_lo, ou_hi), "fGn": (fgn_lo, fgn_hi)}


def in_range(alpha: float, lo: float, hi: float) -> bool:
    return bool(np.isfinite(alpha) and (alpha >= lo) and (alpha <= hi))


def tag_directional(
    *,
    admissible: bool,
    alpha: float,
    exp_lo: float,
    exp_hi: float,
    forbid_lo: float,
    forbid_hi: float,
    allow_boundary: bool,
) -> str:
    if not admissible:
        return "BOUNDARY"
    if in_range(alpha, forbid_lo, forbid_hi):
        return "FAIL"
    if in_range(alpha, exp_lo, exp_hi):
        return "OK"
    return "BOUNDARY" if allow_boundary else "FAIL"


def main() -> int:
    p = argparse.ArgumentParser(description="S-0007: OU+fGn mixtures → crossover certification with fixed early/late windows.")
    p.add_argument("--seed", type=int, default=S0007Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    p.add_argument("--min_r2", type=float, default=S0007Config.min_r2)
    p.add_argument("--k2_min_at_tmax", type=float, default=S0007Config.k2_min_at_tmax)
    p.add_argument("--alpha_drift_max", type=float, default=S0007Config.alpha_drift_max)
    args = p.parse_args()

    cfg = S0007Config(
        seed=int(args.seed),
        min_r2=float(args.min_r2),
        k2_min_at_tmax=float(args.k2_min_at_tmax),
        alpha_drift_max=float(args.alpha_drift_max),
    )

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)

    ranges = expected_ranges(cfg)
    ou_lo, ou_hi = ranges["OU"]
    fgn_lo, fgn_hi = ranges["fGn"]
    fgn_target = 2.0 * float(cfg.fgn_H)

    print(
        f"[S-0007] early=[{cfg.early_t_min:.1f},{cfg.early_t_max:.1f}] late=[{cfg.late_t_min:.1f},{cfg.late_t_max:.1f}] "
        f"min_r2={cfg.min_r2:.3f} k2_min@tmax={cfg.k2_min_at_tmax:.3g} drift_max={cfg.alpha_drift_max:.3g} "
        f"targets: OU~{cfg.ou_alpha_target:.2f}±{cfg.ou_alpha_tol:.2f}, fGn~{fgn_target:.2f}±{cfg.fgn_alpha_tol:.2f}"
    )

    omega_ou = simulate_ou(cfg, rng=np.random.default_rng(cfg.seed + 1))
    omega_fgn = simulate_fgn(cfg, rng=np.random.default_rng(cfg.seed + 2))

    families: List[Tuple[str, bool, bool]] = [
        ("MIX", False, False),
        ("MIX_C3", True, False),
        ("MIX_C4", False, True),
        ("MIX_C3_C4", True, True),
    ]

    rows: List[List[object]] = []
    any_fail = False

    for fam_name, use_c3, use_c4 in families:
        for eps0 in cfg.eps_list:
            eps = float(eps0)

            omega = omega_ou + (eps * omega_fgn)
            omega = center_global_dc(omega)

            if use_c4:
                eps_key = int(round(1000.0 * eps))
                rng_c4 = np.random.default_rng(cfg.seed + 1000 + eps_key + (1 if use_c3 else 0))
                omega = add_shot_in_omega(omega, cfg, rng=rng_c4)
                omega = center_global_dc(omega)

            if use_c3:
                omega = fir_smooth_per_traj(omega, m=int(cfg.c3_m))
                omega = center_global_dc(omega)

            dphi = delta_phi_from_omega(omega, cfg.dt)
            k2 = k2_direct(dphi)

            # window-local series (needed for drift measure too)
            tw_e, k2w_e = _window_local_series(
                t_abs=t, k2_abs=k2, dt=cfg.dt, t_min=cfg.early_t_min, t_max=cfg.early_t_max, min_points=cfg.min_points
            )
            tw_l, k2w_l = _window_local_series(
                t_abs=t, k2_abs=k2, dt=cfg.dt, t_min=cfg.late_t_min, t_max=cfg.late_t_max, min_points=cfg.min_points
            )

            early_fit = fit_window_shifted(
                t_abs=t,
                k2_abs=k2,
                dt=cfg.dt,
                t_min=cfg.early_t_min,
                t_max=cfg.early_t_max,
                min_r2=cfg.min_r2,
                k2_min_at_tmax=cfg.k2_min_at_tmax,
                min_points=cfg.min_points,
            )
            late_fit = fit_window_shifted(
                t_abs=t,
                k2_abs=k2,
                dt=cfg.dt,
                t_min=cfg.late_t_min,
                t_max=cfg.late_t_max,
                min_r2=cfg.min_r2,
                k2_min_at_tmax=cfg.k2_min_at_tmax,
                min_points=cfg.min_points,
            )

            # Drift gate: mixture-only restraint (eps>0). Baseline eps=0 must not be vetoed by drift.
            early_drift = drift_in_window(tw_e, k2w_e, min_points=cfg.min_points) if tw_e.size else float("nan")
            late_drift = drift_in_window(tw_l, k2w_l, min_points=cfg.min_points) if tw_l.size else float("nan")

            if eps > 0.0 and float(cfg.alpha_drift_max) > 0.0:
                if np.isfinite(early_drift) and early_drift > float(cfg.alpha_drift_max):
                    early_fit["pass"] = False
                if np.isfinite(late_drift) and late_drift > float(cfg.alpha_drift_max):
                    late_fit["pass"] = False

            if eps == 0.0:
                early_tag = tag_directional(
                    admissible=bool(early_fit["pass"]),
                    alpha=float(early_fit["alpha"]),
                    exp_lo=ou_lo,
                    exp_hi=ou_hi,
                    forbid_lo=fgn_lo,
                    forbid_hi=fgn_hi,
                    allow_boundary=False,
                )
                late_tag = tag_directional(
                    admissible=bool(late_fit["pass"]),
                    alpha=float(late_fit["alpha"]),
                    exp_lo=ou_lo,
                    exp_hi=ou_hi,
                    forbid_lo=fgn_lo,
                    forbid_hi=fgn_hi,
                    allow_boundary=False,
                )
                case_fail = (early_tag != "OK") or (late_tag != "OK")
            else:
                # Mixture: no wrong claims; boundary allowed.
                early_tag = tag_directional(
                    admissible=bool(early_fit["pass"]),
                    alpha=float(early_fit["alpha"]),
                    exp_lo=ou_lo,
                    exp_hi=ou_hi,
                    forbid_lo=fgn_lo,
                    forbid_hi=fgn_hi,
                    allow_boundary=True,
                )
                late_tag = tag_directional(
                    admissible=bool(late_fit["pass"]),
                    alpha=float(late_fit["alpha"]),
                    exp_lo=fgn_lo,
                    exp_hi=fgn_hi,
                    forbid_lo=ou_lo,
                    forbid_hi=ou_hi,
                    allow_boundary=True,
                )
                case_fail = (early_tag == "FAIL") or (late_tag == "FAIL")

            if case_fail:
                any_fail = True

            case_verdict = "FAIL" if case_fail else ("BOUNDARY" if (early_tag == "BOUNDARY" or late_tag == "BOUNDARY") else "OK")
            case_name = f"{fam_name}:eps={eps:.2f}"

            print(
                f"[S-0007:{case_name}] "
                f"early: alpha={early_fit['alpha']:.3f} r2={early_fit['r2']:.3f} drift={early_drift:.3g} k2_tmax={early_fit['k2_tmax']:.3g} "
                f"exp=[{ou_lo:.3f},{ou_hi:.3f}] -> {early_tag} | "
                f"late: alpha={late_fit['alpha']:.3f} r2={late_fit['r2']:.3f} drift={late_drift:.3g} k2_tmax={late_fit['k2_tmax']:.3g} "
                f"exp=[{(ou_lo if eps==0.0 else fgn_lo):.3f},{(ou_hi if eps==0.0 else fgn_hi):.3f}] -> {late_tag} | "
                f"verdict={case_verdict}"
            )

            rows.append(
                [
                    case_name,
                    fam_name,
                    eps,
                    int(bool(use_c3)),
                    int(bool(use_c4)),
                    float(early_fit["alpha"]),
                    float(early_fit["r2"]),
                    float(early_drift),
                    float(early_fit["k2_tmax"]),
                    int(bool(early_fit["pass"])),
                    early_tag,
                    float(late_fit["alpha"]),
                    float(late_fit["r2"]),
                    float(late_drift),
                    float(late_fit["k2_tmax"]),
                    int(bool(late_fit["pass"])),
                    late_tag,
                    case_verdict,
                ]
            )

    if any_fail:
        print("[S-0007] AUDIT FAILED: one or more admissible windows made a wrong claim OR eps=0 baseline was not OK in both windows.")
        exit_code = 2
    else:
        print("[S-0007] AUDIT PASSED: no wrong claims; mixed/crossover regions stayed boundary; baseline OU OK.")
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
            "family",
            "eps",
            "use_c3",
            "use_c4",
            "early_alpha",
            "early_r2",
            "early_drift",
            "early_k2_tmax",
            "early_admissible",
            "early_tag",
            "late_alpha",
            "late_r2",
            "late_drift",
            "late_k2_tmax",
            "late_admissible",
            "late_tag",
            "case_verdict",
        ]]
        out_rows.extend(rows)
        _write_csv_with_provenance_header(cases_path, header_kv, out_rows)
        print(f"Wrote (untracked): {cases_path}")

        audit_path = out_dir / cfg.out_audit_csv
        audit_rows = [
            ["field", "value"],
            ["exit_code", exit_code],
            ["seed", int(cfg.seed)],
            ["dt", float(cfg.dt)],
            ["n_steps", int(cfg.n_steps)],
            ["n_trajectories", int(cfg.n_trajectories)],
            ["eps_list", ",".join([str(float(x)) for x in cfg.eps_list])],
            ["early_t_min", float(cfg.early_t_min)],
            ["early_t_max", float(cfg.early_t_max)],
            ["late_t_min", float(cfg.late_t_min)],
            ["late_t_max", float(cfg.late_t_max)],
            ["min_r2", float(cfg.min_r2)],
            ["k2_min_at_tmax", float(cfg.k2_min_at_tmax)],
            ["min_points", int(cfg.min_points)],
            ["alpha_drift_max_mixture_only", float(cfg.alpha_drift_max)],
        ]
        _write_csv_with_provenance_header(audit_path, header_kv, audit_rows)
        print(f"Wrote (untracked): {audit_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())