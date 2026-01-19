#!/usr/bin/env python3
"""
S-0009 — Higher-order odd-cumulant boundary (κ₅ phase-structure diagnostic)

This toy certifies that controlled κ₅ ≠ 0 at the innovation level yields a
sign-sensitive, auditable phase bias in the complex coherence ⟨exp(iΔϕ(t))⟩
that reverses under κ₅→−κ₅, while the magnitude envelope remains explainable
by an even-cumulant (κ₂, κ₄) truncation.

Construction notes (diagnostic, not ontological):
  - ω innovations are i.i.d. (stationary, memoryless).
  - K5± innovations are discrete with:
        mean=0, Var=1, μ3=0, μ4=3 (=> κ4=0), and μ5 = ±2 (=> κ5 = ±2).
    This isolates an odd 5th cumulant effect without introducing κ3.

Prediction model (cumulant-truncated complex mean):
    log Z(t) ≈ -κ2(t)/2 + κ4(t)/24 + i κ5(t)/120
  where Z(t) := ⟨exp(iΔϕ(t))⟩ and κn(t) are empirical cumulants of Δϕ(t)
  across the ensemble at fixed t.

Exit codes:
  0 — PASS: GAUSS magnitude closure passes; K5± magnitude closure passes;
            K5± phase bias is sign-coherent and reverses.
  2 — FAIL: Any required audit condition fails.
  3 — INCONCLUSIVE: Coherence floor too low in the audit window (no sign claim allowed).

Conventions:
  - Deterministic seeds only.
  - Fixed audit window; no scanning / optimization.
  - Outputs written only under toys/outputs/ and only when --write_outputs is passed.
  - CSV outputs include provenance header lines (# key=value).
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
class S0009Config:
    seed: int = 9009

    dt: float = 0.02
    n_steps: int = 4096
    n_trajectories: int = 1024

    # Fixed audit window (absolute times)
    t_min: float = 40.0
    t_max: float = 70.0

    # Magnitude-closure audit tolerances over the window
    # (operate on log c to make multiplicative errors comparable)
    max_median_abs_dlogc: float = 0.08
    max_p95_abs_dlogc: float = 0.20
    max_abs_drift_slope: float = 0.06  # slope of dlogc vs t within window

    # Phase-bias coherence thresholds (window-restricted)
    # Require strong sign coherence: fraction of points with expected sign.
    residual_sign_bin_frac: float = 0.70

    # Coherence floor guard: if |Z| too small in window, phase sign is meaningless
    coherence_floor: float = 1.0e-3

    out_dir: str = "toys/outputs"
    out_compare_csv: str = "s0009_envelope_compare.csv"
    out_phase_png: str = "s0009_phase_bias.png"
    out_error_png: str = "s0009_error_diagnostic.png"
    out_audit_csv: str = "s0009_audit.csv"


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


def simulate_omega_gauss(cfg: S0009Config, rng: np.random.Generator) -> np.ndarray:
    # innovations mean=0, Var=1
    return rng.standard_normal(size=(int(cfg.n_trajectories), int(cfg.n_steps))).astype(float)


def _discrete_innovations(
    *,
    n: Tuple[int, int],
    support: np.ndarray,
    probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    support = np.asarray(support, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if support.ndim != 1 or probs.ndim != 1 or support.size != probs.size:
        raise ValueError("support/probs mismatch")
    if not np.isclose(np.sum(probs), 1.0):
        raise ValueError("probs must sum to 1")
    if np.any(probs < 0):
        raise ValueError("probs must be nonnegative")

    idx = rng.choice(np.arange(support.size), size=n, p=probs)
    return support[idx].astype(float)


def simulate_omega_k5plus(cfg: S0009Config, rng: np.random.Generator) -> np.ndarray:
    # Support is the sign-flipped twin of K5- so that μ5 (and κ5) flips sign,
    # while mean=0, Var=1, μ3=0, μ4=3 are preserved.
    #
    # K5- support/probs (all positive):
    #   x = [-3, -1, 0, +1, +2]
    #   p = [1/60, 1/3, 1/3, 1/4, 1/15]
    #
    # Negating x yields K5+ with μ5 = +2.
    support = np.array([+3.0, +1.0, 0.0, -1.0, -2.0], dtype=float)
    probs = np.array([1.0 / 60.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 15.0], dtype=float)

    omega = _discrete_innovations(n=(int(cfg.n_trajectories), int(cfg.n_steps)), support=support, probs=probs, rng=rng)
    return omega


def simulate_omega_k5minus(cfg: S0009Config, rng: np.random.Generator) -> np.ndarray:
    # Discrete innovations satisfying:
    #   mean=0, Var=1, μ3=0, μ4=3 (=> κ4=0), and μ5 = -2 (=> κ5 = -2).
    support = np.array([-3.0, -1.0, 0.0, +1.0, +2.0], dtype=float)
    probs = np.array([1.0 / 60.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 15.0], dtype=float)

    omega = _discrete_innovations(n=(int(cfg.n_trajectories), int(cfg.n_steps)), support=support, probs=probs, rng=rng)
    return omega


def delta_phi_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(np.asarray(omega, dtype=float), axis=1) * float(dt)


def complex_coherence(delta_phi: np.ndarray) -> np.ndarray:
    # Z(t) = mean exp(i Δϕ) across trajectories at each time
    z = np.mean(np.exp(1j * delta_phi), axis=0)
    return z


def cumulants_2_4_5(delta_phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Empirical cumulants of Δϕ(t) across ensemble at each t.
    x = np.asarray(delta_phi, dtype=float)
    mu = np.mean(x, axis=0)

    xc = x - mu[None, :]

    m2 = np.mean(xc**2, axis=0)
    m3 = np.mean(xc**3, axis=0)
    m4 = np.mean(xc**4, axis=0)
    m5 = np.mean(xc**5, axis=0)

    k2 = m2
    k4 = m4 - 3.0 * (m2**2)
    k5 = m5 - 10.0 * m3 * m2

    return k2, k4, k5


def predict_complex_from_cumulants(k2: np.ndarray, k4: np.ndarray, k5: np.ndarray) -> np.ndarray:
    # log Z ≈ -k2/2 + k4/24 + i k5/120
    logZ = (-0.5 * k2) + (k4 / 24.0) + 1j * (k5 / 120.0)
    return np.exp(logZ)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 4:
        return float("nan")
    xv = x[good]
    yv = y[good]
    A = np.vstack([np.ones_like(xv), xv]).T
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    return float(coef[1])


def audit_case(
    *,
    name: str,
    t: np.ndarray,
    z_meas: np.ndarray,
    z_pred: np.ndarray,
    cfg: S0009Config,
    expect_phase_sign: int,  # +1 for K5+, -1 for K5-, 0 for GAUSS (near zero)
) -> Dict[str, float | str | int]:
    t = np.asarray(t, dtype=float)
    z_meas = np.asarray(z_meas, dtype=complex)
    z_pred = np.asarray(z_pred, dtype=complex)

    mask = _audit_mask(t, cfg.t_min, cfg.t_max)

    c_meas = np.abs(z_meas)
    c_pred = np.abs(z_pred)

    # Coherence floor guard (window)
    c_floor = float(cfg.coherence_floor)
    if np.any(mask):
        cmin = float(np.min(c_meas[mask]))
    else:
        cmin = float("nan")

    # If coherence is too low across the window, phase sign is meaningless.
    incoherent = bool(np.any(mask) and (cmin < c_floor))

    eps = 1e-18
    dlogc = np.log(np.maximum(c_meas, eps)) - np.log(np.maximum(c_pred, eps))
    dlogc_w = dlogc[mask] if np.any(mask) else np.asarray([], dtype=float)

    med_abs = float(np.median(np.abs(dlogc_w))) if dlogc_w.size else float("nan")
    p95_abs = float(np.quantile(np.abs(dlogc_w), 0.95)) if dlogc_w.size else float("nan")
    drift_slope = _linear_slope(t[mask], dlogc_w) if dlogc_w.size else float("nan")

    mag_ok = bool(
        (not incoherent)
        and np.isfinite(med_abs)
        and np.isfinite(p95_abs)
        and np.isfinite(drift_slope)
        and (med_abs <= float(cfg.max_median_abs_dlogc))
        and (p95_abs <= float(cfg.max_p95_abs_dlogc))
        and (abs(drift_slope) <= float(cfg.max_abs_drift_slope))
    )

    # Phase bias:
    # Use angle(Z_meas) in window. For sign coherence, avoid wrap issues by using sin(angle),
    # which preserves sign of small biases and is stable under 2π wraps.
    ang = np.angle(z_meas)
    s = np.sin(ang)
    s_w = s[mask] if np.any(mask) else np.asarray([], dtype=float)

    if (not s_w.size) or incoherent:
        sign_frac = float("nan")
        median_s = float("nan")
        phase_ok = False
    else:
        median_s = float(np.median(s_w))
        if expect_phase_sign == 0:
            # GAUSS: expect near-zero median; we treat it as "OK" if small in magnitude.
            sign_frac = float(np.mean(np.abs(s_w) < 0.25))  # descriptive only
            phase_ok = bool(abs(median_s) < 0.10)
        else:
            want = float(expect_phase_sign)
            sign_frac = float(np.mean((s_w * want) > 0.0))
            phase_ok = bool(sign_frac >= float(cfg.residual_sign_bin_frac))

    verdict: str
    if incoherent:
        verdict = "INCONCLUSIVE"
    elif mag_ok and phase_ok:
        verdict = "OK"
    else:
        verdict = "FAIL"

    return {
        "case": name,
        "coherence_min": cmin,
        "median_abs_dlogc": med_abs,
        "p95_abs_dlogc": p95_abs,
        "drift_slope": drift_slope,
        "mag_ok": int(mag_ok),
        "median_sin_phase": median_s,
        "sign_fraction": sign_frac,
        "phase_ok": int(bool(phase_ok)),
        "verdict": verdict,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="S-0009: κ5 phase-structure boundary demo (odd cumulant; magnitude stable).")
    p.add_argument("--seed", type=int, default=S0009Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0009Config(seed=int(args.seed))

    rng_g = np.random.default_rng(cfg.seed + 1)
    rng_p = np.random.default_rng(cfg.seed + 2)

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)

    print(
        f"[S-0009] window=[{cfg.t_min:.1f},{cfg.t_max:.1f}] "
        f"tol(med,p95,drift)=({cfg.max_median_abs_dlogc:.3g},{cfg.max_p95_abs_dlogc:.3g},{cfg.max_abs_drift_slope:.3g}) "
        f"sign_frac_min={cfg.residual_sign_bin_frac:.3g} coherence_floor={cfg.coherence_floor:.3g}"
    )

    rng_g = np.random.default_rng(cfg.seed + 1)
    rng_p = np.random.default_rng(cfg.seed + 2)

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)

    omega_k5p = simulate_omega_k5plus(cfg, rng=rng_p)
    omega_k5m = -omega_k5p

    cases: List[Tuple[str, np.ndarray, int]] = [
        ("GAUSS", simulate_omega_gauss(cfg, rng=rng_g), 0),
        ("K5+", omega_k5p, +1),
        ("K5-", omega_k5m, -1),
    ]

    audit_rows: List[Dict[str, float | str | int]] = []
    compare_table: List[List[object]] = []

    # For plots: store window-restricted series
    plot_series: Dict[str, Dict[str, np.ndarray]] = {}

    any_fail = False
    any_inconclusive = False

    for name, omega, phase_sign in cases:
        dphi = delta_phi_from_omega(omega, cfg.dt)

        z_meas = complex_coherence(dphi)
        k2, k4, k5 = cumulants_2_4_5(dphi)
        z_pred = predict_complex_from_cumulants(k2, k4, k5)

        res = audit_case(name=name, t=t, z_meas=z_meas, z_pred=z_pred, cfg=cfg, expect_phase_sign=phase_sign)
        audit_rows.append(res)

        c_meas = np.abs(z_meas)
        c_pred = np.abs(z_pred)
        ang = np.angle(z_meas)
        s = np.sin(ang)

        mask = _audit_mask(t, cfg.t_min, cfg.t_max)
        dlogc = np.log(np.maximum(c_meas, 1e-18)) - np.log(np.maximum(c_pred, 1e-18))

        plot_series[name] = {
            "t_w": t[mask],
            "dlogc_w": dlogc[mask],
            "sinphi_w": s[mask],
            "cmin": np.array([float(res["coherence_min"])]),
        }

        print(
            f"[S-0009:{name}] "
            f"coh_min={float(res['coherence_min']):.3g} "
            f"med|dlogc|={float(res['median_abs_dlogc']):.3g} p95|dlogc|={float(res['p95_abs_dlogc']):.3g} drift={float(res['drift_slope']):.3g} "
            f"mag_ok={int(res['mag_ok'])} "
            f"median_sin={float(res['median_sin_phase']):.3g} sign_frac={float(res['sign_fraction']):.3g} phase_ok={int(res['phase_ok'])} "
            f"verdict={res['verdict']}"
        )

        if str(res["verdict"]) == "FAIL":
            any_fail = True
        elif str(res["verdict"]) == "INCONCLUSIVE":
            any_inconclusive = True

        # CSV rows (full-length; kept small/standard)
        if not compare_table:
            compare_table.append(
                [
                    "t",
                    "case",
                    "c_meas",
                    "c_pred",
                    "dlogc",
                    "angle_meas_rad",
                    "sin_angle_meas",
                    "k2",
                    "k4",
                    "k5",
                ]
            )
        for i in range(int(cfg.n_steps)):
            compare_table.append(
                [
                    float(t[i]),
                    name,
                    float(c_meas[i]),
                    float(c_pred[i]),
                    float(dlogc[i]),
                    float(ang[i]),
                    float(s[i]),
                    float(k2[i]),
                    float(k4[i]),
                    float(k5[i]),
                ]
            )

    if any_fail:
        print("[S-0009] AUDIT FAILED")
        exit_code = 2
    elif any_inconclusive:
        print("[S-0009] INCONCLUSIVE")
        exit_code = 3
    else:
        print("[S-0009] AUDIT PASSED")
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

        # Compare CSV (large but deterministic; untracked)
        compare_path = out_dir / cfg.out_compare_csv
        _write_csv_with_provenance_header(compare_path, header_kv, compare_table)
        print(f"Wrote (untracked): {compare_path}")

        # Audit CSV (small summary)
        audit_path = out_dir / cfg.out_audit_csv
        audit_table: List[List[object]] = [
            ["field", "value"],
            ["exit_code", exit_code],
            ["seed", int(cfg.seed)],
            ["dt", float(cfg.dt)],
            ["n_steps", int(cfg.n_steps)],
            ["n_trajectories", int(cfg.n_trajectories)],
            ["t_min", float(cfg.t_min)],
            ["t_max", float(cfg.t_max)],
            ["max_median_abs_dlogc", float(cfg.max_median_abs_dlogc)],
            ["max_p95_abs_dlogc", float(cfg.max_p95_abs_dlogc)],
            ["max_abs_drift_slope", float(cfg.max_abs_drift_slope)],
            ["residual_sign_bin_frac", float(cfg.residual_sign_bin_frac)],
            ["coherence_floor", float(cfg.coherence_floor)],
        ]
        for row in audit_rows:
            case = str(row["case"])
            for k in [
                "coherence_min",
                "median_abs_dlogc",
                "p95_abs_dlogc",
                "drift_slope",
                "mag_ok",
                "median_sin_phase",
                "sign_fraction",
                "phase_ok",
                "verdict",
            ]:
                audit_table.append([f"{case}.{k}", row[k]])
        _write_csv_with_provenance_header(audit_path, header_kv, audit_table)
        print(f"Wrote (untracked): {audit_path}")

        # Plots (matplotlib only when writing outputs)
        import matplotlib.pyplot as plt  # noqa: PLC0415

        # Phase bias plot (sin(angle) vs t in window)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for name in ["GAUSS", "K5+", "K5-"]:
            ax.plot(plot_series[name]["t_w"], plot_series[name]["sinphi_w"], label=name)
        ax.set_xlabel("t")
        ax.set_ylabel("sin(angle(Z_meas))")
        ax.set_title("S-0009 phase-bias diagnostic (window-restricted)")
        ax.legend()
        phase_path = out_dir / cfg.out_phase_png
        fig.savefig(phase_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote (untracked): {phase_path}")

        # Error diagnostic plot (dlogc vs t in window)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for name in ["GAUSS", "K5+", "K5-"]:
            ax.plot(plot_series[name]["t_w"], plot_series[name]["dlogc_w"], label=name)
        ax.set_xlabel("t")
        ax.set_ylabel("log(c_meas) - log(c_pred)")
        ax.set_title("S-0009 magnitude-closure error (window-restricted)")
        ax.legend()
        err_path = out_dir / cfg.out_error_png
        fig.savefig(err_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote (untracked): {err_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
