#!/usr/bin/env python3
"""
s0004_sensitivity_sigma_scale.py

S-0004-SENS: Sweep sigma_scale (scale-mixture spread) and map the
dimensionless boundary metric:

    rho(t*) := |k4_direct(t*)| / k2_direct(t*)^2

We categorize each point with a 4-color logic:
- Invalid (Gray): gauss_ok == False
- Noise Floor (Blue): gauss_ok AND closure.pass == True
- Transition (Orange): gauss_ok AND closure.pass == False AND not certified_fail
- Failure (Red): gauss_ok AND certified_fail == True

Certified_fail additionally requires fail-by-margin + collapse direction.

We also implement the 2-consecutive-point rule to identify rho_crit bracket:
    [rho_last_nonfail, rho_first_fail]
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Import the already-certified machinery from S-0004
from s0004_k4_even_cumulant_boundary_demo import (  # type: ignore
    DemoConfig,
    _mask_window,
    _run_utc_iso,
    _git_commit_short,
    simulate_omega_gauss,
    simulate_omega_k4,
    run_subcase,
    audit_gaussian_identity,
    gauss_sanity_certificate,
)

# ---------------------------------------------------------------------
# Sweep config (kept separate from DemoConfig to avoid perturbing S-0004)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SensConfig:
    # Number of sweep points
    n_points: int = 40

    # Log sweep bounds for sigma_scale (spread multiplier)
    # Choose a range that spans "nearly Gaussian" to "definitely non-Gaussian"
    sigma_min: float = 0.85
    sigma_max: float = 2.50

    # 2-consecutive rule
    consec_fail_required: int = 2

    # Outputs
    out_dir: str = "toys/outputs"
    out_csv: str = "s0004_sensitivity.csv"
    out_png: str = "s0004_sensitivity.png"


# ---------------------------------------------------------------------
# Categorization + bracket finding
# ---------------------------------------------------------------------

def _certified_fail(cfg: DemoConfig, sub: Dict[str, Any]) -> bool:
    """
    Certified fail:
      - closure.pass is False
      - fail-by-margin (median and p95 exceed thresholds)
      - collapse direction OK (Jensen-consistent): resid_log_median >= 0
    """
    clo = sub["closure"]
    if bool(clo.get("pass", False)):
        return False

    med = float(clo.get("log_err_median", float("nan")))
    p95 = float(clo.get("log_err_p95", float("nan")))
    fail_by_margin = (med >= float(cfg.k4_fail_median_min)) and (p95 >= float(cfg.k4_fail_p95_min))

    resid = float(sub.get("resid_log_median", float("nan")))
    collapse_ok = resid >= float(cfg.k4_resid_sign_median_min)

    return bool(fail_by_margin and collapse_ok)


def _category(gauss_ok: bool, closure_pass: bool, certified_fail: bool) -> str:
    if not gauss_ok:
        return "Invalid"
    if closure_pass:
        return "NoiseFloor"
    if certified_fail:
        return "Failure"
    return "Transition"


def _find_rho_bracket(
    rows_sorted: List[Dict[str, Any]],
    consec_required: int = 2,
) -> Dict[str, Any]:
    """
    rows_sorted must be sorted by rho asc, and include:
      - category in {"Invalid","NoiseFloor","Transition","Failure"}
      - gauss_ok bool
      - rho float

    We find the first index i such that rows i..i+consec_required-1 are all Failure.
    Bracket is:
      rho_first_fail = rho[i]
      rho_last_nonfail = max rho < rho_first_fail among gauss_ok == True and category != Failure
    """
    fail_idx = None
    consec = 0
    for i, r in enumerate(rows_sorted):
        if (r.get("gauss_ok", False) is True) and (r.get("category") == "Failure") and np.isfinite(r["rho"]):
            consec += 1
            if consec >= consec_required:
                fail_idx = i - (consec_required - 1)
                break
        else:
            consec = 0

    if fail_idx is None:
        return {
            "found": False,
            "rho_first_fail": float("nan"),
            "rho_last_nonfail": float("nan"),
            "fail_index": -1,
            "note": f"No {consec_required}-consecutive Failure crossing found.",
        }

    rho_first_fail = float(rows_sorted[fail_idx]["rho"])

    rho_last_nonfail = float("nan")
    for j in range(fail_idx - 1, -1, -1):
        rj = rows_sorted[j]
        if (rj.get("gauss_ok", False) is True) and (rj.get("category") != "Failure") and np.isfinite(rj["rho"]):
            rho_last_nonfail = float(rj["rho"])
            break

    return {
        "found": True,
        "rho_first_fail": rho_first_fail,
        "rho_last_nonfail": rho_last_nonfail,
        "fail_index": int(fail_idx),
        "note": "",
    }


# ---------------------------------------------------------------------
# Main sweep runner
# ---------------------------------------------------------------------

def main() -> int:
    cfg = DemoConfig()
    sens = SensConfig()

    run_utc = _run_utc_iso()
    git_commit = _git_commit_short()

    out_dir = Path(sens.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    # ---- GAUSS validity gate (run once; recorded for every point) ----
    omega_gauss = simulate_omega_gauss(cfg, rng)
    res_gauss = run_subcase(cfg, omega_gauss, center_method=cfg.center_method_gauss)

    id_gauss = audit_gaussian_identity(res_gauss["t"], res_gauss["c_meas_lin"], res_gauss["k2_direct"], cfg)

    sanity = gauss_sanity_certificate(
        omega0=res_gauss["omega0"],
        R=res_gauss["R"],
        k2_pred=res_gauss["k2_pred"],
        k2_direct=res_gauss["k2_direct"],
        t=res_gauss["t"],
        cfg=cfg,
    )

    gauss_ok = bool(sanity.get("sanity_ok", False)) and bool(id_gauss.get("pass", False)) and bool(
        res_gauss["closure"].get("pass", False)
    )

    # Determine t* (must match S-0004 logic: last point in audit window)
    t = np.asarray(res_gauss["t"], dtype=float)
    mask = _mask_window(t, cfg)
    if np.count_nonzero(mask) < 1:
        idx_tstar = 0
    else:
        idx_tstar = int(np.where(mask)[0][-1])
    tstar = float(t[idx_tstar])

    # ---- Sweep sigma_scale (log-spaced) ----
    sigmas = np.exp(np.linspace(math.log(sens.sigma_min), math.log(sens.sigma_max), int(sens.n_points)))

    rows: List[Dict[str, Any]] = []
    for sigma_s in sigmas:
        omega_k4 = simulate_omega_k4(cfg, rng, sigma_s=float(sigma_s))
        res_k4 = run_subcase(cfg, omega_k4, center_method=cfg.center_method_k4)

        clo = res_k4["closure"]
        closure_pass = bool(clo.get("pass", False))
        cert_fail = _certified_fail(cfg, res_k4)

        cat = _category(gauss_ok=gauss_ok, closure_pass=closure_pass, certified_fail=cert_fail)

        # rho computed inside run_subcase (direct cumulants at t*)
        rho = float(res_k4.get("rho_tstar", float("nan")))
        k2_t = float(res_k4.get("k2_direct_tstar", float("nan")))
        k4_t = float(res_k4.get("k4_direct_tstar", float("nan")))

        rows.append(
            {
                "sigma_scale": float(sigma_s),
                "rho": rho,
                "category": cat,
                "gauss_ok": bool(gauss_ok),
                "closure_pass": bool(closure_pass),
                "certified_fail": bool(cert_fail),
                "log_err_median": float(clo.get("log_err_median", float("nan"))),
                "log_err_p95": float(clo.get("log_err_p95", float("nan"))),
                "drift_slope": float(clo.get("drift_slope", float("nan"))),
                "drift_bins_used": int(clo.get("drift_bins_used", 0)),
                "resid_log_median": float(res_k4.get("resid_log_median", float("nan"))),
                "k2_direct_tstar": k2_t,
                "k4_direct_tstar": k4_t,
                "tstar": tstar,
                "idx_tstar": int(res_k4.get("idx_tstar", idx_tstar)),
                "run_utc": run_utc,
                "git_commit": git_commit,
            }
        )

    # Sort by rho for bracket detection
    rows_sorted = sorted(rows, key=lambda r: (float("inf") if not np.isfinite(r["rho"]) else r["rho"]))
    bracket = _find_rho_bracket(rows_sorted, consec_required=int(sens.consec_fail_required))

    # ---- Write CSV ----
    csv_path = out_dir / sens.out_csv
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sigma_scale",
                "rho_tstar",
                "category",
                "gauss_ok",
                "closure_pass",
                "certified_fail",
                "log_err_median",
                "log_err_p95",
                "drift_slope",
                "drift_bins_used",
                "resid_log_median",
                "k2_direct_tstar",
                "k4_direct_tstar",
                "tstar",
                "dt",
                "n_steps",
                "n_trajectories",
                "seed",
                "center_method_gauss",
                "center_method_k4",
                "audit_t_min",
                "audit_t_max",
                "k4_fail_median_min",
                "k4_fail_p95_min",
                "k4_resid_sign_median_min",
                "gauss_identity_ok",
                "gauss_sanity_ok",
                "gauss_closure_ok",
                "rho_last_nonfail",
                "rho_first_fail",
                "rho_bracket_found",
                "run_utc",
                "git_commit",
            ]
        )

        for r in rows_sorted:
            w.writerow(
                [
                    f"{r['sigma_scale']:.10g}",
                    f"{r['rho']:.10g}",
                    r["category"],
                    int(bool(r["gauss_ok"])),
                    int(bool(r["closure_pass"])),
                    int(bool(r["certified_fail"])),
                    f"{r['log_err_median']:.10g}",
                    f"{r['log_err_p95']:.10g}",
                    f"{r['drift_slope']:.10g}",
                    int(r["drift_bins_used"]),
                    f"{r['resid_log_median']:.10g}",
                    f"{r['k2_direct_tstar']:.10g}",
                    f"{r['k4_direct_tstar']:.10g}",
                    f"{r['tstar']:.10g}",
                    f"{cfg.dt:.10g}",
                    cfg.n_steps,
                    cfg.n_trajectories,
                    cfg.seed,
                    cfg.center_method_gauss,
                    cfg.center_method_k4,
                    f"{cfg.audit_window_t_min:.10g}",
                    f"{cfg.audit_window_t_max:.10g}",
                    f"{cfg.k4_fail_median_min:.10g}",
                    f"{cfg.k4_fail_p95_min:.10g}",
                    f"{cfg.k4_resid_sign_median_min:.10g}",
                    int(bool(id_gauss.get("pass", False))),
                    int(bool(sanity.get("sanity_ok", False))),
                    int(bool(res_gauss["closure"].get("pass", False))),
                    f"{bracket['rho_last_nonfail']:.10g}",
                    f"{bracket['rho_first_fail']:.10g}",
                    int(bool(bracket["found"])),
                    r["run_utc"],
                    r["git_commit"],
                ]
            )

    # ---- Plot (4-color categorical overlay) ----
    fig_path = out_dir / sens.out_png

    # Color map as requested
    color_map = {
        "Invalid": "gray",
        "NoiseFloor": "blue",
        "Transition": "orange",
        "Failure": "red",
    }

    # Build arrays
    rho_vals = np.array([r["rho"] for r in rows_sorted], dtype=float)
    med_err = np.array([r["log_err_median"] for r in rows_sorted], dtype=float)
    cats = [r["category"] for r in rows_sorted]

    plt.figure()
    for cat in ["Invalid", "NoiseFloor", "Transition", "Failure"]:
        m = np.array([c == cat for c in cats], dtype=bool)
        if np.any(m):
            plt.scatter(rho_vals[m], med_err[m], s=22, c=color_map[cat], label=cat, alpha=0.9)

    # Bracket shading/lines
    if bool(bracket["found"]) and np.isfinite(bracket["rho_first_fail"]):
        rf = float(bracket["rho_first_fail"])
        rn = float(bracket["rho_last_nonfail"])
        if np.isfinite(rn):
            plt.axvspan(rn, rf, alpha=0.12, color="orange")
            plt.axvline(rn, linestyle="--", linewidth=1.2, color="orange")
        plt.axvline(rf, linestyle="--", linewidth=1.2, color="red")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\rho(t^*) = |\kappa_4(t^*)|/\kappa_2(t^*)^2$")
    plt.ylabel(r"median $|\Delta \log c|$ in audit window")
    plt.title("S-0004-SENS: Sensitivity map (sigma_scale sweep) with 4-color validity logic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    # ---- Console summary ----
    print(f"[S-0004-SENS] gauss_ok={int(gauss_ok)}  t*={tstar:.6g}  idx_t*={idx_tstar}")
    if bracket["found"]:
        print(
            f"[S-0004-SENS] rho bracket (2-consecutive rule): "
            f"[{bracket['rho_last_nonfail']:.6g}, {bracket['rho_first_fail']:.6g}]"
        )
    else:
        print(f"[S-0004-SENS] rho bracket not found: {bracket['note']}")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {fig_path}")

    # If GAUSS gate fails, treat as invalid run
    if not gauss_ok:
        return 3

    # Otherwise: success even if bracket not found (it may be outside sweep range)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())