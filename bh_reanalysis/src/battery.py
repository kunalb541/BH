"""Run the BH reanalysis battery.

Loads cells, runs T1-T5, declares T6 not testable, writes JSON results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .load_bh_data import load_all_cells
from .predictive_causal import run_t1
from .observer_geometry import run_t2
from .threshold_shift import run_t3, run_t3_sensitivity
from .temporal_evolution import run_t4
from .intervention_fork import run_t5


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _to_jsonable(o):
    """Convert numpy types so json.dump works."""
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_jsonable(x) for x in o]
    if isinstance(o, tuple):
        return [_to_jsonable(x) for x in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cells = load_all_cells()
    print(f"Loaded {len(cells)} cells")

    # T1
    print("Running T1 (predictive vs causal dissociation)...")
    t1 = run_t1(cells)
    with (RESULTS_DIR / "t1_predictive_causal.json").open("w") as f:
        json.dump(_to_jsonable(t1), f, indent=2)
    print(f"  T1 dissociation rate (primary): {t1['dissociation_rate_primary']:.3f}")
    print(f"  T1 P1.1 pass: {t1['P1.1_pass']}")

    # T2
    print("Running T2 (observer geometry)...")
    t2 = run_t2(cells)
    with (RESULTS_DIR / "t2_observer_angles.json").open("w") as f:
        json.dump(_to_jsonable(t2), f, indent=2)
    for L, p in t2["per_L"].items():
        print(f"  L={L}: med within={p['median_within_angle']:.3f}, "
              f"med across={p['median_across_angle']:.3f}, "
              f"KS p={p['ks_pvalue']:.4g}")
    print(f"  T2 P2.1 pass (within<across): {t2['P2.1_overall_pass']}")
    print(f"  T2 P2.2 pass (magnitude floor): {t2['P2.2_overall_pass']}")
    print(f"  T2 P2.3 pass (KS distinguishable): {t2['P2.3_overall_pass']}")

    # T3
    print("Running T3 (J/U direction shift)...")
    t3 = run_t3(cells, observer_name="gap")
    t3_sens = run_t3_sensitivity(cells)
    with (RESULTS_DIR / "t3_band_edge.json").open("w") as f:
        json.dump(_to_jsonable({"primary": t3, "sensitivity": t3_sens}), f, indent=2)
    print(f"  T3 P3.1 pass: {t3['P3.1_pass']}")
    print(f"  T3 P3.2 pass: {t3['P3.2_pass']}")

    # T4
    print("Running T4 (temporal evolution)...")
    t4 = run_t4(cells)
    with (RESULTS_DIR / "t4_temporal.json").open("w") as f:
        json.dump(_to_jsonable(t4), f, indent=2)
    print(f"  T4 P4.1 (L=6 monotonic): {t4['P4.1_pass']}")
    print(f"  T4 P4.2 (L=8/9 tau2>=tau3): {t4['P4.2_pass']}")

    # T5
    print("Running T5 (scoring fork)...")
    t5 = run_t5(cells)
    with (RESULTS_DIR / "t5_fork.json").open("w") as f:
        json.dump(_to_jsonable(t5), f, indent=2)
    print(f"  T5 3-way disagreement rate: {t5['rate_3way_disagreement']:.3f}")
    print(f"  T5 P5.1 pass: {t5['P5.1_pass']}")
    print(f"  T5 P5.2 pass: {t5['P5.2_pass']}")

    # T6
    t6 = {
        "status": "NOT TESTABLE on existing data",
        "reason": (
            "Existing simulation data contains only one intervention type "
            "(additional dephasing). Particle injection, phase imprinting, "
            "and local heating were never simulated."
        ),
        "recommended_followup": (
            "Simulate at least one additional intervention type (e.g., local "
            "heating via amplitude damping, or per-site particle injection) "
            "at the same (L, J/U, tau) cells. Then run T6 from the brief on "
            "the augmented dataset."
        ),
        "locked_outcome": "DECLARATION (pre-registered as not-testable)",
    }
    with (RESULTS_DIR / "t6_hierarchy.json").open("w") as f:
        json.dump(_to_jsonable(t6), f, indent=2)
    print(f"  T6: {t6['status']}")

    print("\nAll tests complete. Results in", RESULTS_DIR)
    return {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5, "t6": t6}


if __name__ == "__main__":
    main()
