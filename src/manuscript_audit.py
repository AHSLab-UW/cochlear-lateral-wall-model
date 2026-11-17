"""Reproduce a clearly labelled hybrid reconstruction of the reference draft."""

import hashlib
import json

import numpy as np

from src.model_equations import (
    Inputs, Parameters, STATE_NAMES, baseline, drug_concentration, endolymph,
    observables, related_code_closure, rhs, simulate,
)
from src.pike_trace_diagnostic import ROOT


OUTPUT = ROOT / "results" / "exploratory" / "manuscript_audit.json"
MANUSCRIPT = (ROOT / "text" / "reference draft" / "Cochlear_Biophysical_Model_draft"
              / "Cochlear_Biophysical_Model_13State_Manuscript_v4.tex")

STATE_RANGES = {
    "flow": (.15, 2.5), "oxygen": (.005, .130), "atp_sync": (.1, 1.5),
    "atp_marginal": (.1, 1.5), "strial_volume": (.010, .250),
    "k_sync": (40., 180.), "k_intrastrial": (.5, 60.), "k_marginal": (60., 180.),
    "ep": (-50., 130.), "endolymph_atp": (.001, 5.),
    "met_adaptation": (0., 1.), "trauma": (0., 1.),
}
OPEN_UPPER = {"met_adaptation", "trauma"}


def sampled_state_ranges(times, trajectory):
    """Compare samples against the manuscript's declared state ranges."""
    times = np.asarray(times, dtype=float)
    trajectory = np.asarray(trajectory, dtype=float)
    if (times.ndim != 1 or len(times) == 0 or trajectory.shape != (len(STATE_NAMES), len(times))
            or not np.isfinite(times).all() or not np.isfinite(trajectory).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("times and trajectory must be finite, aligned and strictly ordered")
    result = {}
    for i, name in enumerate(STATE_NAMES):
        if name not in STATE_RANGES:
            continue
        low, high = STATE_RANGES[name]
        values = trajectory[i]
        outside = (values < low) | (values >= high if name in OPEN_UPPER else values > high)
        invalid = np.flatnonzero(outside)
        result[name] = {
            "declared_lower": low, "declared_upper": high,
            "upper_inclusive": name not in OPEN_UPPER,
            "sampled_minimum": float(np.min(values)),
            "sampled_maximum": float(np.max(values)),
            "outside_sample_count": int(len(invalid)),
            "first_outside_time_min": float(times[invalid[0]]) if len(invalid) else None,
        }
    return result


def build_report():
    p = Parameters()
    closure = related_code_closure()
    initial = baseline(p)
    times = np.linspace(0., 125., 2501)
    forcing = lambda t: Inputs(drug=drug_concentration(t, 5., 5.542, 4.2, p),
                               furosemide_protocol=True)
    trajectories = {method: simulate(times, forcing, closure, p, boundaries=(5., 5.542), method=method)
                    for method in ("Radau", "BDF")}
    sol = trajectories["Radau"]
    ep_i, volume_i = int(np.argmin(sol[8])), int(np.argmax(sol[4]))
    k = np.array([endolymph(y, p)[1] for y in sol.T])
    output = {
        "status": "exploratory hybrid manuscript reconstruction, not validated or fitted",
        "manuscript": str(MANUSCRIPT.relative_to(ROOT)),
        "manuscript_sha256": hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest(),
        "state_names": STATE_NAMES,
        "closure": closure.description,
        "protocol": {"start_min": 5., "stop_min": 5.542,
                     "dose_proxy": 4.2, "furosemide_boundary_active_for_entire_run": True,
                     "sampling_step_min": .05, "time_window_min": [0., 125.]},
        "limitations": [
            "The draft leaves U_C/U_M undefined; the selected closure and basal paracellular leak come from related 19-state code, not verified manuscript source.",
            "Delta P2X conductance is interpreted as max(0, current minus baseline); nkcc_avail in perilymph flux is interpreted as h(D).",
            "The draft's 80 mg/kg 4.2 dose proxy is used without additional infusion scaling; start and stop times are explicit illustrative assumptions.",
            "Displayed rounded parameters produce a baseline voltage derivative, not an exact equilibrium; no unstated recalibration was applied.",
            "This is not a reproduction of the unavailable original draft-generating code, independent validation, or physical ion/water/charge closure.",
            "The scenario uses the phenomenological treated boundary K loss and relaxation for the entire furosemide protocol.",
            "The manuscript's declared ranges are compared at samples, not enforced as invariant physiological limits; solver agreement and sampled checks are not a global stability proof.",
        ],
        "reference_state": dict(zip(STATE_NAMES, map(float, initial))),
        "reference_derivatives": dict(zip(STATE_NAMES, map(float, rhs(0., initial, Inputs(), closure, p)))),
        "reference_observables": observables(initial, Inputs(), closure, p),
        "sampled_declared_state_ranges": {
            method: sampled_state_ranges(times, values) for method, values in trajectories.items()},
        "sampled_output": {
            "ep_nadir_mV": float(sol[8, ep_i]),
            "ep_nadir_time_min": float(times[ep_i]),
            "ep_nadir_time_after_injection_min": float(times[ep_i] - 5.),
            "endolymph_k_minimum_mM": float(np.min(k)),
            "strial_volume_maximum": float(sol[4, volume_i]),
            "strial_volume_peak_time_after_injection_min": float(times[volume_i] - 5.),
            "total_k_change": float(sol[10, -1] - sol[10, 0]),
            "max_abs_radau_bdf_by_state": dict(zip(STATE_NAMES,
                map(float, np.max(np.abs(sol - trajectories["BDF"]), axis=1)))),
        },
    }
    return output


def main():
    report = build_report()
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
