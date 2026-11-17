"""Assumption-explicit retrospective benchmarks without changing the freeze."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np

from src.manuscript_audit import sampled_state_ranges
from src.model_equations import (
    Inputs, STATE_NAMES, baseline, broadband_sound_multiplier, observables,
    related_code_closure, rhs, simulate,
)
from src.parameter_freeze import FREEZE, load_freeze
from src.pike_trace_diagnostic import ROOT


EVIDENCE = ROOT / "refs/research_evidence/tier2_zhang_hou_observations.json"
OUTPUT = ROOT / "results/tier2_zhang_hou_benchmarks.json"
LEVELS = (100, 90, 80, 70, 60)
TONE_DRIVES = {60: 1.028, 70: 1.047, 80: 1.071, 90: 1.095, 100: 1.131}
HEALTH = (
    ("Reference", 1.0, 1.0, 0.55),
    ("Moderate deficit", 0.75, 0.75, 0.35),
    ("Larger deficit", 0.50, 0.50, 0.15),
)
TRAUMA = (
    ("Lower susceptibility", 0.10, 47.5),
    ("Reference susceptibility", 0.20, 95.0),
    ("Higher susceptibility", 0.40, 190.0),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_image(paper, page, expected_size):
    with fitz.open(paper) as document:
        images = document[page - 1].get_images()
        if len(images) != 1:
            raise ValueError("source page must contain exactly one figure raster")
        image = fitz.Pixmap(document, images[0][0])
        if image.n != 3 or [image.width, image.height] != expected_size:
            raise ValueError("source image geometry or colors changed")
        return np.frombuffer(image.samples, dtype=np.uint8).reshape(
            image.height, image.width, 3).copy()


def extract_observations():
    evidence = json.loads(EVIDENCE.read_text())
    zhang = evidence["zhang"]
    levels = zhang["fig5"]
    image = source_image(ROOT / zhang["paper"], levels["pdf_page"],
                         levels["embedded_image_size_pixels"])
    animals = {}
    for animal, pixels in levels["animals_y_pixels"].items():
        # Keep the manually inspected marker coordinates auditable.
        for x, y in zip(levels["x_pixels"], pixels):
            patch = image[y - 15:y + 16, x - 15:x + 16].astype(int)
            if not np.any((patch[:, :, 0] - patch[:, :, 1] > 45)
                          & (patch[:, :, 0] - patch[:, :, 2] > 30)):
                raise ValueError(f"measured red marker missing for {animal}")
        animals[animal] = (
            (np.asarray(pixels) - levels["zero_mV_y_pixel"]) * -10
            / (levels["minus_10_mV_y_pixel"] - levels["zero_mV_y_pixel"])
        ).tolist()
    matrix = np.asarray(list(animals.values()))
    traces = zhang["fig4"]
    image = source_image(ROOT / zhang["paper"], traces["pdf_page"],
                         traces["embedded_image_size_pixels"])
    time = np.arange(-3, 14.01, 0.5)
    extracted = {}
    for channel in ("isp", "ep"):
        left, right = traces[f"{channel}_x_bounds"]
        top, bottom = traces["trace_y_bounds"]
        patch = image[top:bottom, left:right + 1].astype(int)
        mask = ((patch[:, :, 0] > 100)
                & (patch[:, :, 0] - patch[:, :, 1] > 40)
                & (patch[:, :, 0] - patch[:, :, 2] > 30))
        columns = np.flatnonzero(mask.any(axis=0))
        if len(columns) < 80 or np.max(np.diff(columns)) > 4:
            raise ValueError("source trace has missing or fragmented red pixels")
        y = np.array([np.median(np.flatnonzero(mask[:, col])) + top for col in columns])
        x = columns + left
        x_time = (x - traces[f"{channel}_onset_x_pixel"]) / (
            traces["five_seconds_scale_pixels"] / 5)
        if time[0] < x_time[0] or time[-1] > x_time[-1]:
            raise ValueError("digitization would extrapolate beyond source trace")
        extracted[channel] = (
            (np.interp(time, x_time, y) - traces["zero_y_pixel"])
            * -3 / traces["three_mV_scale_pixels"]
        ).tolist()
    return {
        "metadata": evidence,
        "zhang_levels": {
            "level_dB_SPL": levels["level_dB_SPL"], "animals_delta_isp_mV": animals,
            "mean_delta_isp_mV": matrix.mean(axis=0).tolist(),
            "sd_delta_isp_mV": matrix.std(axis=0, ddof=1).tolist(),
        },
        "zhang_pulse": {"time_seconds": time.tolist(), **extracted},
    }


def tone_forcing(seconds, levels, drives):
    if len(levels) != len(drives) or not levels:
        raise ValueError("aligned, nonempty tone levels and drives required")
    starts = 4 + 40 * np.arange(len(levels) * 4)
    index = int(np.searchsorted(starts, seconds, side="right") - 1)
    if index < 0:
        return Inputs()
    elapsed = seconds - starts[index]
    envelope = max(0., min(elapsed / .5, 1., (10 - elapsed) / .5))
    return Inputs(sound_multiplier=1 + (drives[index // 4] - 1) * envelope)


def outputs(times, states, forcing, p):
    closure = related_code_closure()
    obs = [observables(y, forcing(float(t)), closure, p)
           for t, y in zip(times, states.T)]
    return {
        "ep_mV": states[8].tolist(),
        "isp_mV": [row["intrastrial_voltage"] for row in obs],
        "k_is_mM": states[6].tolist(),
        "k_el_mM": [row["endolymph_k"] for row in obs],
        "flow": states[0].tolist(),
        "trauma": states[12].tolist(),
    }


def run_zhang(p, levels, drives, *, method="Radau", step_seconds=.1):
    starts = 4 + 40 * np.arange(4 * len(levels))
    seconds = np.linspace(0, starts[-1] + 40,
                          round((starts[-1] + 40) / step_seconds) + 1)
    times = seconds / 60
    forcing = lambda t: tone_forcing(t * 60, levels, drives)
    boundaries = [(start + edge) / 60 for start in starts for edge in (0, .5, 9.5, 10)]
    states = simulate(times, forcing, related_code_closure(), p,
                      boundaries=boundaries, method=method, rtol=1e-9, atol=1e-11)
    result = outputs(times, states, forcing, p)
    aligned_seconds = np.linspace(-3, 14, round(17 / step_seconds) + 1)
    averages = []
    for block, level in enumerate(levels):
        averaged = {}
        for key in ("ep_mV", "isp_mV", "k_is_mM", "k_el_mM"):
            values = np.asarray(result[key])
            pulses = []
            for start in starts[block * 4:(block + 1) * 4]:
                indices = np.rint((start + aligned_seconds) / step_seconds).astype(int)
                pulses.append(values[indices] - values[round(start / step_seconds)])
            averaged[key] = np.mean(pulses, axis=0).tolist()
        on = (aligned_seconds >= 0) & (aligned_seconds <= 10)
        plateau = (aligned_seconds >= 7) & (aligned_seconds <= 9)
        averages.append({
            "level_dB_SPL": level, "time_seconds": aligned_seconds.tolist(),
            "delta": averaged,
            "peak_isp_mV": float(np.min(np.asarray(averaged["isp_mV"])[on])),
            "peak_ep_mV": float(np.min(np.asarray(averaged["ep_mV"])[on])),
            "plateau_isp_mV": float(np.mean(np.asarray(averaged["isp_mV"])[plateau])),
            "plateau_ep_mV": float(np.mean(np.asarray(averaged["ep_mV"])[plateau])),
            "max_k_is_mM": float(np.max(np.asarray(averaged["k_is_mM"])[on])),
        })
    return {
        "levels_dB_SPL": list(levels), "drives": list(drives),
        "method": method, "step_seconds": step_seconds,
        "time_seconds": seconds.tolist(), "outputs": result,
        "averages": averages,
        "ranges": sampled_state_ranges(times, states),
        "inventory_max_change": float(np.max(np.abs(states[10] - states[10, 0]))),
    }


def noise_forcing(time):
    active = (0 <= time < 180) or (1440 <= time < 1620)
    return Inputs(sound_multiplier=broadband_sound_multiplier(120) if active else 1)


def run_hou(p, *, method="Radau", step_min=1.):
    closure = related_code_closure()
    settle_times = np.linspace(0, 1440, 289)
    settle = simulate(settle_times, lambda _: Inputs(), closure, p,
                      method=method, rtol=1e-8, atol=1e-10)
    initial = settle[:, -1]
    times = np.linspace(0, 1740, round(1740 / step_min) + 1)
    exposed = simulate(times, noise_forcing, closure, p, initial=initial,
                       boundaries=(180., 1440., 1620.), method=method,
                       rtol=1e-8, atol=1e-10)
    quiet = simulate(times, lambda _: Inputs(), closure, p, initial=initial,
                     method=method, rtol=1e-8, atol=1e-10)
    result = outputs(times, exposed, noise_forcing, p)
    indices = [round(t / step_min) for t in (0, 180, 1440, 1620, 1740)]
    return {
        "method": method, "step_min": step_min,
        "time_min": times.tolist(), "outputs": result,
        "quiet_ep_mV": quiet[8].tolist(), "quiet_flow": quiet[0].tolist(),
        "quiet_start_state": initial.tolist(),
        "quiet_start_rhs_max_abs": float(np.max(np.abs(rhs(0, initial, Inputs(), closure, p)))),
        "settle_ranges": sampled_state_ranges(settle_times, settle),
        "ranges": sampled_state_ranges(times, exposed),
        "inventory_max_change": float(np.max(np.abs(exposed[10] - exposed[10, 0]))),
        "endpoints": [
            {
                "time_min": float(times[i]), "ep_mV": float(exposed[8, i]),
                "flow": float(exposed[0, i]), "trauma": float(exposed[12, i]),
                "noise_minus_quiet_ep_mV": float(exposed[8, i] - quiet[8, i]),
            } for i in indices
        ],
    }


def run_benchmarks():
    p, frozen = load_freeze()
    evidence = extract_observations()
    zhang = {}
    for label, levels, drives in (
        ("fig4_broadband", [90], [broadband_sound_multiplier(90)]),
        ("fig4_tone", [90], [1.076]),
        ("fig5_broadband", LEVELS, [broadband_sound_multiplier(x) for x in LEVELS]),
        ("fig5_tone", LEVELS, [TONE_DRIVES[x] for x in LEVELS]),
    ):
        zhang[label] = run_zhang(p, levels, drives)
    hou = []
    for health, flow, permeability, autoregulation in HEALTH:
        for susceptibility, constriction, repair in TRAUMA:
            changes = {
                "flow_base": flow, "oxygen_permeability": permeability,
                "autoregulation": autoregulation,
                "trauma_vasoconstriction": constriction, "repair_time": repair,
            }
            run = run_hou(replace(p, **changes))
            run.update({"health": health, "susceptibility": susceptibility,
                        "parameter_changes": changes})
            hou.append(run)
    checks = []
    for key in ("fig4_tone", "fig5_tone", "fig4_broadband", "fig5_broadband"):
        original = zhang[key]
        check = run_zhang(p, original["levels_dB_SPL"], original["drives"], method="BDF")
        discrepancy = max(
            float(np.max(np.abs(np.asarray(original["outputs"][key])
                                - np.asarray(check["outputs"][key]))))
            for key in original["outputs"])
        if discrepancy > 1e-4:
            raise RuntimeError(f"Zhang solver disagreement: {discrepancy}")
        checks.append({"protocol": key, "Radau_BDF_max_output_difference": discrepancy})
    for index in (0, 4, 8):
        original = hou[index]
        check = run_hou(replace(p, **original["parameter_changes"]), method="BDF")
        discrepancy = max(
            float(np.max(np.abs(np.asarray(original["outputs"][key])
                                - np.asarray(check["outputs"][key]))))
            for key in original["outputs"])
        if discrepancy > .001:
            raise RuntimeError(f"Hou solver disagreement: {discrepancy}")
        checks.append({"hou_grid_index": index, "Radau_BDF_max_output_difference": discrepancy})
    refined = run_zhang(p, [90], [1.076], step_seconds=.05)
    grid_error = max(abs(refined["averages"][0][key] - zhang["fig4_tone"]["averages"][0][key])
                     for key in ("peak_isp_mV", "peak_ep_mV"))
    if grid_error > 1e-4:
        raise RuntimeError("Zhang pulse grid extrema changed")
    refined_hou = run_hou(replace(p, **hou[8]["parameter_changes"]), step_min=.5)
    hou_grid_error = max(abs(a["ep_mV"] - b["ep_mV"])
                        for a, b in zip(hou[8]["endpoints"], refined_hou["endpoints"]))
    if hou_grid_error > .001:
        raise RuntimeError("Hou endpoint grid discrepancy")
    if any(run["inventory_max_change"] > 1e-9 for run in [*zhang.values(), *hou]):
        raise RuntimeError("closed noise protocol changed total potassium")
    report = {
        "status": "Retrospective conditional Tier 2 benchmarks; not independent validation or fitting",
        "provenance": {
            "freeze_sha256": sha256(FREEZE), "script_sha256": sha256(Path(__file__)),
            "evidence_sha256": sha256(EVIDENCE),
            "papers_sha256": {study: sha256(ROOT / evidence["metadata"][study]["paper"])
                             for study in ("zhang", "hou")},
        },
        "reference_parameters": asdict(p),
        "observations": evidence, "zhang": zhang, "hou": hou,
        "checks": checks, "zhang_peak_grid_error": grid_error,
        "hou_endpoint_grid_error_mV": hou_grid_error,
        "assumptions": [
            "No optimization, scenario selection for agreement, or change to the frozen candidate.",
            "Zhang: 4 pulses/level, 10 s including linear 0.5-s excess-drive ramps, 30-s quiet gaps; no extra interlevel delay.",
            "Fig.5 source order 100 to 60 dB; all states carried forward; no preceding anoxia simulation.",
            "Tone drives are previously outcome-calibrated effective inputs, not independent acoustic calibration.",
            "Fig.4 uses 1.076; Fig.5 90-dB drive is 1.095; cohorts/maps are not pooled.",
            "Observed ISP and EP traces were sequential compartment recordings; no correlation coefficient or independent-channel likelihood is computed.",
            "Hou: health and susceptibility bundles are assumed, not ages, treatment effects or fitted mouse parameters.",
            "Each Hou scenario settles in quiet for 1440 min from baseline(p) at conserved total K before two 180-min, 120-dB exposures starting 1440 min apart.",
            "No chronological aging, chronic damage accumulation, pericyte treatment or VEGF mechanism is simulated.",
            "Hou flow/EP comparisons normalize to respective healthy control means; source control uncertainty and covariance are not propagated.",
            "All nine Hou cases retained, including any declared-range excursions; sensitivity grid is not a confidence region.",
        ],
    }
    load_freeze()
    return report


def main():
    report = run_benchmarks()
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}")
    print("Zhang peaks:", report["zhang"]["fig4_tone"]["averages"][0]["peak_isp_mV"],
          report["zhang"]["fig4_tone"]["averages"][0]["peak_ep_mV"])
    for run in report["hou"]:
        print(run["health"], run["susceptibility"], run["endpoints"][-2:])


if __name__ == "__main__":
    main()
