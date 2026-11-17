"""Create and validate a separate freeze for the six-study joint candidate."""

from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from src.model_equations import Parameters, baseline, related_code_closure
from src.fit_parameters import OUTPUT as FIT, PARAMETERS
from src.pike_trace_diagnostic import ROOT
from src.salt_1979_digitize import load_evidence
from src.salt_trace_residuals import fig3_data_hash


FREEZE = ROOT / "results/parameter_freeze.json"
MODEL = ROOT / "src/model_equations.py"
TRACE_CODE = ROOT / "src/salt_trace_residuals.py"
FIT_CODE = ROOT / "src/fit_parameters.py"
FREEZE_CODE = ROOT / "src/parameter_freeze.py"
ARCHIVED_MODEL = ROOT / "results/frozen_model.py"
HISTORICAL_FREEZE = ROOT / "results/initial_candidate.json"
STATE = "user-authorized six-study nine-parameter exploratory candidate freeze"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_freeze():
    fit = json.loads(FIT.read_text())
    if fit.get("status") != "completed six-study nine-parameter local joint fit; Salt Figure 5 excluded":
        raise ValueError("completed six-study fit report required")
    optimizer = fit["optimizer"]
    if not optimizer["success"] or not np.isfinite(optimizer["final_optimizer_score"]):
        raise ValueError("successful finite optimizer result required for freeze")
    exclusion = fit["salt_fig5_exclusion"]
    if any(exclusion.get(key) is not False for key in (
            "used_for_initialization", "used_for_start_selection", "used_in_objective",
            "data_or_image_hash_in_active_provenance")):
        raise ValueError("fit report does not certify complete Figure 5 exclusion")
    if set(fit["parameters"]) != set(PARAMETERS):
        raise ValueError("fit parameter names differ from the nine-parameter schema")
    parameters = Parameters(**fit["all_model_parameter_values"])
    expected_initial = baseline(parameters).tolist()
    if not np.allclose(fit["initial_state"], expected_initial, rtol=0, atol=1e-12):
        raise ValueError("fit report initial state does not match model parameters")
    if fit["provenance"].get("excluded_fig5_hashes") is not True:
        raise ValueError("Figure 5 hashes must be absent from active fit provenance")
    return {
        "status": STATE,
        "freeze_date": "2026-11-16",
        "candidate": "reference 13-state reconstruction; six-study local joint calibration",
        "fit_source": str(FIT.relative_to(ROOT)),
        "fit_sha256": sha256(FIT),
        "model_source": str(MODEL.relative_to(ROOT)),
        "model_sha256": sha256(MODEL),
        "archived_model_source": str(ARCHIVED_MODEL.relative_to(ROOT)),
        "closure": related_code_closure().description,
        "parameters": asdict(parameters),
        "fitted_parameter_values": fit["parameters"],
        "parameter_bounds": fit["bounds"],
        "initial_state": expected_initial,
        "drug_exposure_proxy": fit["drug_exposure_proxy"],
        "optimizer": optimizer,
        "objective_components": {
            name: component["weighted_sum_squares"]
            for name, component in fit["components"].items()
        },
        "fig3_provenance": fit["fig3_provenance"],
        "fit_code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (FIT_CODE, TRACE_CODE, FREEZE_CODE)
        },
        "fit_provenance": fit["provenance"],
        "historical_freeze_preserved": {
            "path": str(HISTORICAL_FREEZE.relative_to(ROOT)),
            "sha256": sha256(HISTORICAL_FREEZE),
        },
        "environment_at_freeze": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "limitations": list(fit["limitations"]),
    }


def validate_freeze_data(data, *, fit_path=FIT, model_path=MODEL,
                         archived_model_path=ARCHIVED_MODEL):
    if data.get("status") != STATE:
        raise ValueError("unrecognized six-study model freeze")
    if sha256(fit_path) != data["fit_sha256"]:
        raise RuntimeError("fit report differs from frozen hash; refusing to run")
    if sha256(model_path) != data["model_sha256"]:
        raise RuntimeError("active model source differs from frozen hash; refusing to run")
    if sha256(archived_model_path) != data["model_sha256"]:
        raise RuntimeError("archived model differs from frozen hash; refusing to run")
    fit = json.loads(Path(fit_path).read_text())
    if data["fitted_parameter_values"] != fit["parameters"]:
        raise RuntimeError("frozen parameter values differ from fit report")
    if data["parameters"] != fit["all_model_parameter_values"]:
        raise RuntimeError("frozen model parameters differ from fit report")
    if data["drug_exposure_proxy"] != fit["drug_exposure_proxy"]:
        raise RuntimeError("frozen drug exposure proxies differ from fit report")
    if set(data["parameters"]) != {item.name for item in fields(Parameters)}:
        raise ValueError("frozen parameter set differs from model parameter schema")
    parameters = Parameters(**data["parameters"])
    if data["closure"] != related_code_closure().description:
        raise RuntimeError("hybrid closure differs from frozen description")
    if not np.array_equal(np.asarray(data["initial_state"]), baseline(parameters)):
        raise RuntimeError("initial state differs from frozen nominal baseline")
    if set(data["drug_exposure_proxy"]) != set(PARAMETERS[:2]):
        raise ValueError("drug exposure proxy is incomplete")
    if data["fit_provenance"].get("excluded_fig5_hashes") is not True:
        raise RuntimeError("fit provenance does not exclude Figure 5 hashes")
    for relative, expected in data["fit_code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"frozen fit code differs from hash: {relative}")
    for relative, expected in fit["provenance"]["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"frozen objective code differs from hash: {relative}")
    evidence = load_evidence()
    if fig3_data_hash(evidence) != data["fig3_provenance"]["fig3_data_sha256"]:
        raise RuntimeError("Salt Figure 3 extracted data differ from frozen hash")
    image_hashes = {
        evidence["source"]["rendering"]["images"][key]["file"]:
        evidence["source"]["rendering"]["images"][key]["sha256"]
        for key in ("fig3_page", "fig3_crop")
    }
    if data["fig3_provenance"]["fig3_source_image_sha256"] != image_hashes:
        raise RuntimeError("frozen Salt Figure 3 image provenance differs from evidence")
    for relative, expected in data["fig3_provenance"]["fig3_source_image_sha256"].items():
        path = ROOT / relative
        if path.exists() and sha256(path) != expected:
            raise RuntimeError(f"Salt Figure 3 source image differs from hash: {relative}")
    old_freeze = data["historical_freeze_preserved"]
    if sha256(ROOT / old_freeze["path"]) != old_freeze["sha256"]:
        raise RuntimeError("historical freeze changed after new freeze creation")
    return parameters, data


def load_freeze(path=FREEZE):
    data = json.loads(Path(path).read_text())
    return validate_freeze_data(data)


def main():
    if FREEZE.exists() or ARCHIVED_MODEL.exists():
        raise FileExistsError("new freeze or model snapshot already exists; neither is overwritten")
    snapshot = build_freeze()
    model_bytes = MODEL.read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != snapshot["model_sha256"]:
        raise RuntimeError("model changed while creating freeze")
    with ARCHIVED_MODEL.open("xb") as file:
        file.write(model_bytes)
    with FREEZE.open("x", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=2, allow_nan=False)
        file.write("\n")
    load_freeze()
    print(f"Wrote {FREEZE}")


if __name__ == "__main__":
    main()
