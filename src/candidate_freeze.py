"""Create and verify the outcome-informed reference candidate evaluation freeze."""

from dataclasses import asdict, fields
import hashlib
import json
import platform

import numpy as np
import scipy

from src.model_equations import Parameters, baseline, related_code_closure
from src.kinetics_fit import ALL_NAMES
from src.wang_noise_cohorts import OUTPUT as FIT
from src.wang_noise_traces import NAMES, make_parameters
from src.pike_trace_diagnostic import ROOT


FREEZE = ROOT / "results" / "initial_candidate.json"
MODEL = ROOT / "src" / "model_equations.py"
ARCHIVED_MODEL = ROOT / "results" / "model_initialization.py"
STATE = "user-authorized exploratory reference candidate evaluation freeze"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_freeze():
    fit = json.loads(FIT.read_text())
    if fit["status"] != "conditional exploratory two-study noise optimization":
        raise ValueError("completed conditional two-study candidate required")
    drug = [fit["drug_parameters_frozen"][key] for key in ALL_NAMES]
    sound = [fit["sound_parameters"][key] for key in NAMES]
    p = make_parameters(sound, drug)
    return {
        "status": STATE,
        "freeze_date": "2026-11-16",
        "model": "reference 13-state manuscript reconstruction with explicit related-code hybrid closure",
        "model_source": str(MODEL.relative_to(ROOT)),
        "model_sha256": sha256(MODEL),
        "fit_source": str(FIT.relative_to(ROOT)),
        "fit_sha256": sha256(FIT),
        "closure": related_code_closure().description,
        "parameters": asdict(p),
        "drug_exposure_proxy": {name: float(value) for name, value in zip(ALL_NAMES[:2], drug[:2])},
        "initial_state": baseline(p).tolist(),
        "environment_at_freeze": {"python": platform.python_version(),
                                  "numpy": np.__version__, "scipy": scipy.__version__},
        "limitations": [
            "User-authorized prospective computational freeze, not a preregistration, blinded model selection, independent validation or proof of biological validity.",
            "Selected parameters were informed by the drug objective and conditional 1990/1992 noise fit; two sound parameters are at numerical bounds and joint optimality is not established.",
            "The draft leaves key closure definitions unspecified; the selected implementation uses a related-code hybrid closure and phenomenological drug K boundary.",
            "The nominal initial vector is not exactly stationary. Draft state ranges are violated in some fitted trajectories and are not proven invariants.",
            "Salt and Konishi and other proposed evaluation outcomes were inspected during earlier exploratory development; subsequent comparisons are nonblinded.",
            "Tier 2 protocol and observation mappings require explicit documentation. No parameter tuning is permitted under this freeze.",
        ],
    }


def load_freeze(path=FREEZE):
    data = json.loads(path.read_text())
    if data["status"] != STATE:
        raise ValueError("unrecognized model freeze")
    if sha256(MODEL) != data["model_sha256"]:
        raise RuntimeError("model source differs from frozen hash; refusing to run")
    if sha256(ARCHIVED_MODEL) != data["model_sha256"]:
        raise RuntimeError("archived model differs from frozen hash; refusing to run")
    if set(data["parameters"]) != {f.name for f in fields(Parameters)}:
        raise ValueError("frozen parameter set differs from model parameter schema")
    p = Parameters(**data["parameters"])
    if data["closure"] != related_code_closure().description:
        raise RuntimeError("hybrid closure differs from frozen description")
    if not np.array_equal(np.asarray(data["initial_state"]), baseline(p)):
        raise RuntimeError("initial state differs from frozen nominal baseline")
    required = set(ALL_NAMES[:2])
    if set(data["drug_exposure_proxy"]) != required:
        raise ValueError("frozen drug exposure proxy is incomplete")
    return p, data


def main():
    snapshot = build_freeze()
    if FREEZE.exists() or ARCHIVED_MODEL.exists():
        raise FileExistsError("freeze or archived model already exists; neither will be overwritten")
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
