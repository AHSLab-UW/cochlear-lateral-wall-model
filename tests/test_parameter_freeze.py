import copy
import hashlib
import json
import unittest

from src.parameter_freeze import (
    FREEZE,
    STATE,
    load_freeze,
    validate_freeze_data,
)
from src.fit_parameters import OUTPUT, PARAMETERS


class referenceSixStudyFreezeTests(unittest.TestCase):
    def test_new_freeze_is_separate_and_matches_fit(self):
        parameters, frozen = load_freeze()
        fit = json.loads(OUTPUT.read_text())
        self.assertEqual(FREEZE.name, "parameter_freeze.json")
        self.assertEqual(frozen["status"], STATE)
        self.assertEqual(frozen["fitted_parameter_values"], fit["parameters"])
        self.assertEqual(tuple(fit["parameters"]), PARAMETERS)
        self.assertEqual(frozen["initial_state"], fit["initial_state"])
        self.assertEqual(frozen["parameters"], fit["all_model_parameter_values"])
        self.assertEqual(
            frozen["drug_exposure_proxy"], fit["drug_exposure_proxy"])
        self.assertEqual(frozen["fit_sha256"], hashlib.sha256(
            OUTPUT.read_bytes()).hexdigest())
        self.assertEqual(len(parameters.__dict__), len(frozen["parameters"]))

    def test_mutated_hash_or_initial_state_is_rejected(self):
        _, frozen = load_freeze()
        corrupted = copy.deepcopy(frozen)
        corrupted["fit_sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            validate_freeze_data(corrupted)
        corrupted = copy.deepcopy(frozen)
        corrupted["initial_state"][8] += 1.0
        with self.assertRaises(RuntimeError):
            validate_freeze_data(corrupted)

    def test_freeze_is_not_the_historical_freeze(self):
        _, frozen = load_freeze()
        self.assertNotEqual(
            frozen["historical_freeze_preserved"]["path"], str(FREEZE.name))
        self.assertNotIn("salt_figure_5", frozen["objective_components"])
        self.assertTrue(frozen["fit_provenance"]["excluded_fig5_hashes"])


if __name__ == "__main__":
    unittest.main()
