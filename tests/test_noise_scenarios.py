import unittest

import numpy as np

from src.model_equations import STATE_NAMES
from src.parameter_freeze import load_freeze
from src.noise_scenarios import (
    SOUND_LEVELS_DB_SPL,
    run_scenario,
)


class referenceSixStudyModelOnlyScenarioTests(unittest.TestCase):
    def test_scenario_uses_verified_freeze_and_returns_finite_13_state_outputs(self):
        parameters, freeze = load_freeze()
        result = run_scenario(
            115.0, parameters, np.asarray(freeze["initial_state"]), sample_interval_min=5.0)

        self.assertIn(115.0, SOUND_LEVELS_DB_SPL)
        self.assertEqual(result["scenario_id"], "sound_115_dB_SPL")
        self.assertEqual(
            set(result["output_states"]),
            {"t_0_min", "t_20_min", "t_120_min"},
        )
        self.assertEqual(set(result["output_states"]["t_0_min"]), set(STATE_NAMES))
        for state in result["output_states"].values():
            self.assertTrue(np.isfinite(list(state.values())).all())
        self.assertAlmostEqual(result["output_states"]["t_0_min"]["ep"], 80.0)
        self.assertTrue(np.isfinite(list(result["metrics"].values())).all())
        self.assertEqual(result["input"]["exposure_stop_min"], 20.0)
        self.assertEqual(result["input"]["total_simulation_min"], 120.0)

    def test_time_trace_contains_dense_finite_outputs_and_matches_snapshots(self):
        parameters, freeze = load_freeze()
        result = run_scenario(
            115.0,
            parameters,
            np.asarray(freeze["initial_state"]),
            sample_interval_min=0.25,
            include_time_trace=True,
        )

        trace = result["time_trace"]
        times = np.asarray(trace["time_min"])
        self.assertEqual(len(times), 481)
        self.assertTrue(np.all(np.diff(times) > 0))
        for key in ("ep_mV", "marginal_k_mM", "endolymph_k_mM", "trauma_state"):
            values = np.asarray(trace[key])
            self.assertEqual(values.shape, times.shape)
            self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(trace["ep_mV"][0], 80.0)
        self.assertAlmostEqual(trace["ep_mV"][80], result["metrics"]["ep_at_exposure_end_mV"])
        self.assertAlmostEqual(trace["ep_mV"][-1], result["metrics"]["ep_at_recovery_end_mV"])
        self.assertAlmostEqual(trace["marginal_k_mM"][80],
                               result["output_states"]["t_20_min"]["k_marginal"])


if __name__ == "__main__":
    unittest.main()
