import unittest

import numpy as np

from src.structural_sensitivity import (
    RELATIVE_CHANGES,
    STRUCTURAL_PARAMETERS,
    _summarize_noise,
)
from src.model_equations import baseline
from src.parameter_freeze import load_freeze


class structuralSensitivityTests(unittest.TestCase):
    def test_selected_symbols_map_to_the_frozen_model_fields(self):
        self.assertEqual(
            STRUCTURAL_PARAMETERS,
            {
                "phi_C": "sync_volume",
                "phi_M": "marginal_volume",
                "v_S_0": "strial_volume_base",
                "q_K": "k_flux",
                "c_a": "atp_cleft_factor",
            },
        )
        self.assertEqual(RELATIVE_CHANGES, (-0.20, 0.20))

    def test_noise_summary_reports_finite_state_diagnostics(self):
        parameters, _ = load_freeze()
        summary = _summarize_noise(parameters, baseline(parameters))
        self.assertEqual(set(summary), {"105", "115", "125"})
        for metrics in summary.values():
            self.assertTrue(np.isfinite(list(metrics.values())).all())
            self.assertGreater(metrics["maximum_k_m_during_exposure_mM"], 0)
            self.assertGreaterEqual(
                metrics["post_offset_ep_peak_time_min"], 20.0
            )


if __name__ == "__main__":
    unittest.main()
