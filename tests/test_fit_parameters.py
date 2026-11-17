import copy
import unittest

import numpy as np

from src.fit_parameters import (
    PARAMETERS,
    START,
    evaluate_all,
    load_start,
)
from src.salt_trace_residuals import fig3_data_hash
from src.wang_noise_traces import load_noise
from src.pike_trace_diagnostic import load_anchors
from src.salt_1979_digitize import load_evidence


class referenceSixStudyJointFitTests(unittest.TestCase):
    def test_seed_is_existing_pre_figure_five_salt_table_fit(self):
        self.assertEqual(START.name, "salt_table3_fit.json")
        values = load_start()
        self.assertEqual(values.shape, (9,))
        self.assertEqual(len(PARAMETERS), 9)

    def test_fig5_data_and_hash_alterations_do_not_change_objective(self):
        evidence = load_evidence()
        altered = copy.deepcopy(evidence)
        altered["figures"]["fig5"]["series"]["95_dba"]["points"][0][1] += 500
        altered["figures"]["fig5"]["series"]["105_dba"]["points"].reverse()
        altered["source"]["rendering"]["images"]["fig5_page"]["sha256"] = "0" * 64
        altered["source"]["rendering"]["images"]["fig5_crop"]["sha256"] = "f" * 64
        self.assertEqual(fig3_data_hash(evidence), fig3_data_hash(altered))
        arguments = (load_start(), load_anchors(), load_noise())
        residual, audit = evaluate_all(*arguments, evidence)
        changed_residual, changed_audit = evaluate_all(*arguments, altered)
        np.testing.assert_array_equal(residual, changed_residual)
        self.assertEqual(audit["score"], changed_audit["score"])
        self.assertEqual(audit["fig3_data_sha256"], changed_audit["fig3_data_sha256"])
        self.assertNotIn("salt_figure_5", audit["components"])

    def test_objective_accepts_extraction_without_figure_five(self):
        evidence = load_evidence()
        evidence["figures"].pop("fig5")
        residual, audit = evaluate_all(
            load_start(), load_anchors(), load_noise(), evidence)
        self.assertTrue(np.isfinite(residual).all())
        self.assertEqual(
            set(audit["components"]),
            {"pike_bosher", "asakuma", "rybak", "wang_1992", "wang_1990",
             "salt_table_iii", "salt_figure_3"},
        )


if __name__ == "__main__":
    unittest.main()
