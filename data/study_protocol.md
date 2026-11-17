# Working study protocol: cochlear model

Status: **not preregistered; the six-study 13-state candidate is frozen
for exploratory analysis, but no Tier 2 dataset or evaluation protocol is
certified.** The original 14-state design below remains historical context,
not the current fitted candidate. This is a working protocol.

## Current candidate and prospective evaluation gate

The current model and six-study fit are recorded in
`results/parameter_freeze.json`
(SHA-256 `ca8ef7fe46139b55a9402429ef710e692a3016e40bf7fc64f20087989e9646c4`).
`results/six_study_model_only_forecast_20261116.json`
(SHA-256 `bccd2792b09b24444999d2f78c36c7d8c5364258c47578a867ec536713c1e7d7`)
records the model-only 105, 115 and 125 dB SPL, 20-minute exposure and
100-minute recovery grid, with an 80 mV initial EP. These forecasts were
produced after outcome-informed model development; they are neither a
preregistered study-matched test nor evidence of predictive success.

Before any independent outcome is accessed:

1. A separate data custodian should screen candidate studies using only
   protocol and provenance metadata, without conveying results, figures,
   abstracts containing outcome summaries, or cohort-level measurements.
   Record possible earlier inspection, overlapping animals, sources or
   development influence; exclude uncertifiable candidates rather than
   assuming blinding. The present repository contains previously inspected
   source inventories, so file presence alone does not establish a holdout.
2. Select a genuinely uninspected, independent cohort or study without
   looking at outcomes. Record species, preparation, broadband stimulus,
   calibration, duration, recovery timing, baseline and measured endpoint
   metadata. Any mismatch to the existing model-only forecast must be
   disclosed; identical numeric dBA and dB SPL values are not interchangeable.
3. Before release, freeze and timestamp the chosen source/cohort, stimulus
   mapping, initial-state and observation operators, endpoint times and units,
   digitization and exclusion rules, handling of missing observations and
   uncertainty, discrepancy metrics and interpretation. Do not select or
   change these based on outcomes. The current forecast protocol explicitly
   permits no direct comparison; a separate, prospective comparison protocol
   is required before unlocking data. Do not invent a success threshold.
4. Preserve hashes of the snapshot, prediction, protocol and subsequently
   released unmodified data. Record who had access and when, acknowledge
   limits of the access audit, and report every prespecified endpoint.
   If a result informs any model revision, it becomes development evidence;
   use a new, untouched holdout for evaluation of the revision.

No source or cohort has yet met these gates, and no evaluation result can
be claimed. Because the model's nominal baseline is not an exact rest
equilibrium, any normalization or matched quiet-control convention must be
fixed before outcome release rather than silently fitted to the comparison.

## 1. Scope and scientific contract

The proposed system has the 14 differential states listed in
[parameter_dict.md](parameter_dict.md). Plasma glucose, arterial oxygen and
furosemide are inputs, not additional differential states. Every prediction
must specify an input protocol, initial conditions, parameter set, observation
mapping, units and model version.

The primary objective is to test whether a physically coherent, reduced model
can jointly account for selected electrochemical and physiological responses
without adjusting its parameters to the evaluation measurements. A mechanism
is a model hypothesis unless supported independently; reproducing an outcome
does not establish that the proposed mechanism is unique.

## 2. Important limitation of the evidence split

The two-tier separation in [evidence_hierarchy.md](evidence_hierarchy.md) is a
restriction on the data used for calibration, not evidence that the present
researchers or model design are blind to Tier 2. That file already describes
several desired outcomes, and the reference draft and existing evidence index may
contain outcomes or parameter choices informed by them. Audit all inherited
equations, constants, fitting decisions and protocols for this provenance
before designating anything independently held out.

If architecture or parameter choices were informed by Tier 2 outcomes, report
the evaluation as **previously known, outcome-informed testing**, not a
prospective or independently blinded validation. Obtain genuinely independent
data, not used in model development or selection, for a stronger external
test. Do not equate splitting measurements from one paper with independence
without checking shared experiments, animals and analysis.

## 3. Evidence and parameter ledger

For each claim or numerical target, record the primary source and exact
location, species, preparation, sample, intervention, measurement, units,
digitization method, uncertainty, and whether the quantity is direct,
derived, a proxy, an assumption or unavailable. Record when a source is not
accessible. Link each fitted or fixed model parameter to its evidence and
record whether it is directly measured, indirectly inferred, assumed, or
structurally unidentifiable.

Only approved Tier 1 observations enter the calibration objective or choose
parameter values. Record any existing parameter values whose provenance is
unknown; do not silently inherit them. Tier 2 stimuli and demographics may
be used to specify experimental inputs, but any condition-specific input
derived from an evaluation outcome must be labeled and treated as an
assumption or sensitivity analysis, not as a zero-retuning prediction.

For the aging scenario, a change in receptor conductance must have an
independently justified magnitude or range determined before comparing the
aging outcomes. Otherwise report a conditional parameter sweep, not an
independently parameterized prediction.

## 4. Mathematical and calibration gate

Before freezing, specify all differential and algebraic equations, domains,
compartments, volumes, initial conditions, units, inputs and observation
operators. Verify:

1. Dimensional consistency and physically admissible parameter ranges.
2. The potassium inventory identity, including volume changes and all
   transfers; conservation at rest and during acoustic trauma, and an
   explicitly accounted sink under the diuretic condition.
3. Whether a separate endolymph concentration follows from the represented
   inventories and volumes. Conservation of total potassium alone does not
   imply conservation of endolymph concentration.
4. The distinction between circuit state, source conductance and the actual
   experimental measurement of input resistance.
5. Evidence support and alternative interpretations for the carrier,
   oxygen-extraction, energetic, osmotic, pharmacologic and injury closures.

Predefine the Tier 1 objective, data transformations, uncertainty weights,
parameter bounds and optimization procedure. Distinguish reported
uncertainties from investigator-chosen tolerances. Investigate identifiability
and report correlated, weakly constrained and assumed parameters. Conduct
multistart or other appropriate optimization checks and retain convergence
diagnostics. Failure to identify a parameter is not fixed by calling it
calibrated.

## 5. Freeze and reproducibility gate

Before quantitative Tier 2 comparison, produce an immutable, dated snapshot
containing the equations and implementation; a complete parameter table with
values, units and provenance; initial states; solver and tolerances;
calibration data and scripts; observation operators; evaluation inputs; and
the analysis and figure-generation scripts. Record file hashes and software
versions and verify that the snapshot regenerates Tier 1 results.

For each evaluation, specify before comparison:

- The independent source and preparation, stimulus, output and observation
  operator, including any modeled-to-measured proxy assumptions.
- Endpoints, units, baseline normalization, time windows, data extraction
  and uncertainty rules, and how missing or censored observations are handled.
- The discrepancy metrics and, only if justified independently, numerical
  acceptance thresholds. If evidence does not justify a threshold, report the
  discrepancy and uncertainty without inventing a pass/fail criterion.
- Applicable parameter uncertainty and protocol uncertainty, and whether
  measured experimental inputs or a declared plausible input range are used.

Record which evaluation outcomes were already known or used in model design.
The freeze prevents subsequent fitting; it does not retroactively create
blinding or independent model selection.

## 6. Verification and stability gate

Test units, state ordering, conservation identities, admissible state domains,
equilibria, input protocols, solver failures and the reproducibility of saved
results. Check numerical convergence across tolerances and suitable solver
families. Analyze local stability where differentiable; explicitly address the
piecewise minimum and stimulus switches. Investigate relevant parameter
uncertainty and boundary behavior, and distinguish mathematical proofs from
finite numerical experiments. Do not infer global stability from a local
Jacobian or from a small selection of trajectories.

Correct implementation errors with an auditable change record and repeat
affected checks. Changes to model structure, parameter selection, evaluation
metrics or acceptance criteria after observing Tier 2 outcomes invalidate the
original confirmatory interpretation; label subsequent analyses exploratory
and seek an untouched test set.

## 7. Reporting gate

Report all prespecified comparisons, including mismatches, uncertain or
unavailable comparisons, alternative explanations and limitations. Separate
directly observed quantities from hidden model states and mechanisms; do not
claim that a response curve proves a unique physiological pathway. Report
species and preparation differences, uncertainty, failure modes, code and data
availability, and source licensing constraints. The manuscript is a draft
until its source support, results, figures, author requirements and author
approval have been checked.
