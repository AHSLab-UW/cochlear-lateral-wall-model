"""Exploratory 14-state cochlear ODE, not a calibrated physiological model.

Coefficients are demonstrative assumptions except for a conditional
reference-voltage anchor. See README.md and the prototype parameter ledger
before interpreting any simulation.
"""

from dataclasses import dataclass
import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from src.potassium_balance import PotassiumFluxes, PotassiumState, evaluate_potassium_balance

STATE_NAMES = (
    "flow", "oxygen", "glucose", "atp_sync", "atp_marginal", "volume",
    "k_sync", "k_intrastrial", "k_marginal", "k_total", "ep",
    "endolymph_atp", "met_adaptation", "trauma",
)


@dataclass(frozen=True)
class Inputs:
    """Prescribed inputs; drug, sound and injury are fractions, not doses.

    ``atp_influx`` is a prescribed bulk ATP concentration rate (µM/min),
    not an injected amount or a pharmacokinetic/mixing model.
    """

    glucose: float = 5.0
    oxygen: float = 1.1
    drug: float = 0.0
    sound: float = 0.0
    injury: float = 0.0
    atp_influx: float = 0.0

    def __post_init__(self) -> None:
        for name in ("glucose", "oxygen", "drug", "sound", "injury", "atp_influx"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.glucose < 0 or self.oxygen < 0 or self.atp_influx < 0:
            raise ValueError("substrate inputs and ATP influx must be nonnegative")
        for name in ("drug", "sound", "injury"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True)
class Parameters:
    """Hypothetical rates and couplings, in minutes and the units in README."""

    oxygen_tau: float = 2.0
    glucose_vmax: float = 1.0
    glucose_km: float = 5.0
    glucose_efflux: float = 0.08
    glucose_use: float = 0.1
    energy_tau: float = 3.0
    water_rate: float = 0.005
    sigma_k: float = 0.2
    sigma_g: float = 0.1
    volume_relaxation: float = 0.1
    potassium_flux: float = 0.2
    injury_leak: float = 0.001
    drug_sink: float = 0.02
    voltage_tau: float = 0.1
    atp_release: float = 0.02
    baseline_source: float = 80.0
    p2x_max_relative_conductance: float = 0.1
    p2x_half_activation: float = 5.0
    endolymph_volume_coupling: float = 1.0
    strial_reference_volume: float = 0.1
    syncytial_volume: float = 0.2
    marginal_volume: float = 0.3

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("oxygen_tau", "glucose_km", "energy_tau", "voltage_tau",
                     "p2x_half_activation", "strial_reference_volume",
                     "syncytial_volume", "marginal_volume"):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        if self.sigma_k > 1 or self.sigma_g > 1:
            raise ValueError("reflection coefficients must be at most one")
        if self.endolymph_volume_coupling > 1:
            raise ValueError("endolymph volume coupling must be at most one")


def baseline(parameters: Parameters = Parameters()) -> np.ndarray:
    """Demonstration equilibrium; numbers are not a cross-species data fit."""
    volume = parameters.strial_reference_volume
    total = (parameters.syncytial_volume * 100.0 + volume * 5.0
             + parameters.marginal_volume * 100.0 + 150.0)
    return np.array([1.0, 1.0, 5.0, 1.0, 1.0, volume, 100.0, 5.0,
                     100.0, total, 80.0, 0.013, 0.0, 0.0])


def _unpack(state: Sequence[float]) -> np.ndarray:
    y = np.asarray(state, dtype=float)
    if y.shape != (14,) or not np.isfinite(y).all():
        raise ValueError("state must contain exactly 14 finite values")
    if np.any(y[[0, 1, 2, 3, 4, 5, 6, 7, 8]] <= 0):
        raise ValueError("flow, substrates, ATP proxies, volume and K must be positive")
    if y[11] < 0 or not 0 <= y[12] <= 1 or not 0 <= y[13] <= 1:
        raise ValueError("ATP must be nonnegative and adaptation and injury in [0,1]")
    return y


def potassium_state(y: Sequence[float], p: Parameters = Parameters()) -> PotassiumState:
    return PotassiumState(p.syncytial_volume, float(y[5]), p.marginal_volume,
                          1.0, p.strial_reference_volume,
                          float(y[6]), float(y[7]), float(y[8]), float(y[9]),
                          p.endolymph_volume_coupling)


def relative_circuit_conductance(state: Sequence[float], inputs: Inputs,
                                 p: Parameters = Parameters()) -> float:
    """Local positive voltage-slope conductance relative to resting baseline.

    This dimensionless conductance is not measured cochlear partition
    conductance and cannot be expressed in ohms without a separate scale.
    """
    y = _unpack(state)
    atp = y[11]
    reference = 0.013
    half = p.p2x_half_activation
    activation = atp / (half + atp)
    reference_activation = reference / (half + reference)
    conductance = (1 + 0.05 * inputs.sound * (1 - y[12])
                   + p.p2x_max_relative_conductance * (activation - reference_activation))
    if conductance <= 0:
        raise ValueError("relative circuit conductance must be positive")
    return conductance


def relative_input_resistance(state: Sequence[float], inputs: Inputs,
                              p: Parameters = Parameters()) -> float:
    """Instantaneous inverse slope conductance normalized to resting baseline."""
    return 1.0 / relative_circuit_conductance(state, inputs, p)


def rhs(t: float, state: Sequence[float], inputs: Inputs,
        p: Parameters = Parameters()) -> np.ndarray:
    """Evaluate the piecewise-smooth ODE; invalid states raise ValueError."""
    del t
    y = _unpack(state)
    flow, oxygen, glucose, ac, am, volume, kc, ks, km, _, ep, atp, adapt, trauma = y
    kstate = potassium_state(y, p)
    ke = kstate.endolymph.concentration
    if ke <= 0:
        raise ValueError("endolymph potassium must be positive")

    glucose_support = 2.0 * glucose / (5.0 + glucose)
    oxygen_support = 2.0 * oxygen / (1.0 + oxygen)
    substrate = min(glucose_support, oxygen_support)
    flow_rate = (1.0 + 0.2 * (1.0 - (ac + am) / 2.0) - 0.3 * trauma - flow) / 2.0
    oxygen_rate = (flow * inputs.oxygen - oxygen) / p.oxygen_tau - 0.05 * substrate
    glucose_rate = (
        p.glucose_vmax * inputs.glucose / (p.glucose_km + inputs.glucose)
        - p.glucose_efflux * glucose - p.glucose_use * substrate
    )
    ac_rate = (substrate - ac) / p.energy_tau
    am_rate = (substrate + 0.05 * inputs.drug - am) / p.energy_tau
    volume_rate = (
        p.water_rate * (2 * p.sigma_k * (ks - 5.0)
                        - p.sigma_g * (inputs.glucose - glucose))
        - p.volume_relaxation * (volume - p.strial_reference_volume)
    )
    flux = p.potassium_flux
    flows = PotassiumFluxes(
        j_cs=flux * kc / 100.0 + p.injury_leak * trauma * (kc - ks),
        j_sm=flux * ks / 5.0 * am * (1 - inputs.drug),
        j_me=flux * km / 100.0,
        j_ec=flux * ke / 150.0,
        j_b=0.0,
        j_b_to_s=0.0,
        j_p=p.drug_sink * inputs.drug * ke / 150.0,
    )
    kr = evaluate_potassium_balance(kstate, flows, volume_rate)

    source = p.baseline_source + 20.0 * math.log((kc / ks) / 20.0)
    conductance = relative_circuit_conductance(y, inputs, p)
    voltage_rate = (source * am - ep * conductance) / p.voltage_tau
    atp_rate = (0.013 - atp) / 2.0 + p.atp_release * inputs.sound + inputs.atp_influx
    adaptation_rate = inputs.sound * (1 - adapt) - adapt / 2.0
    trauma_rate = 0.1 * inputs.injury * (1 - trauma) - trauma / 30.0
    return np.array([flow_rate, oxygen_rate, glucose_rate, ac_rate, am_rate,
                     volume_rate, kr.k_c_rate, kr.k_s_rate, kr.k_m_rate,
                     kr.total_amount_rate, voltage_rate, atp_rate,
                     adaptation_rate, trauma_rate])


def simulate(times: Sequence[float], forcing: Callable[[float], Inputs],
             initial: Sequence[float] | None = None,
             parameters: Parameters = Parameters(), *, method: str = "Radau",
             rtol: float = 1e-7, atol: float = 1e-9):
    """Integrate over ordered time points; callers split discontinuous inputs."""
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("times must be finite and strictly increasing")
    y0 = _unpack(baseline(parameters) if initial is None else initial)
    result = solve_ivp(lambda time, y: rhs(time, y, forcing(time), parameters),
                       (t[0], t[-1]), y0, t_eval=t, method=method,
                       rtol=rtol, atol=atol)
    if not result.success:
        raise RuntimeError(f"ODE integration failed: {result.message}")
    return result
