"""Reconstruction of the displayed 13-state reference manuscript equations.

This is not a port of the related 19-state C implementation. See the candidate
ledger for missing definitions, parameter provenance, and known discrepancies.
"""

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


STATE_NAMES = (
    "flow", "oxygen", "atp_sync", "atp_marginal", "strial_volume",
    "k_sync", "k_intrastrial", "k_marginal", "ep", "endolymph_atp",
    "total_potassium", "met_adaptation", "trauma",
)


@dataclass(frozen=True)
class Parameters:
    flow_base: float = 1.0
    flow_min: float = 0.12
    flow_smoothing: float = 0.01
    autoregulation: float = 0.55
    flow_tau: float = 1.5
    oxygen_permeability: float = 1.0
    arterial_oxygen: float = 0.130
    oxygen_base: float = 0.070
    oxygen_km: float = 0.035
    oxygen_transport: float = 0.081
    oxygen_stoichiometry: float = 6.0
    oxygen_scaling: float = 0.0045
    metabolism_base: float = 0.18
    demand_gain: float = 0.35
    atp_base: float = 1.0
    sync_atp_tau: float = 2.0
    marginal_atp_tau: float = 1.5
    sync_pump_fraction: float = 0.65
    marginal_pump_fraction: float = 0.78
    sync_atp_km: float = 0.40
    marginal_atp_km: float = 0.45
    marginal_k_km: float = 4.0
    strial_volume_base: float = 0.035
    sync_volume: float = 0.4
    marginal_volume: float = 0.042
    water_permeability: float = 0.00024
    potassium_reflection: float = 0.95
    volume_restoration: float = 0.095
    volume_span: float = 0.07
    k_sync_base: float = 140.0
    k_strial_base: float = 1.5
    k_marginal_base: float = 145.0
    k_endolymph_base: float = 150.0
    k_plasma: float = 4.2
    k_flux: float = 2.8
    kir_fraction: float = 1.0
    blb_leak: float = 0.0
    drug_k_loss: float = 0.0165
    drug_k_relax: float = 0.022
    thermal_voltage: float = 26.7
    pump_voltage: float = 15.0
    voltage_offset: float = 10.16
    intrastrial_voltage_offset: float = 4.87
    sync_resistance: float = 1.0
    strial_conductance: float = 1.0
    drug_conductance_loss: float = 0.44
    met_conductance: float = 0.3
    met_reversal: float = -70.0
    p2x_reversal: float = 0.0
    capacitance: float = 0.03
    ep_base: float = 80.0
    resistance_series_fraction: float = 0.42201480
    drug_onset: float = 0.22
    drug_fast: float = 1.55
    drug_slow: float = 48.0
    drug_slow_fraction: float = 0.27
    drug_hill: float = 2.6
    atp_endolymph_base: float = 0.0074
    atp_release: float = 0.1
    atp_degradation: float = 1.5
    atp_cleft_factor: float = 30.0
    p2x_max: float = 0.9797
    p2x_half: float = 12.34
    p2x_hill: float = 1.0
    adapt_rate: float = 0.145
    adapt_recovery: float = 90.0
    adapt_half: float = 0.105
    adapt_hill: float = 2.2
    adapt_rest: float = 0.23
    adapt_evoked: float = 0.68
    adapt_p2x_fraction: float = 0.78
    adapt_p2x_half: float = 0.006
    trauma_half: float = 0.272
    trauma_hill: float = 14.0
    trauma_time: float = 26.0
    repair_time: float = 95.0
    trauma_vasoconstriction: float = 0.76
    trauma_k_leak: float = 0.052

    def __post_init__(self):
        if not all(math.isfinite(x) for x in vars(self).values()):
            raise ValueError("parameters must be finite")
        positive = ("flow_tau", "flow_smoothing", "oxygen_base", "oxygen_km", "atp_base",
                    "sync_atp_tau", "marginal_atp_tau", "sync_atp_km", "marginal_atp_km",
                    "marginal_k_km", "strial_volume_base", "sync_volume", "marginal_volume",
                    "volume_span", "k_sync_base", "k_strial_base", "k_marginal_base",
                    "k_endolymph_base", "thermal_voltage", "capacitance", "met_conductance",
                    "drug_onset", "drug_fast", "drug_slow", "drug_hill", "p2x_half", "p2x_hill",
                    "adapt_recovery", "adapt_half", "adapt_hill", "adapt_p2x_half",
                    "trauma_half", "trauma_hill", "trauma_time", "repair_time")
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError("time constants, reference concentrations, volumes and denominators must be positive")
        fractions = ("sync_pump_fraction", "marginal_pump_fraction", "potassium_reflection",
                     "drug_conductance_loss", "resistance_series_fraction", "drug_slow_fraction",
                     "adapt_rest", "adapt_evoked", "adapt_p2x_fraction", "trauma_vasoconstriction")
        if any(not 0 <= getattr(self, name) <= 1 for name in fractions):
            raise ValueError("fractional parameters must be in [0,1]")
        nonnegative = ("flow_base", "flow_min", "autoregulation", "oxygen_permeability",
                       "arterial_oxygen", "oxygen_transport", "oxygen_stoichiometry", "oxygen_scaling",
                       "metabolism_base", "demand_gain", "water_permeability", "volume_restoration",
                       "k_plasma", "k_flux", "kir_fraction", "blb_leak", "drug_k_loss", "drug_k_relax",
                       "pump_voltage", "strial_conductance", "sync_resistance", "atp_endolymph_base", "atp_release",
                       "atp_degradation", "atp_cleft_factor", "p2x_max", "adapt_rate", "trauma_k_leak")
        if any(getattr(self, name) < 0 for name in nonnegative):
            raise ValueError("rates, conductances and concentrations must be nonnegative")
        if self.k_sync_base == self.k_strial_base:
            raise ValueError("baseline Kir gradient must be nonzero")
        if self.ep_base <= self.met_reversal:
            raise ValueError("reference EP must exceed MET reversal")


@dataclass(frozen=True)
class Inputs:
    """External inputs; furosemide flag controls the manuscript boundary term."""
    drug: float = 0.0
    sound_multiplier: float = 1.0
    atp_delivery: float = 0.0
    furosemide_protocol: bool = False

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.drug, self.sound_multiplier, self.atp_delivery)):
            raise ValueError("inputs must be finite")
        if self.drug < 0 or self.sound_multiplier < 1 or self.atp_delivery < 0:
            raise ValueError("drug and ATP delivery must be nonnegative and sound multiplier >= 1")


def baseline(p: Parameters = Parameters()) -> np.ndarray:
    total = (p.k_endolymph_base + p.sync_volume * p.k_sync_base
             + p.strial_volume_base * p.k_strial_base + p.marginal_volume * p.k_marginal_base)
    return np.array([p.flow_base, p.oxygen_base, p.atp_base, p.atp_base,
                     p.strial_volume_base, p.k_sync_base, p.k_strial_base,
                     p.k_marginal_base, p.ep_base, p.atp_endolymph_base, total, 0., 0.])


def drug_concentration(time: float, start: float, stop: float, level: float,
                       p: Parameters = Parameters()) -> float:
    """Finite-infusion forcing from the draft, with concentration in IC50 units."""
    if not all(math.isfinite(v) for v in (time, start, stop, level)):
        raise ValueError("protocol arguments must be finite")
    if start < 0 or stop <= start or level < 0:
        raise ValueError("require 0 <= start < stop and nonnegative level")
    if time < start:
        return 0.0
    rise = -math.expm1(-(min(time, stop) - start) / p.drug_onset)
    elapsed = max(0., time - stop)
    decay = ((1. - p.drug_slow_fraction) * math.exp(-elapsed / p.drug_fast)
             + p.drug_slow_fraction * math.exp(-elapsed / p.drug_slow))
    return level * rise * decay


def endolymph(state, p: Parameters = Parameters()):
    y = np.asarray(state, dtype=float)
    if y.shape != (13,) or not np.isfinite(y).all():
        raise ValueError("state must contain 13 finite values")
    volume = 1. + p.strial_volume_base - y[4]
    if volume <= 0 or y[4] <= 0 or np.any(y[[0, 1, 2, 3, 5, 6, 7]] <= 0):
        raise ValueError("positive flow, oxygen, ATP proxies, volumes and potassium are required")
    ke = (y[10] - p.sync_volume * y[5] - y[4] * y[6] - p.marginal_volume * y[7]) / volume
    if ke <= 0:
        raise ValueError("endolymph potassium must be positive")
    return volume, ke


def validate_sampled_domain(state, p: Parameters = Parameters(), *, tolerance=1e-8):
    """Check declared nonnegative/bounded states at an output sample."""
    y = np.asarray(state, dtype=float)
    endolymph(y, p)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if y[9] < -tolerance or np.any(y[[11, 12]] < -tolerance) or np.any(y[[11, 12]] > 1. + tolerance):
        raise ValueError("sampled ATP, adaptation or trauma is outside its domain")


@dataclass(frozen=True)
class MissingEquationClosure:
    """Explicit assumptions for equations and a parameter missing from the draft."""
    workloads: Callable[[np.ndarray, float, Parameters], tuple[float, float]]
    paracellular_leak: float
    description: str

    def __post_init__(self):
        if not callable(self.workloads):
            raise ValueError("workload closure must be callable")
        if not math.isfinite(self.paracellular_leak) or self.paracellular_leak < 0:
            raise ValueError("paracellular leak must be finite and nonnegative")
        if not self.description.strip():
            raise ValueError("closure provenance must be specified")


def related_code_workloads(y, ke, p):
    """Workload definitions from the related 19-state code, not the manuscript."""
    ac, am, ks = y[2], y[3], y[6]
    uc = (ac / (p.sync_atp_km + ac) * (p.sync_atp_km + p.atp_base) / p.atp_base
          * ke / p.k_endolymph_base)
    um = (am / (p.marginal_atp_km + am)
          * (p.marginal_atp_km + p.atp_base) / p.atp_base
          * ks / (p.marginal_k_km + ks) * (p.marginal_k_km + p.k_strial_base) / p.k_strial_base)
    return uc, um


def _hill_pair(value, half, exponent):
    """Return activation and its complement without overflowing Hill powers."""
    if not math.isfinite(value) or value < 0:
        raise ValueError("Hill input must be finite and nonnegative")
    if value == 0:
        return 0., 1.
    x = exponent * (math.log(value) - math.log(half))
    if x >= 0:
        e = math.exp(-x)
        return 1. / (1. + e), e / (1. + e)
    e = math.exp(x)
    return e / (1. + e), 1. / (1. + e)


def trauma_derivative(sound_multiplier, trauma, p):
    """Manuscript Hill-type acoustic-trauma accumulation and repair."""
    activation, _ = _hill_pair(max(0., sound_multiplier - 1.), p.trauma_half, p.trauma_hill)
    return activation * (1. - trauma) / p.trauma_time - trauma / p.repair_time


def related_code_closure():
    """Return a named hybrid closure; this is not manuscript-source validation."""
    return MissingEquationClosure(
        workloads=related_code_workloads, paracellular_leak=0.,
        description=("U_C, U_M, marginal K half-saturation (4 mM) and intact basal "
                     "paracellular leak adapted from related 19-state C model commit "
                     "64ae8344e45dcb45b8b9d2bc9fb0528cd6876aac; not specified in manuscript."),
    )


def broadband_sound_multiplier(spl_db):
    """Empirical broadband SPL-to-drive transform from the manuscript."""
    if not math.isfinite(spl_db):
        raise ValueError("SPL must be finite")
    x = (spl_db - 110.5) * math.log(10.) / 15.5
    activation = 1. / (1. + math.exp(-x)) if x >= 0 else math.exp(x) / (1. + math.exp(x))
    return 1. + .32 * activation


def observables(state, inputs: Inputs, closure: MissingEquationClosure,
                p: Parameters = Parameters()):
    y = np.asarray(state, dtype=float)
    ve, ke = endolymph(y, p)
    flow, oxygen, ac, am, vs, kc, ks, km, ep, atp, _, adapt, trauma = y
    uc, um = closure.workloads(y, ke, p)
    if not all(math.isfinite(x) and x >= 0 for x in (uc, um)):
        raise ValueError("workloads must be finite and nonnegative")
    _, availability = _hill_pair(inputs.drug, 1., p.drug_hill)
    kir = p.thermal_voltage * math.log(kc / ks)
    kir0 = p.thermal_voltage * math.log(p.k_sync_base / p.k_strial_base)
    api = p.thermal_voltage * math.log(km / ke)
    receptor_atp = p.atp_endolymph_base + p.atp_cleft_factor * max(0., atp - p.atp_endolymph_base)

    def conductance(a):
        occupancy, _ = _hill_pair(a, p.p2x_half, p.p2x_hill)
        return p.p2x_max * occupancy

    gp2x = conductance(receptor_atp)
    gp2x0 = conductance(p.atp_endolymph_base)
    excess = max(0., inputs.sound_multiplier - 1.) if p.met_conductance > 0 else 0.
    met_effective = (1. - p.adapt_rest * adapt * (1. - trauma)
                     + excess * (1. - p.adapt_evoked * adapt) + .35 * trauma)
    gs = p.strial_conductance * (1. - p.drug_conductance_loss * (1. - availability))
    strial_voltage = (p.kir_fraction * kir + api
                      + p.pump_voltage * (ac + am) / 2. - p.voltage_offset)
    circuit_current = (p.met_conductance * met_effective * max(0., ep - p.met_reversal)
                       + gp2x * (ep - p.p2x_reversal))
    intrastrial_voltage = (p.kir_fraction * kir - p.intrastrial_voltage_offset
                           - p.sync_resistance * circuit_current)
    if p.met_conductance <= 0:
        raise ValueError("manuscript flux normalization requires positive MET conductance")
    denominator = p.ep_base - p.met_reversal
    if denominator <= 0:
        raise ValueError("reference EP must exceed MET reversal")
    j_kir = (p.kir_fraction * p.k_flux * (kc - ks)
             / (p.k_sync_base - p.k_strial_base) * abs(kir / kir0))
    j_uptake = p.k_flux * availability * um
    j_apical = p.k_flux * km / p.k_marginal_base
    j_met = (p.k_flux * met_effective * max(0., ep - p.met_reversal)
             / denominator * ke / p.k_endolymph_base)
    j_p2x = (p.k_flux * (gp2x - gp2x0) / p.met_conductance
             * (ep - p.p2x_reversal) / denominator * ke / p.k_endolymph_base)
    j_para = (closure.paracellular_leak + p.trauma_k_leak * trauma) * (kc - ks)
    j_blb = p.blb_leak * (p.k_plasma - ks)
    j_peri = (p.drug_k_loss * (ke - p.k_plasma) * (1. - availability)
              + p.drug_k_relax * (ke - p.k_endolymph_base) * availability
              if inputs.furosemide_protocol else 0.)
    conductance_total = gs + p.met_conductance * met_effective + gp2x
    if conductance_total <= 0:
        raise ValueError("partition conductance must be positive")
    return {
        "endolymph_volume": ve, "endolymph_k": ke, "nkcc_availability": availability,
        "kir_voltage": kir, "apical_voltage": api, "strial_voltage": strial_voltage,
        "circuit_current": circuit_current, "intrastrial_voltage": intrastrial_voltage,
        "effective_strial_conductance": gs, "p2x_conductance": gp2x,
        "baseline_p2x_conductance": gp2x0, "effective_met": met_effective,
        "sync_workload": uc, "marginal_workload": um, "kir_flux": j_kir,
        "marginal_uptake": j_uptake, "apical_flux": j_apical, "met_flux": j_met,
        "p2x_flux": j_p2x, "paracellular_flux": j_para, "blb_flux": j_blb,
        "perilymph_flux": j_peri,
        "resistance_ratio": (p.resistance_series_fraction + (1. - p.resistance_series_fraction)
                             * (p.strial_conductance + p.met_conductance + gp2x0)
                             / conductance_total),
    }


def rhs(time, state, inputs: Inputs, closure: MissingEquationClosure,
        p: Parameters = Parameters()):
    del time
    y = np.asarray(state, dtype=float)
    o = observables(y, inputs, closure, p)
    flow, oxygen, ac, am, vs, kc, ks, km, ep, atp, _, adapt, trauma = y
    del kc, km
    avg = (ac + am) / 2.
    raw_flow = p.flow_base * (1. + p.autoregulation * (p.atp_base - avg) / p.atp_base)
    raw_flow *= 1. - p.trauma_vasoconstriction * trauma
    x = raw_flow - p.flow_min
    target_flow = p.flow_min + .5 * (x + math.hypot(x, p.flow_smoothing))
    safe_flow = max(flow, 1e-4)
    extraction = safe_flow * -math.expm1(-p.oxygen_permeability / safe_flow) / -math.expm1(-1.)
    oxidative = oxygen / (p.oxygen_km + oxygen) * (p.oxygen_km + p.oxygen_base) / p.oxygen_base
    metabolism = p.metabolism_base * oxidative * max(0.2, 1. + p.demand_gain * (p.atp_base - avg))
    dc = (1. - p.sync_pump_fraction) * ac + p.sync_pump_fraction * o["sync_workload"]
    dm = ((1. - p.marginal_pump_fraction) * am
          + p.marginal_pump_fraction * o["nkcc_availability"] * o["marginal_workload"])
    volume_rate = (p.water_permeability * 2. * p.potassium_reflection * (ks - p.k_strial_base)
                   - p.volume_restoration * (vs - p.strial_volume_base)
                   * (1. + ((vs - p.strial_volume_base) / p.volume_span) ** 2))
    kc_rate = (o["met_flux"] + o["p2x_flux"] - o["kir_flux"] - o["paracellular_flux"]) / p.sync_volume
    ks_rate = ((o["kir_flux"] + o["paracellular_flux"] - o["marginal_uptake"] + o["blb_flux"]) / vs
               - ks * volume_rate / vs)
    km_rate = (o["marginal_uptake"] - o["apical_flux"]) / p.marginal_volume
    voltage_rate = (o["effective_strial_conductance"] * (o["strial_voltage"] - ep)
                    - p.met_conductance * o["effective_met"] * max(0., ep - p.met_reversal)
                    - o["p2x_conductance"] * (ep - p.p2x_reversal)) / p.capacitance
    ve = o["endolymph_volume"]
    excess = max(0., inputs.sound_multiplier - 1.) if p.met_conductance > 0 else 0.
    atp_rate = ((p.atp_degradation * p.atp_endolymph_base * ve
                 + p.atp_release * (1. - .55 * adapt) * excess + inputs.atp_delivery) / ve
                - p.atp_degradation * atp + atp * volume_rate / ve)
    mechanical, _ = _hill_pair(excess, p.adapt_half, p.adapt_hill)
    dg = max(0., o["p2x_conductance"] - o["baseline_p2x_conductance"])
    purinergic = dg / (p.adapt_p2x_half + dg)
    adapt_rate = (p.adapt_rate * ((1. - p.adapt_p2x_fraction) * mechanical
                                  + p.adapt_p2x_fraction * purinergic) * (1. - adapt)
                  - adapt / p.adapt_recovery)
    trauma_rate = trauma_derivative(inputs.sound_multiplier, trauma, p)
    out = np.array([(target_flow - flow) / p.flow_tau,
                    p.oxygen_transport * extraction * (p.arterial_oxygen - oxygen)
                    - p.oxygen_stoichiometry * p.oxygen_scaling * metabolism,
                    (oxidative * 2. / (1. + ac / p.atp_base) - dc) / p.sync_atp_tau,
                    (oxidative * 2. / (1. + am / p.atp_base) - dm) / p.marginal_atp_tau,
                    volume_rate, kc_rate, ks_rate, km_rate, voltage_rate, atp_rate,
                    o["blb_flux"] - o["perilymph_flux"], adapt_rate, trauma_rate])
    if not np.isfinite(out).all():
        raise ValueError("nonfinite derivatives")
    return out


def simulate(times, forcing: Callable[[float], Inputs], closure: MissingEquationClosure,
             p: Parameters = Parameters(), *, initial=None, boundaries=(), method="Radau",
             rtol=1e-8, atol=1e-10):
    """Integrate across explicitly supplied forcing discontinuities."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times must be finite, strictly increasing and have at least two entries")
    boundary_array = np.asarray(boundaries, dtype=float)
    if boundary_array.ndim != 1 or not np.isfinite(boundary_array).all():
        raise ValueError("boundaries must be a finite one-dimensional sequence")
    if np.any(boundary_array <= times[0]) or np.any(boundary_array >= times[-1]):
        raise ValueError("boundaries must lie strictly inside the simulation interval")
    if len(boundary_array) > 1 and np.any(np.diff(boundary_array) <= 0):
        raise ValueError("boundaries must be strictly increasing")
    state = baseline(p) if initial is None else np.asarray(initial, dtype=float).copy()
    validate_sampled_domain(state, p)
    result = np.empty((13, len(times)))
    result[:, 0] = state
    edges = [times[0], *boundary_array, times[-1]]
    for left, right in zip(edges[:-1], edges[1:]):
        indices = np.flatnonzero((times > left) & (times <= right))
        evaluation = np.unique(np.r_[times[indices], right])
        def segment_rhs(t, y):
            evaluation_time = np.nextafter(right, left) if t == right and right != times[-1] else t
            return rhs(t, y, forcing(float(evaluation_time)), closure, p)

        sol = solve_ivp(segment_rhs,
                        (left, right), state, t_eval=evaluation,
                        method=method, rtol=rtol, atol=atol)
        if not sol.success or sol.y.shape != (13, len(evaluation)) or not np.isfinite(sol.y).all():
            raise RuntimeError(f"integration failed on [{left}, {right}]: {sol.message}")
        if len(indices):
            result[:, indices] = sol.y[:, np.searchsorted(evaluation, times[indices])]
        state = sol.y[:, -1]
    for state in result.T:
        validate_sampled_domain(state, p)
    return result
