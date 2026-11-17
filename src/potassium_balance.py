"""Provisional algebraic potassium mass-balance kernel.

This module implements bookkeeping only; it does not prescribe transport or
water-flux laws and is not a complete cochlear model.
"""

from dataclasses import dataclass
import math


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class EndolymphInventory:
    """Algebraic endolymph inventory; volumes are normalized to V_E,0."""

    phi_e: float
    amount: float
    concentration: float


@dataclass(frozen=True, slots=True)
class PotassiumState:
    """Potassium state using dimensionless compartment volumes.

    ``k_*`` are concentrations in mM. ``total_amount`` is in
    mM·V_E,0, and each ``phi_*`` is the corresponding volume divided by
    V_E,0. C and M volumes are fixed. ``endolymph_volume_coupling`` is
    the fraction of S expansion taken from E; zero assumes fixed E volume,
    one is the optional closed-fluid-space closure in formulation section 2.
    """

    phi_c: float
    phi_s: float
    phi_m: float
    phi_e_reference: float
    phi_s_reference: float
    k_c: float
    k_s: float
    k_m: float
    total_amount: float
    endolymph_volume_coupling: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "phi_c",
            "phi_s",
            "phi_m",
            "phi_e_reference",
            "phi_s_reference",
            "k_c",
            "k_s",
            "k_m",
            "total_amount",
            "endolymph_volume_coupling",
        ):
            _require_finite(name, getattr(self, name))
        for name in ("phi_c", "phi_s", "phi_m", "phi_e_reference", "phi_s_reference"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("k_c", "k_s", "k_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.total_amount < 0:
            raise ValueError("total_amount must be nonnegative")
        if not 0 <= self.endolymph_volume_coupling <= 1:
            raise ValueError("endolymph_volume_coupling must be between zero and one")
        if self.endolymph.phi_e <= 0:
            raise ValueError("algebraic endolymph volume must be positive")
        if self.endolymph.amount < 0:
            raise ValueError("residual endolymph potassium amount must be nonnegative")

    @property
    def endolymph(self) -> EndolymphInventory:
        """Return the residual endolymph volume, amount, and concentration."""
        phi_e = self.phi_e_reference - self.endolymph_volume_coupling * (
            self.phi_s - self.phi_s_reference
        )
        if phi_e <= 0:
            raise ValueError("algebraic endolymph volume must be positive")
        amount = (
            self.total_amount
            - self.phi_c * self.k_c
            - self.phi_s * self.k_s
            - self.phi_m * self.k_m
        )
        return EndolymphInventory(phi_e, amount, amount / phi_e)


@dataclass(frozen=True, slots=True)
class PotassiumFluxes:
    """Prescribed amount fluxes in mM·V_E,0/min.

    Positive signs follow the named directions: CS (C→S), SM (S→M),
    ME (M→E), EC (E→C), B (blood→modeled cochlea), and P
    (modeled cochlea→external sink). ``j_b_to_s`` is the portion of the
    blood-boundary flux entering S; ``j_b - j_b_to_s`` enters endolymph.
    Values may be signed to represent reversal of the named direction.
    """

    j_cs: float
    j_sm: float
    j_me: float
    j_ec: float
    j_b: float
    j_b_to_s: float
    j_p: float

    def __post_init__(self) -> None:
        for name in (
            "j_cs",
            "j_sm",
            "j_me",
            "j_ec",
            "j_b",
            "j_b_to_s",
            "j_p",
        ):
            _require_finite(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class PotassiumRates:
    """Algebraic rates; concentration rates are in mM/min."""

    phi_s_rate: float
    phi_e_rate: float
    k_c_rate: float
    k_s_rate: float
    k_m_rate: float
    total_amount_rate: float
    endolymph_amount_rate: float
    endolymph_concentration_rate: float


def evaluate_potassium_balance(
    state: PotassiumState,
    fluxes: PotassiumFluxes,
    phi_s_rate: float,
) -> PotassiumRates:
    """Evaluate the compartment concentration and total-inventory rates.

    ``phi_s_rate`` is dimensionless volume fraction per minute. All fluxes
    are prescribed; no transport, boundary, or swelling parameters are
    inferred here. The endolymph amount rate is the residual implied by
    total-pool accounting, equivalently Eq. (6) of formulation section 2.
    """
    _require_finite("phi_s_rate", phi_s_rate)
    state.endolymph  # Revalidate the algebraic closure at the calculation boundary.

    k_c_rate = (-fluxes.j_cs + fluxes.j_ec) / state.phi_c
    k_s_rate = (
        fluxes.j_cs
        - fluxes.j_sm
        + fluxes.j_b_to_s
        - state.k_s * phi_s_rate
    ) / state.phi_s
    k_m_rate = (fluxes.j_sm - fluxes.j_me) / state.phi_m
    total_amount_rate = fluxes.j_b - fluxes.j_p
    endolymph_amount_rate = (
        fluxes.j_me
        - fluxes.j_ec
        + (fluxes.j_b - fluxes.j_b_to_s)
        - fluxes.j_p
    )
    phi_e_rate = -state.endolymph_volume_coupling * phi_s_rate
    endolymph_concentration_rate = (
        endolymph_amount_rate - state.endolymph.concentration * phi_e_rate
    ) / state.endolymph.phi_e

    rates = PotassiumRates(
        phi_s_rate=phi_s_rate,
        phi_e_rate=phi_e_rate,
        k_c_rate=k_c_rate,
        k_s_rate=k_s_rate,
        k_m_rate=k_m_rate,
        total_amount_rate=total_amount_rate,
        endolymph_amount_rate=endolymph_amount_rate,
        endolymph_concentration_rate=endolymph_concentration_rate,
    )
    for name in rates.__dataclass_fields__:
        _require_finite(name, getattr(rates, name))
    return rates
