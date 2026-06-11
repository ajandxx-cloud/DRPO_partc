from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class UtilityConfig:
    base_util: float
    home_util: float
    outside_option_util: Optional[float]
    beta_price: float
    beta_walk: float
    beta_time: float
    distance_unit: str = "km"
    travel_time_unit: str = "second"


def _coef(value: Optional[float]) -> float:
    return 0.0 if value is None else float(value)


def _value(value: Optional[float]) -> float:
    return 0.0 if value is None else float(value)


def utility_outside(outside_option_util: float) -> float:
    return float(outside_option_util)


def utility_with_price(nonprice_utility: float, beta_price: float, price: float) -> float:
    return float(nonprice_utility) + float(beta_price) * float(price)


def utility_home_nonprice(
    base_util: float,
    home_util: float,
    beta_time: Optional[float],
    predicted_travel_time_home: Optional[float],
) -> float:
    return (
        float(base_util)
        + float(home_util)
        + _coef(beta_time) * _value(predicted_travel_time_home)
    )


def utility_home(
    base_util: float,
    home_util: float,
    beta_time: Optional[float],
    beta_price: float,
    predicted_travel_time_home: Optional[float],
    price_home: float,
) -> float:
    return utility_with_price(
        utility_home_nonprice(
            base_util,
            home_util,
            beta_time,
            predicted_travel_time_home,
        ),
        beta_price,
        price_home,
    )


def utility_meeting_point_nonprice(
    base_util: float,
    beta_walk: Optional[float],
    beta_time: Optional[float],
    walking_distance: Optional[float],
    predicted_travel_time_mp: Optional[float],
) -> float:
    return (
        float(base_util)
        + _coef(beta_walk) * _value(walking_distance)
        + _coef(beta_time) * _value(predicted_travel_time_mp)
    )


def utility_meeting_point(
    base_util: float,
    beta_walk: Optional[float],
    beta_time: Optional[float],
    beta_price: float,
    walking_distance: Optional[float],
    predicted_travel_time_mp: Optional[float],
    price_mp: float,
) -> float:
    return utility_with_price(
        utility_meeting_point_nonprice(
            base_util,
            beta_walk,
            beta_time,
            walking_distance,
            predicted_travel_time_mp,
        ),
        beta_price,
        price_mp,
    )


def mnl_probabilities(utilities: Sequence[float]) -> np.ndarray:
    values = np.asarray(utilities, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("MNL utilities must contain at least one option.")
    if not np.isfinite(values).all():
        raise ValueError("MNL utilities must be finite.")

    vmax = float(np.max(values))
    exp_values = np.exp(values - vmax)
    denom = float(np.sum(exp_values))
    if denom <= 0.0 or not np.isfinite(denom):
        raise ValueError("MNL denominator is not finite and positive.")
    return exp_values / denom


def validate_utility_config(config: UtilityConfig, strict_units: bool = False) -> None:
    fields = [
        config.base_util,
        config.home_util,
        config.beta_price,
        config.beta_walk,
        config.beta_time,
    ]
    if config.outside_option_util is not None:
        fields.append(config.outside_option_util)
    if not np.isfinite(np.asarray(fields, dtype=np.float64)).all():
        raise ValueError("Utility parameters must be finite.")
    if config.beta_price >= 0:
        raise ValueError("beta_price must be negative.")
    if config.beta_walk >= 0:
        raise ValueError("beta_walk must be negative.")
    if config.beta_time >= 0:
        raise ValueError("beta_time must be negative.")
    if strict_units:
        if config.distance_unit != "km":
            raise ValueError("Final Yanjiao distance_unit must be 'km'.")
        if config.travel_time_unit != "second":
            raise ValueError("Final Yanjiao travel_time_unit must be 'second'.")


def validate_choice_utility_policy(
    choice_util_matrix,
    final_yanjiao_mode: bool = False,
    allow_derived_choice_utility: bool = False,
) -> None:
    if (
        final_yanjiao_mode
        and choice_util_matrix is not None
        and not allow_derived_choice_utility
    ):
        raise ValueError(
            "Final Yanjiao experiments must not use opaque choice_util_matrix. "
            "Use walking_distance_matrix, or set allow_derived_choice_utility "
            "only when metadata proves the matrix is beta_walk * distance."
        )
