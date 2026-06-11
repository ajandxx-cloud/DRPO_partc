import numpy as np
import pytest

from Src.Utils.passenger_utility import (
    UtilityConfig,
    mnl_probabilities,
    utility_home,
    utility_meeting_point,
    utility_outside,
    validate_choice_utility_policy,
    validate_utility_config,
)


def test_home_utility_exact_formula():
    assert utility_home(
        base_util=1.2,
        home_util=0.5,
        beta_time=-0.01,
        beta_price=-0.2,
        predicted_travel_time_home=600.0,
        price_home=8.0,
    ) == pytest.approx(1.2 + 0.5 - 0.01 * 600.0 - 0.2 * 8.0)


def test_meeting_point_utility_exact_formula():
    assert utility_meeting_point(
        base_util=1.2,
        beta_walk=-1.5,
        beta_time=-0.01,
        beta_price=-0.2,
        walking_distance=0.4,
        predicted_travel_time_mp=540.0,
        price_mp=6.0,
    ) == pytest.approx(1.2 - 1.5 * 0.4 - 0.01 * 540.0 - 0.2 * 6.0)


def test_outside_option_probability_enters_denominator():
    utilities = [
        utility_outside(-0.3),
        utility_home(1.0, 0.4, -0.01, -0.2, 300.0, 5.0),
        utility_meeting_point(1.0, -1.0, -0.01, -0.2, 0.2, 260.0, 4.0),
    ]

    probs = mnl_probabilities(utilities)

    assert probs.shape == (3,)
    assert np.isclose(probs.sum(), 1.0)
    assert 0.0 < probs[0] < 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beta_price": 0.0},
        {"beta_walk": 0.0},
        {"beta_time": 0.0},
    ],
)
def test_utility_coefficients_must_be_negative(kwargs):
    params = dict(
        base_util=1.0,
        home_util=0.2,
        outside_option_util=-1.0,
        beta_price=-0.2,
        beta_walk=-1.0,
        beta_time=-0.01,
    )
    params.update(kwargs)

    with pytest.raises(ValueError):
        validate_utility_config(UtilityConfig(**params))


def test_final_yanjiao_unit_validation():
    cfg = UtilityConfig(
        base_util=1.0,
        home_util=0.2,
        outside_option_util=-1.0,
        beta_price=-0.2,
        beta_walk=-1.0,
        beta_time=-0.01,
        distance_unit="meter",
    )

    with pytest.raises(ValueError):
        validate_utility_config(cfg, strict_units=True)


def test_final_yanjiao_rejects_opaque_choice_utility():
    matrix = np.zeros((2, 2), dtype=float)

    with pytest.raises(ValueError):
        validate_choice_utility_policy(matrix, final_yanjiao_mode=True)

    validate_choice_utility_policy(
        matrix,
        final_yanjiao_mode=True,
        allow_derived_choice_utility=True,
    )
