import types

import numpy as np

from Environments.OOH.customerchoice import customerchoicemodel
from Environments.OOH.containers import Location, ParcelPoint
from Src.Utils.passenger_utility import mnl_probabilities


def euclidean(a, b):
    return float(np.hypot(a.x - b.x, a.y - b.y))


def test_customerchoice_nonprice_utilities_match_target_formula():
    model = customerchoicemodel(
        base_util=1.0,
        dist_scaler=1.0,
        euclidean=euclidean,
        dist_mat=[],
        n_cust=1,
        outside_option_util=-0.5,
        travel_time_weight=-0.01,
        walk_distance_weight=-2.0,
    )
    customer = types.SimpleNamespace(
        id_num=0,
        home=Location(0.0, 0.0, 0, 0),
        home_util=0.4,
        incentiveSensitivity=-0.2,
    )
    pp = ParcelPoint(Location(0.3, 0.4, 1, 0), 1, 1)

    home_util = model.base_util + customer.home_util + model.travel_time_weight * 300.0
    mp_util = model.mnl(customer, pp, travel_time=240.0)

    assert home_util == 1.0 + 0.4 - 0.01 * 300.0
    assert mp_util == 1.0 - 2.0 * 0.5 - 0.01 * 240.0


def test_outside_option_changes_mnl_denominator():
    outside = -0.5
    home = 1.0 + 0.4 - 0.2 * 5.0
    mp = 1.0 - 2.0 * 0.5 - 0.2 * 4.0

    with_outside = mnl_probabilities([outside, home, mp])
    without_outside = mnl_probabilities([home, mp])

    assert with_outside[1] < without_outside[0]
    assert 0.0 < with_outside[0] < 1.0


def test_negative_price_coefficient_reduces_utility_when_price_increases():
    beta_price = -0.2
    base_nonprice = 1.4

    low_price_util = base_nonprice + beta_price * 5.0
    high_price_util = base_nonprice + beta_price * 8.0

    assert high_price_util < low_price_util
