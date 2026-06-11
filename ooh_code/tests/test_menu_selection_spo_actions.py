import types

import numpy as np

from Environments.OOH.containers import Location, ParcelPoint
from Src.Algorithms.DSPO_MenuSelection_SPO import DSPO_MenuSelection_SPO


def make_customer():
    return types.SimpleNamespace(
        id_num=1,
        home=Location(0.0, 0.0, 1, 0),
        home_util=0.2,
        incentiveSensitivity=-0.2,
        service_time=1.0,
    )


def make_pps():
    return [
        ParcelPoint(Location(1.0, 0.0, 10, 0), 2, 10),
        ParcelPoint(Location(2.0, 0.0, 11, 0), 2, 11),
        ParcelPoint(Location(3.0, 0.0, 12, 0), 2, 12),
    ]


def make_algo(selected_indices, customerchoice=None, travel_times=None):
    algo = object.__new__(DSPO_MenuSelection_SPO)
    algo.load_data = False
    algo.initial_phase = False
    algo.use_travel_time_prediction = travel_times is not None
    algo.use_menu_selection = True
    algo.menu_size = 1
    algo.pricing_model_ready = False
    algo.cost_multiplier = 1.0
    algo.l0_home = 1.0
    algo.l_mp = 0.5
    algo.revenue = 0.0
    algo.base_util = 0.0
    algo.min_p = -100.0
    algo.max_p = 100.0
    algo.max_discount_per_customer = 100.0
    algo.config = types.SimpleNamespace(
        outside_option_util=None,
        env=types.SimpleNamespace(
            customerchoice=customerchoice,
            set_travel_times=lambda value: None,
        ),
    )
    algo.get_feature_rep_infer = lambda fleet: np.zeros((1, 1, 1), dtype=np.float32)
    algo._predict_marginal_costs = lambda feat, home, pps: np.array([3.0, 1.0, 2.0, 4.0])
    algo._enumerate_best_menu_spoa = lambda *args, **kwargs: (list(selected_indices), 0.0, None)
    algo.cheapestInsertionCosts = lambda loc, fleet: float(loc.id_num)
    algo.mnl = lambda customer, pp: 0.0
    if travel_times is not None:
        algo.get_travel_time_prediction = lambda feat, home, pps: travel_times
    return algo


def test_selected_first_pp_price_does_not_overwrite_home_slot():
    algo = make_algo(selected_indices=[0])
    pps = make_pps()
    state = [make_customer(), {"fleet": []}, {"parcelpoints": pps}, 0]

    prices = algo.get_action_pricing(state, training=False)

    assert prices.shape == (4,)
    assert prices[0] != 0.0
    assert prices[1] != 0.0
    assert prices[2] == 0.0
    assert prices[3] == 0.0


def test_selected_menu_travel_time_uses_original_pp_index():
    seen_travel_times = []

    class ChoiceModel:
        travel_time_weight = -0.01

        def mnl(self, customer, pp, travel_time=None):
            seen_travel_times.append(travel_time)
            return 0.0

    algo = make_algo(
        selected_indices=[2],
        customerchoice=ChoiceModel(),
        travel_times={"home": 5.0, "ooh": [10.0, 20.0, 30.0]},
    )
    pps = make_pps()
    state = [make_customer(), {"fleet": []}, {"parcelpoints": pps}, 0]

    algo.get_action_pricing(state, training=False)

    assert seen_travel_times == [30.0]


def test_reference_price_probs_include_outside_option_denominator():
    algo = object.__new__(DSPO_MenuSelection_SPO)
    algo.config = types.SimpleNamespace(outside_option_util=0.0)

    probs = algo._compute_ref_price_probs(home_util=0.0, pp_utils=[0.0, 0.0])

    assert np.allclose(probs, np.array([0.25, 0.25, 0.25], dtype=np.float32))
    assert np.isclose(float(probs.sum()), 0.75)


def test_done_update_skips_hgs_for_depot_only_episode():
    algo = object.__new__(DSPO_MenuSelection_SPO)
    algo.initial_phase = False
    algo.load_data = True
    algo.n_layers = 1
    algo.grid_dim = 1
    algo.features = np.ones((1, 1))
    algo.cap_features = np.ones((1, 1))
    algo.reopt_HGS_final = lambda data: (_ for _ in ()).throw(AssertionError("HGS should be skipped"))

    cost = algo.update({"id": 0, "time": 0}, state=None, done=True)

    assert cost == 0.0
    assert algo.features.shape == (0, 1)
    assert algo.cap_features.shape == (0, 1)
