import types

import numpy as np
import torch
from torch import nn

from Src.Algorithms.DSPO_plus_SPO import DSPO_plus_SPO
from Src.Algorithms.DRPO import DRPO
from Environments.OOH.containers import Fleet, Location, ParcelPoint, Vehicle


def make_algo_stub():
    stub = object.__new__(DSPO_plus_SPO)
    stub.base_util = -1.0
    stub.revenue = 50.0
    stub.min_p = -5.0
    stub.max_p = 3.5
    stub.config = types.SimpleNamespace(
        outside_option_util=-1.0,
        home_failure=0.1,
        failure_cost=20.0,
        env=types.SimpleNamespace(depot=Location(0.0, 0.0, 0, 0)),
    )
    stub._safe_exp = DSPO_plus_SPO._safe_exp
    stub.load_data = False
    stub.cost_multiplier = 1.0
    stub.l0_home = 2.5
    stub.l_mp = 0.75
    return stub


def test_drpo_is_public_alias_for_spo_implementation():
    assert issubclass(DRPO, DSPO_plus_SPO)


def test_pricing_oracle_lifted_shape_and_outside_option():
    algo = make_algo_stub()
    costs = np.asarray([12.0, 8.0, 9.5], dtype=np.float32)
    customer_info = {
        "base_util": -1.0,
        "home_util": 1.4,
        "incentive_sensitivity": -0.25,
        "revenue": 50.0,
    }
    pps_info = [{"util": -1.2}, {"util": -1.6}]

    out = algo._pricing_oracle_lifted_np(costs, customer_info, pps_info)

    assert out is not None
    assert out["prices"].shape == (3,)
    assert out["probs"].shape == (3,)
    assert out["lifted"].shape == (4,)
    assert np.isfinite(out["lifted"]).all()
    assert np.isfinite(out["r0"])
    # Outside option is included in the denominator, so internal probabilities
    # need not sum to one and no separate outside component appears in w.
    assert 0.0 < float(out["probs"].sum()) < 1.0
    assert np.isclose(out["lifted"][-1], -out["r0"])


def test_lifted_spo_loss_backpropagates_to_cost_predictions():
    algo = make_algo_stub()
    customer_info = {
        "base_util": -1.0,
        "home_util": 1.4,
        "incentive_sensitivity": -0.25,
        "revenue": 50.0,
    }
    pps_info = [{"util": -1.2}, {"util": -1.6}]
    c_true = torch.tensor([12.0, 8.0, 9.5], dtype=torch.float32)
    c_hat = torch.tensor([11.5, 8.7, 10.2], dtype=torch.float32, requires_grad=True)

    true_oracle = algo._pricing_oracle_lifted_np(c_true.detach().numpy(), customer_info, pps_info)
    pseudo_oracle = algo._pricing_oracle_lifted_np((2.0 * c_hat - c_true).detach().numpy(), customer_info, pps_info)
    assert true_oracle is not None
    assert pseudo_oracle is not None

    y_true = torch.cat([c_true, torch.ones(1)])
    y_hat = torch.cat([c_hat, torch.ones(1)])
    w_true = torch.tensor(true_oracle["lifted"], dtype=torch.float32)
    w_pseudo = torch.tensor(pseudo_oracle["lifted"], dtype=torch.float32)
    loss = (torch.dot(y_true, w_pseudo - w_true) + 2.0 * torch.dot(y_hat, w_true - w_pseudo)) / 4.0

    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert c_hat.grad is not None
    assert torch.isfinite(c_hat.grad).all()
    assert torch.count_nonzero(c_hat.grad).item() > 0


def test_huber_keeps_full_weight_when_spo_batch_is_invalid():
    class TinyPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))

        def forward(self, feat, cap_feat):
            return feat @ self.weight

    algo = object.__new__(DSPO_plus_SPO)
    algo.supervised_ml = TinyPredictor()
    algo.criterion = nn.MSELoss()
    algo.optimizer = torch.optim.SGD(algo.supervised_ml.parameters(), lr=0.1)
    algo.spo_training_data = {0: {"invalid": True}}
    algo.spo_huber_loss_history = []
    algo._calculate_spo_loss_for_batch = lambda *args, **kwargs: None

    feat = torch.tensor([[1.0]], dtype=torch.float32)
    cap_feat = torch.tensor([0.0], dtype=torch.float32)
    target = torch.tensor([[0.0]], dtype=torch.float32)

    loss = DSPO_plus_SPO.self_supervised_update(
        algo, feat, cap_feat, target, spo_weight=0.7, record_loss=True
    )

    assert np.isclose(loss, 1.0)
    assert torch.isclose(
        algo.supervised_ml.weight.detach(),
        torch.tensor([[0.8]], dtype=torch.float32),
    ).all()


def test_terminal_replacement_labels_match_by_arrival_time_not_location_id():
    algo = make_algo_stub()
    # The served final location has the meeting-point id 200, while the
    # passenger's home id is 10. The label construction must still find it by
    # arrival time and produce one cost per current menu option.
    served_mp = Location(6.0, 0.0, 200, 7)
    other = Location(12.0, 0.0, 99, 3)
    fleet = Fleet([Vehicle([served_mp, other], 4, 0)])
    customer = types.SimpleNamespace(
        home=Location(0.0, 4.0, 10, 7),
        service_time=1.0,
    )
    near_pp = ParcelPoint(Location(5.5, 0.0, 200, 0), 3, 200)
    far_pp = ParcelPoint(Location(0.0, 20.0, 201, 0), 3, 201)

    labels = algo._calculate_global_costs_for_all_options_spo(
        customer, [near_pp, far_pp], fleet, global_cost_base=0.0, arrival_time=7
    )

    assert labels is not None
    assert len(labels) == 3
    assert all(np.isfinite(labels))
    assert labels[1] < labels[2]
