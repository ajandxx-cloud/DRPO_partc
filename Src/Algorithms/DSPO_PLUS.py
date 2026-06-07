#!/usr/bin/env python3
"""
DSPO_PLUS: original DSPO + option-level SPO+ training, without menu selection.

This implementation is deliberately aligned with the current DSPO.py in this
repository:
    - same candidate parcel-point set;
    - same DSPO cost construction;
    - same Lambert-W pricing structure;
    - same MNL utility functions;
    - no menu screening / no top-L filtering.

The only methodological change is the learning objective after the original
initial DSPO phase: the original Huber prediction loss is blended with an
option-level SPO+ proxy.

Two price-bound modes are supported for the 2x2 experiment:
    clip: reproduce the original DSPO final price clipping [min_p, max_p];
    wide: use a large finite symmetric range [-wide_price_bound, wide_price_bound].
"""

from typing import Dict, List, Optional
from math import e

import numpy as np
import numpy.ma as ma
import torch
from torch import float32
from scipy.special import lambertw

from Src.Algorithms.DSPO import DSPO


class DSPO_PLUS(DSPO):
    """DSPO with a decision-focused SPO+ training objective."""

    PRICE_MODE = "clip"  # subclasses can override with "wide"

    def __init__(self, config):
        super(DSPO_PLUS, self).__init__(config)

        self.price_mode = getattr(config, "price_bound_mode", self.PRICE_MODE)
        self.wide_price_bound = float(getattr(config, "wide_price_bound", 1000.0))
        if self.wide_price_bound <= 0:
            raise ValueError("wide_price_bound must be positive")

        self.spo_warmup_episodes = int(getattr(config, "spo_warmup_episodes", 5))
        self.spo_rampup_episodes = max(1, int(getattr(config, "spo_rampup_episodes", 10)))
        self.max_spo_loss_weight = float(getattr(config, "spo_loss_weight", 0.3))
        self.spo_loss_scale = float(getattr(config, "spo_loss_scale", 1.0))
        self.spo_batch_size = int(getattr(config, "spo_batch_size", min(64, config.batch_size)))
        self.spo_max_replay_size = int(getattr(config, "spo_replay_size", config.buffer_size))
        self.spo_grad_clip_norm = float(getattr(config, "spo_grad_clip_norm", 5.0))

        self.spo_replay_data: List[Dict] = []
        self.spo_episode_count = 0
        self._spo_data_logged = False
        self._spo_weight_logged = False
        self._last_spo_debug: Dict[str, float] = {}

        print(
            "DSPO_PLUS initialized: "
            f"price_mode={self.price_mode}, "
            f"wide_price_bound={self.wide_price_bound}, "
            f"spo_warmup={self.spo_warmup_episodes}, "
            f"spo_rampup={self.spo_rampup_episodes}, "
            f"max_spo_weight={self.max_spo_loss_weight}, "
            f"external_option={self.external_option}, "
            "menu_selection=False"
        )

    @staticmethod
    def _safe_exp(x: float) -> float:
        return float(np.exp(np.clip(float(x), -700.0, 700.0)))

    def _is_wide_mode(self) -> bool:
        return str(self.price_mode).lower() == "wide"

    def _clip_price_vector(self, prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=np.float64)
        if self._is_wide_mode():
            return np.clip(prices, -self.wide_price_bound, self.wide_price_bound)
        return np.clip(prices, self.min_p, self.max_p)

    # ------------------------------------------------------------------
    # DSPO-aligned pricing.
    # ------------------------------------------------------------------
    def get_action_pricing(self, state, training):
        """Same DSPO pricing action space; stores SPO+ records during training."""
        customer = state[0]
        fleet = state[1]

        if self.load_data:
            mask = ma.masked_array(
                state[2]["parcelpoints"],
                mask=self.adjacency[customer.id_num],
            )
            pps = mask[mask.mask].data
        else:
            pps = state[2]["parcelpoints"]

        if self.initial_phase:
            return np.around(np.zeros(len(pps) + 1), decimals=2)

        cur_feat = self.get_feature_rep_infer(fleet["fleet"])
        costs = self.get_prediction(cur_feat, customer.home, pps)

        theta = self.init_theta - (state[3] * self.cool_theta)
        mltplr = self.cost_multiplier

        home_cheapest = self.cheapestInsertionCosts(customer.home, fleet)
        homeCosts = (
            customer.service_time * mltplr
            + ((1.0 - theta) * home_cheapest + theta * (costs[1] - costs[0]))
        )
        sum_mnl = self._safe_exp(
            self.base_util
            + customer.home_util
            + customer.incentiveSensitivity * (homeCosts - self.revenue)
        )

        pp_costs = np.full(len(pps), 1000000000.0, dtype=np.float64)
        pp_cheapest = np.full(len(pps), np.inf, dtype=np.float64)
        pp_utils = np.zeros(len(pps), dtype=np.float64)
        valid_pp_mask = np.zeros(len(pps), dtype=bool)

        for idx, pp in enumerate(pps):
            pp_utils[idx] = float(self.mnl(customer, pp))
            if pp.remainingCapacity > 0:
                valid_pp_mask[idx] = True
                pp_cheapest[idx] = self.cheapestInsertionCosts(pp.location, fleet)
                pp_costs[idx] = mltplr * (
                    (1.0 - theta) * pp_cheapest[idx]
                    + theta * (costs[idx + 2] - costs[0])
                )
                sum_mnl += self._safe_exp(
                    pp_utils[idx] + customer.incentiveSensitivity * (pp_costs[idx] - self.revenue)
                )

        lambertw0 = (
            lambertw(self.adjust_lambert_sum_for_external_option(sum_mnl) / e).real + 1.0
        ) / customer.incentiveSensitivity

        a_hat = np.zeros(len(pps) + 1, dtype=np.float64)
        a_hat[0] = homeCosts - self.revenue - lambertw0
        for idx, pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                a_hat[idx + 1] = pp_costs[idx] - self.revenue - lambertw0

        a_hat = self._clip_price_vector(a_hat)

        if training:
            self._store_spo_decision_record(
                cur_feat=cur_feat,
                customer=customer,
                pps=pps,
                theta=theta,
                home_cheapest=home_cheapest,
                pp_cheapest=pp_cheapest,
                valid_pp_mask=valid_pp_mask,
                pp_utils=pp_utils,
            )

        return np.around(a_hat, decimals=2)

    # ------------------------------------------------------------------
    # SPO+ replay data.
    # ------------------------------------------------------------------
    def _store_spo_decision_record(
        self,
        cur_feat: torch.Tensor,
        customer,
        pps,
        theta: float,
        home_cheapest: float,
        pp_cheapest: np.ndarray,
        valid_pp_mask: np.ndarray,
        pp_utils: np.ndarray,
    ) -> None:
        """Store one DSPO decision instance for option-level SPO+ training."""
        mltplr = self.cost_multiplier
        true_costs_full = np.full(len(pps) + 1, np.inf, dtype=np.float32)
        true_costs_full[0] = float(customer.service_time * mltplr + home_cheapest)

        for idx in range(len(pps)):
            if valid_pp_mask[idx]:
                true_costs_full[idx + 1] = float(mltplr * pp_cheapest[idx])

        record = {
            "cur_feat": cur_feat.squeeze(0).detach().cpu().numpy().astype(np.float32),
            "home_id_num": int(customer.home.id_num),
            "home_time": float(customer.home.time),
            "home_service_time": float(customer.service_time),
            "home_util": float(customer.home_util),
            "incentive_sensitivity": float(customer.incentiveSensitivity),
            "theta": float(theta),
            "home_cheapest": float(home_cheapest),
            "pp_cheapest": pp_cheapest.astype(np.float32),
            "valid_pp_mask": valid_pp_mask.astype(bool),
            "pp_utils": pp_utils.astype(np.float32),
            "pp_loc_id_nums": [int(pp.location.id_num) for pp in pps],
            "pp_remaining_caps": [float(pp.remainingCapacity) for pp in pps],
            "true_costs_full": true_costs_full,
        }

        self.spo_replay_data.append(record)
        if len(self.spo_replay_data) > self.spo_max_replay_size:
            overflow = len(self.spo_replay_data) - self.spo_max_replay_size
            del self.spo_replay_data[:overflow]

        if not self._spo_data_logged:
            print(f"[DSPO_PLUS] collected first SPO+ decision record, dim={len(true_costs_full)}")
            self._spo_data_logged = True

    def _recompute_dspo_cost_vector_tensor(self, record: Dict) -> torch.Tensor:
        """Recompute the DSPO option-level cost vector with gradients."""
        device = self.device
        cur_feat = torch.tensor(record["cur_feat"], dtype=float32, device=device)
        if cur_feat.dim() == 4:
            cur_feat = cur_feat.squeeze(0)
        if cur_feat.dim() != 3:
            raise ValueError(f"Expected cur_feat to be 3D, got {cur_feat.shape}")

        time_int = min(int(record["home_time"] / self.interval), self.n_layers - 1)

        option_feats: List[torch.Tensor] = []
        option_caps: List[torch.Tensor] = []

        # Base state, matching DSPO.get_prediction index 0.
        option_feats.append(cur_feat.clone())
        option_caps.append(torch.tensor(1000000.0, dtype=float32, device=device))

        # Home option, matching DSPO.get_prediction index 1.
        home_feat = cur_feat.clone()
        home_cell = self.customer_cell[record["home_id_num"]]
        home_feat[time_int][home_cell[0]][home_cell[1]] += 1
        option_feats.append(home_feat)
        option_caps.append(torch.tensor(1000000.0, dtype=float32, device=device))

        # PP options, matching DSPO.get_prediction index 2+idx.
        for loc_id, rem_cap in zip(record["pp_loc_id_nums"], record["pp_remaining_caps"]):
            pp_feat = cur_feat.clone()
            pp_cell = self.customer_cell[loc_id]
            pp_feat[time_int][pp_cell[0]][pp_cell[1]] += 1
            option_feats.append(pp_feat)
            option_caps.append(torch.tensor(float(rem_cap) - 1.0, dtype=float32, device=device))

        preds: List[torch.Tensor] = []
        for feat, cap in zip(option_feats, option_caps):
            pred = self.supervised_ml(feat.unsqueeze(0).to(device), cap.unsqueeze(0).to(device)).squeeze()
            preds.append(pred)

        pred_vec = torch.stack(preds).reshape(-1)
        base_pred = pred_vec[0]

        theta = torch.tensor(record["theta"], dtype=float32, device=device)
        mltplr = torch.tensor(self.cost_multiplier, dtype=float32, device=device)
        home_service = torch.tensor(
            record["home_service_time"] * self.cost_multiplier,
            dtype=float32,
            device=device,
        )
        home_cheapest = torch.tensor(record["home_cheapest"], dtype=float32, device=device)

        home_cost = home_service + (1.0 - theta) * home_cheapest + theta * (pred_vec[1] - base_pred)

        pp_cheapest = torch.tensor(record["pp_cheapest"], dtype=float32, device=device)
        pp_costs = mltplr * ((1.0 - theta) * pp_cheapest + theta * (pred_vec[2:] - base_pred))

        return torch.cat([home_cost.reshape(1), pp_costs.reshape(-1)])

    # ------------------------------------------------------------------
    # SPO+ oracle and loss.
    # ------------------------------------------------------------------
    def _oracle_choice_probs(
        self,
        costs_np: np.ndarray,
        home_util: float,
        pp_utils: List[float],
        incentive_sensitivity: float,
    ) -> Optional[np.ndarray]:
        """DSPO pricing oracle followed by internal MNL probabilities.

        If the external option is enabled, the returned probabilities are only
        for internal options and therefore may sum to less than one.
        """
        costs_np = np.asarray(costs_np, dtype=np.float64).reshape(-1)
        K = int(costs_np.shape[0])
        if K <= 0 or len(pp_utils) != K - 1:
            return None

        s = float(incentive_sensitivity)
        if not np.isfinite(s) or abs(s) < 1e-8:
            return None

        terms = [self.base_util + float(home_util) + s * (costs_np[0] - self.revenue)]
        for j in range(K - 1):
            terms.append(float(pp_utils[j]) + s * (costs_np[j + 1] - self.revenue))

        terms = np.asarray(terms, dtype=np.float64)
        tmax = float(np.max(terms))
        sum_mnl = self._safe_exp(tmax) * float(np.sum(np.exp(terms - tmax)))
        if not np.isfinite(sum_mnl) or sum_mnl <= 0:
            return None

        lambertw0 = (lambertw(self.adjust_lambert_sum_for_external_option(sum_mnl) / np.e).real + 1.0) / s
        if not np.isfinite(lambertw0):
            return None

        prices = np.zeros(K, dtype=np.float64)
        prices[0] = costs_np[0] - self.revenue - lambertw0
        for j in range(K - 1):
            prices[j + 1] = costs_np[j + 1] - self.revenue - lambertw0
        prices = self._clip_price_vector(prices)

        v_terms = [self.base_util + float(home_util) + s * prices[0]]
        for j in range(K - 1):
            v_terms.append(float(pp_utils[j]) + s * prices[j + 1])

        v_terms = np.asarray(v_terms, dtype=np.float64)
        if self.external_option:
            vmax = float(max(np.max(v_terms), self.get_external_utility()))
            exp_terms = np.exp(v_terms - vmax)
            denom = float(np.sum(exp_terms) + np.exp(self.get_external_utility() - vmax))
        else:
            vmax = float(np.max(v_terms))
            exp_terms = np.exp(v_terms - vmax)
            denom = float(np.sum(exp_terms))
        if not np.isfinite(denom) or denom <= 0:
            return None

        probs = exp_terms / denom
        return probs[:K].astype(np.float32)

    def _calculate_spo_loss_from_replay(self) -> Optional[torch.Tensor]:
        """Calculate mini-batch SPO+ proxy loss from stored DSPO decisions."""
        if len(self.spo_replay_data) == 0:
            return None

        sample_size = min(int(self.spo_batch_size), len(self.spo_replay_data))
        sample_indices = np.random.choice(len(self.spo_replay_data), sample_size, replace=False)

        losses: List[torch.Tensor] = []
        valid_samples = 0

        for replay_idx in sample_indices:
            record = self.spo_replay_data[int(replay_idx)]

            try:
                c_hat_full = self._recompute_dspo_cost_vector_tensor(record)
            except Exception as exc:
                print(f"[DSPO_PLUS] failed to recompute cost vector: {exc}")
                continue

            true_full = np.asarray(record["true_costs_full"], dtype=np.float32)
            valid_pp_mask = np.asarray(record["valid_pp_mask"], dtype=bool)
            valid_indices = [0] + [idx + 1 for idx, valid in enumerate(valid_pp_mask) if valid]
            if len(valid_indices) <= 1:
                continue

            valid_indices_t = torch.tensor(valid_indices, dtype=torch.long, device=c_hat_full.device)
            c_hat = c_hat_full.index_select(0, valid_indices_t)
            c_true_np = true_full[valid_indices]

            if not np.all(np.isfinite(c_true_np)) or not torch.isfinite(c_hat).all():
                continue

            valid_pp_utils = [float(record["pp_utils"][idx]) for idx, valid in enumerate(valid_pp_mask) if valid]
            c_hat_np = c_hat.detach().cpu().numpy().astype(np.float64)
            pseudo_np = 2.0 * c_hat_np - c_true_np.astype(np.float64)

            w_true_np = self._oracle_choice_probs(
                c_true_np,
                record["home_util"],
                valid_pp_utils,
                record["incentive_sensitivity"],
            )
            if w_true_np is None:
                continue

            w_surr_np = self._oracle_choice_probs(
                pseudo_np,
                record["home_util"],
                valid_pp_utils,
                record["incentive_sensitivity"],
            )
            if w_surr_np is None:
                continue

            if len(w_true_np) != c_hat.numel() or len(w_surr_np) != c_hat.numel():
                continue

            w_true = torch.tensor(w_true_np, dtype=float32, device=c_hat.device)
            w_surr = torch.tensor(w_surr_np, dtype=float32, device=c_hat.device)
            c_true = torch.tensor(c_true_np, dtype=float32, device=c_hat.device)

            # SPO+ surrogate with oracle outputs treated as stop-gradient:
            # c_true^T w*(2 c_hat - c_true)
            # + 2 c_hat^T [w*(c_true) - w*(2 c_hat - c_true)]
            spo_loss_i = (c_true * w_surr).sum() + 2.0 * torch.dot(c_hat, (w_true - w_surr))
            spo_loss_i = spo_loss_i / float(c_hat.numel())

            if torch.isfinite(spo_loss_i) and spo_loss_i.requires_grad:
                losses.append(spo_loss_i)
                valid_samples += 1

        if len(losses) == 0:
            return None

        self._last_spo_debug = {
            "valid_spo_samples": float(valid_samples),
            "sample_size": float(sample_size),
        }
        return torch.stack(losses).mean() * self.spo_loss_scale

    # ------------------------------------------------------------------
    # Training hooks.
    # ------------------------------------------------------------------
    def update(self, data, state, done=False):
        was_initial_phase = self.initial_phase if done else None
        cost = super(DSPO_PLUS, self).update(data, state, done)
        if done and not was_initial_phase:
            self.spo_episode_count += 1
        return cost

    def optimize(self):
        feat, cap_feat, target = self.memory.sample(batch_size=self.config.batch_size)
        spo_weight = self._get_current_spo_weight()

        if (not self._spo_weight_logged) and spo_weight > 0:
            print(
                f"[DSPO_PLUS] SPO+ enabled: weight={spo_weight:.4f}, "
                f"episode={self.spo_episode_count}, replay={len(self.spo_replay_data)}"
            )
            self._spo_weight_logged = True

        loss = self.self_supervised_update(feat, cap_feat, target, spo_weight=spo_weight)
        dbg = self._last_spo_debug
        if dbg:
            print(
                f"DSPO_PLUS loss={loss:.4f}, spo_weight={spo_weight:.3f}, "
                f"valid_spo_samples={int(dbg.get('valid_spo_samples', 0))}/"
                f"{int(dbg.get('sample_size', 0))}"
            )
        else:
            print(f"DSPO_PLUS Huber loss={loss:.4f}, spo_weight={spo_weight:.3f}")
        return loss

    def self_supervised_update(self, feat, cap_feat, target, spo_weight: float = 0.0):
        self.optimizer.zero_grad()
        outputs = self.supervised_ml(feat, cap_feat)
        huber_loss = self.criterion(outputs, target)

        self._last_spo_debug = {}
        spo_loss = None
        if spo_weight > 0.0:
            spo_loss = self._calculate_spo_loss_from_replay()
            if spo_loss is not None and (not spo_loss.requires_grad or not torch.isfinite(spo_loss)):
                spo_loss = None

        if spo_loss is not None and spo_weight > 0.0:
            total_loss = (1.0 - spo_weight) * huber_loss + spo_weight * spo_loss
            self._last_spo_debug.update(
                {
                    "huber_loss": float(huber_loss.item()),
                    "spo_loss": float(spo_loss.detach().item()),
                    "total_loss": float(total_loss.detach().item()),
                }
            )
        else:
            total_loss = huber_loss

        total_loss.backward()
        if self.spo_grad_clip_norm is not None and self.spo_grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.supervised_ml.parameters(), self.spo_grad_clip_norm)
        self.optimizer.step()
        return float(total_loss.item())

    def _get_current_spo_weight(self) -> float:
        if self.initial_phase:
            return 0.0
        if len(self.spo_replay_data) == 0:
            return 0.0
        if self.spo_episode_count < self.spo_warmup_episodes:
            return 0.0

        ramp = (self.spo_episode_count - self.spo_warmup_episodes) / float(self.spo_rampup_episodes)
        ramp = min(max(ramp, 0.0), 1.0)
        return float(self.max_spo_loss_weight * ramp)
