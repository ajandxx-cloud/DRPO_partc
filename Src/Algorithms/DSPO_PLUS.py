#!/usr/bin/env python3
"""
DSPO_PLUS: DSPO + SPO+ without menu selection.

This class keeps the original DSPO pricing decision space:
    action = [home_price, pp_1_price, ..., pp_K_price]
for the same candidate parcel-point set used by DSPO.  No top-L menu
selection is introduced.

Training:
    - The base DSPO Huber objective is preserved through DSPO.update().
    - After the initial DSPO phase, a decision-focused SPO+ proxy is mixed
      into the supervised update.
    - The SPO+ oracle decision vector is the MNL choice-probability vector
      induced by the same Lambert-W pricing rule used by DSPO.

The SPO+ term is computed on option-level cost vectors:
    c = [home_cost, pp_1_cost, ..., pp_K_cost]
rather than on a scalar batch output.  This is the key difference from the
older DSPO_plus_SPO prototype.

Price-bound modes:
    - ``clip``: reproduce the original DSPO conservative bounds
      (home minimum 0.5, PP minimum -3.5, final clip to DSPO config bounds).
    - ``wide``: use a large but finite symmetric bound [-B, B] and remove the
      original conservative home/PP floors.  This preserves boundedness for
      SPO while making the bound unlikely to affect pricing in normal runs.
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
    """
    DSPO baseline enhanced with an SPO+ decision-focused training objective.

    Important design choice
    -----------------------
    This class deliberately does NOT perform menu screening.  It uses the
    same candidate parcel-point set and the same Lambert-W pricing rule as
    DSPO.  The only change is the learning objective after the initial DSPO
    warm-up: Huber cost-prediction loss is combined with an SPO+ proxy that
    directly trains the option-level cost predictions used by the DSPO
    pricing rule.
    """

    PRICE_MODE = "clip"  # subclasses override with "wide"

    def __init__(self, config):
        super(DSPO_PLUS, self).__init__(config)

        # SPO+ schedule.  Existing parser names are reused for compatibility.
        self.spo_warmup_episodes = getattr(config, "spo_warmup_episodes", 5)
        self.spo_rampup_episodes = max(1, getattr(config, "spo_rampup_episodes", 10))
        self.max_spo_loss_weight = getattr(config, "spo_loss_weight", 0.7)

        # Replay buffer for option-level decision records.
        self.spo_replay_data: List[Dict] = []
        self.spo_max_replay_size = getattr(config, "spo_replay_size", config.buffer_size)
        self.spo_batch_size = getattr(config, "spo_batch_size", min(64, config.batch_size))
        self.spo_grad_clip_norm = getattr(config, "spo_grad_clip_norm", 5.0)

        # Optional scalar for balancing SPO+ scale against Huber.
        self.spo_loss_scale = getattr(config, "spo_loss_scale", 1.0)

        # Price-bound mode for the 2x2 experiments.
        self.price_mode = getattr(config, "price_bound_mode", self.PRICE_MODE)
        self.wide_price_bound = float(getattr(config, "wide_price_bound", 100.0))
        if self.wide_price_bound <= 0:
            raise ValueError("wide_price_bound must be positive")

        self.spo_episode_count = 0
        self.spo_huber_loss_history: List[float] = []
        self._spo_weight_logged = False
        self._spo_data_logged = False
        self._last_spo_debug: Dict[str, float] = {}

        print(
            "DSPO_PLUS initialized: "
            f"spo_warmup={self.spo_warmup_episodes}, "
            f"spo_rampup={self.spo_rampup_episodes}, "
            f"max_spo_weight={self.max_spo_loss_weight}, "
            f"spo_batch_size={self.spo_batch_size}, "
            f"price_mode={self.price_mode}, "
            f"wide_bound={self.wide_price_bound}, "
            "menu_selection=False"
        )

    @staticmethod
    def _safe_exp(x: float) -> float:
        return float(np.exp(np.clip(float(x), -700.0, 700.0)))

    # ------------------------------------------------------------------
    # Price post-processing helpers.
    # ------------------------------------------------------------------
    def _is_wide_mode(self) -> bool:
        return str(self.price_mode).lower() == "wide"

    def _postprocess_home_price(self, price: float) -> float:
        """Apply DSPO-compatible or wide-bound home-price post-processing."""
        safety_margin = 0.1
        if price < 0:
            price = price * (1.0 - safety_margin)
        if not self._is_wide_mode():
            price = max(price, 0.5)
        return float(price)

    def _postprocess_pp_price(self, price: float) -> float:
        """Apply DSPO-compatible or wide-bound PP-price post-processing."""
        safety_margin = 0.1
        if price < 0:
            price = price * (1.0 - safety_margin)
        if not self._is_wide_mode():
            price = max(price, -3.5)
        return float(price)

    def _clip_price_vector(self, prices: np.ndarray) -> np.ndarray:
        """Final finite price bounds used by deployment and the SPO+ oracle."""
        prices = np.asarray(prices, dtype=np.float64)
        if self._is_wide_mode():
            return np.clip(prices, -self.wide_price_bound, self.wide_price_bound)
        max_discount = -3.5
        return np.clip(prices, max(max_discount, self.min_p), self.max_p)

    # ------------------------------------------------------------------
    # DSPO-aligned pricing rule.
    # ------------------------------------------------------------------
    def get_action_pricing(self, state, training):
        """
        Same action space and pricing logic as DSPO.get_action_pricing.

        Difference from DSPO:
        when training=True and the initial phase is over, store the full
        option-level decision record needed for SPO+ training.
        """
        customer = state[0]
        fleet = state[1]
        parcelpoint_state = state[2]
        arrival_time = state[3]

        customerchoice_model = None
        if hasattr(self.config, "env") and self.config.env is not None:
            customerchoice_model = getattr(self.config.env, "customerchoice", None)

        if self.load_data:
            mask = ma.masked_array(
                parcelpoint_state["parcelpoints"],
                mask=self.adjacency[customer.id_num],
            )
            pps = mask[mask.mask].data
        else:
            pps = parcelpoint_state["parcelpoints"]

        # Keep original DSPO initial phase behavior: zero prices.
        if self.initial_phase:
            return np.around(np.zeros(len(pps) + 1), decimals=2)

        cur_feat = self.get_feature_rep_infer(fleet["fleet"])
        ml_costs = self.get_prediction(cur_feat, customer.home, pps)

        travel_times = None
        if self.use_travel_time_prediction:
            travel_times = self.get_travel_time_prediction(cur_feat, customer.home, pps)
            if hasattr(self.config, "env") and self.config.env is not None:
                self.config.env.set_travel_times(travel_times)

        theta = self.init_theta - (arrival_time * self.cool_theta)
        mltplr = self.cost_multiplier

        # Original DSPO home-cost expression is mirrored exactly here.
        # Note: the route-cost term is intentionally not multiplied by
        # mltplr, because DSPO.py currently uses this formula.
        home_cheapest = self.cheapestInsertionCosts(customer.home, fleet)
        home_cost = (
            (self.l0_home + customer.service_time) * mltplr
            + ((1.0 - theta) * home_cheapest + theta * (ml_costs[1] - ml_costs[0]))
        )

        home_base_util = self._home_utility(customer, customerchoice_model, travel_times)
        s = customer.incentiveSensitivity
        sum_mnl = self._safe_exp(home_base_util + s * (home_cost - self.revenue))

        pp_costs = np.full(len(pps), 1000000000.0, dtype=np.float64)
        pp_cheapest = np.full(len(pps), np.inf, dtype=np.float64)
        pp_utils = np.zeros(len(pps), dtype=np.float64)
        valid_pp_mask = np.zeros(len(pps), dtype=bool)

        for idx, pp in enumerate(pps):
            pp_utils[idx] = self._pp_utility(customer, pp, idx, customerchoice_model, travel_times)

            if pp.remainingCapacity > 0:
                valid_pp_mask[idx] = True
                pp_cheapest[idx] = self.cheapestInsertionCosts(pp.location, fleet)
                pp_costs[idx] = (
                    self.l_mp * mltplr
                    + mltplr
                    * ((1.0 - theta) * pp_cheapest[idx] + theta * (ml_costs[idx + 2] - ml_costs[0]))
                )
                sum_mnl += self._safe_exp(pp_utils[idx] + s * (pp_costs[idx] - self.revenue))

        outside_option_util = getattr(self.config, "outside_option_util", None)
        if outside_option_util is not None:
            sum_mnl += self._safe_exp(outside_option_util)

        lambertw0 = (lambertw(sum_mnl / e).real + 1.0) / s

        a_hat = np.zeros(len(pps) + 1, dtype=np.float64)
        a_hat[0] = self._postprocess_home_price(home_cost - self.revenue - lambertw0)

        for idx, pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                a_hat[idx + 1] = self._postprocess_pp_price(pp_costs[idx] - self.revenue - lambertw0)

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
                home_base_util=home_base_util,
                pp_utils=pp_utils,
            )

        return np.around(a_hat, decimals=2)

    def _home_utility(self, customer, customerchoice_model, travel_times) -> float:
        home_travel_time = None
        if travel_times is not None and "home" in travel_times:
            home_travel_time = travel_times["home"]

        home_base_util = self.base_util + customer.home_util
        if customerchoice_model is not None:
            if customerchoice_model.travel_time_weight is not None and home_travel_time is not None:
                home_base_util += customerchoice_model.travel_time_weight * home_travel_time

        return float(home_base_util)

    def _pp_utility(self, customer, pp, pp_idx: int, customerchoice_model, travel_times) -> float:
        if customerchoice_model is not None:
            ooh_travel_time = None
            if travel_times is not None and "ooh" in travel_times and pp_idx < len(travel_times["ooh"]):
                ooh_travel_time = travel_times["ooh"][pp_idx]
            return float(customerchoice_model.mnl(customer, pp, travel_time=ooh_travel_time))
        return float(self.mnl(customer, pp))

    # ------------------------------------------------------------------
    # SPO+ data and differentiable DSPO cost-vector reconstruction.
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
        home_base_util: float,
        pp_utils: np.ndarray,
    ) -> None:
        """Store one DSPO decision instance for later SPO+ training."""
        mltplr = self.cost_multiplier

        true_costs_full = np.full(len(pps) + 1, np.inf, dtype=np.float32)

        # Match the existing DSPO home cost scale exactly.
        true_costs_full[0] = float((self.l0_home + customer.service_time) * mltplr + home_cheapest)

        for idx, pp in enumerate(pps):
            if valid_pp_mask[idx]:
                true_costs_full[idx + 1] = float(self.l_mp * mltplr + mltplr * pp_cheapest[idx])

        record = {
            "cur_feat": cur_feat.squeeze(0).detach().cpu().numpy().astype(np.float32),
            "home_id_num": int(customer.home.id_num),
            "home_time": float(customer.home.time),
            "home_service_time": float(customer.service_time),
            "incentive_sensitivity": float(customer.incentiveSensitivity),
            "theta": float(theta),
            "home_cheapest": float(home_cheapest),
            "pp_cheapest": pp_cheapest.astype(np.float32),
            "valid_pp_mask": valid_pp_mask.astype(bool),
            "home_base_util": float(home_base_util),
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
        """
        Recompute the DSPO option-level cost vector with gradients.

        Returns:
            Tensor of shape [1 + len(pps)], ordered as [home, pp_0, ..., pp_K].
        """
        device = self.device
        cur_feat = torch.tensor(record["cur_feat"], dtype=float32, device=device)
        if cur_feat.dim() == 4:
            cur_feat = cur_feat.squeeze(0)
        if cur_feat.dim() != 3:
            raise ValueError(f"Expected cur_feat to be 3D, got {cur_feat.shape}")

        time_int = min(int(record["home_time"] / self.interval), self.n_layers - 1)

        option_feats: List[torch.Tensor] = []
        option_caps: List[torch.Tensor] = []

        # Baseline current route feature, matching DSPO.get_prediction index 0.
        option_feats.append(cur_feat.clone())
        option_caps.append(torch.tensor(1000000.0, dtype=float32, device=device))

        # Home option, matching DSPO.get_prediction index 1.
        home_feat = cur_feat.clone()
        home_cell = self.customer_cell[record["home_id_num"]]
        home_feat[time_int][home_cell[0]][home_cell[1]] += 1
        option_feats.append(home_feat)
        option_caps.append(torch.tensor(1000000.0, dtype=float32, device=device))

        # Parcel-point options, matching DSPO.get_prediction index 2+idx.
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
            (self.l0_home + record["home_service_time"]) * self.cost_multiplier,
            dtype=float32,
            device=device,
        )
        home_cheapest = torch.tensor(record["home_cheapest"], dtype=float32, device=device)

        # Mirrors DSPO.py exactly: no mltplr on the home route-cost term.
        home_cost = home_service + (1.0 - theta) * home_cheapest + theta * (pred_vec[1] - base_pred)

        pp_cheapest = torch.tensor(record["pp_cheapest"], dtype=float32, device=device)
        pp_marginal_pred = pred_vec[2:] - base_pred
        pp_service = torch.tensor(self.l_mp * self.cost_multiplier, dtype=float32, device=device)

        pp_costs = pp_service + mltplr * ((1.0 - theta) * pp_cheapest + theta * pp_marginal_pred)

        return torch.cat([home_cost.reshape(1), pp_costs.reshape(-1)])

    # ------------------------------------------------------------------
    # SPO+ oracle and loss.
    # ------------------------------------------------------------------
    def _oracle_choice_probs(
        self,
        costs_np: np.ndarray,
        home_base_util: float,
        pp_utils: List[float],
        incentive_sensitivity: float,
    ) -> Optional[np.ndarray]:
        """
        DSPO pricing oracle followed by MNL probabilities.

        The returned decision vector w has the same dimension as costs_np and
        excludes the outside option.  The outside option, when configured, is
        included in the denominator exactly as in DSPO pricing.
        """
        costs_np = np.asarray(costs_np, dtype=np.float64).reshape(-1)
        K = int(costs_np.shape[0])
        if K <= 0:
            return None

        s = float(incentive_sensitivity)
        if not np.isfinite(s) or abs(s) < 1e-8:
            return None

        if len(pp_utils) != K - 1:
            return None

        outside_option_util = getattr(self.config, "outside_option_util", None)
        outside_option_util = float(outside_option_util) if outside_option_util is not None else None

        terms = [float(home_base_util) + s * (costs_np[0] - self.revenue)]
        for j in range(K - 1):
            terms.append(float(pp_utils[j]) + s * (costs_np[j + 1] - self.revenue))
        if outside_option_util is not None:
            terms.append(outside_option_util)

        terms = np.asarray(terms, dtype=np.float64)
        tmax = float(np.max(terms))
        sum_mnl = self._safe_exp(tmax) * float(np.sum(np.exp(terms - tmax)))
        if not np.isfinite(sum_mnl) or sum_mnl <= 0:
            return None

        lambertw0 = (lambertw(sum_mnl / np.e).real + 1.0) / s
        if not np.isfinite(lambertw0):
            return None

        prices = np.zeros(K, dtype=np.float64)
        prices[0] = self._postprocess_home_price(costs_np[0] - self.revenue - lambertw0)

        for j in range(K - 1):
            prices[j + 1] = self._postprocess_pp_price(costs_np[j + 1] - self.revenue - lambertw0)

        prices = self._clip_price_vector(prices)

        v_terms = [float(home_base_util) + s * prices[0]]
        for j in range(K - 1):
            v_terms.append(float(pp_utils[j]) + s * prices[j + 1])
        if outside_option_util is not None:
            v_terms.append(outside_option_util)

        v_terms = np.asarray(v_terms, dtype=np.float64)
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

            # Decision vector = home + all feasible parcel points.  No menu screening.
            valid_indices = [0] + [idx + 1 for idx, valid in enumerate(valid_pp_mask) if valid]
            if len(valid_indices) <= 1:
                continue

            valid_indices_t = torch.tensor(valid_indices, dtype=torch.long, device=c_hat_full.device)
            c_hat = c_hat_full.index_select(0, valid_indices_t)

            c_true_np = true_full[valid_indices]
            if not np.all(np.isfinite(c_true_np)):
                continue

            if not torch.isfinite(c_hat).all():
                continue

            valid_pp_utils = [
                float(record["pp_utils"][idx])
                for idx, valid in enumerate(valid_pp_mask)
                if valid
            ]

            c_hat_np = c_hat.detach().cpu().numpy().astype(np.float64)
            pseudo_np = 2.0 * c_hat_np - c_true_np.astype(np.float64)

            w_true_np = self._oracle_choice_probs(
                c_true_np,
                record["home_base_util"],
                valid_pp_utils,
                record["incentive_sensitivity"],
            )
            if w_true_np is None:
                continue

            w_surr_np = self._oracle_choice_probs(
                pseudo_np,
                record["home_base_util"],
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

            # SPO+ surrogate:
            #   c_true^T w*(2 c_hat - c_true)
            #   + 2 c_hat^T (w*(c_true) - w*(2 c_hat - c_true))
            # Oracle outputs are stop-gradient, yielding the SPO+ subgradient.
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
        return torch.stack(losses).mean() * float(self.spo_loss_scale)

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
        self.spo_huber_loss_history.append(float(huber_loss.item()))
        if len(self.spo_huber_loss_history) > 100:
            self.spo_huber_loss_history.pop(0)

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
