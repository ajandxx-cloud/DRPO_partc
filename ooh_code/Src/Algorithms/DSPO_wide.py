#!/usr/bin/env python3
"""
DSPO_wide: DSPO baseline with a wide but finite price range.

This algorithm keeps the original DSPO candidate set, cost construction,
Lambert-W pricing structure, and MNL choice model, but replaces the original
small price truncation with a large finite bound [-wide_price_bound,
wide_price_bound].  This supports the 2x2 diagnostic experiment:
    DSPO_clip vs DSPO_wide
    DSPO_PLUS_clip vs DSPO_PLUS_wide
"""

from math import e

import numpy as np
import numpy.ma as ma
from scipy.special import lambertw

from Src.Algorithms.DSPO import DSPO


class DSPO_wide(DSPO):
    """Original DSPO with a wide finite price bound instead of the default clip."""

    @staticmethod
    def _safe_exp(x: float) -> float:
        return float(np.exp(np.clip(float(x), -700.0, 700.0)))

    def __init__(self, config):
        super(DSPO_wide, self).__init__(config)
        self.wide_price_bound = float(getattr(config, "wide_price_bound", 1000.0))
        if self.wide_price_bound <= 0:
            raise ValueError("wide_price_bound must be positive")
        self.wide_price_min = -self.wide_price_bound
        self.wide_price_max = self.wide_price_bound
        print(
            f"DSPO_wide initialized: price_range=[{self.wide_price_min}, {self.wide_price_max}]"
        )

    def get_action_pricing(self, state, training):
        """
        Same DSPO pricing calculation, but with wide finite bounds.

        The original DSPO-specific home floor (0.5), OOH floor (-3.5), and
        final clip to [max(-3.5, min_p), max_p] are removed.  Instead, prices
        are clipped only to [-wide_price_bound, wide_price_bound].
        """
        customerchoice_model = None
        if hasattr(self.config, "env") and self.config.env is not None:
            customerchoice_model = getattr(self.config.env, "customerchoice", None)

        if self.load_data:
            mask = ma.masked_array(
                state[2]["parcelpoints"],
                mask=self.adjacency[state[0].id_num],
            )
            pps = mask[mask.mask].data
        else:
            pps = state[2]["parcelpoints"]

        if self.initial_phase:
            return np.around(np.zeros(len(pps) + 1), decimals=2)

        cur_feat = self.get_feature_rep_infer(state[1]["fleet"])
        costs = self.get_prediction(cur_feat, state[0].home, pps)

        travel_times = None
        if self.use_travel_time_prediction:
            travel_times = self.get_travel_time_prediction(cur_feat, state[0].home, pps)
            if hasattr(self.config, "env") and self.config.env is not None:
                self.config.env.set_travel_times(travel_times)

        theta = self.init_theta - (state[3] * self.cool_theta)
        mltplr = self.cost_multiplier

        homeCosts = (
            (self.l0_home + state[0].service_time) * mltplr
            + ((1.0 - theta) * self.cheapestInsertionCosts(state[0].home, state[1])
               + theta * (costs[1] - costs[0]))
        )

        if customerchoice_model is not None:
            home_travel_time = None
            if travel_times is not None and "home" in travel_times:
                home_travel_time = travel_times["home"]
            home_base_util = self.base_util + state[0].home_util
            if customerchoice_model.travel_time_weight is not None and home_travel_time is not None:
                home_base_util += customerchoice_model.travel_time_weight * home_travel_time
            sum_mnl = self._safe_exp(
                home_base_util + state[0].incentiveSensitivity * (homeCosts - self.revenue)
            )
        else:
            sum_mnl = self._safe_exp(
                self.base_util
                + state[0].home_util
                + state[0].incentiveSensitivity * (homeCosts - self.revenue)
            )

        pp_costs = np.full((len(pps), 1), 1000000000.0)
        for idx, pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                if customerchoice_model is not None:
                    ooh_travel_time = None
                    if travel_times is not None and "ooh" in travel_times and idx < len(travel_times["ooh"]):
                        ooh_travel_time = travel_times["ooh"][idx]
                    util = customerchoice_model.mnl(state[0], pp, travel_time=ooh_travel_time)
                else:
                    util = self.mnl(state[0], pp)

                pp_costs[idx] = (
                    self.l_mp * mltplr
                    + mltplr
                    * ((1.0 - theta) * self.cheapestInsertionCosts(pp.location, state[1])
                       + theta * (costs[idx + 2] - costs[0]))
                )
                sum_mnl += self._safe_exp(
                    util + state[0].incentiveSensitivity * (pp_costs[idx] - self.revenue)
                )

        outside_option_util = getattr(self.config, "outside_option_util", None)
        if outside_option_util is not None:
            sum_mnl += self._safe_exp(outside_option_util)

        lambertw0 = (lambertw(sum_mnl / e).real + 1.0) / state[0].incentiveSensitivity

        a_hat = np.zeros(len(pps) + 1)
        safety_margin = 0.1

        home_price_base = homeCosts - self.revenue - lambertw0
        if home_price_base < 0:
            home_price_base = home_price_base * (1.0 - safety_margin)
        a_hat[0] = home_price_base

        for idx, pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                pp_price_base = pp_costs[idx] - self.revenue - lambertw0
                if pp_price_base < 0:
                    pp_price_base = pp_price_base * (1.0 - safety_margin)
                a_hat[idx + 1] = pp_price_base

        a_hat = np.clip(a_hat, self.wide_price_min, self.wide_price_max)
        return np.around(a_hat, decimals=2)
