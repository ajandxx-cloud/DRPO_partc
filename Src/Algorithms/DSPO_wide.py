#!/usr/bin/env python3
"""
DSPO_wide: original DSPO baseline with a wide but finite price range.

This class preserves the current DSPO.py candidate set, cost construction,
Lambert-W pricing structure, and MNL utility.  The only change is the final
price clipping range:
    original DSPO: [min_p, max_p]
    DSPO_wide:    [-wide_price_bound, wide_price_bound]
"""

from math import e, exp

import numpy as np
import numpy.ma as ma
from scipy.special import lambertw

from Src.Algorithms.DSPO import DSPO


class DSPO_wide(DSPO):
    """Original DSPO with a large finite price bound."""

    def __init__(self, config):
        super(DSPO_wide, self).__init__(config)
        self.wide_price_bound = float(getattr(config, "wide_price_bound", 1000.0))
        if self.wide_price_bound <= 0:
            raise ValueError("wide_price_bound must be positive")
        print(f"DSPO_wide initialized: price_range=[{-self.wide_price_bound}, {self.wide_price_bound}]")

    def get_action_pricing(self, state, training):
        if self.initial_phase:
            if self.load_data:
                mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])
                pps = mask[mask.mask].data
            else:
                pps = state[2]["parcelpoints"]
            a_hat = np.zeros(len(pps) + 1)
            return np.around(a_hat, decimals=2)

        if self.load_data:
            mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])
            pps = mask[mask.mask].data
        else:
            pps = state[2]["parcelpoints"]

        pp_costs = np.full(len(pps), 1000000000.0)

        cur_feat = self.get_feature_rep_infer(state[1]["fleet"])
        costs = self.get_prediction(cur_feat, state[0].home, pps)

        theta = self.init_theta - (state[3] * self.cool_theta)
        mltplr = self.cost_multiplier

        homeCosts = state[0].service_time * mltplr + (
            (1.0 - theta) * self.cheapestInsertionCosts(state[0].home, state[1])
            + theta * (costs[1] - costs[0])
        )
        sum_mnl = exp(
            self.base_util
            + state[0].home_util
            + state[0].incentiveSensitivity * (homeCosts - self.revenue)
        )

        for idx, pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                util = self.mnl(state[0], pp)
                pp_costs[idx] = mltplr * (
                    (1.0 - theta) * self.cheapestInsertionCosts(pp.location, state[1])
                    + theta * (costs[idx + 2] - costs[0])
                )
                sum_mnl += exp(util + state[0].incentiveSensitivity * (pp_costs[idx] - self.revenue))

        lambertw0 = (lambertw(sum_mnl / e).real + 1.0) / state[0].incentiveSensitivity

        a_hat = np.zeros(len(pps) + 1)
        a_hat[0] = homeCosts - self.revenue - lambertw0
        for idx, pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                a_hat[idx + 1] = pp_costs[idx] - self.revenue - lambertw0

        a_hat = np.clip(a_hat, -self.wide_price_bound, self.wide_price_bound)
        return np.around(a_hat, decimals=2)
