#!/usr/bin/env python3
"""
DSPO_PLUS_wide: DSPO_PLUS with a wide but finite price range.

This variant preserves the DSPO_PLUS SPO+ training logic, but replaces the
original DSPO-style small clipping rule with a large finite price range
[-wide_price_bound, wide_price_bound].  The same wide pricing rule is used
both for online deployment and for the SPO+ oracle, so training and deployment
remain aligned.
"""

from Src.Algorithms.DSPO_PLUS import DSPO_PLUS


class DSPO_PLUS_wide(DSPO_PLUS):
    """DSPO_PLUS with wide finite pricing bounds for the 2x2 experiment."""

    PRICE_MODE = "wide"
