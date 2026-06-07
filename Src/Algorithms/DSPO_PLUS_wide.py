#!/usr/bin/env python3
"""
DSPO_PLUS_wide: DSPO_PLUS with a wide but finite price range.

This variant preserves DSPO_PLUS SPO+ training and changes only the final price
bound used by both deployment and the SPO+ oracle:
    [-wide_price_bound, wide_price_bound]
"""

from Src.Algorithms.DSPO_PLUS import DSPO_PLUS


class DSPO_PLUS_wide(DSPO_PLUS):
    """DSPO_PLUS with a large finite price bound."""

    PRICE_MODE = "wide"
