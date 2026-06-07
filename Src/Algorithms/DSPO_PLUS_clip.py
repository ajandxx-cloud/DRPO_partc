#!/usr/bin/env python3
"""
DSPO_PLUS_clip: DSPO_PLUS with the original DSPO price truncation.

This is the clipped SPO+ variant for the 2x2 diagnostic experiment:
    DSPO_clip      vs DSPO_wide
    DSPO_PLUS_clip vs DSPO_PLUS_wide
"""

from Src.Algorithms.DSPO_PLUS import DSPO_PLUS


class DSPO_PLUS_clip(DSPO_PLUS):
    """DSPO_PLUS with default original DSPO clipping [min_p, max_p]."""

    PRICE_MODE = "clip"
