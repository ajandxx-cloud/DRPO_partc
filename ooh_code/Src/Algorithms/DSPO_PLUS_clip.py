#!/usr/bin/env python3
"""
DSPO_PLUS_clip: DSPO_PLUS with the original DSPO-style price truncation.

This is the clipped SPO+ variant for the 2x2 diagnostic experiment:
    DSPO_clip, DSPO_wide, DSPO_PLUS_clip, DSPO_PLUS_wide.
"""

from Src.Algorithms.DSPO_PLUS import DSPO_PLUS


class DSPO_PLUS_clip(DSPO_PLUS):
    """DSPO_PLUS with the default clipped pricing rule."""

    PRICE_MODE = "clip"
