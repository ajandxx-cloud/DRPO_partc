#!/usr/bin/env python3
"""
DSPO_clip: original DSPO baseline with the original price truncation.

This thin wrapper exists so the 2x2 diagnostic experiment can explicitly run:
    DSPO_clip      vs DSPO_wide
    DSPO_PLUS_clip vs DSPO_PLUS_wide
"""

from Src.Algorithms.DSPO import DSPO


class DSPO_clip(DSPO):
    """Original DSPO with its default price clipping [min_p, max_p]."""
    pass
