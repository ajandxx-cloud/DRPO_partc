#!/usr/bin/env python3
"""
DSPO_clip: original DSPO baseline with the original pricing truncation.

This wrapper is intentionally thin.  It exists so experiments can explicitly
refer to the clipped baseline in a 2x2 design:
    DSPO_clip, DSPO_wide, DSPO_PLUS_clip, DSPO_PLUS_wide.
"""

from Src.Algorithms.DSPO import DSPO


class DSPO_clip(DSPO):
    """Original DSPO with its default price clipping behavior."""
    pass
