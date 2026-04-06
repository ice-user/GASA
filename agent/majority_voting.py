"""
agent/majority_voting.py
-------------------------
Time-weighted majority voting over a circular prediction buffer.

Each prediction is assigned an exponentially increasing weight
so that the most recent prediction counts most.

    weight_i = exp(alpha * i)   where i=0 is oldest, i=K-1 is newest
    alpha = 0.3  (default)
"""

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class MajorityVoter:
    """
    Circular vote buffer with time-weighted majority voting.

    Parameters
    ----------
    k     : int   – voting window size (default 5)
    alpha : float – exponential decay coefficient (default 0.3)

    Usage
    -----
        voter = MajorityVoter(k=5)
        voter.push("cpu_bound")
        decision = voter.vote()   # → "cpu_bound" or None if buffer empty
    """

    def __init__(self, k: int = 5, alpha: float = 0.3):
        self.k     = k
        self.alpha = alpha
        self._buf  = deque(maxlen=k)
        logger.info("MajorityVoter initialised (k=%d, alpha=%.2f).", k, alpha)

    def push(self, label: str):
        """Add a new prediction to the buffer."""
        self._buf.append(label)
        logger.debug("Vote buffer: %s", list(self._buf))

    def vote(self):
        """
        Compute the time-weighted winning class.

        Returns
        -------
        str  – winning class label
        None – if the buffer is empty
        """
        if not self._buf:
            return None

        buf   = list(self._buf)
        n     = len(buf)
        # weights: oldest → smallest, newest → largest
        weights = np.exp(self.alpha * np.arange(n))

        tally = {}
        for label, w in zip(buf, weights):
            tally[label] = tally.get(label, 0.0) + w

        winner = max(tally, key=tally.__getitem__)
        logger.debug("Tally=%s  winner=%s", tally, winner)
        return winner

    def clear(self):
        """Reset the buffer (e.g. after a forced scheduler switch)."""
        self._buf.clear()
