"""Pseudo-labelling and retraining shared by the replay (stream.py) and the live detector.

Windows the novelty model finds normal become benign examples; windows of a peer
the novelty model flags in CONSECUTIVE adjacent windows become attack examples.
The supervised model is periodically refitted on its base training data plus
the most recent POOL examples of each kind.
"""
from collections import defaultdict, deque

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

CONSECUTIVE = 2
POOL = 10_000


def fit_supervised(x, y, seed=0):
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed).fit(x, y)


class PseudoLabeler:
    def __init__(self, window=1.0, consecutive=CONSECUTIVE, pool=POOL):
        self.window, self.consecutive = window, consecutive
        self.benign, self.attack = deque(maxlen=pool), deque(maxlen=pool)
        self.streak, self.last_seen = defaultdict(int), {}

    def observe(self, x, flagged, peer, start):
        """One scored window (feature vector x), in time order."""
        if self.last_seen.get(peer, -np.inf) < start - self.window:
            self.streak[peer] = 0          # a gap breaks the streak
        self.last_seen[peer] = start
        self.streak[peer] = self.streak[peer] + 1 if flagged else 0
        if not flagged:
            self.benign.append(x)
        elif self.streak[peer] >= self.consecutive:
            self.attack.append(x)

    def training_set(self, base_x, base_y):
        """Base data plus the pseudo-labelled pools, or None if there is nothing new."""
        if not self.benign:
            return None
        xs, ys = [base_x, np.array(self.benign)], [base_y, np.zeros(len(self.benign), int)]
        if self.attack:
            xs.append(np.array(self.attack))
            ys.append(np.ones(len(self.attack), int))
        return np.concatenate(xs).astype(np.float32), np.concatenate(ys)
