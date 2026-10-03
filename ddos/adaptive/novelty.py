"""Novelty detection against one network's own benign traffic.

The detector learns what normal windows look like on the protected server
(a calibration period assumed free of attacks) and flags windows far from all
of them: mean distance to the k nearest calibration windows, on log-scaled,
standardised window features. Unlike a classifier it needs no attack examples,
so it can flag attack types no training set contained.

The threshold is a high quantile of the calibration windows' own scores
(each scored against the others), i.e. the expected false-positive rate on
benign traffic like the calibration period. A distance-based score is used
because it keeps growing outside the calibration range; isolation forests,
tried first, give a far-out window the same score as the most extreme
benign one and missed CIC-IDS2017's HTTP flood entirely.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class KnnNovelty:
    def __init__(self, features, k=5, quantile=0.999):
        self.features, self.k, self.quantile = features, k, quantile

    def _x(self, df):
        return self.scaler.transform(np.log1p(np.abs(df[self.features].to_numpy(np.float64))))

    def fit(self, benign):
        raw = np.log1p(np.abs(benign[self.features].to_numpy(np.float64)))
        self.scaler = StandardScaler().fit(raw)
        x = self.scaler.transform(raw)
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(x) - 1)).fit(x)
        dist, _ = self.nn.kneighbors(x, n_neighbors=self.nn.n_neighbors + 1)
        self.threshold = float(np.quantile(dist[:, 1:].mean(axis=1), self.quantile))   # skip itself
        return self

    def score(self, df):
        dist, _ = self.nn.kneighbors(self._x(df))
        return dist.mean(axis=1)

    def flag(self, df):
        return self.score(df) > self.threshold
