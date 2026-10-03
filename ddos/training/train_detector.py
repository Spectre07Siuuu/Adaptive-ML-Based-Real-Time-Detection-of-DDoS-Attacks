"""Train the model the live detector loads.

    python -m ddos.training.train_detector

Histogram gradient boosting on the 26 window features, attack vs normal: in
phase 2 it transferred best across datasets and scores one window in ~1 ms.
It is trained on every labelled window of the lab and both CIC datasets (each
capped as in phase 1), because coverage of attack types mattered more than
anything else in the cross-dataset results. Evaluation happened in phases 1-2;
the live lab sessions of phase 3 are new data.

The bundle also keeps the training windows, so the live detector can retrain
on them plus its pseudo-labels (--adaptive).
"""
import joblib
import numpy as np
import pandas as pd

from ddos.adaptive.online import fit_supervised
from ddos.config import PROCESSED_DIR, DETECTOR_MODEL, WINDOW_SECONDS
from ddos.features.window import FEATURES
from ddos.training.cross_dataset import cap

DATASETS = ["lab", "cicids2017", "cicddos2019"]


def main():
    frames = [cap(pd.read_csv(PROCESSED_DIR / f"{name}_windows.csv")) for name in DATASETS]
    data = pd.concat(frames, ignore_index=True)
    x = data[FEATURES].to_numpy(np.float32)
    y = (data["label"] > 0).to_numpy(int)
    model = fit_supervised(x, y)
    DETECTOR_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "window": WINDOW_SECONDS,
                 "trained_on": DATASETS, "rows": len(data), "base_x": x, "base_y": y}, DETECTOR_MODEL)
    print(f"Trained on {len(data)} windows ({y.mean():.1%} attack) -> {DETECTOR_MODEL}")


if __name__ == "__main__":
    main()
