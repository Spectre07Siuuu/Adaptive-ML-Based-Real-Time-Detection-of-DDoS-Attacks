"""Phase 4: a lab-trained detector deployed on a new network, static vs adaptive.

    python -m ddos.adaptive.stream            # CIC-IDS2017 and CIC-DDoS2019 as the new networks

The windows of a public dataset are replayed in time order, as the live
detector would see them. Both detectors start from the same
gradient boosting model, trained on the lab only, plus a KnnNovelty model fitted
on the first CALIBRATION seconds of the new network's traffic (no labels; that
period has no attacks in either dataset, which the report checks). A window is
an attack if the supervised model or the novelty model says so.

The static detector never changes. The adaptive one retrains the supervised
model every RETRAIN seconds on the lab data plus pseudo-labels from the stream:
windows the novelty model finds normal become benign examples, windows of a
peer flagged by the novelty model in CONSECUTIVE windows in a row become attack
examples. No ground-truth label is used for adapting; labels are only used to
score both detectors.

The lab-only starting model transfers very differently from one random seed to
the next (phase 2), so everything runs for SEEDS seeds. Rates are reported per
window and with the live detector's rule (a peer counts as attacking only from
its second flagged window in a row, as the detector blocks).

Reports: reports/adaptive/ (<dataset>_blocks.csv per 10 minutes, summary.csv,
adaptation.png)
"""
import argparse
from collections import defaultdict, deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ddos.adaptive.novelty import KnnNovelty
from ddos.config import PROCESSED_DIR, REPORTS_DIR, WINDOW_SECONDS, ALL_LABEL_NAMES
from ddos.features.window import FEATURES
from ddos.training.cross_dataset import cap

CALIBRATION = 1800   # seconds of unlabelled traffic that define "normal" on the new network
STEP = 60            # seconds of stream scored at a time
RETRAIN = 600        # adaptive detector: seconds between retrains
CONSECUTIVE = 2      # novelty flags in a row before a peer's windows count as attack examples
POOL = 10_000        # most recent pseudo-labelled windows kept per class
BLOCK = 600          # reporting interval
OUT = REPORTS_DIR / "adaptive"


SEEDS = 3


def fit_supervised(x, y, seed):
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed).fit(x, y)


class AdaptiveDetector:
    def __init__(self, base, novelty, adaptive, seed):
        self.base_x = base[FEATURES].to_numpy(np.float32)
        self.base_y = (base["label"] > 0).to_numpy(int)
        self.seed = seed
        self.model = fit_supervised(self.base_x, self.base_y, seed)
        self.novelty, self.adaptive = novelty, adaptive
        self.benign, self.attack = deque(maxlen=POOL), deque(maxlen=POOL)
        self.streak, self.last_seen = defaultdict(int), {}
        self.retrains = 0

    def step(self, batch):
        """Decide on a batch of windows (time order); remember pseudo-labels if adaptive."""
        x = batch[FEATURES].to_numpy(np.float32)
        sup = self.model.predict(x).astype(bool)
        nov = self.novelty.flag(batch)
        if self.adaptive:
            for row, flagged, peer, start in zip(x, nov, batch["peer_ip"], batch["window_start"]):
                if self.last_seen.get(peer, -np.inf) < start - WINDOW_SECONDS:
                    self.streak[peer] = 0          # a gap breaks the streak
                self.last_seen[peer] = start
                self.streak[peer] = self.streak[peer] + 1 if flagged else 0
                if not flagged:
                    self.benign.append(row)
                elif self.streak[peer] >= CONSECUTIVE:
                    self.attack.append(row)
        return sup, nov

    def retrain(self):
        if not self.adaptive or not self.benign:
            return
        extra = [np.array(self.benign)] + ([np.array(self.attack)] if self.attack else [])
        labels = [np.zeros(len(self.benign), int)] + ([np.ones(len(self.attack), int)] if self.attack else [])
        self.model = fit_supervised(np.concatenate([self.base_x, *extra]),
                                    np.concatenate([self.base_y, *labels]), self.seed)
        self.retrains += 1


def run(name, base, seed):
    data = pd.read_csv(PROCESSED_DIR / f"{name}_windows.csv").sort_values("window_start", kind="stable")
    t0 = data["window_start"].min()
    calib = data[data["window_start"] < t0 + CALIBRATION]
    stream = data[data["window_start"] >= t0 + CALIBRATION].copy()
    novelty = KnnNovelty(FEATURES).fit(calib)
    detectors = {"static": AdaptiveDetector(base, novelty, False, seed),
                 "adaptive": AdaptiveDetector(base, novelty, True, seed)}
    decisions = {k: {"sup": [], "nov": []} for k in detectors}

    next_retrain = t0 + CALIBRATION + RETRAIN
    starts = stream["window_start"].to_numpy()
    for lo in np.arange(t0 + CALIBRATION, starts.max() + STEP, STEP):
        if lo >= next_retrain:
            for d in detectors.values():
                d.retrain()
            next_retrain += RETRAIN
        batch = stream[(starts >= lo) & (starts < lo + STEP)]
        if batch.empty:
            continue
        for k, d in detectors.items():
            sup, nov = d.step(batch)
            decisions[k]["sup"].append(sup)
            decisions[k]["nov"].append(nov)

    for k in detectors:
        stream[f"{k}_sup"] = np.concatenate(decisions[k]["sup"])
        stream[f"{k}_nov"] = np.concatenate(decisions[k]["nov"])
        stream[k] = stream[f"{k}_sup"] | stream[f"{k}_nov"]
    stream["block"] = ((stream["window_start"] - t0 - CALIBRATION) // BLOCK).astype(int)
    return stream, {"calibration windows": len(calib), "attacks in calibration": int((calib["label"] > 0).sum()),
                    "retrains": detectors["adaptive"].retrains,
                    "pseudo-attack windows": len(detectors["adaptive"].attack)}


def in_streak(df, col):
    """True from a peer's second flagged window in a row (the live detector's blocking rule)."""
    d = df[["peer_ip", "window_start", col]].sort_values(["peer_ip", "window_start"])
    prev_flag = d.groupby("peer_ip")[col].shift(1, fill_value=False)
    prev_start = d.groupby("peer_ip")["window_start"].shift(1)
    return (d[col] & prev_flag & (d["window_start"] - prev_start <= WINDOW_SECONDS)).reindex(df.index)


def rates(df, col):
    attack, benign = df["label"] > 0, df["label"] == 0
    return {"false_positive": 100 * df.loc[benign, col].mean() if benign.any() else np.nan,
            "detection": 100 * df.loc[attack, col].mean() if attack.any() else np.nan}


DETECTORS = [("static", "static: supervised or novelty"), ("adaptive", "adaptive: supervised or novelty"),
             ("static_sup", "static: supervised alone"), ("adaptive_sup", "adaptive: supervised alone"),
             ("static_nov", "novelty alone")]


def summarise(name, stream, info, seed):
    rows = []
    for col, label in DETECTORS:
        stream[f"{col}_streak"] = in_streak(stream, col)
        r, rs = rates(stream, col), rates(stream, f"{col}_streak")
        per_type = {f"detection {ALL_LABEL_NAMES[l]} %": 100 * stream.loc[stream["label"] == l, col].mean()
                    for l in sorted(stream["label"].unique()) if l > 0}
        rows.append({"dataset": name, "seed": seed, "detector": label,
                     "false positive %": r["false_positive"], "detection %": r["detection"],
                     "false positive, 2-window rule %": rs["false_positive"],
                     "detection, 2-window rule %": rs["detection"], **per_type, **info})
    blocks = []
    for b, g in stream.groupby("block"):
        row = {"seed": seed, "block": b, "minutes": b * BLOCK // 60,
               "benign": int((g["label"] == 0).sum()), "attack": int((g["label"] > 0).sum())}
        for col in ("static", "adaptive", "static_sup", "adaptive_sup"):
            r = rates(g, col)
            row[f"{col} FP %"] = r["false_positive"]
            row[f"{col} detection %"] = r["detection"]
        blocks.append(row)
    return pd.DataFrame(rows), pd.DataFrame(blocks)


def plot(blocks, path):
    """Supervised model's false positives over time, static vs adaptive, mean over seeds."""
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(1, len(blocks), figsize=(5.5 * len(blocks), 3.6), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (name, b) in zip(axes[0], blocks.items()):
        mean = b.groupby("minutes").mean(numeric_only=True).reset_index()
        for col, color, marker, label in [("static_sup FP %", "#eb6834", "s", "static"),
                                          ("adaptive_sup FP %", "#2a78d6", "o", "adaptive")]:
            ax.plot(mean["minutes"], mean[col], color=color, lw=2, marker=marker, ms=5, label=label)
        for m in mean.loc[mean["attack"] > 0, "minutes"]:
            ax.axvspan(m, m + BLOCK // 60, color=grid, zorder=0)
        ax.set_title(f"{name}: lab model on a new network", loc="left", color=ink)
        ax.set_xlabel("Minutes after calibration (grey: attacks in progress)", color=muted)
        ax.set_ylabel("Benign windows flagged by the\nsupervised model, % (mean of seeds)", color=muted)
        ax.set_ylim(bottom=0)
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.tick_params(colors=muted)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", nargs="*", default=["cicids2017", "cicddos2019"])
    parser.add_argument("--seeds", type=int, default=SEEDS)
    args = parser.parse_args()
    lab = pd.read_csv(PROCESSED_DIR / "lab_windows.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, all_blocks = [], {}
    for name in args.datasets:
        blocks = []
        for seed in range(args.seeds):
            stream, info = run(name, cap(lab, seed=seed), seed)
            summary, b = summarise(name, stream, info, seed)
            summaries.append(summary)
            blocks.append(b)
            print(f"{name} seed {seed}: {info}", flush=True)
        all_blocks[name] = pd.concat(blocks)
        all_blocks[name].round(2).to_csv(OUT / f"{name}_blocks.csv", index=False)
    results = pd.concat(summaries)
    results.round(2).to_csv(OUT / "results.csv", index=False)

    metrics = [c for c in results.columns if c.endswith("%")]
    table = results.groupby(["dataset", "detector"], sort=False)[metrics].agg(["mean", "min", "max"])
    table.round(2).to_csv(OUT / "summary.csv")
    pd.set_option("display.width", 250)
    for name in args.datasets:
        print(f"\n{name} (mean [min-max] over {args.seeds} seeds):")
        sub = table.loc[name]
        for det in sub.index:
            cells = [f"{m.replace(' %', '')}: {sub.loc[det, (m, 'mean')]:.1f} "
                     f"[{sub.loc[det, (m, 'min')]:.1f}-{sub.loc[det, (m, 'max')]:.1f}]"
                     for m in metrics if not np.isnan(sub.loc[det, (m, 'mean')])]
            print(f"  {det:32s} " + " | ".join(cells))
    plot(all_blocks, OUT / "adaptation.png")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
