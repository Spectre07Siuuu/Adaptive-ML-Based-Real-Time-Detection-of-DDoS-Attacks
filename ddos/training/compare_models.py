"""Phase 2: deep sequence models against tree baselines, within and across datasets.

    python -m ddos.training.compare_models                      # full run, ~1 hour on 2 cores
    python -m ddos.training.compare_models --seeds 1 --epochs 3 # quick check

Models (attack vs normal):
  RF, HGB (window)     Random Forest / histogram gradient boosting on the 26 window features
  RF (sequence)        Random Forest on the networks' own input, flattened (32 packets x 14)
  CNN, LSTM, Transformer  deep_models.py on the packet sequences

All models see the same windows, the same train/test split per dataset as
cross_dataset.py (lab: held-out session; others: held-out 10-minute blocks) and
the same per-dataset training cap. 15% of the training rounds are set aside to
pick the networks' best epoch; the trees train on the same 85%. Each model is
trained on each dataset alone and on all three together, for every seed, and
tested on every dataset's test part. Speed is measured on the lab-only models.

Reports: reports/models/ (results.csv per run, summary.csv, efficiency.csv,
detection_by_attack.csv, comparison.png)
"""
import argparse
import copy
import io
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

from ddos.config import PROCESSED_DIR, REPORTS_DIR, ALL_LABEL_NAMES
from ddos.features.window import FEATURES
from ddos.training.cross_dataset import cap
from ddos.training.deep_models import build
from ddos.training.train_baselines import split

DATASETS = ["lab", "cicids2017", "cicddos2019"]
NETWORKS = ["CNN", "LSTM", "Transformer"]
MODELS = ["RF (window)", "HGB (window)", "RF (sequence)", *NETWORKS]
OUT = REPORTS_DIR / "models"


# ---------- data ----------

def load():
    data = {}
    for name in DATASETS:
        win = pd.read_csv(PROCESSED_DIR / f"{name}_windows.csv")
        seq = np.load(PROCESSED_DIR / f"{name}_sequences.npz")
        if len(seq["label"]) != len(win) or (seq["label"] != win["label"].to_numpy()).any():
            raise SystemExit(f"{name}: sequences do not match the windows; run build_sequences")
        train, test, groups, _ = split(win)
        data[name] = {"win": win, "x": seq["x"].astype(np.float32), "length": seq["length"].astype(np.int64),
                      "train": np.flatnonzero(train), "test": np.flatnonzero(test), "groups": groups}
    return data


def training_rows(d, seed):
    """Capped training rows of one dataset, split into fit / validation by round."""
    rows = cap(d["win"].iloc[d["train"]], seed=seed).index.to_numpy()
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
                    .split(rows, groups=d["groups"][rows]))
    return rows[fit], rows[val]


def gather(data, parts, kind):
    """Stack (inputs, y) for [(dataset, rows), ...]; kind is window, sequence or flat."""
    xs, ls, ys = [], [], []
    for name, rows in parts:
        d = data[name]
        if kind == "window":
            xs.append(d["win"].loc[rows, FEATURES].to_numpy(np.float32))
        else:
            xs.append(d["x"][rows])
            ls.append(d["length"][rows])
        ys.append((d["win"]["label"].to_numpy()[rows] > 0).astype(np.int64))
    x, y = np.concatenate(xs), np.concatenate(ys)
    if kind == "flat":
        return x.reshape(len(x), -1), y
    if kind == "sequence":
        return (x, np.concatenate(ls)), y
    return x, y


# ---------- networks ----------

def net_predict(model, inputs, batch=4096):
    x, lengths = inputs
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(torch.from_numpy(x[i:i + batch]), torch.from_numpy(lengths[i:i + batch])) > 0)
    return torch.cat(out).numpy().astype(int)


def train_network(name, fit, val, seed, epochs, batch=256, patience=4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    (x, lengths), y = fit
    model = build(name, x.shape[2], x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xt, lt, yt = torch.from_numpy(x), torch.from_numpy(lengths), torch.from_numpy(y).float()
    best, best_state, stale, used = -1.0, None, 0, 0
    for epoch in range(epochs):
        model.train()
        order = torch.from_numpy(rng.permutation(len(y)))
        for i in range(0, len(y), batch):
            idx = order[i:i + batch]
            opt.zero_grad()
            loss_fn(model(xt[idx], lt[idx]), yt[idx]).backward()
            opt.step()
        used = epoch + 1
        score = f1_score(val[1], net_predict(model, val[0]), average="macro", zero_division=0)
        if score > best + 1e-4:
            best, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, used


# ---------- one model, one setting ----------

KIND = {"RF (window)": "window", "HGB (window)": "window", "RF (sequence)": "flat",
        **{n: "sequence" for n in NETWORKS}}


def fit_model(name, data, sources, seed, epochs):
    fit_parts, val_parts = [], []
    for s in sources:
        f, v = training_rows(data[s], seed)
        fit_parts.append((s, f))
        val_parts.append((s, v))
    kind = KIND[name]
    fit, val = gather(data, fit_parts, kind), gather(data, val_parts, kind)
    start = time.perf_counter()
    if name in NETWORKS:
        model, epochs_used = train_network(name, fit, val, seed, epochs)
    else:
        if name.startswith("RF"):
            model = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1,
                                           random_state=seed)
        else:
            model = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
        model.fit(*fit)
        epochs_used = None
    return model, time.perf_counter() - start, epochs_used


def predict(name, model, inputs):
    return net_predict(model, inputs) if name in NETWORKS else model.predict(inputs)


def scores(y_label, pred):
    y, flagged = y_label > 0, pred.astype(bool)
    row = {"macro_f1": f1_score(y, flagged, average="macro", zero_division=0) * 100,
           "detection": flagged[y].mean() * 100 if y.any() else np.nan,
           "false_positive": flagged[~y].mean() * 100 if (~y).any() else np.nan}
    per_type = {ALL_LABEL_NAMES[l]: flagged[y_label == l].mean() * 100 for l in np.unique(y_label) if l > 0}
    return row, per_type


# ---------- speed and size ----------

def efficiency(name, model, data):
    d = data["lab"]
    inputs, _ = gather(data, [("lab", d["test"])], KIND[name])
    if name in NETWORKS:
        params = sum(p.numel() for p in model.parameters())
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        size = buf.tell()
        one = lambda i: net_predict(model, (inputs[0][i:i + 1], inputs[1][i:i + 1]))
        many = lambda: net_predict(model, inputs)
    else:
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1   # one window at a time: thread start-up would dominate
        params = (sum(t.tree_.node_count for t in model.estimators_) if hasattr(model, "estimators_")
                  else sum(len(p.nodes) for it in model._predictors for p in it))
        size = len(pickle.dumps(model))
        one = lambda i: model.predict(inputs[i:i + 1])
        many = lambda: model.predict(inputs)
    picks = np.random.default_rng(0).choice(len(d["test"]), 300, replace=False)
    times = []
    for i in picks:
        start = time.perf_counter()
        one(i)
        times.append(time.perf_counter() - start)
    start = time.perf_counter()
    many()
    batch_s = time.perf_counter() - start
    return {"Model": name, "Parameters / tree nodes": params, "Size (KB)": round(size / 1024, 1),
            "Latency, 1 window (ms)": round(np.median(times) * 1e3, 3),
            "Throughput (windows/s)": round(len(d["test"]) / batch_s)}


# ---------- report ----------

def summarise(results):
    """Mean +- std over seeds of the three evaluation kinds, per model."""
    r = results.copy()
    r["kind"] = np.where(r["Trained on"] == "all datasets", "all datasets",
                         np.where(r["Trained on"] == r["Test"] + " only", "within dataset", "across datasets"))
    per_seed = r.groupby(["Model", "kind", "Seed"])["macro_f1"].mean().reset_index()
    table = per_seed.groupby(["Model", "kind"])["macro_f1"].agg(["mean", "std"]).reset_index()
    return table.pivot(index="Model", columns="kind", values=["mean", "std"]).reindex(MODELS)


def plot(summary, eff, path):
    kinds = [("within dataset", "#2a78d6", "o"), ("across datasets", "#eb6834", "s"),
             ("all datasets", "#1baf7a", "D")]
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor("#fcfcfb")
    rows = np.arange(len(MODELS))[::-1]
    for (kind, color, marker), dy in zip(kinds, (0.18, 0, -0.18)):
        mean, std = summary[("mean", kind)].to_numpy(), summary[("std", kind)].fillna(0).to_numpy()
        ax.errorbar(mean, rows + dy, xerr=std, fmt=marker, color=color, ms=8, elinewidth=2,
                    capsize=0, label=kind, mec="#fcfcfb", mew=1.5)
    ax.set_yticks(rows, MODELS, color=ink)
    ax.set_xlabel("Binary macro-F1 % (mean over seeds; bars: ±1 std)", color=muted)
    ax.set_xlim(0, 101)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, labelcolor=ink,
              handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
    ax.set_title("Detection quality", loc="left", color=ink, pad=28)
    lat = eff.set_index("Model").reindex(MODELS)["Latency, 1 window (ms)"]
    bx.barh(rows, lat.to_numpy(), color="#2a78d6", height=0.5)
    for r, v in zip(rows, lat.to_numpy()):
        bx.text(v * 1.15, r, f"{v:.2f} ms", va="center", color=muted, fontsize=9)
    bx.set_xscale("log")
    bx.set_yticks(rows, [""] * len(MODELS))
    bx.set_xlabel("Latency for one window, ms (log scale)", color=muted)
    bx.set_xlim(right=lat.max() * 8)
    bx.set_title("Speed (lab test set, 1 window at a time)", loc="left", color=ink, pad=28)
    for a in (ax, bx):
        a.set_facecolor("#fcfcfb")
        a.grid(axis="x", color=grid, linewidth=0.8)
        a.set_axisbelow(True)
        a.tick_params(colors=muted)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.spines["bottom"].set_color(grid)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30, help="maximum epochs (early stopping)")
    parser.add_argument("--models", nargs="*", default=MODELS, help="subset of models to run")
    args = parser.parse_args()

    data = load()
    settings = {**{f"{n} only": [n] for n in DATASETS}, "all datasets": DATASETS}
    OUT.mkdir(parents=True, exist_ok=True)
    rows, per_type_rows, eff = [], [], []
    for name in args.models:
        for seed in range(args.seeds):
            for setting, sources in settings.items():
                model, train_s, epochs_used = fit_model(name, data, sources, seed, args.epochs)
                for target in DATASETS:
                    d = data[target]
                    inputs, _ = gather(data, [(target, d["test"])], KIND[name])
                    labels = d["win"]["label"].to_numpy()[d["test"]]
                    row, per_type = scores(labels, predict(name, model, inputs))
                    rows.append({"Model": name, "Seed": seed, "Trained on": setting, "Test": target,
                                 **{k: round(v, 2) for k, v in row.items()},
                                 "Train time (s)": round(train_s, 2), "Epochs": epochs_used})
                    per_type_rows += [{"Model": name, "Seed": seed, "Trained on": setting, "Test": target,
                                       "Attack": k, "Detection %": round(v, 2)} for k, v in per_type.items()]
                if seed == 0 and setting == "lab only":
                    eff.append(efficiency(name, model, data) | {"Train time, lab (s)": round(train_s, 1)})
                print(f"{name:14s} seed {seed} {setting:17s} {train_s:7.1f}s"
                      + (f" ({epochs_used} epochs)" if epochs_used else ""), flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "results.csv", index=False)
    pd.DataFrame(per_type_rows).to_csv(OUT / "detection_by_attack.csv", index=False)
    eff = pd.DataFrame(eff)
    eff.to_csv(OUT / "efficiency.csv", index=False)
    summary = summarise(results)
    summary.round(2).to_csv(OUT / "summary.csv")

    pd.set_option("display.width", 200)
    print("\nBinary macro-F1 % (mean ± std over seeds):")
    for model in summary.index:
        cells = [f"{summary.loc[model, ('mean', k)]:6.2f} ± {summary.loc[model, ('std', k)]:5.2f}"
                 for k in ("within dataset", "across datasets", "all datasets")]
        print(f"  {model:14s} within {cells[0]}   across {cells[1]}   all {cells[2]}")
    print("\n" + eff.to_string(index=False))
    if set(MODELS) <= set(args.models):
        plot(summary, eff, OUT / "comparison.png")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
