"""Cross-dataset evaluation, attack vs normal.

    python -m ddos.training.cross_dataset                # per-peer windows: lab + 2 CIC datasets
    python -m ddos.training.cross_dataset --view flows   # SDN-style flow view: + the SDN dataset

The windows view is the detector's own feature set (26 features), available for
packet captures only. The flows view (7 features, features/flows.py) is much
coarser but also covers the SDN dataset, which has no packets.

Every dataset is split once into train / test without leakage: the lab holds out
its last session, the public datasets hold out a quarter of their 10-minute
rounds. Every model is then scored on the same test parts, trained on:

  - one dataset (the diagonal is the usual within-dataset result),
  - all datasets except the test one (leave-one-dataset-out),
  - all datasets together.

Training sets are capped at CAP windows per dataset (stratified), so each dataset
weighs the same and SVM stays tractable. The task is binary because the attack
types differ between datasets; detection rates per attack type are saved too.
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ddos.config import PROCESSED_DIR, REPORTS_DIR, ALL_LABEL_NAMES
from ddos.features.flows import FLOW_FEATURES
from ddos.features.window import FEATURES
from ddos.training.train_baselines import MODELS, split

VIEWS = {
    "windows": (FEATURES, ["lab", "cicids2017", "cicddos2019"]),
    "flows": (FLOW_FEATURES, ["lab", "cicids2017", "cicddos2019", "sdn"]),
}
CAP = 20_000


def load(view):
    data = {}
    for name in VIEWS[view][1]:
        path = PROCESSED_DIR / f"{name}_{view}.csv"
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue
        df = pd.read_csv(path)
        train, test, _, how = split(df)
        data[name] = (df[train], df[test])
        print(f"{name:12s} train {train.sum():7d}  test {test.sum():7d}  ({how}); "
              f"attack share {(df['label'] > 0).mean():.1%}")
    return data


def cap(df, n=CAP, seed=42):
    if len(df) <= n:
        return df
    # Keep the class mix of the dataset, attack type included
    return pd.concat([g.sample(max(1, round(n * len(g) / len(df))), random_state=seed)
                      for _, g in df.groupby("label")])


def scores(test, pred):
    y = (test["label"] > 0).to_numpy()
    flagged = pred.astype(bool)
    row = {
        "Macro F1 %": f1_score(y, flagged, average="macro", zero_division=0) * 100,
        "Detection %": flagged[y].mean() * 100 if y.any() else np.nan,
        "False Positive %": flagged[~y].mean() * 100 if (~y).any() else np.nan,
    }
    per_type = {ALL_LABEL_NAMES[l]: flagged[(test["label"] == l).to_numpy()].mean() * 100
                for l in sorted(test["label"].unique()) if l > 0}
    return {k: round(v, 2) for k, v in row.items()}, per_type


def heatmap(table, title, path):
    fig, ax = plt.subplots(figsize=(1.5 * len(table.columns) + 3.5, 0.55 * len(table) + 1.8))
    im = ax.imshow(table.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(table)), table.index)
    ax.set_xlabel("Test dataset (held-out part)")
    ax.set_ylabel("Trained on")
    for i in range(len(table)):
        for j in range(len(table.columns)):
            v = table.iat[i, j]
            ax.text(j, i, "-" if pd.isna(v) else f"{v:.1f}", ha="center", va="center",
                    color="white" if not pd.isna(v) and v > 60 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--view", choices=list(VIEWS), default="windows")
    args = parser.parse_args()
    features = VIEWS[args.view][0]
    out = REPORTS_DIR / "cross_dataset" / args.view

    data = load(args.view)
    names = list(data)
    settings = {f"{n} only": [n] for n in names}
    if len(names) > 2:
        settings.update({f"all but {n}": [m for m in names if m != n] for n in names})
    settings["all datasets"] = names

    out.mkdir(parents=True, exist_ok=True)
    rows, per_type_rows = [], []
    for model_name, make in MODELS.items():
        for setting, sources in settings.items():
            train = pd.concat([cap(data[s][0]) for s in sources])
            model = make()
            start = time.perf_counter()
            model.fit(train[features], (train["label"] > 0).astype(int))
            fit_s = time.perf_counter() - start
            for target in names:
                test = data[target][1]
                metrics, per_type = scores(test, model.predict(test[features]))
                rows.append({"Model": model_name, "Trained on": setting, "Test": target,
                             **metrics, "Train Time(s)": round(fit_s, 2)})
                per_type_rows += [{"Model": model_name, "Trained on": setting, "Test": target,
                                   "Attack": k, "Detection %": round(v, 2)} for k, v in per_type.items()]
            print(f"{model_name:13s} {setting:22s} fit {fit_s:6.1f}s")

    results = pd.DataFrame(rows)
    results.to_csv(out / "results.csv", index=False)
    pd.DataFrame(per_type_rows).to_csv(out / "detection_by_attack.csv", index=False)

    order = list(settings)
    for model_name in MODELS:
        sub = results[results["Model"] == model_name]
        slug = model_name.lower().replace(" ", "_")
        for metric in ["Macro F1 %", "Detection %", "False Positive %"]:
            table = sub.pivot(index="Trained on", columns="Test", values=metric).loc[order, names]
            if metric == "Macro F1 %":
                print(f"\n{model_name}: binary macro-F1 % on each held-out test part")
                print(table.to_string())
                heatmap(table, f"{model_name}, {args.view} view: macro-F1 %", out / f"macro_f1_{slug}.png")
            table.to_csv(out / f"{slug}_{metric.split()[0].lower()}.csv")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
