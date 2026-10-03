"""Train RF / KNN / SVM on the lab window dataset, with splits that do not leak.

    python -m ddos.training.train_baselines
    python -m ddos.training.train_baselines --test-session s3

Rows are grouped by (session, round), so an attack and the benign gap after it
stay on one side of every split. With two or more sessions the test set is one
whole held-out session (the last one by default); with a single session it is
about a quarter of its rounds. Cross-validation on the training side is grouped too.

Besides accuracy and macro-F1 it reports the two numbers mitigation depends on:
detection rate (attack windows flagged as any attack) and false-positive rate
(Normal windows flagged as an attack, i.e. benign hosts that would get blocked).
"""
import argparse
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ddos.config import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, ALL_LABEL_NAMES, WINDOW_SECONDS
from ddos.features.window import FEATURES

DATA = PROCESSED_DIR / "lab_windows.csv"
OUT_REPORTS = REPORTS_DIR / "lab"
OUT_MODELS = MODELS_DIR / "lab"

# Scaling lives inside each pipeline, so a saved model can never be paired with the wrong scaler
MODELS = {
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                    n_jobs=-1, random_state=42),
    "KNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVM": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced",
                                                       random_state=42)),
}


def split(df, test_session=None):
    groups = (df["session"] + ":" + df["round"].astype(str)).to_numpy()
    sessions = sorted(df["session"].unique())
    if len(sessions) > 1:
        test_session = test_session or sessions[-1]
        test = (df["session"] == test_session).to_numpy()
        how = f"held-out session {test_session}"
    else:
        folds = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        _, test_idx = next(folds.split(df, df["label"], groups))
        test = np.zeros(len(df), dtype=bool)
        test[test_idx] = True
        how = "a quarter of the rounds held out (single session)"
    return ~test, test, groups, how


def detection_rates(y_true, y_pred):
    attack, flagged = y_true > 0, y_pred > 0
    detection = flagged[attack].mean() if attack.any() else np.nan
    false_pos = flagged[~attack].mean() if (~attack).any() else np.nan
    return detection, false_pos


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--test-session", help="session to hold out (default: the last one)")
    args = parser.parse_args()

    df = pd.read_csv(DATA)
    train, test, groups, how = split(df, args.test_session)
    X_train, y_train = df.loc[train, FEATURES], df.loc[train, "label"].to_numpy()
    X_test, y_test = df.loc[test, FEATURES], df.loc[test, "label"].to_numpy()

    seen = X_test.merge(X_train.drop_duplicates(), how="inner")
    print(f"Split: {how}")
    print(f"  train {len(X_train)} windows, test {len(X_test)} windows "
          f"({len(seen) / len(X_test):.1%} of test rows also occur verbatim in train)")

    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    (OUT_REPORTS / "figures").mkdir(exist_ok=True)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    label_ids = sorted(df["label"].unique())
    label_names = [ALL_LABEL_NAMES[i] for i in label_ids]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    results, reports, trained = [], [], {}
    for name, make in MODELS.items():
        print(f"\n{'=' * 40}\n{name}")
        model = make()
        cv_f1 = cross_val_score(make(), X_train, y_train, groups=groups[train],
                                cv=cv, scoring="f1_macro").mean()

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start
        trained[name] = model
        start = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_time = time.perf_counter() - start

        detection, false_pos = detection_rates(y_test, y_pred)
        report = classification_report(y_test, y_pred, labels=label_ids,
                                       target_names=label_names, zero_division=0)
        print(report)
        print(f"Detection rate {detection:.2%}, false-positive rate {false_pos:.2%}")
        reports.append(f"{name}\n{report}")

        results.append({
            "Model": name,
            "Accuracy %": round(accuracy_score(y_test, y_pred) * 100, 2),
            "Macro F1 %": round(f1_score(y_test, y_pred, average="macro") * 100, 2),
            "CV Macro F1 %": round(cv_f1 * 100, 2),
            "Detection %": round(detection * 100, 2),
            "False Positive %": round(false_pos * 100, 2),
            "Train Time(s)": round(train_time, 2),
            "Pred (us/window)": round(predict_time / len(X_test) * 1e6, 2),
        })

        slug = name.lower().replace(" ", "_")
        cm = confusion_matrix(y_test, y_pred, labels=label_ids)
        disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
        disp.plot(cmap="Blues", values_format="d", colorbar=False)
        disp.ax_.set_title(f"{name} ({how})")
        plt.tight_layout()
        plt.savefig(OUT_REPORTS / "figures" / f"confusion_{slug}.png", dpi=200)
        plt.close()

        # Everything the live detector needs to use the model the same way
        joblib.dump({"model": model, "features": FEATURES, "window": WINDOW_SECONDS,
                     "labels": {i: ALL_LABEL_NAMES[i] for i in label_ids},
                     "train_sessions": sorted(df.loc[train, "session"].unique())},
                    OUT_MODELS / f"{slug}.joblib")

    table = pd.DataFrame(results)
    print(f"\n{'=' * 40}\nMODEL COMPARISON ({how}):")
    print(table.to_string(index=False))
    table.to_csv(OUT_REPORTS / "model_comparison.csv", index=False)
    (OUT_REPORTS / "classification_reports.txt").write_text(f"Split: {how}\n\n" + "\n".join(reports))

    importance = pd.Series(trained["Random Forest"].feature_importances_, index=FEATURES)
    importance = importance.sort_values(ascending=False)
    importance.to_csv(OUT_REPORTS / "feature_importance.csv", header=["importance"])
    print("\nTop features (Random Forest):")
    print(importance.head(10).to_string())
    print(f"\nSaved reports to {OUT_REPORTS} and models to {OUT_MODELS}")


if __name__ == "__main__":
    main()
