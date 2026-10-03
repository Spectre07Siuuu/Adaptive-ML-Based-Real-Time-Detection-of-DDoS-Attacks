import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
import joblib
import time

from ddos.config import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, LABEL_NAMES

# =========================
# Load Data
# =========================
df = pd.read_csv(PROCESSED_DIR / "final_balanced_dataset.csv")
X  = df.drop("label", axis=1)
y  = df["label"]

# =========================
# Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Scaler
# =========================
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

# =========================
# Models
# =========================
models = {
    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "scaled": False
    },
    "KNN": {
        "model": KNeighborsClassifier(n_neighbors=5),
        "scaled": True
    },
    "SVM": {
        "model": SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),
        "scaled": True
    }
}

results     = []
label_names = list(LABEL_NAMES.values())

for name, config in models.items():
    print(f"\n{'='*40}")
    print(f"Training {name}...")

    model  = config["model"]
    scaled = config["scaled"]

    tr_X = X_train_scaled if scaled else X_train
    te_X = X_test_scaled  if scaled else X_test

    # Train
    start      = time.time()
    model.fit(tr_X, y_train)
    train_time = time.time() - start

    # Predict
    start        = time.time()
    y_pred       = model.predict(te_X)
    predict_time = time.time() - start

    # Metrics
    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall    = recall_score(y_test, y_pred, average="macro")
    cv        = cross_val_score(
                    model,
                    X_train_scaled if scaled else X_train,
                    y_train,
                    cv=5,
                    scoring="accuracy"
                ).mean()

    # Output
    print(f"Accuracy   : {acc*100:.2f}%")
    print(f"CV Score   : {cv*100:.2f}%")
    print(f"Macro F1   : {f1*100:.2f}%")
    print(f"Precision  : {precision*100:.2f}%")
    print(f"Recall     : {recall*100:.2f}%")
    print(f"Train Time : {train_time:.2f}s")
    print(f"Pred Time  : {predict_time*1000:.4f}ms")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save all models
    joblib.dump(model, MODELS_DIR / f"{name.replace(' ','_')}_model.pkl")

    results.append({
        "Model":          name,
        "Accuracy %":     round(acc       * 100, 2),
        "CV Score %":     round(cv        * 100, 2),
        "Macro F1 %":     round(f1        * 100, 2),
        "Precision %":    round(precision * 100, 2),
        "Recall %":       round(recall    * 100, 2),
        "Train Time(s)":  round(train_time,       2),
        "Pred Time(ms)":  round(predict_time*1000, 4)
    })

# =========================
# Feature Importance — RF
# =========================
rf_model    = models["Random Forest"]["model"]
importances = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(f"\n{'='*40}")
print("Feature Importance (Random Forest):")
print(importances)

# =========================
# Comparison Table
# =========================
print(f"\n{'='*40}")
print("MODEL COMPARISON TABLE:")
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# =========================
# Save CSV
# =========================
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
df_results.to_csv(REPORTS_DIR / "model_comparison.csv",   index=False)
importances.to_csv(REPORTS_DIR / "feature_importance.csv", header=True)

print("\nSaved files:")
print("  Random_Forest_model.pkl")
print("  KNN_model.pkl")
print("  SVM_model.pkl")
print("  scaler.pkl")
print("  model_comparison.csv")
print("  feature_importance.csv")
