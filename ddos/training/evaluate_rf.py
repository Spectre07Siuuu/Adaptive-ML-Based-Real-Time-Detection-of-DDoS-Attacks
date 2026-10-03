import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

from ddos.config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, LABEL_NAMES

# =========================
# Load Dataset
# =========================
df = pd.read_csv(PROCESSED_DIR / 'final_balanced_dataset.csv')

X = df.drop('label', axis=1)
y = df['label']

# =========================
# Load Model
# =========================
# train_models.py saves with joblib and fits the RF on UNSCALED features,
# so load with joblib and do not apply scaler.pkl here.
rf = joblib.load(MODELS_DIR / 'Random_Forest_model.pkl')

# =========================
# Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Prediction
# =========================
y_pred = rf.predict(X_test)

# =========================
# Classification Report
# =========================
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=list(LABEL_NAMES.values())
    )
)

# =========================
# Confusion Matrix
# =========================
classes = [
    'Normal',
    'ICMP',
    'UDP',
    'SYN'
]

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(cm, cmap='Blues')

ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))

ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

ax.set_title("Random Forest Confusion Matrix")

plt.colorbar(im)

for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(
            j,
            i,
            cm[i, j],
            ha='center',
            va='center',
            color='white' if cm[i, j] > cm.max()/2 else 'black'
        )

plt.tight_layout()

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    FIGURES_DIR / "confusion_matrix_rf.png",
    dpi=300
)

print("\nConfusion Matrix Saved:")
print(FIGURES_DIR / "confusion_matrix_rf.png")

# =========================
# ROC Curve
# =========================
y_test_bin = label_binarize(
    y_test,
    classes=[0, 1, 2, 3]
)

y_score = rf.predict_proba(X_test)

colors = [
    'blue',
    'red',
    'green',
    'orange'
]

fig, ax = plt.subplots(figsize=(8, 6))

for i, cls in enumerate(classes):

    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i],
        y_score[:, i]
    )

    roc_auc = auc(fpr, tpr)

    ax.plot(
        fpr,
        tpr,
        color=colors[i],
        linewidth=2,
        label=f"{cls} (AUC = {roc_auc:.4f})"
    )

ax.plot(
    [0, 1],
    [0, 1],
    'k--',
    linewidth=1
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")

ax.set_title("Random Forest ROC Curve")

ax.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "roc_curve_rf.png",
    dpi=300
)

print("\nROC Curve Saved:")
print(FIGURES_DIR / "roc_curve_rf.png")

print("\nDone!")
