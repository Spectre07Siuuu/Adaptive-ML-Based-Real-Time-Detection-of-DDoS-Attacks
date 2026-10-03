import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

from ddos.config import PROCESSED_DIR, MODELS_DIR

# Trains the RF + scaler pair loaded by realtime/predict_server.py.
# Unlike train_models.py, the RF here is trained on SCALED features.
df = pd.read_csv(PROCESSED_DIR / 'final_balanced_dataset.csv')

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

acc = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"Accuracy: {acc*100:.2f}%")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

with open(MODELS_DIR / 'rf_model_new.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=4)

with open(MODELS_DIR / 'scaler_new.pkl', 'wb') as f:
    pickle.dump(scaler, f, protocol=4)

print("Models saved!")
