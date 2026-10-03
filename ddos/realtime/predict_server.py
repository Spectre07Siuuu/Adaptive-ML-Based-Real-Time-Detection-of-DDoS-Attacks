import socket
import pickle
import numpy as np
import json

from ddos.config import MODELS_DIR, LABEL_NAMES

MODEL_PATH = MODELS_DIR / "rf_model_new.pkl"
SCALER_PATH = MODELS_DIR / "scaler_new.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("[SERVER] ML Model loaded! Listening on port 9999...")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('127.0.0.1', 9999))
server.listen(5)

label_map = LABEL_NAMES

while True:
    conn, addr = server.accept()
    data = conn.recv(4096).decode()
    try:
        features = json.loads(data)
        scaled = scaler.transform([features])
        prediction = int(model.predict(scaled)[0])
        result = {"prediction": prediction, "label": label_map[prediction]}
        conn.send(json.dumps(result).encode())
    except Exception as e:
        conn.send(json.dumps({"prediction": 0, "label": "Normal"}).encode())
    conn.close()
