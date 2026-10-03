from pathlib import Path
import os

# Project root = the folder that contains the `ddos` package
ROOT = Path(__file__).resolve().parent.parent

# Set DDOS_DATA_DIR to keep the large pcaps outside OneDrive (or on /mnt/e/... in WSL)
DATA_DIR      = Path(os.environ.get("DDOS_DATA_DIR", ROOT / "data"))
PCAP_DIR      = DATA_DIR / "raw" / "pcap"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"

LABEL_NAMES = {0: "Normal", 1: "ICMP Flood", 2: "UDP Flood", 3: "SYN Flood"}
# Attack types that occur only in the public datasets (ddos/datasets/catalog.py)
ALL_LABEL_NAMES = {**LABEL_NAMES, 4: "HTTP Flood", 5: "Reflection",
                   6: "ACK Flood"}   # 6: lab only, never in any training data (phase 5 zero-day test)

# Mininet lab captures (ddos/lab): one folder per session with capture.pcap, events.csv, meta.json
LAB_PCAP_DIR = PCAP_DIR / "lab"

# Lab sessions run with the live detector (phase 3); kept apart from the training captures
MITIGATION_DIR = PCAP_DIR / "mitigation"

# Model bundle loaded by the live detector (ddos/training/train_detector.py)
DETECTOR_MODEL = MODELS_DIR / "detector.joblib"

# Public datasets (ddos/datasets): one folder per dataset, sliced to the victim's traffic
EXTERNAL_DIR = DATA_DIR / "raw" / "external"

# Per-peer time windows, shared by offline extraction and the live detector
WINDOW_SECONDS = 1.0
