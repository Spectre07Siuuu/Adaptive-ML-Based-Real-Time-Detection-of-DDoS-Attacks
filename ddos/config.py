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

# Mininet lab captures (ddos/lab): one folder per session with capture.pcap, events.csv, meta.json
LAB_PCAP_DIR = PCAP_DIR / "lab"

# Per-peer time windows, shared by offline extraction and the live detector
WINDOW_SECONDS = 1.0
