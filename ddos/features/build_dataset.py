"""Build the labelled window dataset from the Mininet lab sessions.

    python -m ddos.features.build_dataset            # every session in data/raw/pcap/lab
    python -m ddos.features.build_dataset s1 s2

Labels come from events.csv, not from the file name: a window from an attacker
IP that overlaps one of its attacks gets that attack's label, and windows from
every other peer are Normal. Attacker windows outside any attack are dropped.

Each row also gets its `round` (0 = warm-up, n = attack n plus the benign gap
after it), which the training scripts use to split without leakage.
"""
import argparse
import csv
import json

import numpy as np
import pandas as pd

from ddos.config import LAB_PCAP_DIR, PROCESSED_DIR, WINDOW_SECONDS, LABEL_NAMES
from ddos.features.packets import read_pcap
from ddos.features.window import WindowAggregator, FEATURES

GRACE = 1.0   # seconds around each attack: process start-up, packets still in flight

OUTPUT = PROCESSED_DIR / "lab_windows.csv"


def load_session(session_dir):
    meta = json.loads((session_dir / "meta.json").read_text())
    with open(session_dir / "events.csv", newline="") as f:
        events = pd.DataFrame(list(csv.DictReader(f)))
    if not events.empty:
        events = events.astype({"round": int, "label": int, "pps": int,
                                "start_ts": float, "end_ts": float})
    return meta, events


def label_windows(df, events, attackers, window):
    """Add label / attack_rate / round columns; drop attacker windows outside attacks."""
    df["label"] = 0
    df["attack_rate"] = ""
    matched = np.zeros(len(df), dtype=bool)
    for ev in events.itertuples():
        hit = ((df["peer_ip"] == ev.ip)
               & (df["window_start"] + window > ev.start_ts - GRACE)
               & (df["window_start"] < ev.end_ts + GRACE))
        df.loc[hit, "label"] = ev.label
        df.loc[hit, "attack_rate"] = ev.rate
        matched |= hit.to_numpy()

    unmatched = df["peer_ip"].isin(attackers).to_numpy() & ~matched
    df = df[~unmatched].copy()

    if events.empty:
        df["round"] = 0
    else:
        # A round starts with the first window that can carry its attack's label, so
        # no attack window lands in the round before (that would leak across a split)
        starts = events.groupby("round")["start_ts"].min().sort_index()
        reach = df["window_start"].to_numpy() + window + GRACE
        idx = np.searchsorted(starts.to_numpy(), reach, side="left")
        df["round"] = np.where(idx > 0, starts.index.to_numpy()[np.maximum(idx - 1, 0)], 0)
    return df, int(unmatched.sum())


def extract_session(session_dir, window=WINDOW_SECONDS):
    meta, events = load_session(session_dir)
    agg = WindowAggregator([meta["victim"]], window)
    rows = []
    for pkt in read_pcap(session_dir / "capture.pcap"):
        rows.extend(agg.add(pkt))
    rows.extend(agg.flush())

    df = pd.DataFrame(rows, columns=["window_start", "peer_ip", *FEATURES])
    df, unmatched = label_windows(df, events, set(meta["attackers"].values()), window)
    df.insert(0, "session", meta["session"])
    stats = {"late_packets": agg.late, "unmatched_attacker_windows": unmatched,
             "tcpdump": meta.get("tcpdump", {})}
    return df, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sessions", nargs="*", help="session folders (default: all)")
    args = parser.parse_args()

    names = args.sessions or sorted(p.name for p in LAB_PCAP_DIR.iterdir()
                                    if (p / "capture.pcap").exists())
    if not names:
        raise SystemExit(f"No sessions in {LAB_PCAP_DIR}; run ddos.lab.capture first")

    frames = []
    for name in names:
        print(f"Extracting {name}...")
        df, stats = extract_session(LAB_PCAP_DIR / name)
        counts = df["label"].map(LABEL_NAMES).value_counts().to_dict()
        print(f"  {len(df)} windows {counts}")
        print(f"  {stats}")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    cols = ["session", "round", "window_start", "peer_ip", *FEATURES, "attack_rate", "label"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data[cols].to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(data)} windows to {OUTPUT}")
    print(data["label"].map(LABEL_NAMES).value_counts().to_string())


if __name__ == "__main__":
    main()
