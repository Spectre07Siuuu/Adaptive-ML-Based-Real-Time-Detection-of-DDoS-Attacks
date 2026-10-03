"""Packet sequences for every labelled window, for the deep learning models.

    python -m ddos.features.build_sequences                  # lab + every public dataset
    python -m ddos.features.build_sequences cicids2017

Reads the same captures as build_dataset / build_windows and writes
data/processed/<name>_sequences.npz. Row i of the file is row i of
<name>_windows.csv (labels, rounds and splits come from there), so the
window-feature models and the sequence models see exactly the same windows.
CIC-DDoS2019 takes ~15 minutes (the 29 GB capture is read again).
"""
import argparse
import json

import numpy as np
import pandas as pd

from ddos.config import EXTERNAL_DIR, LAB_PCAP_DIR, PROCESSED_DIR, WINDOW_SECONDS
from ddos.datasets.build_windows import host_packets
from ddos.datasets.catalog import DATASETS
from ddos.features.packets import read_pcap
from ddos.features.sequences import SequenceAggregator, PACKET_FEATURES, SEQ_LEN


def sequences(packets, victim):
    agg = SequenceAggregator([victim], WINDOW_SECONDS)
    rows = []
    for pkt in packets:
        rows.extend(agg.add(pkt))
    rows.extend(agg.flush())
    return rows


def collect(name):
    """All windows' sequences for one dataset, keyed by (session, window_start, peer_ip)."""
    found = {}
    if name == "lab":
        sessions = pd.read_csv(PROCESSED_DIR / "lab_windows.csv", usecols=["session"])["session"].unique()
        for session in sessions:
            meta = json.loads((LAB_PCAP_DIR / session / "meta.json").read_text())
            print(f"  {session}...", flush=True)
            for r in sequences(read_pcap(LAB_PCAP_DIR / session / "capture.pcap"), meta["victim"]):
                found[(session, r["window_start"], r["peer_ip"])] = r
    else:
        ds = DATASETS[name]
        for r in sequences(host_packets([EXTERNAL_DIR / p for p in ds.inputs], ds.victim), ds.victim):
            found[(name, r["window_start"], r["peer_ip"])] = r
    return found


def build(name):
    windows = pd.read_csv(PROCESSED_DIR / f"{name}_windows.csv",
                          usecols=["session", "round", "window_start", "peer_ip", "label"])
    found = collect(name)
    keys = list(zip(windows["session"], windows["window_start"], windows["peer_ip"]))
    missing = sum(k not in found for k in keys)
    if missing:
        raise SystemExit(f"{name}: {missing} labelled windows have no sequence; rebuild the windows first")
    out = PROCESSED_DIR / f"{name}_sequences.npz"
    np.savez_compressed(
        out,
        x=np.stack([found[k]["seq"] for k in keys]).astype(np.float16),
        length=np.array([found[k]["length"] for k in keys], dtype=np.int16),
        label=windows["label"].to_numpy(np.int8),
        features=np.array(PACKET_FEATURES),
    )
    lengths = np.array([found[k]["length"] for k in keys])
    print(f"  {len(keys)} windows -> {out} (median {np.median(lengths):.0f} packets, "
          f"{(lengths == SEQ_LEN).mean():.0%} full)")


def main():
    names = ["lab", *DATASETS]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", nargs="*", help=f"any of {', '.join(names)} (default: all)")
    args = parser.parse_args()
    unknown = set(args.datasets) - set(names)
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    for name in args.datasets or names:
        print(f"{name}:", flush=True)
        build(name)


if __name__ == "__main__":
    main()
