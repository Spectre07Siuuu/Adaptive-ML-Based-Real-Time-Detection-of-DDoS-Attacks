"""Window and flow datasets from the public captures, with the lab's features.

    python -m ddos.datasets.build_windows               # every dataset whose inputs exist
    python -m ddos.datasets.build_windows cicddos2019 --refresh

Two steps. Extract streams the captures (pcap, pcapng or zip, no unpacking) once
through the lab's WindowAggregator and the SDN-style FlowAggregator, and caches
the unlabelled rows next to the capture; this is the slow part (minutes for
CIC-DDoS2019). Label applies catalog.py and writes data/processed/<name>_windows.csv
(same columns as lab_windows.csv) and <name>_flows.csv.
Windows of non-DDoS attacks (label -1) and attacker windows outside every
published attack are dropped. `round` is a 10-minute block, the grouping unit
for within-dataset splits.
"""
import argparse
import socket
from datetime import datetime, timezone

import pandas as pd

from ddos.config import EXTERNAL_DIR, PROCESSED_DIR, WINDOW_SECONDS, ALL_LABEL_NAMES
from ddos.datasets.catalog import DATASETS
from ddos.datasets.slice_pcap import records, sources, to_host
from ddos.features.build_dataset import label_windows
from ddos.features.flows import FlowAggregator, FLOW_FEATURES, FLOW_SECONDS
from ddos.features.packets import parse_frame
from ddos.features.window import WindowAggregator, FEATURES

GRACE = 2.0    # catalog times are measured to the second (see catalog.py)
BLOCK = 600    # seconds per round
COLUMNS = ["session", "round", "window_start", "peer_ip", *FEATURES, "attack_rate", "label"]
FLOW_COLUMNS = ["session", "round", "window_start", "peer_ip", "dst", "proto", *FLOW_FEATURES,
                "attack_rate", "label"]


def host_packets(paths, host):
    """Parsed packets to or from `host`; other frames are skipped before parsing."""
    hosts, seen = {socket.inet_aton(host)}, 0
    for _, stream in sources(paths, None):
        for linktype, sec, usec, _, data in records(stream):
            seen += 1
            if seen % 10_000_000 == 0:
                print(f"  {seen:,} packets read, at {datetime.fromtimestamp(sec, timezone.utc):%H:%M} UTC",
                      flush=True)
            if to_host(data, linktype, hosts):
                pkt = parse_frame(data, sec + usec / 1e6, linktype)
                if pkt is not None:
                    yield pkt


def extract(ds, refresh=False):
    """Unlabelled (windows, flows) for the dataset, from the cache when possible."""
    caches = [EXTERNAL_DIR / ds.name / f"{view}_unlabelled.csv" for view in ("windows", "flows")]
    if all(c.exists() for c in caches) and not refresh:
        return tuple(pd.read_csv(c) for c in caches)
    windows, flows = WindowAggregator([ds.victim], WINDOW_SECONDS), FlowAggregator([ds.victim])
    window_rows, flow_rows = [], []
    for pkt in host_packets([EXTERNAL_DIR / p for p in ds.inputs], ds.victim):
        window_rows.extend(windows.add(pkt))
        flow_rows.extend(flows.add(pkt))
    window_rows.extend(windows.flush())
    flow_rows.extend(flows.flush())
    if windows.late:
        print(f"  {windows.late} packets arrived out of order and were dropped")
    frames = (pd.DataFrame(window_rows, columns=["window_start", "peer_ip", *FEATURES]),
              pd.DataFrame(flow_rows, columns=["window_start", "peer_ip", "dst", "proto", *FLOW_FEATURES]))
    for df, cache in zip(frames, caches):
        df.to_csv(cache, index=False)
    return frames


def label(ds, df, window=WINDOW_SECONDS):
    events = pd.DataFrame([{"round": i, "ip": ds.attacker, "label": a.label, "rate": a.name,
                            "start_ts": ds.epoch(a.start), "end_ts": ds.epoch(a.end)}
                           for i, a in enumerate(ds.attacks, start=1)])
    df, unmatched = label_windows(df.copy(), events, {ds.attacker}, window, grace=GRACE)
    per_attack = df[df["peer_ip"] == ds.attacker].groupby("attack_rate").size().to_dict()
    df = df[df["label"] >= 0].copy()
    df["round"] = ((df["window_start"] - df["window_start"].min()) // BLOCK).astype(int)
    df["attack_rate"] = ""
    df.insert(0, "session", ds.name)
    return df, {"unmatched_attacker_windows": unmatched, "attacker_windows_per_attack": per_attack}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", nargs="*", help=f"any of {', '.join(DATASETS)} (default: all available)")
    parser.add_argument("--refresh", action="store_true", help="re-read the captures, ignore the cache")
    args = parser.parse_args()
    unknown = set(args.datasets) - set(DATASETS)
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    names = args.datasets or [n for n, ds in DATASETS.items()
                              if all((EXTERNAL_DIR / p).exists() for p in ds.inputs)]
    if not names:
        raise SystemExit(f"No captures in {EXTERNAL_DIR}; see README (Phase 1)")

    for name in names:
        ds = DATASETS[name]
        print(f"{name}: extracting windows and flows for {ds.victim}...", flush=True)
        windows, flows = extract(ds, args.refresh)
        for view, raw, window, cols in [("windows", windows, WINDOW_SECONDS, COLUMNS),
                                        ("flows", flows, FLOW_SECONDS, FLOW_COLUMNS)]:
            df, stats = label(ds, raw, window)
            out = PROCESSED_DIR / f"{name}_{view}.csv"
            df[cols].to_csv(out, index=False)
            print(f"  {view}: {len(df)} rows {df['label'].map(ALL_LABEL_NAMES).value_counts().to_dict()}")
            print(f"  {stats}")
            print(f"  saved {out}")


if __name__ == "__main__":
    main()
