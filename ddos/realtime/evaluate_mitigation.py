"""Score a lab session run with the live detector (capture.py --detector).

    python -m ddos.realtime.evaluate_mitigation m1

For each attack in events.csv: was the attacker blocked while attacking, how
many seconds after the attack started, and how much of its traffic still
reached the victim's switch port. Attack traffic is counted in the session's
own capture of that port, so blocked packets (dropped at the attacker's switch
port) do not appear. Also: blocks of benign clients, and the share of benign
windows the model flagged.

Output: reports/mitigation/<session>_attacks.csv, <session>_summary.csv
"""
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

from ddos.config import MITIGATION_DIR, REPORTS_DIR
from ddos.features.packets import read_pcap

OUT = REPORTS_DIR / "mitigation"
FLOW_SETUP = 0.5   # seconds allowed for a drop flow to take effect


def attack_packets(path, victim, attackers):
    """Packets per (attacker, second) arriving at the victim's port."""
    counts = Counter()
    for pkt in read_pcap(path):
        if pkt.dst == victim and pkt.src in attackers:
            counts[(pkt.src, int(pkt.ts))] += 1
    return counts


def delivered(counts, ip, start, end):
    return sum(c for (src, sec), c in counts.items() if src == ip and start <= sec < end)


def evaluate(session_dir):
    meta = json.loads((session_dir / "meta.json").read_text())
    events = pd.read_csv(session_dir / "events.csv")
    log = [json.loads(line) for line in (session_dir / "detector.jsonl").read_text().splitlines()]
    blocks = pd.DataFrame([e for e in log if e["event"] == "block"], columns=["ts", "peer", "seconds"])
    blocks["until"] = blocks["ts"] + blocks["seconds"]
    attackers = set(meta["attackers"].values())
    counts = attack_packets(session_dir / "capture.pcap", meta["victim"], attackers)

    rows = []
    for ev in events.itertuples():
        mine = blocks[blocks["peer"] == ev.ip]
        active = mine[(mine["ts"] <= ev.start_ts) & (mine["until"] > ev.start_ts)]
        during = mine[(mine["ts"] > ev.start_ts) & (mine["ts"] <= ev.end_ts)]
        duration = ev.end_ts - ev.start_ts
        # Seconds of the attack covered by any block
        covered = sum(max(0.0, min(b.until, ev.end_ts) - max(b.ts, ev.start_ts)) for b in mine.itertuples())
        first = during["ts"].min() if len(during) else np.nan
        got = delivered(counts, ev.ip, int(ev.start_ts), int(ev.end_ts) + 1)
        if len(active):
            status, before_rate = "already blocked", np.nan
        elif len(during):
            status = "blocked"
            before = delivered(counts, ev.ip, int(ev.start_ts), int(first))
            before_rate = before / max(first - ev.start_ts, 1.0)
        else:
            status, before_rate = "missed", got / max(duration, 1.0)
        expected = (before_rate if before_rate == before_rate and before_rate > 0 else ev.pps) * duration
        rows.append({
            "round": ev.round, "attacker": ev.ip, "attack": ev.attack, "rate": ev.rate,
            "duration_s": round(duration, 1), "status": status,
            "time_to_block_s": round(first - ev.start_ts, 2) if status == "blocked" else np.nan,
            "blocked_share_%": round(100 * covered / duration, 1),
            "packets_delivered": got,
            "traffic_stopped_%": round(100 * (1 - min(got / expected, 1.0)), 1) if expected else np.nan,
        })
    attacks = pd.DataFrame(rows)

    clients = set(meta["clients"].values())
    false_blocks = blocks[~blocks["peer"].isin(attackers)]
    scores = pd.read_csv(session_dir / "scores.csv")
    benign = scores[scores["peer_ip"].isin(clients)]
    summary = {
        "session": meta["session"],
        "attacks": len(attacks),
        "blocked while attacking %": round(100 * attacks["status"].isin(["blocked", "already blocked"]).mean(), 1),
        "missed": int((attacks["status"] == "missed").sum()),
        "median time to block (s)": round(attacks["time_to_block_s"].median(), 2),
        "max time to block (s)": round(attacks["time_to_block_s"].max(), 2),
        "attack traffic stopped % (mean)": round(attacks["traffic_stopped_%"].mean(), 1),
        "benign windows flagged %": round(100 * benign["flagged"].mean(), 3) if len(benign) else np.nan,
        "benign clients blocked": int(false_blocks["peer"].nunique()),
        "false blocks": len(false_blocks),
        "detector errors": sum(e["event"] == "error" for e in log),
        "socket kernel drops": next((e.get("kernel_drops") for e in log if e["event"] == "stop"), None),
    }
    return attacks, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("session")
    args = parser.parse_args()
    attacks, summary = evaluate(MITIGATION_DIR / args.session)
    OUT.mkdir(parents=True, exist_ok=True)
    attacks.to_csv(OUT / f"{args.session}_attacks.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT / f"{args.session}_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(attacks.to_string(index=False))
    print()
    for k, v in summary.items():
        print(f"  {k:34s} {v}")
    by_type = attacks.groupby(["attack", "rate"])["time_to_block_s"].median().unstack()
    print("\nMedian time to block (s) by attack and rate:\n" + by_type.round(2).to_string())


if __name__ == "__main__":
    main()
