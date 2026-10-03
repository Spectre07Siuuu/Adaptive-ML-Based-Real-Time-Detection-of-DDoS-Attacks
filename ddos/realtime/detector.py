"""Live DDoS detection with automatic mitigation (phase 3).

    sudo .venv/bin/python -m ddos.realtime.detector --iface s1-eth1 --victim 10.0.0.100

Sniffs the protected server's switch port, builds the same per-peer 1 s windows
the model was trained on (WindowAggregator), and scores every window as soon as
it closes. A peer flagged in `--consecutive` windows in a row gets an OVS drop
flow on its source address. The flow carries a hard timeout, so OVS lifts the
block by itself; a peer that is still attacking is caught and blocked again.

Nothing fails open: a window that cannot be scored is logged as an error, never
reported as normal. --replay feeds a capture instead of a live port (timestamps
from the file) and --dry-run logs the blocks without calling ovs-ofctl; together
they check a model on a recorded session without root or Mininet. Logs: --log (JSON lines: start, block, error, stop with
socket statistics) and --scores (CSV, one row per scored window).
"""
import argparse
import json
import signal
import socket
import struct
import subprocess
import time
from collections import defaultdict
from types import SimpleNamespace

import joblib
import numpy as np

from ddos.config import DETECTOR_MODEL
from ddos.features.packets import parse_frame, read_pcap, DLT_EN10MB
from ddos.features.window import WindowAggregator

ETH_P_ALL = 0x0003
SOL_PACKET, PACKET_STATISTICS = 263, 6
COOKIE = "0xdd05"   # marks the detector's flows, so they can be listed and removed


class Mitigator:
    """Decides when a peer is blocked, and blocks it with ovs-ofctl."""

    def __init__(self, switch, consecutive=2, block_seconds=60, whitelist=(),
                 run=subprocess.run, log=lambda **event: None):
        self.switch, self.consecutive, self.block_seconds = switch, consecutive, block_seconds
        self.whitelist = set(whitelist)
        self.run, self.log = run, log
        self.streak = defaultdict(int)
        self.blocked_until = {}

    def update(self, now, peer, flagged, score):
        """Record one scored window; return True if the peer was blocked now."""
        if not flagged or self.blocked_until.get(peer, 0) > now:
            self.streak[peer] = 0   # after a block lifts, the peer needs a fresh streak
            return False
        self.streak[peer] += 1
        if peer in self.whitelist or self.streak[peer] < self.consecutive:
            return False
        flow = (f"cookie={COOKIE},priority=100,ip,nw_src={peer},"
                f"hard_timeout={self.block_seconds},actions=drop")
        result = self.run(["ovs-ofctl", "add-flow", self.switch, flow], capture_output=True, text=True)
        if result.returncode != 0:
            self.log(event="error", ts=now, peer=peer, error=f"ovs-ofctl: {result.stderr.strip()}")
            return False
        self.blocked_until[peer] = now + self.block_seconds
        self.log(event="block", ts=now, peer=peer, score=round(score, 4), windows=self.streak[peer],
                 seconds=self.block_seconds)
        self.streak[peer] = 0
        return True

    def clear(self):
        self.run(["ovs-ofctl", "del-flows", self.switch, f"cookie={COOKIE}/-1"], capture_output=True)


class Detector:
    def __init__(self, bundle, victim, mitigator, threshold, log, scores):
        self.model, self.features = bundle["model"], bundle["features"]
        self.agg = WindowAggregator([victim], bundle["window"])
        self.mitigator, self.threshold, self.log, self.scores = mitigator, threshold, log, scores

    def packet(self, buf, now):
        pkt = parse_frame(buf, now, DLT_EN10MB)
        self.score(self.agg.add(pkt) if pkt else self.agg.advance(now), now)

    def tick(self, now):
        self.score(self.agg.advance(now), now)

    def score(self, rows, now):
        if not rows:
            return
        try:
            x = np.array([[r[f] for f in self.features] for r in rows], dtype=np.float32)
            probs = self.model.predict_proba(x)[:, 1]
        except Exception as e:   # keep sniffing; the window is reported as unscored
            self.log(event="error", ts=now, error=f"scoring failed: {e!r}",
                     windows=[(r["window_start"], r["peer_ip"]) for r in rows])
            return
        for r, p in zip(rows, probs):
            flagged = p >= self.threshold
            self.scores.write(f"{r['window_start']:.0f},{r['peer_ip']},{r['n_in']},{p:.4f},{int(flagged)}\n")
            self.mitigator.update(now, r["peer_ip"], flagged, float(p))
        self.scores.flush()


def socket_stats(sock):
    packets, drops = struct.unpack("II", sock.getsockopt(SOL_PACKET, PACKET_STATISTICS, 8))
    return {"packets": packets, "kernel_drops": drops}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--iface", required=True, help="switch port facing the protected server")
    parser.add_argument("--victim", required=True, help="IPv4 address of the protected server")
    parser.add_argument("--switch", default="s1")
    parser.add_argument("--model", default=str(DETECTOR_MODEL))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--consecutive", type=int, default=2, help="flagged windows in a row before blocking")
    parser.add_argument("--block-seconds", type=int, default=60)
    parser.add_argument("--whitelist", nargs="*", default=[], help="addresses never blocked")
    parser.add_argument("--log", default="detector.jsonl")
    parser.add_argument("--scores", default="scores.csv")
    parser.add_argument("--keep-flows", action="store_true", help="leave drop flows in place on exit")
    parser.add_argument("--replay", help="read this capture instead of sniffing --iface")
    parser.add_argument("--dry-run", action="store_true", help="log blocks without calling ovs-ofctl")
    args = parser.parse_args()

    log_file = open(args.log, "a")

    def log(**event):
        log_file.write(json.dumps(event) + "\n")
        log_file.flush()

    bundle = joblib.load(args.model)
    run = (lambda cmd, **kw: SimpleNamespace(returncode=0, stderr="")) if args.dry_run else subprocess.run
    mitigator = Mitigator(args.switch, args.consecutive, args.block_seconds,
                          [args.victim, *args.whitelist], run=run, log=log)
    scores = open(args.scores, "w")
    scores.write("window_start,peer_ip,n_in,score,flagged\n")
    detector = Detector(bundle, args.victim, mitigator, args.threshold, log, scores)

    if args.replay:
        log(event="start", ts=None, replay=args.replay, victim=args.victim, threshold=args.threshold,
            consecutive=args.consecutive, block_seconds=args.block_seconds)
        last = None
        for pkt in read_pcap(args.replay):
            detector.score(detector.agg.add(pkt), pkt.ts)
            last = pkt.ts
        detector.score(detector.agg.flush(), last)
        log(event="stop", ts=last)
        return

    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(ETH_P_ALL))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
    sock.bind((args.iface, 0))
    sock.settimeout(0.2)

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log(event="start", ts=time.time(), iface=args.iface, victim=args.victim,
        threshold=args.threshold, consecutive=args.consecutive, block_seconds=args.block_seconds)
    print(f"Watching {args.iface} for {args.victim}", flush=True)
    while running:
        try:
            buf = sock.recv(65535)
            detector.packet(buf, time.time())
        except socket.timeout:
            detector.tick(time.time())
        except InterruptedError:
            continue
    detector.score(detector.agg.flush(), time.time())
    if not args.keep_flows:
        mitigator.clear()
    log(event="stop", ts=time.time(), **socket_stats(sock))


if __name__ == "__main__":
    main()
