"""Live DDoS detection with automatic mitigation (phases 3 and 5).

    sudo .venv/bin/python -m ddos.realtime.detector --iface s1-eth1 --victim 10.0.0.100
    sudo .venv/bin/python -m ddos.realtime.detector --config deploy/detector.toml

Sniffs the protected server's switch port, builds the same per-peer 1 s windows
the model was trained on (WindowAggregator), and scores every window as soon as
it closes. A peer flagged in `--consecutive` windows in a row gets an OVS drop
flow on its source address. The flow carries a hard timeout, so OVS lifts the
block by itself; a peer that is still attacking is caught and blocked again.

--mode hybrid adds the phase 4 novelty detector: for the first --calibrate
seconds the detector learns this network's normal traffic from the windows the
supervised model calls benign, then also flags windows unlike any of them.
--adaptive also retrains the supervised model every --retrain-seconds on its
training data plus pseudo-labels from the novelty detector, in a background
thread; the new model replaces the old one when it is ready.

Nothing fails open: a window that cannot be scored is logged as an error, never
reported as normal. --replay feeds a capture instead of a live port (timestamps
from the file) and --dry-run logs the blocks without calling ovs-ofctl; together
they check a model on a recorded session without root or Mininet.

Logs: --log (JSON lines: start, calibrated, retrained, block, error, stats every
minute, stop) and --scores (CSV, one row per scored window). Every option can
also be set in a TOML file given with --config (same names, '-' -> '_').
"""
import argparse
import json
import signal
import socket
import struct
import subprocess
import threading
import time
import tomllib
from collections import defaultdict
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from ddos.adaptive.novelty import KnnNovelty
from ddos.adaptive.online import PseudoLabeler, fit_supervised
from ddos.config import DETECTOR_MODEL
from ddos.features.packets import parse_frame, read_pcap, DLT_EN10MB
from ddos.features.window import WindowAggregator

ETH_P_ALL = 0x0003
SOL_PACKET, PACKET_STATISTICS = 263, 6
COOKIE = "0xdd05"        # marks the detector's flows, so they can be listed and removed
MIN_CALIBRATION = 30     # benign windows needed before the novelty detector is fitted
STATS_EVERY = 60


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

    def active(self, now):
        return sum(until > now for until in self.blocked_until.values())

    def clear(self):
        self.run(["ovs-ofctl", "del-flows", self.switch, f"cookie={COOKIE}/-1"], capture_output=True)


class Detector:
    def __init__(self, bundle, victim, mitigator, threshold, log, scores, mode="supervised",
                 calibrate=300.0, adaptive=False, retrain_seconds=120.0, background=True):
        self.model, self.features = bundle["model"], bundle["features"]
        self.window = bundle["window"]
        self.agg = WindowAggregator([victim], self.window)
        self.mitigator, self.threshold, self.log, self.scores = mitigator, threshold, log, scores
        self.hybrid, self.adaptive = mode == "hybrid", adaptive and mode == "hybrid"
        self.calibrate_until, self.calibration = None, []
        self.calibrate_seconds = calibrate
        self.novelty = None
        self.base = (bundle.get("base_x"), bundle.get("base_y"))
        if self.adaptive and self.base[0] is None:
            raise SystemExit("--adaptive needs a model bundle with training data; rerun train_detector")
        self.labeler = PseudoLabeler(self.window)
        self.retrain_seconds, self.next_retrain, self.training = retrain_seconds, None, None
        self.background, self.version = background, 0
        self.counts = defaultdict(int)

    def packet(self, buf, now):
        pkt = parse_frame(buf, now, DLT_EN10MB)
        self.counts["packets"] += 1
        self.score(self.agg.add(pkt) if pkt else self.agg.advance(now), now)

    def tick(self, now):
        self.score(self.agg.advance(now), now)

    def score(self, rows, now):
        if not rows:
            return
        try:
            frame = pd.DataFrame(rows)
            x = frame[self.features].to_numpy(np.float32)
            probs = self.model.predict_proba(x)[:, 1]
            sup = probs >= self.threshold
            nov = self.novelty.flag(frame) if self.novelty is not None else np.zeros(len(rows), bool)
        except Exception as e:   # keep sniffing; the window is reported as unscored
            self.log(event="error", ts=now, error=f"scoring failed: {e!r}",
                     windows=[(r["window_start"], r["peer_ip"]) for r in rows])
            return
        self._calibrate(frame[~sup], now)
        for r, xi, p, s, n in zip(rows, x, probs, sup, nov):
            flagged = bool(s or (self.hybrid and n))
            self.scores.write(f"{r['window_start']:.0f},{r['peer_ip']},{r['n_in']},{p:.4f},"
                              f"{int(n)},{int(flagged)},{self.version}\n")
            self.mitigator.update(now, r["peer_ip"], flagged, float(p))
            if self.adaptive and self.novelty is not None:
                self.labeler.observe(xi, bool(n), r["peer_ip"], r["window_start"])
            self.counts["windows"] += 1
            self.counts["flagged"] += flagged
        self.scores.flush()
        self._maybe_retrain(now)

    def _calibrate(self, benign, now):
        """Collect windows the supervised model calls benign; fit the novelty detector once."""
        if not self.hybrid or self.novelty is not None:
            return
        if self.calibrate_until is None:
            self.calibrate_until = now + self.calibrate_seconds
        self.calibration.append(benign)
        windows = sum(len(f) for f in self.calibration)
        if now >= self.calibrate_until and windows >= MIN_CALIBRATION:
            self.novelty = KnnNovelty(self.features).fit(pd.concat(self.calibration, ignore_index=True))
            self.calibration = []
            self.next_retrain = now + self.retrain_seconds
            self.log(event="calibrated", ts=now, windows=windows, threshold=round(self.novelty.threshold, 4))

    def _maybe_retrain(self, now):
        if not self.adaptive or self.next_retrain is None or now < self.next_retrain:
            return
        if self.training is not None and self.training.is_alive():
            return   # the previous retrain is still running
        self.next_retrain = now + self.retrain_seconds
        data = self.labeler.training_set(*self.base)   # snapshot, taken in this thread
        if data is None:
            return
        attack = len(self.labeler.attack)

        def fit():
            started = time.time()
            model = fit_supervised(*data)
            self.model, self.version = model, self.version + 1   # reference swap
            self.log(event="retrained", ts=now, version=self.version, rows=len(data[1]),
                     pseudo_attack=attack, seconds=round(time.time() - started, 2))

        if self.background:
            self.training = threading.Thread(target=fit, daemon=True)
            self.training.start()
        else:
            fit()

    def stats(self, now):
        stats = dict(self.counts, blocks_active=self.mitigator.active(now), model_version=self.version,
                     calibrated=self.novelty is not None)
        self.counts = defaultdict(int)
        return stats


def socket_stats(sock):
    """Packets the socket received and dropped since the last call (the kernel resets them)."""
    packets, drops = struct.unpack("II", sock.getsockopt(SOL_PACKET, PACKET_STATISTICS, 8))
    return {"socket_packets": packets, "kernel_drops": drops}


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], parents=[pre])
    parser.add_argument("--iface", help="switch port facing the protected server")
    parser.add_argument("--victim", help="IPv4 address of the protected server")
    parser.add_argument("--switch", default="s1")
    parser.add_argument("--model", default=str(DETECTOR_MODEL))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--consecutive", type=int, default=2, help="flagged windows in a row before blocking")
    parser.add_argument("--block-seconds", type=int, default=60)
    parser.add_argument("--whitelist", nargs="*", default=[], help="addresses never blocked")
    parser.add_argument("--mode", choices=["supervised", "hybrid"], default="supervised")
    parser.add_argument("--calibrate", type=float, default=300, help="seconds of traffic that define normal")
    parser.add_argument("--adaptive", action="store_true", help="retrain on pseudo-labels (hybrid mode)")
    parser.add_argument("--retrain-seconds", type=float, default=120)
    parser.add_argument("--log", default="detector.jsonl")
    parser.add_argument("--scores", default="scores.csv")
    parser.add_argument("--keep-flows", action="store_true", help="leave drop flows in place on exit")
    parser.add_argument("--replay", help="read this capture instead of sniffing --iface")
    parser.add_argument("--dry-run", action="store_true", help="log blocks without calling ovs-ofctl")
    if known.config:
        with open(known.config, "rb") as f:
            parser.set_defaults(**tomllib.load(f))
    args = parser.parse_args(argv)
    if not args.victim or not (args.iface or args.replay):
        parser.error("--victim and --iface (or --replay) are required, on the command line or in --config")
    return args


def main(argv=None):
    args = parse_args(argv)
    log_file = open(args.log, "a")

    def log(**event):
        log_file.write(json.dumps(event) + "\n")
        log_file.flush()

    bundle = joblib.load(args.model)
    run = (lambda cmd, **kw: SimpleNamespace(returncode=0, stderr="")) if args.dry_run else subprocess.run
    mitigator = Mitigator(args.switch, args.consecutive, args.block_seconds,
                          [args.victim, *args.whitelist], run=run, log=log)
    scores = open(args.scores, "w")
    scores.write("window_start,peer_ip,n_in,score,novel,flagged,model_version\n")
    detector = Detector(bundle, args.victim, mitigator, args.threshold, log, scores, args.mode,
                        args.calibrate, args.adaptive, args.retrain_seconds,
                        background=not args.replay)   # a replay retrains in line, in stream time
    settings = {k: getattr(args, k) for k in ("threshold", "consecutive", "block_seconds", "mode",
                                               "calibrate", "adaptive", "retrain_seconds")}

    if args.replay:
        log(event="start", ts=None, replay=args.replay, victim=args.victim, **settings)
        last = None
        for pkt in read_pcap(args.replay):
            detector.score(detector.agg.add(pkt), pkt.ts)
            last = pkt.ts
        detector.score(detector.agg.flush(), last)
        log(event="stop", ts=last, **detector.stats(last))
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
    log(event="start", ts=time.time(), iface=args.iface, victim=args.victim, **settings)
    print(f"Watching {args.iface} for {args.victim} ({args.mode}{', adaptive' if args.adaptive else ''})",
          flush=True)
    next_stats = time.time() + STATS_EVERY
    while running:
        try:
            buf = sock.recv(65535)
            detector.packet(buf, time.time())
        except socket.timeout:
            detector.tick(time.time())
        except InterruptedError:
            continue
        if time.time() >= next_stats:
            log(event="stats", ts=time.time(), **detector.stats(time.time()), **socket_stats(sock))
            next_stats += STATS_EVERY
    detector.score(detector.agg.flush(), time.time())
    if not args.keep_flows:
        mitigator.clear()
    log(event="stop", ts=time.time(), **detector.stats(time.time()), **socket_stats(sock))


if __name__ == "__main__":
    main()
