"""Live dashboard for the detector: who is sending what, who is flagged, who is blocked.

    .venv/bin/python -m ddos.realtime.dashboard --log /tmp/demo.jsonl --scores /tmp/demo_scores.csv
    .venv/bin/python -m ddos.realtime.dashboard --session m3              # a live --detector session
    .venv/bin/python -m ddos.realtime.dashboard --session m3 --replay    # replay it at real speed

then open http://127.0.0.1:8050 in a browser.

It only reads the detector's two files (detector.jsonl, scores.csv) as they
grow, so it needs no root and cannot disturb the detector. With --session it
also reads the session's events.csv and shades the real attack periods.
--replay plays a finished session back from its first window (or --skip
seconds in), --speed times real time, so a recorded run can be shown without
Mininet.

Stdlib only; the page draws its own charts, so it works without internet.
"""
import argparse
import bisect
import csv
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from ddos.config import MITIGATION_DIR

PAGE = Path(__file__).with_name("dashboard.html")
HISTORY = 120          # seconds of traffic shown
EVENT_KINDS = {"start", "calibrated", "retrained", "block", "error"}


class Tail:
    """Lines appended to a file since the last call (a partial last line waits)."""

    def __init__(self, path):
        self.path, self.offset, self.rest = Path(path), 0, ""

    def lines(self):
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            if self.path.stat().st_size < self.offset:   # file was replaced: start over
                self.offset, self.rest = 0, ""
            f.seek(self.offset)
            chunk = f.read()
            self.offset = f.tell()
        text = self.rest + chunk
        lines = text.split("\n")
        self.rest = lines.pop()
        return [line for line in lines if line]


class State:
    def __init__(self, log, scores, events=None, replay=False, speed=1.0):
        self.log, self.scores = Tail(log), Tail(scores)
        self.header = None
        self.starts, self.rows = [], []      # rows in window order, starts for bisect
        self.events, self.settings = [], {}
        self.attacks = self._attacks(events)
        self.replay, self.speed, self.opened = replay, speed, time.time()
        self.lock = Lock()

    @staticmethod
    def _attacks(path):
        if not path or not Path(path).exists():
            return []
        with open(path, newline="") as f:
            return [{"ip": r["ip"], "attack": r["attack"], "rate": r["rate"],
                     "start": float(r["start_ts"]), "end": float(r["end_ts"])} for r in csv.DictReader(f)]

    def _read(self):
        for line in self.scores.lines():
            if line.startswith("window_start"):
                self.header = line.split(",")
                continue
            values = dict(zip(self.header or [], line.split(",")))
            if "window_start" not in values:
                continue
            start = float(values["window_start"])
            row = {"t": start, "ip": values["peer_ip"], "n": int(float(values["n_in"])),
                   "score": float(values["score"]), "novel": int(values.get("novel", 0)),
                   "flagged": int(values["flagged"]), "version": int(values.get("model_version", 0))}
            i = bisect.bisect_right(self.starts, start)   # keep order if a row arrives late
            self.starts.insert(i, start)
            self.rows.insert(i, row)
        for line in self.log.lines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "start":
                self.settings = {k: v for k, v in event.items() if k not in ("event", "ts")}
            if event.get("event") in EVENT_KINDS and event.get("ts") is not None:
                self.events.append(event)

    def _now(self):
        if not self.replay:
            return time.time()
        if not self.starts:
            return 0.0
        first = self.starts[0] - 1
        return min(first + (time.time() - self.opened) * self.speed, self.starts[-1] + 2)

    def snapshot(self):
        with self.lock:
            self._read()
            now = self._now()
            lo = bisect.bisect_left(self.starts, now - HISTORY - 1)
            hi = bisect.bisect_right(self.starts, now - 1)
            recent = self.rows[lo:hi]
            past = self.rows[:hi]
            events = [e for e in self.events if e["ts"] <= now]
            blocks = [e for e in events if e["event"] == "block"]

        peers = {}
        for r in recent:   # time order: the last row per source is its current state
            series = peers[r["ip"]]["series"] if r["ip"] in peers else []
            series.append([r["t"], r["n"], r["flagged"]])
            peers[r["ip"]] = {**r, "series": series}
        blocked = {}
        for b in blocks:
            blocked[b["peer"]] = max(blocked.get(b["peer"], 0), b["ts"] + b.get("seconds", 0))
        for ip, until in blocked.items():
            if until > now and ip not in peers:
                peers[ip] = {"ip": ip, "t": None, "n": 0, "score": None, "novel": 0, "flagged": 0, "series": []}
        for ip, p in peers.items():
            p["blocked_for"] = max(0.0, blocked.get(ip, 0) - now)
            p["blocks"] = sum(b["peer"] == ip for b in blocks)

        versions = [e["version"] for e in events if e["event"] == "retrained"]
        return {
            "now": now,
            "history": HISTORY,
            "replay": self.replay,
            "speed": self.speed,
            "finished": self.replay and bool(self.starts) and now >= self.starts[-1] + 2,
            "settings": self.settings,
            "peers": sorted(peers.values(), key=lambda p: p["ip"]),
            # an error's "windows" is a list of unscored windows, too long to ship; a block's is a count
            "events": [{k: v for k, v in e.items() if not (k == "windows" and isinstance(v, list))}
                       for e in events[-40:]][::-1],
            "attacks": [a for a in self.attacks if a["end"] >= now - HISTORY and a["start"] <= now],
            "totals": {
                "windows": len(past),
                "blocks": len(blocks),
                "active_blocks": sum(until > now for until in blocked.values()),
                "sources": len({r["ip"] for r in recent}),
                "model_version": versions[-1] if versions else 0,
                "calibrated": any(e["event"] == "calibrated" for e in events),
                "errors": sum(e["event"] == "error" for e in events),
            },
        }


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/state"):
                body = json.dumps(state.snapshot()).encode()
                kind = "application/json"
            elif self.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                kind = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass   # one request a second; keep the terminal quiet

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--session", help="a detector session in data/raw/pcap/mitigation/")
    parser.add_argument("--log", help="the detector's --log file")
    parser.add_argument("--scores", help="the detector's --scores file")
    parser.add_argument("--replay", action="store_true", help="play a finished session back")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed, times real time")
    parser.add_argument("--skip", type=float, default=0, help="replay: start this many seconds in")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    events = None
    if args.session:
        folder = MITIGATION_DIR / args.session
        args.log, args.scores, events = folder / "detector.jsonl", folder / "scores.csv", folder / "events.csv"
    if not args.log or not args.scores:
        parser.error("give --session, or --log and --scores")
    state = State(args.log, args.scores, events, args.replay, args.speed)
    state.opened -= args.skip / args.speed
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Dashboard on http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
