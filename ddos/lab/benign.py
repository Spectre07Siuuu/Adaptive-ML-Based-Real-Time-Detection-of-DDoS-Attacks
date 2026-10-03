"""Benign traffic from one lab client: web browsing, pings and iperf3 transfers.

Started by ddos.lab.capture inside a Mininet host. Timing is random but seeded.
Benign UDP and ICMP are included so the model cannot treat "any UDP" or "any
ICMP" as an attack.
"""
import argparse
import random
import signal
import subprocess
import sys
import threading
import urllib.request
from subprocess import DEVNULL

from ddos.lab.scenario import HTTP_PORT, WEB_FILES

# Small pages are fetched far more often than large downloads
FILE_WEIGHTS = [40, 30, 15, 10, 5]


def browse(rng, server, stop):
    think = rng.uniform(0.3, 2.0)   # this client's mean pause between requests
    while not stop.is_set():
        name = rng.choices(list(WEB_FILES), FILE_WEIGHTS)[0]
        try:
            with urllib.request.urlopen(f"http://{server}:{HTTP_PORT}/{name}", timeout=5) as r:
                r.read()
        except OSError:
            pass   # the server can be unreachable while it is under attack
        stop.wait(rng.expovariate(1 / think))


def transfers(rng, server, port, stop, children):
    while not stop.wait(rng.uniform(20, 60)):
        seconds = rng.randint(2, 8)
        cmd = ["iperf3", "-c", server, "-p", str(port), "-t", str(seconds)]
        if rng.random() < 0.4:
            cmd += ["-u", "-b", f"{rng.choice([1, 2, 5])}M", "-l", str(rng.choice([128, 512, 1200]))]
        else:
            cmd += ["-b", f"{rng.choice([2, 5, 10, 20])}M"]
        proc = subprocess.Popen(cmd, stdout=DEVNULL, stderr=DEVNULL)
        children.append(proc)
        try:
            proc.wait(timeout=seconds + 15)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--iperf-port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True)
    args = parser.parse_args()

    # capture.py stops us with SIGTERM; exit through the finally block below
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    stop = threading.Event()
    children = []
    try:
        rng = random.Random(args.seed)
        if rng.random() < 0.75:
            children.append(subprocess.Popen(
                ["ping", "-q", "-i", f"{rng.uniform(0.5, 2):.2f}", "-s", str(rng.choice([56, 64, 120])),
                 args.server], stdout=DEVNULL, stderr=DEVNULL))

        threads = [
            threading.Thread(target=browse, args=(random.Random(f"{args.seed}-web"), args.server, stop)),
            threading.Thread(target=transfers, args=(random.Random(f"{args.seed}-iperf"), args.server,
                                                     args.iperf_port, stop, children)),
        ]
        for t in threads:
            t.daemon = True
            t.start()
        stop.wait(args.duration)
    finally:
        stop.set()
        # Mininet does not clean up popen() children, so stop ours explicitly
        for proc in children:
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
