"""Run one labelled capture session in the Mininet lab.

    sudo python3 -m ddos.lab.capture --session s1 --seed 1
    python3 -m ddos.lab.capture --session s1 --seed 1 --dry-run   # print the schedule only

Benign clients run for the whole session. After a benign-only warm-up, attackers
follow the seeded schedule from scenario.py. The real start and end time of every
attack goes to events.csv, so build_dataset labels windows by (source IP, time)
rather than by file, which is what broke the v1 "normal" capture.

Output: data/raw/pcap/lab/<session>/{capture.pcap, events.csv, meta.json, logs/}
If a run crashes, clean up Mininet with `sudo mn -c`.

With --detector the live detector (ddos.realtime.detector, run with the project
venv) watches the victim's port and blocks attackers through OVS during the
session. Those sessions go to data/raw/pcap/mitigation/<session>/, with the
detector's detector.jsonl and scores.csv, and never mix with the training data.
"""
import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from subprocess import DEVNULL

from ddos.config import ROOT, LAB_PCAP_DIR, MITIGATION_DIR
from ddos.lab.scenario import (VICTIM, CLIENTS, ATTACKERS, HTTP_PORT, IPERF_BASE_PORT,
                               WEB_FILES, LABELS, ALL_ATTACKS, build_schedule)

TOOLS = ["mn", "ovs-ofctl", "tcpdump", "hping3", "iperf3"]
WARMUP = 60     # benign-only seconds before the first attack
SNAPLEN = 96    # headers only; Ethernet + IPv4 + TCP with options fit in 94 bytes


def stop(proc, sig=signal.SIGTERM, timeout=5):
    if proc.poll() is None:
        proc.send_signal(sig)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def sleep_until(t):
    time.sleep(max(0.0, t - time.time()))


def make_webroot():
    root = Path(tempfile.mkdtemp(prefix="ddos-web-"))
    for name, size in WEB_FILES.items():
        (root / name).write_bytes(os.urandom(size))
    return root


def run_attacks(net, events, start, logs, procs, times):
    """Start and stop hping3 on the attackers on schedule, recording the real times.

    `procs` and `times` are filled in place so the caller can clean up after Ctrl+C.
    """
    actions = sorted([(start + e.offset, 1, i) for i, e in enumerate(events)] +
                     [(start + e.offset + e.duration, 0, i) for i, e in enumerate(events)])
    for when, is_start, i in actions:
        sleep_until(when)
        ev = events[i]
        if is_start:
            # stdout dropped: hping3 prints every ICMP reply even with -q (~17 MB per attack)
            with open(logs / f"round{ev.round:02d}_{ev.attacker}.log", "w") as log:
                procs[i] = net[ev.attacker].popen(ev.command(VICTIM[1]), stdout=DEVNULL, stderr=log)
            times[i] = [time.time(), None]
            print(f"  [{datetime.now():%H:%M:%S}] round {ev.round:2d}: {ev.attacker} "
                  f"{ev.attack} {ev.rate} ({ev.pps} pps)")
        else:
            stop(procs[i])
            times[i][1] = time.time()


def tcpdump_stats(stderr_text):
    stats = {}
    for key in ("captured", "received by filter", "dropped by kernel"):
        m = re.search(rf"(\d+) packets? {key}", stderr_text)
        if m:
            stats[key.replace(" ", "_")] = int(m.group(1))
    return stats


def start_detector(out, logs, iface, args):
    """Run the live detector in the root namespace; wait until it is sniffing."""
    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        sys.exit(f"--detector needs the project venv ({python}); see README Setup")
    log = out / "detector.jsonl"
    log.unlink(missing_ok=True)
    with open(logs / "detector.log", "w") as console:
        proc = subprocess.Popen([str(python), "-m", "ddos.realtime.detector", "--iface", iface,
                                 "--victim", VICTIM[1], "--switch", "s1",
                                 "--block-seconds", str(args.block_seconds), "--mode", args.detector_mode,
                                 "--calibrate", str(WARMUP - 15),   # learn normal before the first attack
                                 *(["--adaptive"] if args.adaptive else []),
                                 "--log", str(log), "--scores", str(out / "scores.csv")],
                                cwd=str(ROOT), stdout=console, stderr=console)
    deadline = time.time() + 30
    while time.time() < deadline and proc.poll() is None:
        if log.exists() and '"start"' in log.read_text():
            return proc
        time.sleep(0.5)
    stop(proc)
    sys.exit(f"The detector did not start; see {logs / 'detector.log'}")


def write_outputs(out, args, events, times, t0, t_end, dump_stats):
    with open(out / "events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "round", "attacker", "ip", "attack", "label", "rate", "pps",
                    "size", "port", "start_ts", "end_ts"])
        for i, e in enumerate(events):
            if i in times:
                w.writerow([args.session, e.round, e.attacker, e.ip, e.attack, ALL_ATTACKS[e.attack],
                            e.rate, e.pps, e.size, e.port, *times[i]])
    meta = {
        "session": args.session, "seed": args.seed, "repeats": args.repeats, "bw_mbps": args.bw,
        "victim": VICTIM[1], "clients": dict(CLIENTS), "attackers": dict(ATTACKERS),
        "interface": "s1-eth1", "snaplen": SNAPLEN, "warmup_s": WARMUP,
        "start_ts": t0, "end_ts": t_end, "tcpdump": dump_stats,
        "detector": {"block_seconds": args.block_seconds, "mode": args.detector_mode,
                     "adaptive": args.adaptive} if args.detector else None,
        "attacks": args.attacks,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def give_back_to_user(out):
    """Files written under sudo belong to root; hand them to the invoking user."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid is None:
        return
    # Include lab/ itself (created by the first run), or the user cannot delete sessions
    for path in [out.parent, out, *out.rglob("*")]:
        try:
            os.chown(path, int(uid), int(gid))
        except OSError:
            pass   # some mounts (e.g. NTFS with uid=) do not support chown


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--session", required=True, help="output folder name, e.g. s1")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3, help="times each (attack, rate) pair runs")
    parser.add_argument("--bw", type=int, default=100, help="link bandwidth in Mbit/s")
    parser.add_argument("--dry-run", action="store_true", help="print the schedule and exit")
    parser.add_argument("--force", action="store_true", help="overwrite an existing session")
    parser.add_argument("--detector", action="store_true", help="run the live detector with mitigation")
    parser.add_argument("--block-seconds", type=int, default=30, help="how long a detected peer stays blocked")
    parser.add_argument("--detector-mode", choices=["supervised", "hybrid"], default="supervised")
    parser.add_argument("--adaptive", action="store_true", help="detector retrains on pseudo-labels (hybrid)")
    parser.add_argument("--attacks", nargs="+", choices=list(ALL_ATTACKS), default=list(LABELS),
                        help="attack types in the schedule (ack: never seen in training)")
    args = parser.parse_args()

    events, attack_phase = build_schedule(args.seed, args.repeats, attacks=tuple(args.attacks))
    total = WARMUP + attack_phase
    print(f"Session {args.session}: {len(events)} attacks in {events[-1].round} rounds, "
          f"about {total / 60:.0f} min")
    if args.dry_run:
        for e in events:
            print(f"  round {e.round:2d} +{WARMUP + e.offset:6.0f}s {e.duration:3.0f}s  "
                  f"{e.attacker} {' '.join(e.command(VICTIM[1]))}")
        return

    if os.geteuid() != 0:
        sys.exit("Mininet needs root: sudo python3 -m ddos.lab.capture ...")
    missing = [t for t in TOOLS if shutil.which(t) is None]
    if missing:
        sys.exit(f"Missing tools: {' '.join(missing)}\n"
                 "  sudo apt install mininet openvswitch-switch hping3 iperf3")
    out = (MITIGATION_DIR if args.detector else LAB_PCAP_DIR) / args.session
    if (out / "capture.pcap").exists() and not args.force:
        sys.exit(f"{out} already has a capture (use --force to overwrite)")
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    from ddos.lab.topology import build_net, VICTIM_PORT   # imports mininet
    net = build_net(bw=args.bw)
    webroot = make_webroot()
    services, clients, attacks, times = [], [], {}, {}
    dump, dump_err, detector = None, open(logs / "tcpdump.log", "w+"), None
    t0 = time.time()
    try:
        victim = net[VICTIM[0]]
        services.append(victim.popen([sys.executable, "-m", "http.server", str(HTTP_PORT),
                                      "--directory", str(webroot)], stdout=DEVNULL, stderr=DEVNULL))
        for i in range(len(CLIENTS)):
            services.append(victim.popen(["iperf3", "-s", "-p", str(IPERF_BASE_PORT + i)],
                                         stdout=DEVNULL, stderr=DEVNULL))
        # -Z root: tcpdump would otherwise drop to its own user, which cannot write here
        dump = subprocess.Popen(["tcpdump", "-i", VICTIM_PORT, "-n", "-s", str(SNAPLEN),
                                 "-B", "16384", "-Z", "root", "-w", str(out / "capture.pcap"), "ip"],
                                stdout=DEVNULL, stderr=dump_err)
        time.sleep(2)
        if args.detector:
            detector = start_detector(out, logs, VICTIM_PORT, args)
            print(f"  [{datetime.now():%H:%M:%S}] detector running ({args.detector_mode}"
                  f"{', adaptive' if args.adaptive else ''}), blocks last {args.block_seconds}s")

        t0 = time.time()
        for i, (name, _) in enumerate(CLIENTS):
            with open(logs / f"{name}.log", "w") as log:
                clients.append(net[name].popen(
                    [sys.executable, "-m", "ddos.lab.benign", "--server", VICTIM[1],
                     "--iperf-port", str(IPERF_BASE_PORT + i), "--seed", str(args.seed * 100 + i),
                     "--duration", str(total + 30)],
                    cwd=str(ROOT), stdout=log, stderr=log))
        print(f"  [{datetime.now():%H:%M:%S}] benign warm-up, {WARMUP}s")

        run_attacks(net, events, t0 + WARMUP, logs, attacks, times)
        sleep_until(t0 + total)
    except KeyboardInterrupt:
        print("Interrupted, saving what was captured")
    finally:
        t_end = time.time()
        # Mininet does not clean up popen() processes, so stop every one we started
        for proc in [*attacks.values(), *clients, *services]:
            stop(proc)
        if detector is not None:
            stop(detector, signal.SIGINT, timeout=15)
        if dump is not None:
            stop(dump, signal.SIGINT)
        net.stop()
        shutil.rmtree(webroot, ignore_errors=True)
        dump_err.seek(0)
        dump_stats = tcpdump_stats(dump_err.read())
        # Unfinished attacks (after Ctrl+C) end now
        times = {i: [s, e if e is not None else t_end] for i, (s, e) in times.items()}
        write_outputs(out, args, events, times, t0, t_end, dump_stats)
        give_back_to_user(out)

    print(f"Saved {out}  tcpdump: {dump_stats}")
    if dump_stats.get("dropped_by_kernel"):
        print("Warning: tcpdump dropped packets; lower RATES in scenario.py or raise tcpdump -B")


if __name__ == "__main__":
    main()
