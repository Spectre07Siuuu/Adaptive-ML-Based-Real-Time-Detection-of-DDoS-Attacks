"""Lab address plan and the seeded attack schedule for one capture session.

Stdlib only: lab scripts run under `sudo python3`, outside the project venv.
"""
import random
from dataclasses import dataclass

VICTIM = ("victim", "10.0.0.100")
CLIENTS = [(f"c{i}", f"10.0.0.{10 + i}") for i in range(1, 5)]
ATTACKERS = [(f"a{i}", f"10.0.0.{20 + i}") for i in range(1, 4)]

HTTP_PORT = 80
IPERF_BASE_PORT = 5201   # client i talks to the victim's iperf3 server on port 5201 + i
WEB_FILES = {"1k.bin": 1_000, "10k.bin": 10_000, "100k.bin": 100_000,
             "1m.bin": 1_000_000, "5m.bin": 5_000_000}

# Attack name -> label id in config.ALL_LABEL_NAMES. The training sessions use LABELS;
# "ack" (TCP ACK flood) is held back to test the detector on an attack no model has seen.
LABELS = {"icmp": 1, "udp": 2, "syn": 3}
ALL_ATTACKS = {**LABELS, "ack": 6}
RATES = {"low": 500, "medium": 2000, "high": 10000}   # packets per second

# Attack ports and payload sizes overlap with the benign traffic on purpose, so the
# model cannot separate the classes by a single hping3 default value.
SYN_PORTS = [HTTP_PORT, IPERF_BASE_PORT]
UDP_PORTS = [53, 123, 1900, IPERF_BASE_PORT]
PAYLOAD_SIZES = [0, 32, 56, 128, 512, 1200]


@dataclass
class Event:
    round: int
    attacker: str
    ip: str
    attack: str
    rate: str
    size: int        # payload bytes (always 0 for SYN)
    port: int        # destination port (0 for ICMP)
    offset: float    # seconds after the warm-up ends
    duration: float

    @property
    def pps(self):
        return RATES[self.rate]

    def command(self, victim_ip):
        cmd = ["hping3", "-q", "-i", f"u{1_000_000 // self.pps}"]
        if self.attack == "syn":
            return cmd + ["-S", "-p", str(self.port), victim_ip]
        if self.attack == "ack":
            return cmd + ["-A", "-p", str(self.port), victim_ip]
        if self.attack == "udp":
            return cmd + ["--udp", "-p", str(self.port), "-d", str(self.size), victim_ip]
        return cmd + ["--icmp", "-d", str(self.size), victim_ip]


def _event(rng, rnd, attacker, attack, rate, offset, duration):
    name, ip = attacker
    size = 0 if attack in ("syn", "ack") else rng.choice(PAYLOAD_SIZES)
    port = {"syn": SYN_PORTS, "ack": SYN_PORTS, "udp": UDP_PORTS}.get(attack)
    return Event(rnd, name, ip, attack, rate, size, rng.choice(port) if port else 0,
                 offset, duration)


def build_schedule(seed, repeats=3, multi_vector=0.3, duration=(20, 40), gap=(20, 40),
                   attacks=tuple(LABELS)):
    """Every (attack, rate) pair `repeats` times, in random order.

    A round is one attack followed by a benign-only gap. With probability
    `multi_vector` a second attacker runs a different attack in the same round.
    Returns (events, length of the attack phase in seconds).
    """
    rng = random.Random(seed)
    combos = [(attack, rate) for attack in attacks for rate in RATES] * repeats
    rng.shuffle(combos)
    events, t = [], 0.0
    for rnd, (attack, rate) in enumerate(combos, start=1):
        length = rng.uniform(*duration)
        first, second = rng.sample(ATTACKERS, 2)
        events.append(_event(rng, rnd, first, attack, rate, t, length))
        if rng.random() < multi_vector:
            other = rng.choice([a for a in attacks if a != attack])
            events.append(_event(rng, rnd, second, other, rng.choice(list(RATES)), t, length))
        t += length + rng.uniform(*gap)
    return events, t
