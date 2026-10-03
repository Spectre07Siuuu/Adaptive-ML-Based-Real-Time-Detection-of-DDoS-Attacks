"""Packet-sequence view for the deep learning models.

For every window of WindowAggregator (same slots, same peers, so rows join one
to one), the first SEQ_LEN packets between the peer and the protected hosts,
in time order and in both directions, each encoded as PACKET_FEATURES. Windows
with fewer packets are zero-padded; `length` says how many are real.

Ports, TTL and TCP window size are left out on purpose: they identify the
attack tool (hping3, LOIC) rather than the attack, and a model that keys on
them would not transfer. Port behaviour is kept as "is this port new in the
window", which floods share whatever tool sends them.
"""
import math

import numpy as np

from ddos.features.packets import TCP, UDP, ICMP, FIN, SYN, RST, PSH, ACK
from ddos.features.window import WindowAggregator

SEQ_LEN = 32
PACKET_FEATURES = [
    "inbound", "size", "payload", "iat",
    "tcp", "udp", "icmp",
    "syn", "ack", "rst", "fin", "psh",
    "peer_port_new", "server_port_new",
]
MTU = 1500
IAT_SCALE = math.log1p(1e6)   # gaps in microseconds, log-scaled so 1 s -> 1.0


class _Sequence:
    __slots__ = ("n_in", "rows", "last_ts", "peer_ports", "server_ports")

    def __init__(self):
        self.n_in = 0
        self.rows = []
        self.last_ts = None
        self.peer_ports, self.server_ports = set(), set()

    def encode(self, pkt, inbound):
        gap = 0.0 if self.last_ts is None else max(pkt.ts - self.last_ts, 0.0)
        self.last_ts = pkt.ts
        peer_port, server_port = (pkt.sport, pkt.dport) if inbound else (pkt.dport, pkt.sport)
        peer_new = server_new = 0.0
        if pkt.proto != ICMP:
            peer_new = float(peer_port not in self.peer_ports)
            server_new = float(server_port not in self.server_ports)
            self.peer_ports.add(peer_port)
            self.server_ports.add(server_port)
        f = pkt.flags
        self.rows.append((
            float(inbound), min(pkt.size / MTU, 1.0), min(pkt.payload / MTU, 1.0),
            min(math.log1p(gap * 1e6) / IAT_SCALE, 1.0),
            float(pkt.proto == TCP), float(pkt.proto == UDP), float(pkt.proto == ICMP),
            float(bool(f & SYN)), float(bool(f & ACK)), float(bool(f & RST)),
            float(bool(f & FIN)), float(bool(f & PSH)),
            peer_new, server_new,
        ))


class SequenceAggregator(WindowAggregator):
    def __init__(self, protected_ips, window=1.0, length=SEQ_LEN):
        super().__init__(protected_ips, window)
        self.length = length

    def add(self, pkt):
        rows = self.advance(pkt.ts)
        if pkt.ts < self.slot * self.window:
            self.late += 1
            return rows
        if pkt.dst in self.protected and pkt.src not in self.protected:
            peer, inbound = pkt.src, True
        elif pkt.src in self.protected and pkt.dst not in self.protected:
            peer, inbound = pkt.dst, False
        else:
            return rows
        seq = self.peers.get(peer)
        if seq is None:
            seq = self.peers[peer] = _Sequence()
        seq.n_in += inbound
        if len(seq.rows) < self.length:
            seq.encode(pkt, inbound)
        return rows

    def _emit(self):
        start = self.slot * self.window if self.slot is not None else None
        rows = []
        for ip, seq in self.peers.items():
            if not seq.n_in:
                continue   # same rule as WindowAggregator
            x = np.zeros((self.length, len(PACKET_FEATURES)), dtype=np.float32)
            x[:len(seq.rows)] = seq.rows
            rows.append({"window_start": start, "peer_ip": ip, "seq": x, "length": len(seq.rows)})
        self.peers = {}
        return rows
