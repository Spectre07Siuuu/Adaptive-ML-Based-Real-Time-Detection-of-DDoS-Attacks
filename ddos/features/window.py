"""Per-peer time-window features.

A window holds what one remote host (the "peer") exchanged with the protected
server(s) during one fixed time slot. Keying on the peer, not on a packet count
like v1, means every window has one source IP the mitigation step can block.

The same aggregator runs offline (build_dataset) and live (detector), so the
model is served exactly the features it was trained on.
"""
import math
from collections import Counter

from ddos.features.packets import TCP, UDP, ICMP, FIN, SYN, RST, PSH, ACK

FEATURES = [
    "n_in", "n_out", "bytes_in", "bytes_out", "out_in_ratio",
    "mean_size_in", "std_size_in", "min_size_in", "max_size_in", "mean_payload_in",
    "tcp_ratio_in", "udp_ratio_in", "icmp_ratio_in",
    "syn_ratio_in", "ack_ratio_in", "rst_ratio_in", "fin_ratio_in", "psh_ratio_in",
    "synack_ratio_out", "rst_ratio_out",
    "uniq_sport_in", "uniq_dport_in", "sport_entropy_in", "dport_entropy_in",
    "iat_mean_in", "iat_std_in",
]


def _entropy(counter, total):
    return -sum(c / total * math.log2(c / total) for c in counter.values())


def _std(total, sq_total, n):
    return math.sqrt(max(sq_total / n - (total / n) ** 2, 0.0))


class _PeerStats:
    __slots__ = ("n_in", "n_out", "bytes_in", "bytes_out", "size_sq", "size_min", "size_max",
                 "payload_in", "tcp", "udp", "icmp", "syn", "ack", "rst", "fin", "psh",
                 "synack_out", "rst_out", "sports", "dports", "last_ts", "iat", "iat_sq")

    def __init__(self):
        self.n_in = self.n_out = self.bytes_in = self.bytes_out = 0
        self.size_sq = 0
        self.size_min, self.size_max = math.inf, 0
        self.payload_in = 0
        self.tcp = self.udp = self.icmp = 0
        self.syn = self.ack = self.rst = self.fin = self.psh = 0
        self.synack_out = self.rst_out = 0
        self.sports, self.dports = Counter(), Counter()
        self.last_ts = None
        self.iat = self.iat_sq = 0.0

    def add_in(self, pkt):
        self.n_in += 1
        self.bytes_in += pkt.size
        self.size_sq += pkt.size * pkt.size
        self.size_min = min(self.size_min, pkt.size)
        self.size_max = max(self.size_max, pkt.size)
        self.payload_in += pkt.payload
        if pkt.proto == TCP:
            self.tcp += 1
            f = pkt.flags
            self.syn += (f & (SYN | ACK)) == SYN   # connection attempts, not SYN-ACKs
            self.ack += bool(f & ACK)
            self.rst += bool(f & RST)
            self.fin += bool(f & FIN)
            self.psh += bool(f & PSH)
        elif pkt.proto == UDP:
            self.udp += 1
        elif pkt.proto == ICMP:
            self.icmp += 1
        if pkt.proto != ICMP:
            self.sports[pkt.sport] += 1
            self.dports[pkt.dport] += 1
        if self.last_ts is not None:
            gap = pkt.ts - self.last_ts
            self.iat += gap
            self.iat_sq += gap * gap
        self.last_ts = pkt.ts

    def add_out(self, pkt):
        self.n_out += 1
        self.bytes_out += pkt.size
        if pkt.proto == TCP:
            self.synack_out += (pkt.flags & (SYN | ACK)) == (SYN | ACK)
            self.rst_out += bool(pkt.flags & RST)

    def features(self):
        n = self.n_in
        ports = sum(self.sports.values())
        gaps = n - 1
        return {
            "n_in": n,
            "n_out": self.n_out,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "out_in_ratio": self.n_out / n,
            "mean_size_in": self.bytes_in / n,
            "std_size_in": _std(self.bytes_in, self.size_sq, n),
            "min_size_in": self.size_min,
            "max_size_in": self.size_max,
            "mean_payload_in": self.payload_in / n,
            "tcp_ratio_in": self.tcp / n,
            "udp_ratio_in": self.udp / n,
            "icmp_ratio_in": self.icmp / n,
            "syn_ratio_in": self.syn / n,
            "ack_ratio_in": self.ack / n,
            "rst_ratio_in": self.rst / n,
            "fin_ratio_in": self.fin / n,
            "psh_ratio_in": self.psh / n,
            "synack_ratio_out": self.synack_out / self.n_out if self.n_out else 0.0,
            "rst_ratio_out": self.rst_out / self.n_out if self.n_out else 0.0,
            "uniq_sport_in": len(self.sports),
            "uniq_dport_in": len(self.dports),
            "sport_entropy_in": _entropy(self.sports, ports) if ports else 0.0,
            "dport_entropy_in": _entropy(self.dports, ports) if ports else 0.0,
            "iat_mean_in": self.iat / gaps if gaps else 0.0,
            "iat_std_in": _std(self.iat, self.iat_sq, gaps) if gaps else 0.0,
        }


class WindowAggregator:
    """Feed packets in time order; get back one feature row per (window, peer).

    A window is emitted once a packet from a later window arrives, or when
    advance() is called with a later time (the live detector does this on a timer).
    Peers that sent nothing to the protected hosts in a window are not emitted.
    Packets older than the open window (rare reordering) are dropped and counted.
    """

    def __init__(self, protected_ips, window=1.0):
        self.protected = set(protected_ips)
        self.window = window
        self.slot = None
        self.peers = {}
        self.late = 0

    def add(self, pkt):
        rows = self.advance(pkt.ts)
        if pkt.ts < self.slot * self.window:
            self.late += 1
            return rows
        if pkt.dst in self.protected and pkt.src not in self.protected:
            self._peer(pkt.src).add_in(pkt)
        elif pkt.src in self.protected and pkt.dst not in self.protected:
            self._peer(pkt.dst).add_out(pkt)
        return rows

    def advance(self, ts):
        """Close the open window if `ts` is past its end."""
        slot = int(ts // self.window)
        if self.slot is None:
            self.slot = slot
        if slot <= self.slot:
            return []
        rows = self._emit()
        self.slot = slot
        return rows

    def flush(self):
        rows = self._emit()
        if self.slot is not None:
            self.slot += 1
        return rows

    def _peer(self, ip):
        stats = self.peers.get(ip)
        if stats is None:
            stats = self.peers[ip] = _PeerStats()
        return stats

    def _emit(self):
        start = self.slot * self.window if self.slot is not None else None
        rows = [{"window_start": start, "peer_ip": ip, **stats.features()}
                for ip, stats in self.peers.items() if stats.n_in]
        self.peers = {}
        return rows
