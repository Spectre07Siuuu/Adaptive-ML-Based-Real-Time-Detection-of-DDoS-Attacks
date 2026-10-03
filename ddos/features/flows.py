"""SDN-style flow view: one row per directed flow (src, dst, protocol) per 30 s.

The SDN DDoS dataset (Ahuja et al.) has no packets, only OpenFlow flow-table
counters that a Ryu controller polled every 30 s. This view rebuilds the
comparable part of those rows from packet captures, so all four datasets can be
compared on one feature set. It is much coarser than the per-peer windows.
Only flows towards the protected hosts are kept; byte counts include the
14-byte Ethernet header, as OpenFlow counters do.
"""
from ddos.features.packets import TCP, UDP, ICMP
from ddos.features.window import WindowAggregator

FLOW_SECONDS = 30.0
FLOW_FEATURES = ["pktrate", "byterate", "mean_size", "pairflow", "proto_tcp", "proto_udp", "proto_icmp"]
ETHERNET = 14


def flow_row(packets, frame_bytes, pairflow, proto, interval=FLOW_SECONDS):
    return {
        "pktrate": packets / interval,
        "byterate": frame_bytes / interval,
        "mean_size": frame_bytes / packets if packets else 0.0,
        "pairflow": int(pairflow),
        "proto_tcp": int(proto == TCP),
        "proto_udp": int(proto == UDP),
        "proto_icmp": int(proto == ICMP),
    }


class FlowAggregator(WindowAggregator):
    """Same time slots as WindowAggregator; state is per flow instead of per peer."""

    def __init__(self, protected_ips, interval=FLOW_SECONDS):
        super().__init__(protected_ips, interval)
        self.reverse = set()

    def add(self, pkt):
        rows = self.advance(pkt.ts)
        if pkt.ts < self.slot * self.window:
            self.late += 1
            return rows
        if pkt.dst in self.protected and pkt.src not in self.protected:
            counts = self.peers.setdefault((pkt.src, pkt.dst, pkt.proto), [0, 0])
            counts[0] += 1
            counts[1] += pkt.size + ETHERNET
        elif pkt.src in self.protected and pkt.dst not in self.protected:
            self.reverse.add((pkt.dst, pkt.src, pkt.proto))
        return rows

    def _emit(self):
        start = self.slot * self.window if self.slot is not None else None
        rows = [{"window_start": start, "peer_ip": src, "dst": dst, "proto": proto,
                 **flow_row(n, size, (src, dst, proto) in self.reverse, proto, self.window)}
                for (src, dst, proto), (n, size) in self.peers.items()]
        self.peers, self.reverse = {}, set()
        return rows
