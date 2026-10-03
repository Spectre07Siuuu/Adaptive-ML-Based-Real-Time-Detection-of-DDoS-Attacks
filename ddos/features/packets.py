"""Parse IPv4 headers out of raw frames (pcap files now, live sockets in the detector).

Only the headers are read, so captures taken with a small snaplen (-s 96) work.
Sizes come from the IP total-length field, not the captured length.
"""
import socket
import struct
from dataclasses import dataclass

import dpkt

TCP, UDP, ICMP = 6, 17, 1

# TCP flag bits
FIN, SYN, RST, PSH, ACK = 0x01, 0x02, 0x04, 0x08, 0x10

# Link types (offset of the IPv4 header is fixed per type, except Ethernet with VLAN tags)
DLT_EN10MB = 1
DLT_RAW = (12, 14, 101, 228)
DLT_LINUX_SLL = 113    # tcpdump -i any (older libpcap)
DLT_LINUX_SLL2 = 276   # tcpdump -i any (libpcap >= 1.10), used by the v1 captures

ETH_IPV4 = 0x0800
ETH_VLAN = (0x8100, 0x88A8)


@dataclass(slots=True)
class Packet:
    ts: float
    src: str
    dst: str
    proto: int
    size: int       # IP total length
    payload: int    # L4 payload bytes
    sport: int = 0
    dport: int = 0
    flags: int = 0  # TCP flags, 0 for other protocols


def _ipv4_offset(buf, linktype):
    """Byte offset of the IPv4 header, or None if the frame is not IPv4."""
    if linktype == DLT_EN10MB:
        off, ethertype = 14, struct.unpack_from("!H", buf, 12)[0]
        while ethertype in ETH_VLAN:
            ethertype = struct.unpack_from("!H", buf, off + 2)[0]
            off += 4
        return off if ethertype == ETH_IPV4 else None
    if linktype == DLT_LINUX_SLL2:
        return 20 if struct.unpack_from("!H", buf, 0)[0] == ETH_IPV4 else None
    if linktype == DLT_LINUX_SLL:
        return 16 if struct.unpack_from("!H", buf, 14)[0] == ETH_IPV4 else None
    if linktype in DLT_RAW:
        return 0 if buf and buf[0] >> 4 == 4 else None
    raise ValueError(f"unsupported link type {linktype}")


def parse_frame(buf, ts, linktype):
    """Return a Packet for an IPv4 TCP/UDP/ICMP frame, else None (ARP, IPv6, truncated...)."""
    try:
        off = _ipv4_offset(buf, linktype)
        if off is None or buf[off] >> 4 != 4:
            return None
        ihl = (buf[off] & 0x0F) * 4
        size, frag = struct.unpack_from("!H2xH", buf, off + 2)
        proto = buf[off + 9]
        if proto not in (TCP, UDP, ICMP):
            return None
        pkt = Packet(ts, socket.inet_ntoa(buf[off + 12:off + 16]),
                     socket.inet_ntoa(buf[off + 16:off + 20]), proto, size, 0)
        l4 = off + ihl
        if frag & 0x1FFF:
            # Non-first fragment: no L4 header, the whole IP payload is data
            pkt.payload = size - ihl
            return pkt
        if proto == TCP:
            pkt.sport, pkt.dport, data_off, pkt.flags = struct.unpack_from("!HH8xBB", buf, l4)
            pkt.payload = size - ihl - (data_off >> 4) * 4
        elif proto == UDP:
            pkt.sport, pkt.dport = struct.unpack_from("!HH", buf, l4)
            pkt.payload = size - ihl - 8
        else:
            pkt.payload = size - ihl - 8
        pkt.payload = max(pkt.payload, 0)
        return pkt
    except (struct.error, IndexError):
        return None


def read_pcap(path):
    """Yield Packets from a pcap or pcapng file."""
    with open(path, "rb") as f:
        reader = dpkt.pcap.UniversalReader(f)
        linktype = reader.datalink()
        for ts, buf in reader:
            pkt = parse_frame(buf, float(ts), linktype)
            if pkt is not None:
                yield pkt
