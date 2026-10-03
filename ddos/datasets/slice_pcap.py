"""Keep one host's packets from a large capture, cut to their headers.

    python -m ddos.datasets.slice_pcap --host 192.168.10.50 --url https://.../Friday.pcap -o out.pcap
    python -m ddos.datasets.slice_pcap --host 192.168.50.1 PCAP-03-11.zip -o out.pcap

Input is a pcap or pcapng capture from a URL (streamed, never stored), plain files, or zip
archives (members read in natural name order). Output holds the IPv4 packets to
or from --host, truncated to 96 bytes, with original lengths and timestamps
kept. Public captures are 2-10 GB; the slice for one server is a few hundred MB.
"""
import argparse
import io
import re
import socket
import struct
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from ddos.features.packets import ipv4_offset

SNAPLEN = 96

# classic pcap magic -> (byte order, timestamp fraction units per second)
MAGICS = {b"\xd4\xc3\xb2\xa1": ("<", 10**6), b"\xa1\xb2\xc3\xd4": (">", 10**6),
          b"\x4d\x3c\xb2\xa1": ("<", 10**9), b"\xa1\xb2\x3c\x4d": (">", 10**9)}
PCAPNG_SHB = b"\x0a\x0d\x0d\x0a"


def records(f):
    """Yield (linktype, sec, usec, origlen, data) from a classic pcap or pcapng stream."""
    magic = f.read(4)
    if magic in MAGICS:
        yield from _pcap_records(f, magic)
    elif magic == PCAPNG_SHB:
        yield from _pcapng_records(f)
    elif magic:
        raise ValueError(f"not a pcap or pcapng file (magic {magic.hex()})")


def _pcap_records(f, magic):
    order, units = MAGICS[magic]
    head = f.read(20)
    linktype = struct.unpack(order + "I", head[16:20])[0] & 0x0FFFFFFF
    record = struct.Struct(order + "IIII")
    while True:
        h = f.read(16)
        if len(h) < 16:
            return
        sec, frac, caplen, origlen = record.unpack(h)
        data = f.read(caplen)
        if len(data) < caplen:
            return   # file cut off mid-record
        yield linktype, sec, frac * 10**6 // units, origlen, data


def _tsresol(options, order):
    """Timestamp units per second from an interface block's options (default microseconds)."""
    i = 0
    while i + 4 <= len(options):
        code, length = struct.unpack_from(order + "HH", options, i)
        if code == 0:
            break
        if code == 9 and length >= 1:   # if_tsresol
            v = options[i + 4]
            return 2 ** (v & 0x7F) if v & 0x80 else 10 ** v
        i += 4 + (length + 3) // 4 * 4
    return 10**6


def _pcapng_records(f):
    """Enhanced Packet Blocks only; simple packet blocks carry no timestamp and are skipped."""
    rest = f.read(8)                       # rest of the first section header block
    order = "<" if rest[4:8] == b"\x4d\x3c\x2b\x1a" else ">"
    f.read(struct.unpack(order + "I", rest[:4])[0] - 12)
    interfaces = []                        # (linktype, units per second) per interface id
    while True:
        h = f.read(8)
        if len(h) < 8:
            return
        if h[:4] == PCAPNG_SHB:            # new section: byte order may change
            bom = f.read(4)
            order = "<" if bom == b"\x4d\x3c\x2b\x1a" else ">"
            f.read(struct.unpack(order + "I", h[4:8])[0] - 12)
            interfaces = []
            continue
        btype, blen = struct.unpack(order + "II", h)
        body = f.read(blen - 8)
        if len(body) < blen - 8:
            return
        if btype == 1:                     # interface description
            interfaces.append((struct.unpack_from(order + "H", body)[0],
                               _tsresol(body[8:-4], order)))
        elif btype == 6:                   # enhanced packet
            iface, hi, lo, caplen, origlen = struct.unpack_from(order + "IIIII", body)
            linktype, units = interfaces[iface]
            ts = (hi << 32) | lo
            yield linktype, ts // units, ts % units * 10**6 // units, origlen, body[20:20 + caplen]


def to_host(data, linktype, hosts):
    """True if the frame is IPv4 with source or destination in `hosts` (packed addresses)."""
    try:
        off = ipv4_offset(data, linktype)
    except (struct.error, IndexError):
        return False
    if off is None or len(data) < off + 20:
        return False
    return data[off + 12:off + 16] in hosts or data[off + 16:off + 20] in hosts


def natural_key(name):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def sources(paths, url):
    """Yield (name, binary stream) for every capture to read, in time order."""
    if url:
        resp = urllib.request.urlopen(url)
        yield url, io.BufferedReader(resp, buffer_size=1 << 20)
    for path in paths:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                members = [m for m in zf.namelist() if not m.endswith("/")]
                for member in sorted(members, key=natural_key):
                    with zf.open(member) as f:
                        yield f"{path}:{member}", io.BufferedReader(f, buffer_size=1 << 20)
        else:
            with open(path, "rb") as f:
                yield str(path), f


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="*", type=Path, help="pcap files or zip archives")
    parser.add_argument("--url", help="stream a pcap from this URL instead of reading a file")
    parser.add_argument("--host", action="append", required=True, help="IPv4 address to keep (repeatable)")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.inputs and not args.url:
        parser.error("give input files or --url")

    hosts = {socket.inet_aton(h) for h in args.host}
    out_linktype, seen, kept, other_link = None, 0, 0, 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".partial")
    with open(tmp, "wb") as out:
        for name, stream in sources(args.inputs, args.url):
            print(f"Reading {name}", flush=True)
            try:
                for linktype, sec, usec, origlen, data in records(stream):
                    seen += 1
                    if seen % 5_000_000 == 0:
                        when = datetime.fromtimestamp(sec, timezone.utc)
                        print(f"  {seen:,} packets read, {kept:,} kept, at {when:%H:%M} UTC", flush=True)
                    if out_linktype is None:
                        out_linktype = linktype
                        out.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, SNAPLEN, linktype))
                    elif linktype != out_linktype:
                        other_link += 1
                        continue
                    if to_host(data, linktype, hosts):
                        cut = data[:SNAPLEN]
                        out.write(struct.pack("<IIII", sec, usec, len(cut), origlen) + cut)
                        kept += 1
            except ValueError as e:
                print(f"  skipped: {e}", file=sys.stderr)
    if other_link:
        print(f"Warning: {other_link:,} packets on interfaces with another link type were skipped")
    tmp.replace(args.output)   # only a finished slice gets the real name
    print(f"Done: {seen:,} packets read, {kept:,} kept -> {args.output}")


if __name__ == "__main__":
    main()
