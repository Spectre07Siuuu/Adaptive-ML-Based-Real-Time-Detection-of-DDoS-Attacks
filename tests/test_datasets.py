import subprocess
import sys
import zipfile
from datetime import datetime, timezone

import pytest
from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap
from scapy.utils import PcapNgWriter

from ddos.config import ROOT
from ddos.datasets.catalog import DATASETS, Dataset
from ddos.datasets.slice_pcap import records
from ddos.features.packets import read_pcap

VICTIM = "192.168.10.50"


def _frames():
    frames = []
    for i in range(6):
        src, dst = ("10.1.1.1", VICTIM) if i % 2 else ("10.1.1.1", "10.9.9.9")
        pkt = Ether() / IP(src=src, dst=dst) / TCP(sport=1000 + i, dport=80) / Raw(b"x" * 300)
        pkt.time = 1_500_000_000 + i + 0.25
        frames.append(pkt)
    return frames


def _write_pcapng(path, frames):
    with PcapNgWriter(str(path)) as w:
        for f in frames:
            w.write(f)


@pytest.mark.parametrize("fmt", ["pcap", "pcapng"])
def test_records_reads_both_formats(tmp_path, fmt):
    path = tmp_path / f"in.{fmt}"
    frames = _frames()
    wrpcap(str(path), frames) if fmt == "pcap" else _write_pcapng(path, frames)
    with open(path, "rb") as f:
        recs = list(records(f))
    assert len(recs) == 6
    linktype, sec, usec, origlen, data = recs[1]
    assert (linktype, sec, usec, origlen) == (1, 1_500_000_001, 250_000, 354)
    assert data == bytes(frames[1])


def test_slice_keeps_victim_packets_truncated(tmp_path):
    src = tmp_path / "day.pcapng"
    _write_pcapng(src, _frames())
    archive = tmp_path / "day.zip"
    with zipfile.ZipFile(archive, "w") as zf:   # zip input, as CIC-DDoS2019 ships
        zf.write(src, "SAT_0")
    out = tmp_path / "slice.pcap"
    subprocess.run([sys.executable, "-m", "ddos.datasets.slice_pcap", "--host", VICTIM,
                    str(archive), "-o", str(out)], cwd=ROOT, check=True, capture_output=True)

    pkts = list(read_pcap(out))
    assert [p.sport for p in pkts] == [1001, 1003, 1005]     # only packets to the victim
    assert all(p.size == 340 and p.payload == 300 for p in pkts)   # sizes survive the cut
    assert out.stat().st_size == 24 + 3 * (16 + 96)
    assert not out.with_suffix(".partial").exists()


def test_catalog_times_are_local():
    ds = Dataset("x", ["x.pcap"], "", VICTIM, "172.16.0.1", utc_offset=-3)
    utc = datetime.fromtimestamp(ds.epoch("2017-07-07 15:56"), timezone.utc)
    assert (utc.hour, utc.minute) == (18, 56)
    for ds in DATASETS.values():
        for a in ds.attacks:
            assert ds.epoch(a.start) < ds.epoch(a.end)


def test_label_uses_published_schedule():
    import pandas as pd
    from ddos.datasets.build_windows import label
    from ddos.datasets.catalog import Attack
    from ddos.features.window import FEATURES

    ds = Dataset("x", [], "", VICTIM, "172.16.0.1", utc_offset=0, attacks=[
        Attack("scan", -1, "2017-07-07 10:00", "2017-07-07 10:05"),
        Attack("flood", 4, "2017-07-07 11:00", "2017-07-07 11:10")])
    t = ds.epoch("2017-07-07 00:00")
    rows = [(t + 10 * 3600 + 120, "172.16.0.1"),     # during the scan: dropped
            (t + 11 * 3600 + 300, "172.16.0.1"),     # during the flood
            (t + 11 * 3600 + 601, "172.16.0.1"),     # within the 2 s grace after the flood
            (t + 11 * 3600 + 650, "172.16.0.1"),     # attacker outside any attack: dropped
            (t + 11 * 3600 + 300, "192.168.10.3")]   # benign peer during the flood
    df = pd.DataFrame([{"window_start": s, "peer_ip": ip, **dict.fromkeys(FEATURES, 1.0)}
                       for s, ip in rows])
    out, stats = label(ds, df)
    assert list(zip(out["peer_ip"], out["label"])) == [("172.16.0.1", 4), ("172.16.0.1", 4),
                                                      ("192.168.10.3", 0)]
    assert stats["unmatched_attacker_windows"] == 1
    assert stats["attacker_windows_per_attack"] == {"flood": 2, "scan": 1}
    assert ds.epoch("2017-07-07 11:00:30") == ds.epoch("2017-07-07 11:00") + 30
    assert (out["session"] == "x").all() and (out["attack_rate"] == "").all()
