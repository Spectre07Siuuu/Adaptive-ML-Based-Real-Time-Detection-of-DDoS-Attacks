import json
import math
from pathlib import Path

import pytest
from scapy.all import Ether, Dot1Q, IP, IPv6, TCP as ScapyTCP, UDP as ScapyUDP, ICMP as ScapyICMP, ARP, Raw, wrpcap

from ddos.config import PCAP_DIR
from ddos.features import build_dataset
from ddos.features.packets import (Packet, parse_frame, read_pcap, DLT_EN10MB,
                                   TCP, UDP, ICMP, SYN, ACK, PSH)
from ddos.features.window import WindowAggregator, FEATURES
from ddos.lab.scenario import LABELS, RATES, build_schedule

VICTIM = "10.0.0.100"


# ---------- packets.py ----------

def test_parse_tcp_syn():
    frame = bytes(Ether() / IP(src="10.0.0.21", dst=VICTIM) / ScapyTCP(sport=1234, dport=80, flags="S"))
    pkt = parse_frame(frame, 1.5, DLT_EN10MB)
    assert (pkt.src, pkt.dst, pkt.proto, pkt.sport, pkt.dport) == ("10.0.0.21", VICTIM, TCP, 1234, 80)
    assert pkt.flags == SYN and pkt.size == 40 and pkt.payload == 0 and pkt.ts == 1.5


def test_parse_udp_payload_behind_vlan():
    frame = bytes(Ether() / Dot1Q(vlan=5) / IP(src="10.0.0.11", dst=VICTIM)
                  / ScapyUDP(sport=5000, dport=53) / Raw(b"x" * 100))
    pkt = parse_frame(frame, 0.0, DLT_EN10MB)
    assert pkt.proto == UDP and pkt.dport == 53 and pkt.payload == 100 and pkt.size == 128


def test_parse_uses_ip_length_when_truncated():
    frame = bytes(Ether() / IP(src="10.0.0.11", dst=VICTIM) / ScapyICMP() / Raw(b"x" * 500))
    pkt = parse_frame(frame[:60], 0.0, DLT_EN10MB)   # as captured with a small snaplen
    assert pkt.proto == ICMP and pkt.size == 528 and pkt.payload == 500


@pytest.mark.parametrize("frame", [
    bytes(Ether() / ARP()),
    bytes(Ether() / IPv6() / ScapyUDP()),
    bytes(Ether() / IP(proto=47)),   # GRE: not TCP/UDP/ICMP
    b"\x00" * 10,
])
def test_parse_ignores_other_traffic(frame):
    assert parse_frame(frame, 0.0, DLT_EN10MB) is None


def test_read_v1_sll2_capture():
    path = PCAP_DIR / "test" / "ddos_test.pcap"
    if not path.exists():
        pytest.skip("v1 test capture not present")
    first = next(read_pcap(path))
    # tcpdump: 127.0.0.1.42998 > 127.0.0.1.6633: Flags [P.], length 8 (OpenFlow echo)
    assert (first.sport, first.dport, first.payload) == (42998, 6633, 8)
    assert first.flags == PSH | ACK


# ---------- window.py ----------

def P(ts, src, dst, proto=TCP, size=40, flags=0, sport=1000, dport=80, payload=0):
    return Packet(ts, src, dst, proto, size, payload, sport, dport, flags)


def test_window_per_peer_features():
    agg = WindowAggregator([VICTIM], window=1.0)
    rows = []
    for i, ts in enumerate([10.1, 10.2, 10.3]):
        rows += agg.add(P(ts, "10.0.0.21", VICTIM, flags=SYN, sport=1000 + i))
    rows += agg.add(P(10.15, VICTIM, "10.0.0.21", flags=SYN | ACK, sport=80, dport=1000))
    rows += agg.add(P(10.25, VICTIM, "10.0.0.21", flags=SYN | ACK, sport=80, dport=1001))
    rows += agg.add(P(10.4, VICTIM, "10.0.0.99"))                    # outbound only: no row
    rows += agg.add(P(10.5, "10.0.0.11", "10.0.0.12"))               # not to the victim: ignored
    assert rows == []

    rows = agg.add(P(11.0, "10.0.0.11", VICTIM, proto=UDP))          # closes window 10
    assert [r["peer_ip"] for r in rows] == ["10.0.0.21"]
    r = rows[0]
    assert set(FEATURES) <= set(r) and r["window_start"] == 10.0
    assert (r["n_in"], r["n_out"], r["bytes_in"]) == (3, 2, 120)
    assert r["syn_ratio_in"] == 1.0 and r["synack_ratio_out"] == 1.0 and r["tcp_ratio_in"] == 1.0
    assert r["out_in_ratio"] == pytest.approx(2 / 3)
    assert r["uniq_sport_in"] == 3 and r["uniq_dport_in"] == 1
    assert r["sport_entropy_in"] == pytest.approx(math.log2(3)) and r["dport_entropy_in"] == 0.0
    assert r["iat_mean_in"] == pytest.approx(0.1) and r["iat_std_in"] == pytest.approx(0.0, abs=1e-9)

    assert agg.add(P(10.9, "10.0.0.11", VICTIM)) == [] and agg.late == 1
    last = agg.flush()
    assert [(x["peer_ip"], x["udp_ratio_in"], x["window_start"]) for x in last] == [("10.0.0.11", 1.0, 11.0)]


def test_advance_closes_window_without_packets():
    agg = WindowAggregator([VICTIM])
    agg.add(P(5.2, "10.0.0.11", VICTIM))
    assert agg.advance(5.9) == []
    assert [r["peer_ip"] for r in agg.advance(6.0)] == ["10.0.0.11"]


# ---------- build_dataset.py on a synthetic session ----------

def _frame(ts, src, dst, layer):
    pkt = Ether() / IP(src=src, dst=dst) / layer
    pkt.time = ts
    return pkt


def test_extract_session_labels_by_ip_and_time(tmp_path: Path):
    t0 = 1000.0
    frames = []
    for s in range(15):                                  # benign client, every second
        frames.append(_frame(t0 + s + 0.1, "10.0.0.11", VICTIM, ScapyTCP(dport=80, flags="PA") / Raw(b"GET")))
    for i in range(150):                                 # SYN flood 1005.0 - 1008.0
        ts = t0 + 5 + i / 50
        frames.append(_frame(ts, "10.0.0.21", VICTIM, ScapyTCP(sport=2000 + i, dport=80, flags="S")))
        frames.append(_frame(ts + 0.001, VICTIM, "10.0.0.21", ScapyTCP(sport=80, dport=2000 + i, flags="SA")))
    frames.append(_frame(t0 + 13.5, "10.0.0.22", VICTIM, ScapyICMP()))   # attacker, outside any attack
    frames.sort(key=lambda p: p.time)
    wrpcap(str(tmp_path / "capture.pcap"), frames)

    (tmp_path / "meta.json").write_text(json.dumps({
        "session": "t1", "victim": VICTIM, "attackers": {"a1": "10.0.0.21", "a2": "10.0.0.22"}}))
    (tmp_path / "events.csv").write_text(
        "session,round,attacker,ip,attack,label,rate,pps,size,port,start_ts,end_ts\n"
        f"t1,1,a1,10.0.0.21,syn,3,low,50,0,80,{t0 + 5},{t0 + 8}\n")

    df, flows, stats = build_dataset.extract_session(tmp_path)
    attack = df[df["label"] == 3]
    assert sorted(attack["window_start"]) == [t0 + 5, t0 + 6, t0 + 7]
    assert set(attack["peer_ip"]) == {"10.0.0.21"} and set(attack["round"]) == {1}
    assert (attack["synack_ratio_out"] == 1.0).all()
    assert set(df.loc[df["label"] == 0, "peer_ip"]) == {"10.0.0.11"}
    assert stats["unmatched_attacker_windows"] == 1           # the stray 10.0.0.22 packet
    benign_rounds = df[df["label"] == 0].set_index("window_start")["round"]
    assert benign_rounds[t0 + 2] == 0 and benign_rounds[t0 + 4] == 1 and benign_rounds[t0 + 12] == 1
    assert (df["session"] == "t1").all()
    attack_flows = flows[flows["label"] == 3]
    assert set(attack_flows["peer_ip"]) == {"10.0.0.21"} and (attack_flows["pairflow"] == 1).all()
    assert attack_flows["pktrate"].sum() == pytest.approx(150 / 30)


# ---------- lab/scenario.py ----------

def test_schedule_covers_every_attack_and_rate():
    events, length = build_schedule(seed=7, repeats=2)
    assert build_schedule(seed=7, repeats=2)[0] == events            # seeded, repeatable
    primary = {}
    for e in events:
        primary.setdefault(e.round, e)
    pairs = sorted((e.attack, e.rate) for e in primary.values())
    assert pairs == sorted([(a, r) for a in LABELS for r in RATES] * 2)

    by_round = {}
    for e in events:
        by_round.setdefault(e.round, []).append(e)
    for evs in by_round.values():
        assert len({e.ip for e in evs}) == len(evs)                  # one attack per attacker
        assert len({e.attack for e in evs}) == len(evs)
    offsets = [evs[0].offset + evs[0].duration for evs in by_round.values()]
    assert offsets == sorted(offsets) and length > offsets[-1]


def test_attack_commands():
    events, _ = build_schedule(seed=1, repeats=1)
    for e in events:
        cmd = e.command(VICTIM)
        assert cmd[0] == "hping3" and cmd[-1] == VICTIM
        assert cmd[cmd.index("-i") + 1] == f"u{1_000_000 // RATES[e.rate]}"
        assert {"syn": "-S", "udp": "--udp", "icmp": "--icmp"}[e.attack] in cmd


# ---------- flows.py ----------

def test_flow_view_per_directed_flow():
    from ddos.features.flows import FlowAggregator
    agg = FlowAggregator([VICTIM])                                   # 30 s slots
    for ts in (1.0, 2.0, 3.0):
        agg.add(P(ts, "10.0.0.21", VICTIM, proto=UDP, size=100))
    agg.add(P(4.0, VICTIM, "10.0.0.21", proto=UDP, size=60))         # reply: pair flow
    agg.add(P(5.0, "10.0.0.11", VICTIM, proto=TCP, size=40))
    rows = {r["peer_ip"]: r for r in agg.add(P(31.0, "10.0.0.11", VICTIM))}
    assert set(rows) == {"10.0.0.21", "10.0.0.11"}
    udp = rows["10.0.0.21"]
    assert udp["pktrate"] == pytest.approx(3 / 30) and udp["mean_size"] == 114   # + Ethernet
    assert udp["byterate"] == pytest.approx(342 / 30)
    assert (udp["pairflow"], udp["proto_udp"], udp["proto_tcp"], udp["window_start"]) == (1, 1, 0, 0.0)
    assert rows["10.0.0.11"]["pairflow"] == 0 and rows["10.0.0.11"]["proto_tcp"] == 1
    assert [r["window_start"] for r in agg.flush()] == [30.0]


# ---------- sequences.py ----------

def test_sequences_match_windows_and_encode_packets():
    from ddos.features.sequences import SequenceAggregator, PACKET_FEATURES
    pkts = [P(10.0, "10.0.0.21", VICTIM, flags=SYN, sport=1000, size=40),
            P(10.0001, VICTIM, "10.0.0.21", flags=SYN | ACK, sport=80, dport=1000, size=44),
            P(10.0002, "10.0.0.21", VICTIM, flags=SYN, sport=1001, size=40),
            P(10.5, VICTIM, "10.0.0.99"),                                  # outbound only
            P(11.2, "10.0.0.11", VICTIM, proto=ICMP, size=84, payload=56)]
    seq, win = SequenceAggregator([VICTIM], length=4), WindowAggregator([VICTIM])
    s_rows, w_rows = [], []
    for p in pkts:
        s_rows += seq.add(p)
        w_rows += win.add(p)
    s_rows += seq.flush()
    w_rows += win.flush()
    assert [(r["window_start"], r["peer_ip"]) for r in s_rows] == \
           [(r["window_start"], r["peer_ip"]) for r in w_rows]

    first = s_rows[0]
    f = {name: first["seq"][:, i] for i, name in enumerate(PACKET_FEATURES)}
    assert first["length"] == 3 and first["seq"].shape == (4, len(PACKET_FEATURES))
    assert list(f["inbound"]) == [1, 0, 1, 0]                            # last row is padding
    assert list(f["syn"][:3]) == [1, 1, 1] and list(f["ack"][:3]) == [0, 1, 0]
    assert list(f["peer_port_new"][:3]) == [1, 0, 1]                     # 1000, 1000, 1001
    assert list(f["server_port_new"][:3]) == [1, 0, 0]                   # 80 every time
    assert f["iat"][0] == 0 and 0.3 < f["iat"][1] < 0.4                  # 100 us gap, log scale
    icmp = s_rows[1]["seq"][0]
    assert icmp[PACKET_FEATURES.index("icmp")] == 1 and icmp[PACKET_FEATURES.index("peer_port_new")] == 0
