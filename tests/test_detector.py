import io
from types import SimpleNamespace

import numpy as np

from ddos.features.window import FEATURES
from ddos.realtime.detector import Detector, Mitigator

ATTACKER, CLIENT, VICTIM = "10.0.0.21", "10.0.0.11", "10.0.0.100"


class FakeOvs:
    def __init__(self, returncode=0):
        self.calls, self.returncode = [], returncode

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        return SimpleNamespace(returncode=self.returncode, stderr="boom")


def test_blocks_after_consecutive_flags_once_per_timeout():
    ovs, events = FakeOvs(), []
    m = Mitigator("s1", consecutive=2, block_seconds=30, whitelist=[VICTIM], run=ovs,
                  log=lambda **e: events.append(e))
    assert not m.update(0, ATTACKER, True, 0.9)            # one flag is not enough
    assert not m.update(1, ATTACKER, False, 0.1)           # streak broken
    assert not m.update(2, ATTACKER, True, 0.9)
    assert m.update(3, ATTACKER, True, 0.95)               # second in a row: block
    flow = ovs.calls[0][3]
    assert ovs.calls[0][:3] == ["ovs-ofctl", "add-flow", "s1"]
    assert "nw_src=10.0.0.21" in flow and "hard_timeout=30" in flow and "actions=drop" in flow
    assert not m.update(4, ATTACKER, True, 0.9) and not m.update(5, ATTACKER, True, 0.9)  # still blocked
    assert m.update(34, ATTACKER, True, 0.9) is False      # streak restarted after the block
    assert m.update(35, ATTACKER, True, 0.9)               # timeout over: blocked again
    assert [e["event"] for e in events] == ["block", "block"]

    for t in range(5):
        assert not m.update(t, VICTIM, True, 1.0)          # whitelisted
    assert len(ovs.calls) == 2


def test_failed_block_is_logged_not_recorded():
    events = []
    m = Mitigator("s1", consecutive=1, run=FakeOvs(returncode=1), log=lambda **e: events.append(e))
    assert not m.update(0, ATTACKER, True, 0.9)
    assert events[0]["event"] == "error" and "boom" in events[0]["error"]
    assert ATTACKER not in m.blocked_until


class FakeModel:
    def __init__(self, fail=False):
        self.fail = fail

    def predict_proba(self, x):
        if self.fail:
            raise ValueError("bad input")
        p = (x[:, FEATURES.index("n_in")] > 100).astype(float)   # "attack" = more than 100 packets
        return np.column_stack([1 - p, p])


def _detector(model):
    ovs, events, scores = FakeOvs(), [], io.StringIO()
    m = Mitigator("s1", consecutive=1, run=ovs, log=lambda **e: events.append(e))
    bundle = {"model": model, "features": FEATURES, "window": 1.0}
    return Detector(bundle, VICTIM, m, 0.5, lambda **e: events.append(e), scores), ovs, events, scores


def _row(peer, n_in):
    return {"window_start": 10.0, "peer_ip": peer, **dict.fromkeys(FEATURES, 0.0), "n_in": n_in}


def test_scores_windows_and_blocks_flagged_peer():
    det, ovs, events, scores = _detector(FakeModel())
    det.score([_row(ATTACKER, 5000), _row(CLIENT, 12)], 11.0)
    assert [e["peer"] for e in events if e["event"] == "block"] == [ATTACKER]
    lines = scores.getvalue().splitlines()
    assert lines == ["10,10.0.0.21,5000,1.0000,1", "10,10.0.0.11,12,0.0000,0"]


def test_unscorable_window_is_an_error_not_normal():
    det, ovs, events, scores = _detector(FakeModel(fail=True))
    det.score([_row(ATTACKER, 5000)], 11.0)
    assert events[0]["event"] == "error" and "bad input" in events[0]["error"]
    assert scores.getvalue() == "" and not ovs.calls


def test_evaluate_mitigation_on_a_synthetic_session(tmp_path):
    import json
    from scapy.all import Ether, IP, UDP, wrpcap
    from ddos.realtime.evaluate_mitigation import evaluate

    t0 = 1000.0
    frames = []
    for i in range(100):                       # a1: 10 s attack at 10 pps, blocked after 2 s
        if i < 20:
            frames.append(Ether() / IP(src=ATTACKER, dst=VICTIM) / UDP())
            frames[-1].time = t0 + i / 10
    for i in range(50):                        # a2: never blocked, 5 s at 10 pps
        frames.append(Ether() / IP(src="10.0.0.22", dst=VICTIM) / UDP())
        frames[-1].time = t0 + 20 + i / 10
    frames.sort(key=lambda f: f.time)
    wrpcap(str(tmp_path / "capture.pcap"), frames)
    (tmp_path / "meta.json").write_text(json.dumps({
        "session": "m0", "victim": VICTIM, "clients": {"c1": CLIENT},
        "attackers": {"a1": ATTACKER, "a2": "10.0.0.22"}}))
    (tmp_path / "events.csv").write_text(
        "session,round,attacker,ip,attack,label,rate,pps,size,port,start_ts,end_ts\n"
        f"m0,1,a1,{ATTACKER},udp,2,low,10,0,53,{t0},{t0 + 10}\n"
        f"m0,2,a2,10.0.0.22,udp,2,low,10,0,53,{t0 + 20},{t0 + 25}\n")
    log = [{"event": "start", "ts": t0 - 5},
           {"event": "block", "ts": t0 + 2, "peer": ATTACKER, "seconds": 30},
           {"event": "block", "ts": t0 + 3, "peer": CLIENT, "seconds": 30},
           {"event": "stop", "ts": t0 + 30, "kernel_drops": 0}]
    (tmp_path / "detector.jsonl").write_text("\n".join(json.dumps(e) for e in log))
    (tmp_path / "scores.csv").write_text(
        "window_start,peer_ip,n_in,score,flagged\n1000,10.0.0.11,5,0.1,0\n1001,10.0.0.11,5,0.9,1\n")

    attacks, summary = evaluate(tmp_path)
    a1, a2 = attacks.iloc[0], attacks.iloc[1]
    assert (a1.status, a1.time_to_block_s, a1.packets_delivered) == ("blocked", 2.0, 20)
    assert a1["blocked_share_%"] == 80.0 and a1["traffic_stopped_%"] == 80.0
    assert (a2.status, a2.packets_delivered, a2["traffic_stopped_%"]) == ("missed", 50, 0.0)
    assert summary["blocked while attacking %"] == 50.0 and summary["missed"] == 1
    assert summary["benign clients blocked"] == 1 and summary["benign windows flagged %"] == 50.0
