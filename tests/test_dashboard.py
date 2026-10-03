import json
import time

from ddos.realtime.dashboard import State, Tail


def test_tail_returns_only_complete_new_lines(tmp_path):
    f = tmp_path / "log.jsonl"
    f.write_text("a\nb")
    t = Tail(f)
    assert t.lines() == ["a"]
    with open(f, "a") as out:
        out.write("c\nd\n")
    assert t.lines() == ["bc", "d"]
    assert t.lines() == []


def test_state_tracks_sources_blocks_and_replay_clock(tmp_path):
    log, scores = tmp_path / "d.jsonl", tmp_path / "s.csv"
    t0 = 1000.0
    rows = ["window_start,peer_ip,n_in,score,novel,flagged,model_version"]
    rows += [f"{t0 + i:.0f},10.0.0.11,10,0.0100,0,0,0" for i in range(10)]
    rows += [f"{t0 + i:.0f},10.0.0.21,5000,1.0000,1,1,0" for i in (5, 6)]
    scores.write_text("\n".join(rows) + "\n")
    events = [{"event": "start", "ts": t0 - 1, "mode": "hybrid", "block_seconds": 30},
              {"event": "block", "ts": t0 + 7, "peer": "10.0.0.21", "seconds": 30, "windows": 2, "score": 1.0},
              {"event": "error", "ts": t0 + 8, "error": "x", "windows": [[1, "a"]]}]
    log.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    state = State(log, scores, replay=True, speed=1.0)
    state.opened = time.time() - 9            # replay clock at t0 + 8
    s = state.snapshot()
    peers = {p["ip"]: p for p in s["peers"]}
    assert set(peers) == {"10.0.0.11", "10.0.0.21"}
    assert peers["10.0.0.21"]["blocked_for"] > 28 and peers["10.0.0.21"]["blocks"] == 1
    assert [q[1] for q in peers["10.0.0.21"]["series"]] == [5000, 5000]
    assert s["totals"]["active_blocks"] == 1 and s["settings"]["mode"] == "hybrid"
    assert s["events"][0]["event"] == "error" and "windows" not in s["events"][0]   # list dropped
    assert s["events"][1]["windows"] == 2                                            # count kept

    state.opened = time.time() - 3            # earlier in the replay: nothing blocked yet
    s = state.snapshot()
    assert s["totals"]["blocks"] == 0 and {p["ip"] for p in s["peers"]} == {"10.0.0.11"}
