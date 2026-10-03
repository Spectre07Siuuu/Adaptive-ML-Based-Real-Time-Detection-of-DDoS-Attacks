"""The SDN DDoS dataset (Ahuja et al., Mendeley jxpfjc64kr, CC BY 4.0) in the flow view.

    python -m ddos.datasets.sdn

Rows are 30 s snapshots of OpenFlow flow entries polled by a Ryu controller in
Mininet; there are no packets, so this dataset joins only the flow-view
comparison (features/flows.py). Labels are binary; the attack type follows from
the protocol (the attacks are ICMP, UDP and TCP SYN floods). Rows with no
packets in the interval are dropped, as the packet-based flow view never emits
them. `round` is a 30-minute block of `dt`, the grouping unit for splits.

Download (12.5 MB):
  https://data.mendeley.com/public-files/datasets/jxpfjc64kr/files/5b57518c-c0fa-4cd9-bc11-e01bd5a363f4/file_downloaded
  -> data/raw/external/sdn/dataset_sdn.csv
"""
import pandas as pd

from ddos.config import EXTERNAL_DIR, PROCESSED_DIR, ALL_LABEL_NAMES
from ddos.features.flows import flow_row, FLOW_SECONDS
from ddos.features.packets import TCP, UDP, ICMP

SOURCE = EXTERNAL_DIR / "sdn" / "dataset_sdn.csv"
OUTPUT = PROCESSED_DIR / "sdn_flows.csv"
PROTOCOLS = {"TCP": TCP, "UDP": UDP, "ICMP": ICMP}
ATTACK_LABEL = {"ICMP": 1, "UDP": 2, "TCP": 3}   # ids in config.ALL_LABEL_NAMES
BLOCK = 1800


def build():
    raw = pd.read_csv(SOURCE)
    raw = raw[raw["pktperflow"] > 0]
    rows = [{"session": "sdn", "round": dt // BLOCK, "window_start": dt, "peer_ip": src, "dst": dst,
             "proto": PROTOCOLS[proto],
             **flow_row(packets, frame_bytes, pair, PROTOCOLS[proto], FLOW_SECONDS),
             "attack_rate": "", "label": ATTACK_LABEL[proto] if label else 0}
            for dt, src, dst, proto, packets, frame_bytes, pair, label in raw[
                ["dt", "src", "dst", "Protocol", "pktperflow", "byteperflow", "Pairflow", "label"]
            ].itertuples(index=False)]
    return pd.DataFrame(rows)


def main():
    df = build()
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} flow rows to {OUTPUT}")
    print(df["label"].map(ALL_LABEL_NAMES).value_counts().to_string())


if __name__ == "__main__":
    main()
