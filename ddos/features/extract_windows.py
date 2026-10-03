from scapy.all import PcapReader, IP, TCP, UDP, ICMP
import pandas as pd

from ddos.config import PCAP_DIR, PROCESSED_DIR

WINDOW_SIZE = 100

def extract(pcap_path, label):
    windows = []
    buffer  = []
    count   = 0

    with PcapReader(str(pcap_path)) as pcap:
        for pkt in pcap:
            if IP not in pkt:
                continue
            buffer.append(pkt)
            if len(buffer) < WINDOW_SIZE:
                continue

            pkt_count    = len(buffer)
            byte_count   = sum(len(p) for p in buffer)
            syn_count    = sum(1 for p in buffer if TCP in p and p[TCP].flags & 0x02)
            avg_pkt_size = byte_count / pkt_count

            # Protocol counts
            udp_count  = sum(1 for p in buffer if UDP  in p)
            icmp_count = sum(1 for p in buffer if ICMP in p)
            tcp_count  = sum(1 for p in buffer if TCP  in p)

            # Protocol ratios
            udp_ratio  = udp_count  / pkt_count
            icmp_ratio = icmp_count / pkt_count
            tcp_ratio  = tcp_count  / pkt_count

            duration = float(buffer[-1].time - buffer[0].time)
            if duration <= 0:
                duration = 0.001

            tx_kbps  = (byte_count * 8) / (duration * 1000)
            pkt_rate = pkt_count / duration

            windows.append({
                "packets":      pkt_count,
                "bytes":        byte_count,
                "syn_count":    syn_count,
                "avg_pkt_size": round(avg_pkt_size, 4),
                "duration":     round(duration,     6),
                "tx_kbps":      round(tx_kbps,      4),
                "pkt_rate":     round(pkt_rate,      4),
                "udp_ratio":    round(udp_ratio,     4),
                "icmp_ratio":   round(icmp_ratio,    4),
                "tcp_ratio":    round(tcp_ratio,     4),
                "label":        label
            })
            buffer = []
            count += 1

            if count % 5000 == 0:
                print(f"  {pcap_path.name} → {count} windows done")

    return pd.DataFrame(windows)

print("Extracting Normal...")
df_normal = extract(PCAP_DIR / "normal.pcap",     0)
print(f"Normal done: {len(df_normal)} rows")

print("Extracting ICMP Flood...")
df_icmp   = extract(PCAP_DIR / "icmp_flood.pcap", 1)
print(f"ICMP done: {len(df_icmp)} rows")

print("Extracting UDP Flood...")
df_udp    = extract(PCAP_DIR / "udp_flood.pcap",  2)
print(f"UDP done: {len(df_udp)} rows")

print("Extracting SYN Flood...")
df_syn    = extract(PCAP_DIR / "syn_flood.pcap",  3)
print(f"SYN done: {len(df_syn)} rows")

# Merge + Shuffle
df = pd.concat([df_normal, df_icmp, df_udp, df_syn], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(PROCESSED_DIR / "final_dataset.csv", index=False)

print("\nDataset Ready!")
print(df["label"].value_counts())
print(f"\nTotal Rows: {len(df)}")
print("\nFeatures:", list(df.columns))
