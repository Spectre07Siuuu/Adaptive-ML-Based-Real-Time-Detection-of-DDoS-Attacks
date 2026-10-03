import subprocess
import time
import joblib
from datetime import datetime

# Load trained model
model = joblib.load("ddos_model.pkl")

print("🚀 AI DDoS Detector Started...\n")

while True:
    try:
        # Get network stats
        packets = int(subprocess.getoutput("cat /sys/class/net/s1-eth1/statistics/rx_packets"))
        bytes_ = int(subprocess.getoutput("cat /sys/class/net/s1-eth1/statistics/rx_bytes"))

        time.sleep(2)

        packets2 = int(subprocess.getoutput("cat /sys/class/net/s1-eth1/statistics/rx_packets"))
        bytes2 = int(subprocess.getoutput("cat /sys/class/net/s1-eth1/statistics/rx_bytes"))

        pkt_rate = packets2 - packets
        byte_rate = bytes2 - bytes_

        syn_count = pkt_rate // 5   # simple estimation
        avg_pkt_size = byte_rate / pkt_rate if pkt_rate > 0 else 0

        # Prediction
        prediction = model.predict([[pkt_rate, byte_rate, syn_count, avg_pkt_size]])[0]

        now = datetime.now().strftime("%H:%M:%S")

        if prediction == 1:
            print(f"[{now}] ⚠ AI ALERT: DDoS Attack Detected!")

            # Auto Block
            subprocess.call("echo 'link h1 s1 down' | sudo mnexec -a 1", shell=True)

            print("🚫 Attacker Blocked Automatically\n")

        else:
            print(f"[{now}] ✅ SAFE")

    except KeyboardInterrupt:
        print("\nStopped.")
        break
