# Adaptive ML-Based Real-Time Detection of DDoS Attacks

Course project by Md. Asif Mustoba Sazzad, Jaba Anika Kotha and Md. Arefin Iqram.
The proposal is in [docs/Adaptive ML DDoS Detection.pdf](docs/Adaptive%20ML%20DDoS%20Detection.pdf).

The current code is the first prototype. It extracts traffic-window features from Mininet
pcaps, trains RF / KNN / SVM classifiers for 4 classes (Normal, ICMP flood, UDP flood,
SYN flood), and serves the RF model over a local socket.

## Folder structure

```
ddos/                       Python package (run every module from the project root)
  config.py                 all paths and label names, in one place
  features/extract_windows.py     pcap -> 100-packet window features -> final_dataset.csv
  preprocessing/balance_dataset.py  random undersampling -> final_balanced_dataset.csv
  training/train_models.py        RF, KNN, SVM + comparison table + feature importance
  training/evaluate_rf.py         RF confusion matrix + ROC curves -> reports/figures/
  training/train_server_model.py  RF on scaled features, the model used by predict_server
  realtime/predict_server.py      TCP server on 127.0.0.1:9999, JSON features in -> label out
legacy/live_ai_detector.py  early prototype, does not run as-is (see Known issues)
data/raw/pcap/              normal / icmp_flood / udp_flood / syn_flood captures (not in git)
data/raw/pcap/test/         small test captures
data/processed/             final_dataset.csv, final_balanced_dataset.csv
models/                     trained .pkl files (not in git)
reports/                    model_comparison.csv, feature_importance.csv, figures/
docs/                       proposal
```

## Setup

```
pip install -r requirements.txt
```

The pcaps are about 850 MB. To keep them outside OneDrive (or to use the original
`/mnt/e/Os_lab` copy in WSL), set `DDOS_DATA_DIR` to a folder that has the same
`raw/pcap/` and `processed/` layout.

## Pipeline

Run from the project root, in this order:

```
python -m ddos.features.extract_windows        # needs the pcaps; slow (scapy)
python -m ddos.preprocessing.balance_dataset
python -m ddos.training.train_models
python -m ddos.training.evaluate_rf
python -m ddos.training.train_server_model
python -m ddos.realtime.predict_server
```

`predict_server` expects one JSON list of the 10 features, in the column order of
`final_balanced_dataset.csv` (without `label`).

## Data provenance

The pcaps were captured in a Mininet topology in WSL with `tcpdump -i any` (Linux SLL2 link
type). Attack traffic goes from h1 (`10.0.0.1`) to h2 (`10.0.0.2`); there is one attacker IP.

## Known issues (current prototype)

- **Duplicate rows / leakage:** 65.6% of `final_dataset.csv` rows are exact duplicates.
  56% of test-split rows also appear verbatim in the training split, so the reported 95%
  accuracy is optimistic. With duplicates removed, RF scores about 91% accuracy and 0.89 macro-F1.
- **Weak features:** `packets` is always 100, so it carries no information. `bytes`, `tx_kbps`
  and `pkt_rate` are all derived from `avg_pkt_size` and `duration`.
- **Label noise:** every window in a pcap gets that file's label. The attack captures also
  contain loopback/controller traffic and background traffic, and those windows are labelled as attack too.
- **No per-IP view:** a window mixes the packets of all hosts, so the detector cannot
  name the attacker IP to block.
- **predict_server:** if anything goes wrong it answers "Normal" (fail-open), and it reads
  only one `recv(4096)` per request. The client that sends features is not in this repo.
- **legacy/live_ai_detector.py:** it loads a `ddos_model.pkl` that does not exist and uses
  4 features (with `syn_count` faked as `pkt_rate // 5`). It only reacts to class 1, and its
  block command (`echo 'link h1 s1 down' | mnexec`) does not run a Mininet CLI command.
