# Adaptive ML-Based Real-Time Detection of DDoS Attacks

Course project by Md. Asif Mustoba Sazzad, Jaba Anika Kotha and Md. Arefin Iqram.
The proposal is in [docs/Adaptive ML DDoS Detection.pdf](docs/Adaptive%20ML%20DDoS%20Detection.pdf).

The code has two pipelines:

- **v2 (in progress):** follows the proposal. A Mininet lab with separate benign clients and
  attackers, per-source-IP time-window features, and labels taken from the logged attack schedule.
- **v1 (first prototype):** 100-packet windows from four Mininet pcaps, RF / KNN / SVM, and a
  socket server. Its "Normal" class is mostly attack traffic (see Known issues), so its 95%
  accuracy does not measure detection.

## Implementation plan

| Phase | Proposal contribution | Status |
|---|---|---|
| 0 | Clean data and honest evaluation (prerequisite) | done: 3 lab sessions, baselines in reports/lab |
| 1 | 02 Cross-dataset eval: lab + CIC-DDoS2019 + CIC-IDS2017 + Kaggle SDN | done: reports/cross_dataset |
| 2 | 04 DL comparison: 1D-CNN, LSTM, Transformer vs RF/XGBoost | done: reports/models |
| 3 | 03 Real-time closed-loop mitigation (OVS drop flows) | done: reports/mitigation |
| 4 | 01 Adaptive pipeline: drift detection, online learning, novelty detection | todo |
| 5 | 05 Deployment-ready architecture | todo |

## Folder structure

```
ddos/                       Python package (run every module from the project root)
  config.py                 all paths, label names and the window length, in one place
  lab/                      Mininet lab, stdlib only (run with sudo python3)
    scenario.py               address plan + seeded attack schedule
    topology.py               OVS switch without controller, victim on port s1-eth1
    benign.py                 benign client traffic: HTTP, ping, iperf3 TCP/UDP
    capture.py                runs one session -> capture.pcap, events.csv, meta.json
  datasets/                 public datasets (phase 1)
    catalog.py                victim, attacker and published attack times per dataset
    slice_pcap.py             pcap/pcapng/zip/URL reader; keeps one host's packets, cut to headers
    build_windows.py          CIC captures -> <name>_windows.csv and <name>_flows.csv
    sdn.py                    SDN dataset CSV -> sdn_flows.csv
  features/
    packets.py                IPv4 header parser (Ethernet, SLL, SLL2, raw)
    window.py                 per-peer time-window features, shared with the future live detector
    flows.py                  SDN-style 30 s flow view, to compare with the SDN dataset
    sequences.py              first 32 packets of each window, the deep models' input
    build_dataset.py          lab sessions -> labelled lab_windows.csv and lab_flows.csv
    build_sequences.py        every dataset -> <name>_sequences.npz, row-aligned with the windows
    extract_windows.py        v1: pcap -> 100-packet windows -> final_dataset.csv
  training/
    train_baselines.py        v2: RF, KNN, SVM with grouped splits -> reports/lab, models/lab
    cross_dataset.py          v2: train on one/some datasets, test on each -> reports/cross_dataset
    deep_models.py            v2: 1D-CNN, LSTM, Transformer over packet sequences
    compare_models.py         v2: deep models vs tree baselines, quality + speed -> reports/models
    train_detector.py         v2: the live detector's model -> models/detector.joblib
    train_models.py           v1: RF, KNN, SVM + comparison table + feature importance
    evaluate_rf.py            v1: RF confusion matrix + ROC curves -> reports/figures/
    train_server_model.py     v1: RF on scaled features, the model used by predict_server
  preprocessing/balance_dataset.py  v1: random undersampling
  realtime/
    detector.py               v2: live sniffer -> windows -> model -> OVS drop flow (phase 3)
    evaluate_mitigation.py    v2: time to block, traffic stopped, false blocks per session
    predict_server.py         v1: TCP server on 127.0.0.1:9999, JSON features in -> label out
tests/                      pytest suite for the v2 pipeline
legacy/live_ai_detector.py  early prototype, does not run as-is (see Known issues)
data/raw/pcap/              v1 captures (not in git); lab/<session>/ for v2 captures
data/raw/external/          public datasets and their cached unlabelled rows (not in git)
data/processed/             *_windows.csv, *_flows.csv (v2), final_dataset.csv, final_balanced_dataset.csv (v1)
models/                     trained models (not in git); models/lab/ for v2
reports/                    v1 reports; reports/lab/ for v2
docs/                       proposal
```

## Setup

Python packages, in a virtual environment:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

On Ubuntu without the `python3-venv` package, create it with
`python3 -m venv --without-pip .venv` and install with
`pip --python .venv/bin/python install -r requirements-dev.txt`.

Lab tools (needed only for capturing):

```
sudo apt install mininet openvswitch-switch hping3 iperf3
```

The pcaps are large. To keep them elsewhere, set `DDOS_DATA_DIR` to a folder with the same
`raw/pcap/` and `processed/` layout (use `sudo -E` so the capture sees it).

## v2 pipeline

Run from the project root. Capture at least two sessions with different seeds, so the test
set can be a whole held-out session:

```
python3 -m ddos.lab.capture --session s1 --seed 1 --dry-run   # show the schedule (~29 min)
sudo python3 -m ddos.lab.capture --session s1 --seed 1
sudo python3 -m ddos.lab.capture --session s2 --seed 2
sudo python3 -m ddos.lab.capture --session s3 --seed 3

.venv/bin/python -m ddos.features.build_dataset
.venv/bin/python -m ddos.training.train_baselines
.venv/bin/python -m pytest
```

Each session runs four benign clients for the whole time. After a 60 s benign warm-up, three
attackers run SYN, UDP and ICMP floods at 500, 2,000 and 10,000 packets/s, three times each,
sometimes two at once. A benign-only gap of 20–40 s follows every attack. tcpdump records
headers only (`-s 96`) on the victim's switch port, so the capture has no controller or loopback
traffic. One session is roughly 1 GB. If a run crashes, clean up with `sudo mn -c`.

`build_dataset` turns each session into one row per (second, peer IP). A window is labelled
with an attack only when its peer is an attacker that was attacking at that time. Attacker
windows outside any attack are dropped. `train_baselines` keeps each attack and the gap after
it on one side of every split. Besides accuracy and macro-F1, it reports the detection rate and
the false-positive rate (benign windows flagged as attacks, i.e. hosts that would be blocked).

## Phase 1: public datasets and cross-dataset evaluation

| Dataset | What is used | Windows (normal / attack) | Attacks |
|---|---|---|---|
| Lab | 3 sessions, victim 10.0.0.100 | 21,163 (18,244 / 2,919) | SYN, UDP, ICMP floods |
| CIC-IDS2017 | Friday 7 Jul 2017, victim 192.168.10.50 | 6,266 (5,180 / 1,086) | LOIC HTTP flood |
| CIC-DDoS2019 | first day 3 Nov 2018, victim 192.168.50.4 | 3,197 (493 / 2,704) | NetBIOS, LDAP, MSSQL reflection; UDP; SYN |
| SDN (Ahuja et al.) | flow-table rows, no packets | 98,508 flow rows (61,751 / 36,757) | ICMP, UDP, SYN floods |

The packet datasets get the same per-peer window features as the lab, so a model trained on
one can be tested on another. The SDN dataset has no packets, only 30 s OpenFlow counters, so
all four datasets are also compared on a coarse SDN-style flow view (`features/flows.py`:
packet rate, byte rate, mean size, reverse flow, protocol).

Download and build (the CIC captures come from Hugging Face mirrors; the official UNB
downloads need a registration form):

```
# CIC-IDS2017 Friday: streams 8.8 GB, keeps the victim's packets (150 MB, ~20 min)
.venv/bin/python -m ddos.datasets.slice_pcap --host 192.168.10.50 \
    --url https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/main/pcaps/Friday-WorkingHours.pcap \
    -o data/raw/external/cicids2017/friday_victim.pcap
# CIC-DDoS2019 first day: 2 GB zip, read in place (29 GB unpacked, 97% of it the victim's)
curl -L -C - -o data/raw/external/cicddos2019/PCAP-03-11.zip --create-dirs \
    https://huggingface.co/datasets/bencorn/CICDDoS2019/resolve/main/pcaps/03-11/PCAP-03-11.zip
# SDN dataset (12.5 MB, CC BY 4.0)
curl -L -o data/raw/external/sdn/dataset_sdn.csv --create-dirs \
    https://data.mendeley.com/public-files/datasets/jxpfjc64kr/files/5b57518c-c0fa-4cd9-bc11-e01bd5a363f4/file_downloaded

.venv/bin/python -m ddos.datasets.build_windows      # ~13 min for CIC-DDoS2019, then cached
.venv/bin/python -m ddos.datasets.sdn
.venv/bin/python -m ddos.training.cross_dataset              # windows view
.venv/bin/python -m ddos.training.cross_dataset --view flows # flow view, with SDN
```

Ground truth (`datasets/catalog.py`): in both CIC datasets all attack traffic reaches the
victim from one NAT address, so windows are labelled by (peer IP, time), as in the lab. The
published times are local Atlantic Daylight Time (UTC-3); only that offset puts the attacker's
traffic inside the published attacks. The published schedule is also wrong in places, so
attack boundaries come from the capture (first and last second with >= 100 packets/s from
the attacker):

- CIC-IDS2017's port scan ran at 14:51–15:24, not 13:55–14:35. Port scan windows are dropped
  (not DDoS).
- CIC-DDoS2019's SYN flood lasted 8 minutes (11:29–11:37), not until 17:35. For the rest of the
  afternoon the attacker sends ~1 packet/s, which would have been mislabelled as SYN flood.
- CIC-DDoS2019's PortMap and UDP-Lag attacks leave no trace in this capture. Their windows are
  dropped.

### Results

Random Forest, binary macro-F1 % (attack vs normal) on each dataset's held-out part
(lab: session s3; others: a quarter of their 10-minute blocks):

| Trained on | lab | cicids2017 | cicddos2019 |
|---|---|---|---|
| lab only | **100.0** | 44.2 | 82.0 |
| cicids2017 only | 77.0 | **100.0** | 20.1 |
| cicddos2019 only | 67.8 | 94.5 | **98.6** |
| all but lab | *68.6* | 99.9 | 97.1 |
| all but cicids2017 | 100.0 | *83.4* | 96.9 |
| all but cicddos2019 | 100.0 | 99.9 | *22.6* |
| all datasets | 100.0 | 99.9 | 96.9 |

- Within a dataset, detection is easy (98.6–100%).
- Across datasets, a model detects the attack types it was trained on and misses new ones. SYN
  floods are caught at 97–100% whatever the training set. The lab model catches 0.3% of the
  CIC-IDS2017 HTTP flood, which the lab never produces. The CIC-IDS2017 model catches 0% of
  CIC-DDoS2019's reflection and UDP floods.
- Adding datasets does not by itself help on an unseen one. The lab alone detects 90% of
  CIC-DDoS2019's reflection attacks; lab + CIC-IDS2017 detects 0%. Why is not established yet
  (CIC-IDS2017's benign traffic is not more UDP-heavy than the lab's).
- One model trained on all three keeps within-dataset accuracy on each, with 0% false
  positives. Coverage of attack types matters more than the number of datasets. That is the
  case for phase 4 (detecting and adapting to unseen attacks).
- On the flow view, the SDN dataset reaches 81% within itself (96% with its native features
  and a time-grouped split) and 33–62% across datasets.

Per-model tables, detection and false-positive rates, per-attack detection rates and
heatmaps are in `reports/cross_dataset/{windows,flows}/`.

Caveats: the CIC-DDoS2019 victim's benign traffic is small (493 windows) and consists of
replies to its own outbound requests (DNS, web), unlike the lab's clients calling a server. In
the flow view CIC-IDS2017 has only 40 attack rows (one flow every 30 s). Each CIC dataset has a
single attacker address. Published papers that report ~99% on these datasets mostly use random
row splits, which leak between train and test.

## Phase 2: deep learning comparison

The networks read the first 32 packets of each window (`features/sequences.py`): direction,
size, payload, inter-arrival time, protocol, TCP flags, and whether each side's port is new
in the window. Ports, TTL and TCP window size are left out; they identify the attack tool
(hping3, LOIC) more than the attack. Baselines: Random Forest and histogram gradient
boosting (sklearn's XGBoost-style booster) on the 26 window features, and a Random Forest on
the networks' own input, flattened. Every model sees the same windows, splits and training
caps as phase 1. Each model is trained on each dataset alone and on all three, with 3 seeds.

```
.venv/bin/python -m ddos.features.build_sequences   # ~15 min, most of it CIC-DDoS2019
.venv/bin/python -m ddos.training.compare_models    # ~18 min on 2 CPU cores
```

Binary macro-F1 %, mean ± std over 3 seeds:

| Model | Within dataset | Across datasets | Trained on all 3 | Latency, 1 window | Parameters / nodes |
|---|---|---|---|---|---|
| RF (window) | 99.1 ± 0.3 | 63.0 ± 8.0 | 98.5 ± 0.1 | 20.0 ms | 5.0k nodes |
| HGB (window) | **99.3 ± 0.3** | **64.7 ± 2.9** | 98.8 ± 0.2 | 1.2 ms | 6.0k nodes |
| RF (sequence) | 96.9 ± 2.2 | 52.0 ± 7.3 | 98.5 ± 0.1 | 16.7 ms | 10.1k nodes |
| 1D-CNN | 97.8 ± 3.3 | 54.5 ± 4.9 | **99.2 ± 0.3** | **0.35 ms** | 17k params |
| LSTM | 95.5 ± 2.6 | 50.9 ± 6.1 | 97.8 ± 1.3 | 0.50 ms | 21k params |
| Transformer | 98.0 ± 2.8 | 51.2 ± 6.0 | 98.7 ± 0.3 | 0.86 ms | 70k params |

![Model comparison](reports/models/comparison.png)

- Within a dataset and when trained on all three, every model is at 96–99%; the differences
  are within a few points and often within one standard deviation.
- Across datasets every model drops to 51–65%. The networks are not better at transferring
  than the trees. The boosted trees on window features transfer best and vary least
  between seeds.
- Input matters more than architecture: the Random Forest loses 11 points across datasets when
  given the networks' packet sequences instead of the window features.
- Per attack type, models disagree on what transfers. Trained on the lab, the window trees
  catch 94–97% of CIC-DDoS2019's reflection floods; the CNN and LSTM ~50%; the Transformer 0%.
  No model trained on the lab catches more than 36% of the HTTP flood.
- Speed: one window at a time, the 1D-CNN takes 0.35 ms and HGB 1.2 ms; sklearn's Random
  Forest takes 17–20 ms per call, slow for per-second, per-peer scoring. In batches, HGB is
  fastest (~390k windows/s). The Transformer trains slowest (51 s on the lab vs 6 s for the CNN).
- For the live detector (phase 3) this points to HGB on window features, with the 1D-CNN as
  the deep-learning option.

Full per-run results, per-attack detection rates and timings are in `reports/models/`.

## Phase 3: live detection and mitigation

`realtime/detector.py` sniffs the victim's switch port (`s1-eth1`) with a raw socket, builds
the same per-peer 1 s windows as training, and scores each window as it closes with
`models/detector.joblib` (HGB on window features, trained on all three datasets). A peer
flagged in 2 consecutive windows gets an OVS flow
`priority=100,ip,nw_src=<peer>,hard_timeout=<N>,actions=drop`; OVS lifts it after N seconds,
and a peer still attacking is caught again. The victim's own address is never blocked, and a
window that cannot be scored is logged as an error rather than treated as normal.

```
.venv/bin/python -m ddos.training.train_detector
# a ~10 minute lab session with the detector running (needs the project venv):
sudo python3 -m ddos.lab.capture --session m1 --seed 11 --repeats 1 --detector
.venv/bin/python -m ddos.realtime.evaluate_mitigation m1
```

Detector sessions are saved under `data/raw/pcap/mitigation/`, apart from the training
captures. The evaluation reports, per attack, the seconds until the attacker was blocked, the
share of its traffic that still reached the victim, and any benign client that was blocked.

Live run, session m1 (seed 11, a schedule the model never saw; 13 attacks, 10 minutes):

| Result | Value |
|---|---|
| Attacks blocked while running | 13 / 13 |
| Time to block, median (max) | 1.13 s (1.75 s) |
| Attack traffic stopped before reaching the victim | 94.5% on average (91.6–96.7%) |
| Benign clients blocked / benign windows flagged | 0 / 0% |
| Detector errors / packets dropped by the socket | 0 / 0 |

The traffic that got through is the 1–2 s before each block. Two attacks outlasted the 30 s
block and were blocked again (15 blocks for 13 attacks). The whole session's capture holds
350k packets; a training session without the detector holds ~6M. Per-attack results are in
`reports/mitigation/m1_attacks.csv`.

Without root, `--replay <pcap> --dry-run` runs the detector over a recorded capture and logs
the blocks it would install. Replaying lab session s3 (which the detector model has seen in
training, so this checks the plumbing, not accuracy): all 30 attacks were caught, median
1.3–1.8 s after they started (6.3 s for low-rate ICMP), and no benign client window was
flagged. The detector processed ~100k packets/s.

## v1 pipeline

```
python -m ddos.features.extract_windows        # needs the v1 pcaps; slow (scapy)
python -m ddos.preprocessing.balance_dataset
python -m ddos.training.train_models
python -m ddos.training.evaluate_rf
python -m ddos.training.train_server_model
python -m ddos.realtime.predict_server
```

`predict_server` expects one JSON list of the 10 features, in the column order of
`final_balanced_dataset.csv` (without `label`).

The v1 pcaps were captured in a Mininet topology in WSL with `tcpdump -i any` (Linux SLL2
link type). All traffic, benign and attack, goes from h1 (`10.0.0.1`) to h2 (`10.0.0.2`).

## Known issues (v1)

- **"Normal" is not normal:** `normal.pcap` (676 s) opens with ~105 s of a UDP flood still
  running from the previous capture (374k zero-length UDP packets to port 0). Then comes ~90 s
  of ICMP at ~7,000 packets/s, plus 380k OpenFlow loopback packets. Only the last ~500 s is
  ordinary ping traffic. Of the 10,460 Normal windows in `final_dataset.csv`, 3,718 are pure
  UDP flood, 2,557 are ICMP at flood rate and 4,149 are controller loopback; only ~28 are
  benign. Each attack capture also starts with the tail of the previous attack (9% of
  `icmp_flood.pcap` is UDP flood). The models learn which capture a window came from, not
  attack vs. normal.
- **Duplicate rows / leakage:** 65.6% of `final_dataset.csv` rows are exact duplicates.
  56% of test-split rows also appear verbatim in the training split.
- **Weak features:** `packets` is always 100, so it carries no information. `bytes`, `tx_kbps`
  and `pkt_rate` are all derived from `avg_pkt_size` and `duration`.
- **No per-IP view:** a window mixes the packets of all hosts, and benign and attack traffic
  share one source IP, so there is no attacker IP to block.
- **predict_server:** if anything goes wrong it answers "Normal" (fail-open), and it reads
  only one `recv(4096)` per request. The client that sends features is not in this repo.
- **legacy/live_ai_detector.py:** it loads a `ddos_model.pkl` that does not exist and uses
  4 features (with `syn_count` faked as `pkt_rate // 5`). It only reacts to class 1, and its
  block command (`echo 'link h1 s1 down' | mnexec`) does not run a Mininet CLI command.
