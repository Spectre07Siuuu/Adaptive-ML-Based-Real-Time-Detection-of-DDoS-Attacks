"""Ground truth for the public packet datasets, from the publishers' attack schedules.

In both CIC datasets every attack reaches the victim from one NAT address, so a
window is an attack when its peer is that address and it falls inside an
attack below. Times are local (Atlantic Daylight Time, UTC-3): of the offsets
-5..-2, only -3 puts 83% (CIC-IDS2017) and 100% (CIC-DDoS2019) of the attacker's
packets inside the published attacks.

The published schedule gives whole minutes and is sometimes wrong, so DDoS
attacks use the capture's own boundaries: the first and last second in which
the attacker sent >= 100 packets/s (floods run at 700-28,000 packets/s, the
attacker's background traffic at ~1 packet/s and is not labelled). Comments
give the published times where they differ.

Sources:
  CIC-IDS2017  https://www.unb.ca/cic/datasets/ids-2017.html (Friday, 7 July 2017)
  CIC-DDoS2019 https://www.unb.ca/cic/datasets/ddos-2019.html (first day, 3 Nov 2018)
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass
class Attack:
    name: str
    label: int     # id in config.ALL_LABEL_NAMES; -1 = not a DDoS attack, windows dropped
    start: str     # local time "YYYY-MM-DD HH:MM[:SS]"
    end: str


@dataclass
class Dataset:
    name: str
    inputs: list        # captures (pcap, pcapng or zip), relative to config.EXTERNAL_DIR
    source: str         # where the full capture came from
    victim: str
    attacker: str
    utc_offset: int     # hours, local time = UTC + offset
    attacks: list = field(default_factory=list)

    def epoch(self, local):
        tz = timezone(timedelta(hours=self.utc_offset))
        fmt = "%Y-%m-%d %H:%M:%S" if local.count(":") == 2 else "%Y-%m-%d %H:%M"
        return datetime.strptime(local, fmt).replace(tzinfo=tz).timestamp()


HF = "https://huggingface.co/datasets/bencorn"

DATASETS = {
    "cicids2017": Dataset(
        name="cicids2017",
        inputs=["cicids2017/friday_victim.pcap"],   # sliced while streaming (slice_pcap)
        source=f"{HF}/CICIDS2017/resolve/main/pcaps/Friday-WorkingHours.pcap",
        victim="192.168.10.50",
        attacker="172.16.0.1",
        utc_offset=-3,
        attacks=[
            # Published as 13:55-14:35, but the capture shows the main scan at 14:51-15:24
            # (~1,000 ports/s, 99% SYN, 99% RST replies); the window covers both
            Attack("portscan", -1, "2017-07-07 13:55", "2017-07-07 15:30"),
            Attack("ddos_loit", 4, "2017-07-07 15:56:42", "2017-07-07 16:16:13"),  # 15:56-16:16
        ],
    ),
    "cicddos2019": Dataset(
        name="cicddos2019",
        inputs=["cicddos2019/PCAP-03-11.zip"],      # 97% of it is the victim's: not sliced
        source=f"{HF}/CICDDoS2019/resolve/main/pcaps/03-11/PCAP-03-11.zip",
        victim="192.168.50.4",
        attacker="172.16.0.5",
        utc_offset=-3,
        attacks=[
            # PortMap and UDP-Lag leave no burst in the capture (attacker stays at ~1 pps),
            # so their windows are dropped instead of being labelled as attacks
            Attack("portmap", -1, "2018-11-03 09:43", "2018-11-03 09:51"),
            Attack("netbios", 5, "2018-11-03 10:01:25", "2018-11-03 10:09:27"),  # 10:00-10:09
            Attack("ldap", 5, "2018-11-03 10:21:54", "2018-11-03 10:29:56"),     # 10:21-10:30
            Attack("mssql", 5, "2018-11-03 10:34:02", "2018-11-03 10:43:16"),    # 10:33-10:42
            Attack("udp", 2, "2018-11-03 10:52:38", "2018-11-03 11:04:16"),      # 10:53-11:03
            Attack("udplag", -1, "2018-11-03 11:14", "2018-11-03 11:24"),
            Attack("syn", 3, "2018-11-03 11:29:17", "2018-11-03 11:37:18"),      # 11:28-17:35
        ],
    ),
}
