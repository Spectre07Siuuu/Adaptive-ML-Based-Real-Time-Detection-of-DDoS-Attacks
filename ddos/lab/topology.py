"""Mininet topology for the lab.

    sudo python3 -m ddos.lab.topology     # start it with a Mininet CLI

One OVS switch and no controller: a single priority-0 NORMAL flow makes it a
plain learning switch. Captures then hold no OpenFlow traffic (the v1 captures
did), and mitigation can block a peer by adding a higher-priority drop flow.
The victim's link is created first, so its switch port is s1-eth1.
"""
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.topo import Topo

from ddos.lab.scenario import VICTIM, CLIENTS, ATTACKERS

SWITCH = "s1"
VICTIM_PORT = "s1-eth1"


class LabTopo(Topo):
    def build(self, bw=100):
        switch = self.addSwitch(SWITCH, failMode="secure")
        for name, ip in [VICTIM, *CLIENTS, *ATTACKERS]:
            host = self.addHost(name, ip=f"{ip}/24")
            self.addLink(host, switch, bw=bw)


def build_net(bw=100):
    net = Mininet(topo=LabTopo(bw=bw), switch=OVSSwitch, link=TCLink,
                  controller=None, autoSetMacs=True)
    net.start()
    net[SWITCH].cmd(f"ovs-ofctl add-flow {SWITCH} priority=0,actions=normal")
    for host in net.hosts:
        host.cmd("sysctl -qw net.ipv6.conf.all.disable_ipv6=1")
    return net


if __name__ == "__main__":
    setLogLevel("info")
    net = build_net()
    CLI(net)
    net.stop()
