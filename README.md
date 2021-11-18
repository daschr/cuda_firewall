# plain_fwd
Plain packet forwarder between NIC and a tap. Used for getting approximate throughput value for comparison against firewall.

# current progress
- [x] uses multiple queues (currently 2 by default)
- [x] atomics
- [x] stats
- [x] external host pinned or internal membuf 

# current stats*

| line rate | 100Mbits/s | 500 Mbits/s | 1Gbit/s | 5 Gbit/s | 10 Gbit/s | 20 Gbit/s | 40 Gbit/s|
|-----------|:----------:|:-----------:|:-------:|:--------:|:---------:|:---------:|:--------:|
||**reached**|**reached**|**reached**|**reached**|**reached**|*pending*|*pending*|

 <font size="1"> *tested using iperf3 and two Mellanox ConnectX-3 NICs (40GigE)</font> 

# usage

* build dpdk (>=21.08)
* `make all`
* run:
   1. `sudo ./plain_fwd --vdev=net_tap0,iface=fw0  --main-lcore <highest lcore id>`
   2. `ip a add <some ip 1> dev fw0`
   3. on second host: `ip a add <some ip 2> <some connected iface>`
   4. testing
       1. on first host: `numactl -C 4 iperf3 -s`
       2. on second host: `numactl -C 0  iperf3 -c 192.168.8.2 -Z  -t 60`
* please note that for optimal perfomace, one must set the lcores for rx/tx and iperf3 wisely:
    * `cat /sys/class/net/<IFNAME>/device/local_cpulist`
