# Thesis - *Offloading network packet classification to GPUs using CUDA and dpdk*
See [this PDF](https://github.com/daschr/cuda_firewall/blob/e4e9b63af005667067a52d3d302dcd39398bfcf8/thesis.pdf).

# cuda_firewall
Implementing a Firewall using dpdk and CUDA
# current progress
- [x] working bitvector search usng CUDA
- [x] make use of dpdk table api
- [x] simple 5 tuple rule syntax with DROP/ACCEPT actions
- [x] l2 polling on trunk port and l2 forward to correspondending tap iface, if lookup successfully highest priority rule has ACCEPT action
- [x] simple l2 forward of incoming packet from tap to trunk port
- [ ] switch from tap to kni
- [ ] add better stats collection to firewall
- [x] improving speed of bitvector search
- [ ] misc. refactoring

# current stats

## Line rate*
| line rate | 100Mbits/s | 500 Mbits/s | 1Gbit/s | 5 Gbit/s | 10 Gbit/s | 20 Gbit/s | 40 Gbit/s|
|-----------|:----------:|:-----------:|:-------:|:--------:|:---------:|:---------:|:--------:|
||**reached**|**reached**|**reached**|**reached**|**reached**|*pending*|*pending*|

 <font size="1"> *tested using iperf3 and two Mellanox ConnectX-3 NICs (40GigE)</font> 


## Packet rate**

| packet rate | with tap forward | without tap forward |
|-----------|:----------:|:-----------:|
||~2.8Mpps|~12.5Mpps|

 <font size="1"> **tested using pktgen-dpdk using the asynchronous execution model (*async* branch) and two Mellanox ConnectX-3 NICs (40GigE)</font> 

# settings
* use `isolcpus` to isolate at least two adjacent logical cores
* force device managed flow steering, f.e. for Mellanox ConnectX-3: `mlx4_core.log_num_mgm_entry_size=-1`
* example: `GRUB_CMDLINE_LINUX_DEFAULT="quiet isolcpus=2,3 mlx4_core.log_num_mgm_entry_size=-1"`

# usage

* build dpdk (>=21.08)
* `make all`
* run:
   1. `sudo ./firewall -l0-1 --vdev=net_tap0,iface=fw0  rules.txt`
   2. `ip a add <some ip 1> dev fw0`
   3. on second host: `ip a add <some ip 2> <some connected iface>`
   4. now test 
