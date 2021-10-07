# cuda_firewall
Implementing a Firewall using dpdk and CUDA
# current progress
- [x] working bitvector search usng CUDA
- [x] make use of dpdk table api
- [x] simple 5 tuple rule syntax with DROP/ACCEPT actions
- [x] l2 polling on trunk port and l2 forward to correspondending tap iface, if lookup successfully highest priority rule has ACCEPT action
- [x] simple l2 forward of incoming packet from tap to trunk port
- [ ] improving speed of bitvector search
- [ ] misc. refactoring

# current stats*

| line rate | 100Mbits/s | 500 Mbits/s | 1Gbit/s | 5 Gbit/s | 10 Gbit/s | 20 Gbit/s | 40 Gbit/s|
|-----------|:----------:|:-----------:|:-------:|:--------:|:---------:|:---------:|:--------:|
||**reached**|**reached**|**reached**|*pending*|*pending*|*pending*|*pending*|

 <font size="1"> *tested using iperf3 and two Mellanox ConnectX-3 NICs (40GigE)</font> 
