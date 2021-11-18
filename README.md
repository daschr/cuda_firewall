# plain_fwd
Plain packet forwarder between NIC and a dpdk kni interface. Used for getting approximate throughput value for comparison against firewall.

# current progress
- [x] uses multiple queues (currently 2 by default)
- [x] atomics
- [ ] stats
- [ ] external host pinned or internal membuf **NOTE:** seems currently not possible 

# current stats*

| line rate | 100Mbits/s | 500 Mbits/s | 1Gbit/s | 5 Gbit/s | 10 Gbit/s | 20 Gbit/s | 40 Gbit/s|
|-----------|:----------:|:-----------:|:-------:|:--------:|:---------:|:---------:|:--------:|
||**reached**|**reached**|**reached**|**reached**|**reached**|**reached**|*pending*|

 <font size="1"> *tested using iperf3 and two Mellanox ConnectX-3 NICs (40GigE)</font> 

# kni module mods
```diff --git a/kernel/linux/kni/kni_misc.c b/kernel/linux/kni/kni_misc.c
index f4944e1ddf..d8bcfa6044 100644
--- a/kernel/linux/kni/kni_misc.c
+++ b/kernel/linux/kni/kni_misc.c
@@ -128,11 +128,6 @@ kni_thread_single(void *data)
                        }
                }
                up_read(&knet->kni_list_lock);
-#ifdef RTE_KNI_PREEMPT_DEFAULT
-               /* reschedule out for a while */
-               schedule_timeout_interruptible(
-                       usecs_to_jiffies(KNI_KTHREAD_RESCHEDULE_INTERVAL));
-#endif
        }
 
        return 0;
@@ -149,10 +144,6 @@ kni_thread_multiple(void *param)
                        kni_net_rx(dev);
                        kni_net_poll_resp(dev);
                }
-#ifdef RTE_KNI_PREEMPT_DEFAULT
-               schedule_timeout_interruptible(
-                       usecs_to_jiffies(KNI_KTHREAD_RESCHEDULE_INTERVAL));
-#endif
        }
 
        return 0;
diff --git a/lib/kni/rte_kni.c b/lib/kni/rte_kni.c
index fc8f0e7b5a..51b3ce173c 100644
--- a/lib/kni/rte_kni.c
+++ b/lib/kni/rte_kni.c
@@ -25,14 +25,14 @@
 #include <rte_kni_common.h>
 #include "rte_kni_fifo.h"
 
-#define MAX_MBUF_BURST_NUM            32
+#define MAX_MBUF_BURST_NUM            64
 
 /* Maximum number of ring entries */
-#define KNI_FIFO_COUNT_MAX     1024
+#define KNI_FIFO_COUNT_MAX     2048
 #define KNI_FIFO_SIZE          (KNI_FIFO_COUNT_MAX * sizeof(void *) + \
                                        sizeof(struct rte_kni_fifo))
 
-#define KNI_REQUEST_MBUF_NUM_MAX      32
+#define KNI_REQUEST_MBUF_NUM_MAX      64
 
 #define KNI_MEM_CHECK(cond, fail) do { if (cond) goto fail; } while (0)
 
```

# usage

* build dpdk (>=21.11): `meson build -Denable_kmods=true`
* `make all`
* `insmod <path to you kni module>`
* run:
   1. `sudo ./plain_fwd --main-lcore <highest lcore id>`
   2. `ip a add <some ip 1> dev fw0`
   3. on second host: `ip a add <some ip 2> <some connected iface>`
   4. testing
       1. on first host: `numactl -C 4 iperf3 -s`
       2. on second host: `numactl -C 0  iperf3 -c 192.168.8.2 -Z  -t 60`
* please note that for optimal perfomace, one must set the lcores for rx/tx and iperf3 wisely:
    * `cat /sys/class/net/<IFNAME>/device/local_cpulist`
