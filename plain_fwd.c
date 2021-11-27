#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

#include <rte_common.h>
#include <rte_log.h>
#include <rte_malloc.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_eal.h>
#include <rte_launch.h>
#include <rte_atomic.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_interrupts.h>
#include <rte_random.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_metrics.h>
#include <rte_bitrate.h>
#include <rte_latencystats.h>

#include "config.h"
#include "misc.h"
#include "stats.h"

#define RTE_LOGTYPE_APP RTE_LOGTYPE_USER1

struct rte_pktmbuf_extmem ext_mem;
struct rte_mempool *mpool_payload;
uint16_t tap_port_id, trunk_port_id;
struct rte_ether_addr tap_macaddr;
stats_t *port_stats=NULL;
uint8_t running;

void exit_handler(int e) {
    __atomic_store_n(&running, 0, __ATOMIC_RELAXED);

    rte_eal_mp_wait_lcore();

    rte_free(port_stats);
    rte_eal_cleanup();

    exit(EXIT_SUCCESS);
}

static int print_stats(void *arg) {
    stats_t *stats=(stats_t *) arg;
    struct timeval t[2];
    unsigned long ts[2];
    double ts_d;
    int p=1;
    stats_t stats_buf[4];
    memset(stats_buf, 0, sizeof(stats_t)*4);

    gettimeofday(t, NULL);
    ts[0]=1000000*t[0].tv_sec+t[0].tv_usec;

    stats_buf[0]=stats[0];
    stats_buf[2]=stats[1];

    while(__atomic_load_n(&running, __ATOMIC_RELAXED)) {
        rte_delay_ms(1000);
        gettimeofday(t+p, NULL);

#define LOAD(X, Y) __atomic_load(&(X), &(Y), __ATOMIC_RELAXED)
        LOAD(stats[0].pkts_in, stats_buf[p].pkts_in);
        LOAD(stats[0].pkts_out, stats_buf[p].pkts_out);
        LOAD(stats[1].pkts_in, stats_buf[2+p].pkts_in);
        LOAD(stats[1].pkts_out, stats_buf[2+p].pkts_out);
#undef LOAD

        ts[p]=1000000*t[p].tv_sec+t[p].tv_usec;
        ts_d=(double) (ts[p]-ts[p^1])/1000000.0;

#define PPS(X, I) (((double) (stats_buf[I+p].X-stats_buf[I+(p^1)].X))/ts_d)
        printf("[trunk] pkts_in: %.2lfpps pkts_out: %.2lfpps\n", PPS(pkts_in, 0), PPS(pkts_out, 0));

        printf("[fw-tap] pkts_in: %.2lfpps pkts_out: %.2lfpps\n", PPS(pkts_in, 2), PPS(pkts_out, 2));
#undef PPS

        p^=1;
    };

    return 0;
}

static int forward(__rte_unused void *arg) {
    printf("%u launched...\n", rte_lcore_id());
    const uint16_t queue_id=rte_lcore_id()>>1;
    struct rte_mbuf *bufs_rx[BURST_SIZE];

    if(rte_lcore_id()&1) {
        stats_t *stats=port_stats;

        printf("[plain_fwd] lcore_id: %u queue_id: %u\n", rte_lcore_id(), queue_id);

        while(likely(__atomic_load_n(&running, __ATOMIC_RELAXED))) {
            const uint16_t nb_rx = rte_eth_rx_burst(trunk_port_id, queue_id, bufs_rx, BURST_SIZE);

            if(unlikely(nb_rx==0))
                continue;

            for(uint16_t i=0; i<nb_rx; ++i)
                rte_ether_addr_copy(&tap_macaddr, &(rte_pktmbuf_mtod(bufs_rx[i], struct rte_ether_hdr *)->dst_addr));

            const uint16_t nb_tx = rte_eth_tx_burst(tap_port_id, queue_id, bufs_rx, nb_rx);

            __atomic_add_fetch(&stats->pkts_in, nb_rx, __ATOMIC_RELEASE);
            __atomic_add_fetch(&stats->pkts_out, nb_tx, __ATOMIC_RELEASE);

            if(unlikely(nb_tx<nb_rx)) {
                for(uint16_t b=nb_tx; b<nb_rx; ++b)
                    rte_pktmbuf_free(bufs_rx[b]);
            }
        }
    } else {
        stats_t *stats=port_stats+1;

        printf("[tap_tx] lcore_id: %u queue_id: %u\n", rte_lcore_id(), queue_id);

        while(likely(__atomic_load_n(&running, __ATOMIC_RELAXED))) {
            const uint16_t nb_rx = rte_eth_rx_burst(tap_port_id, queue_id, bufs_rx, BURST_SIZE);

            if(unlikely(nb_rx==0))
                continue;

            const uint16_t nb_tx = rte_eth_tx_burst(trunk_port_id, queue_id, bufs_rx, nb_rx);

            __atomic_add_fetch(&stats->pkts_in, nb_rx, __ATOMIC_RELEASE);
            __atomic_add_fetch(&stats->pkts_out, nb_tx, __ATOMIC_RELEASE);

            if(unlikely(nb_tx<nb_rx)) {
                for(uint16_t b=nb_tx; b<nb_rx; ++b)
                    rte_pktmbuf_free(bufs_rx[b]);
            }
        }
    }

    return 0;
}

static uint8_t find_tap_trunk_devs(uint16_t *tap_id, uint16_t *trunk_id) {
    struct rte_eth_dev_info dev_info;
    uint8_t found_ports=0, avail_eths=rte_eth_dev_count_avail();

    for(uint32_t id=0; id<avail_eths&&found_ports!=3; ++id) {
        rte_eth_dev_info_get(id, &dev_info);
        if(!strcmp(dev_info.driver_name, "net_tap")&&!(found_ports&1)) {
            *tap_id=id;
            found_ports|=1;
        } else if((~found_ports)&2) {
            *trunk_id=id;
            found_ports|=2;
        }
    }

    return found_ports!=3;
}

int main(int ac, char *as[]) {
    running=1;

    signal(SIGINT, exit_handler);
    signal(SIGKILL, exit_handler);

    int offset;

    if((offset=rte_eal_init(ac, as))<0)
        rte_exit(EXIT_FAILURE, "Error: could not init EAL.\n");
    ++offset;

    uint16_t avail_eths;

    if((avail_eths=rte_eth_dev_count_avail())<2)
        rte_exit(EXIT_FAILURE, "Error: not enough devices available.\n");

    if(find_tap_trunk_devs(&tap_port_id, &trunk_port_id))
        rte_exit(EXIT_FAILURE, "Error: could not find a tap/trunk port.\n");

    rte_eth_macaddr_get(tap_port_id, &tap_macaddr);

    if(setup_memory(&ext_mem, &mpool_payload)) {
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

#define RX_OC(X) RTE_ETH_RX_OFFLOAD_##X
#define TX_OC(X) RTE_ETH_TX_OFFLOAD_##X

    if(setup_port(trunk_port_id, &ext_mem, mpool_payload, DEFAULT_NB_QUEUES, DEFAULT_NB_QUEUES,
                  RX_OC(IPV4_CKSUM)|RX_OC(TCP_CKSUM)|RX_OC(UDP_CKSUM),
                  TX_OC(IPV4_CKSUM)|TX_OC(TCP_CKSUM)|TX_OC(UDP_CKSUM))
            |setup_port(tap_port_id, &ext_mem, mpool_payload, DEFAULT_NB_QUEUES, DEFAULT_NB_QUEUES,
                        RX_OC(IPV4_CKSUM)|RX_OC(TCP_CKSUM)|RX_OC(UDP_CKSUM),
                        TX_OC(IPV4_CKSUM)|TX_OC(TCP_CKSUM)|TX_OC(UDP_CKSUM))) {
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

#undef RX_OC
#undef TX_OC

    port_stats=rte_malloc("port_stats", sizeof(stats_t)*2, 0);
    memset(port_stats, 0, sizeof(stats_t)*2);

    uint16_t coreid=rte_get_next_lcore(rte_get_main_lcore(), 1, 1);
    for(int16_t i=0; i<DEFAULT_NB_QUEUES-1; ++i) {
        coreid=rte_get_next_lcore(coreid, 1, 1);
        rte_eal_remote_launch(forward, NULL, coreid);
        coreid=rte_get_next_lcore(coreid, 1, 1);
        rte_eal_remote_launch(forward, NULL, coreid);
    }

    coreid=rte_get_next_lcore(rte_get_main_lcore(), 1, 1);
    rte_eal_remote_launch(forward, NULL, coreid);
    forward(NULL);

    rte_eal_mp_wait_lcore();

    rte_eal_cleanup();
    return EXIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif