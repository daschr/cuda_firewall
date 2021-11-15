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
stats_t *port_stats=NULL;
volatile uint8_t running;

void exit_handler(int e) {
    running=0;

    unsigned int i;
    RTE_LCORE_FOREACH_WORKER(i) {
        printf("[exit_handler] waiting for lcore %d...\n", i);
        rte_eal_wait_lcore(i);
    }

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

    while(running) {
        rte_delay_ms(2000);
        gettimeofday(t+p, NULL);
        stats_buf[p]=stats[0];
        stats_buf[2+p]=stats[1];
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

static int plain_fwd(void *arg) {
    struct rte_ether_addr *tap_macaddr=(struct rte_ether_addr *) arg;
    struct rte_mbuf *bufs_rx[BURST_SIZE];
    stats_t *stats=port_stats;

    while(running) {
        const uint16_t nb_rx = rte_eth_rx_burst(trunk_port_id, 0, bufs_rx, BURST_SIZE);

        if(unlikely(nb_rx==0))
            continue;
        for(uint16_t i=0; i<nb_rx; ++i)
            rte_memcpy(&(rte_pktmbuf_mtod(bufs_rx[i], struct rte_ether_hdr *)->dst_addr), tap_macaddr, 6);

        const uint16_t nb_tx = rte_eth_tx_burst(tap_port_id, 0, bufs_rx, nb_rx);

        stats->pkts_in+=nb_rx;
        stats->pkts_out+=nb_tx;

        if(unlikely(nb_tx<nb_rx)) {
            for(uint16_t b=nb_tx; b<nb_rx; ++b)
                rte_pktmbuf_free(bufs_rx[b]);
        }
    }

    return 0;
}

static int tap_tx(__rte_unused void *arg) {
    struct rte_mbuf *bufs_rx[BURST_SIZE];
    stats_t *stats=port_stats+1;

    while(running) {
        const uint16_t nb_rx = rte_eth_rx_burst(tap_port_id, 0, bufs_rx, BURST_SIZE);

        if(unlikely(nb_rx==0))
            continue;

        const uint16_t nb_tx = rte_eth_tx_burst(trunk_port_id, 0, bufs_rx, nb_rx);

        stats->pkts_in+=nb_rx;
        stats->pkts_out+=nb_tx;

        if(unlikely(nb_tx<nb_rx)) {
            for(uint16_t b=nb_tx; b<nb_rx; ++b)
                rte_pktmbuf_free(bufs_rx[b]);
        }
    }

    return 0;
}

static uint8_t find_tap_trunk_devs(uint16_t *tap_id, uint16_t *trunk_id) {
    struct rte_eth_dev_info dev_info;
    uint8_t found_ports=0, avail_eths=rte_eth_dev_count_avail();

    for(uint32_t id=0; id<avail_eths&&found_ports!=3; ++id) {
        rte_eth_dev_info_get(id, &dev_info);
        if(strcmp(dev_info.driver_name, "net_tap")==0&&!(found_ports&1)) {
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
    struct rte_ether_addr tap_macaddr;

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

    if(setup_port(trunk_port_id, &ext_mem, mpool_payload, 1, 1,
                  RX_OC(IPV4_CKSUM)|RX_OC(TCP_CKSUM)|RX_OC(UDP_CKSUM),
                  TX_OC(IPV4_CKSUM)|TX_OC(TCP_CKSUM)|TX_OC(UDP_CKSUM))
            |setup_port(tap_port_id, &ext_mem, mpool_payload, 1, 1,
                        RX_OC(IPV4_CKSUM)|RX_OC(TCP_CKSUM)|RX_OC(UDP_CKSUM),
                        TX_OC(IPV4_CKSUM)|TX_OC(TCP_CKSUM)|TX_OC(UDP_CKSUM))) {
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

#undef RX_OC
#undef TX_OC

    port_stats=rte_malloc("port_stats", sizeof(stats_t)*2, 0);
    memset(port_stats, 0, sizeof(stats_t)*2);

    unsigned int coreid=rte_get_next_lcore(rte_get_main_lcore(), 1, 1);
    rte_eal_remote_launch(tap_tx, NULL, coreid);

    coreid=rte_get_next_lcore(coreid, 1, 1);
    rte_eal_remote_launch(plain_fwd, &tap_macaddr, coreid);

    coreid=rte_get_next_lcore(coreid, 1, 1);
    rte_eal_remote_launch(print_stats, port_stats, coreid);

    RTE_LCORE_FOREACH_WORKER(coreid)
    rte_eal_wait_lcore(coreid);

    rte_eal_cleanup();
    return EXIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
