#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <signal.h>

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

#define RTE_LOGTYPE_APP RTE_LOGTYPE_USER1

struct rte_pktmbuf_extmem ext_mem;
struct rte_mempool *mpool_payload;
uint16_t tap_port_id, trunk_port_id;

volatile uint8_t running;

void exit_handler(int e) {
    running=0;

    for(int i=0; i<1; ++i) {
        printf("[exit_handler] waiting for lcore %d...\n", i);
        rte_eal_wait_lcore(i);
        printf("lcore %d stopped...\n", i);
    }

    rte_eal_cleanup();

    exit(EXIT_SUCCESS);
}

static __rte_noreturn void plain_fwd(struct rte_ether_addr *tap_macaddr) {
    struct rte_mbuf *bufs_rx[BURST_SIZE];

    for(;;) {
        const uint16_t nb_rx = rte_eth_rx_burst(trunk_port_id, 0, bufs_rx, BURST_SIZE);

        if(unlikely(nb_rx==0))
            continue;
        for(uint16_t i=0; i<nb_rx; ++i)
            rte_memcpy(&(rte_pktmbuf_mtod(bufs_rx[i], struct rte_ether_hdr *)->dst_addr), tap_macaddr, 6);

        const uint16_t nb_tx = rte_eth_tx_burst(tap_port_id, 0, bufs_rx, nb_rx);

        if(unlikely(nb_tx<nb_rx)) {
            for(uint16_t b=nb_tx; b<nb_rx; ++b)
                rte_pktmbuf_free(bufs_rx[b]);
        }
    }
}

static int tap_tx(__rte_unused void *arg) {
    struct rte_mbuf *bufs_rx[BURST_SIZE];

    while(running) {
        const uint16_t nb_rx = rte_eth_rx_burst(tap_port_id, 0, bufs_rx, BURST_SIZE);

        if(unlikely(nb_rx==0))
            continue;

        const uint16_t nb_tx = rte_eth_tx_burst(trunk_port_id, 0, bufs_rx, nb_rx);

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
    signal(SIGSEGV, exit_handler);

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

    if(setup_port(trunk_port_id, &ext_mem, mpool_payload,
                  TX_OC(IPV4_CKSUM)|TX_OC(TCP_CKSUM)|TX_OC(UDP_CKSUM),
                  TX_OC(IPV4_CKSUM)|TX_OC(TCP_CKSUM)|TX_OC(UDP_CKSUM))
            |setup_port(tap_port_id, &ext_mem, mpool_payload,
                        TX_OC(IPV4_CKSUM)|TX_OC(TCP_CKSUM)|TX_OC(UDP_CKSUM),
                        TX_OC(IPV4_CKSUM)|TX_OC(TCP_CKSUM)|TX_OC(UDP_CKSUM))) {
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

#undef RX_OC
#undef TX_OC

    rte_eal_wait_lcore(1);
    rte_eal_remote_launch(tap_tx, NULL, 1);

    plain_fwd(&tap_macaddr);

    rte_eal_cleanup();
    return EXIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
