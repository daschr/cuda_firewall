#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>

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
#include <rte_kni.h>

#include "config.h"
#include "misc.h"

#define RTE_LOGTYPE_APP RTE_LOGTYPE_USER1

struct rte_pktmbuf_extmem ext_mem;
struct rte_mempool *mpool_payload;
uint16_t tap_port_id, trunk_port_id;
struct rte_kni *kni;

volatile uint8_t running;

void exit_handler(int e) {
    running=0;
    unsigned int i;

    RTE_LCORE_FOREACH_WORKER(i) {
        printf("[exit_handler] waiting for lcore %d...\n", i);
        rte_eal_wait_lcore(i);
    }

    puts("rte_kni_release");
    if(rte_kni_release(kni)<0) {
        fprintf(stderr, "error releasing kni\n");
    }

    puts("rte_kni_close");
    rte_kni_close();
    sleep(5);
    puts("rte_eal_cleanup");
    rte_eal_cleanup();

    exit(EXIT_SUCCESS);
}

static int fw_ingress(__rte_unused void *arg) {
    struct rte_mbuf *bufs_rx[BURST_SIZE];
    printf("[fw_ingress] lcore_id: %u\n", rte_lcore_id());

    while(running) {
        const unsigned nb_rx = (unsigned) rte_eth_rx_burst(trunk_port_id, 0, bufs_rx, BURST_SIZE);

        if(unlikely(nb_rx==0))
            continue;

        printf("[fw_ingress] transmitting %u pkts\n", nb_rx);
        const unsigned nb_tx = rte_kni_tx_burst(kni, bufs_rx, nb_rx);

        if(unlikely(nb_tx<nb_rx))
            for(uint16_t i=nb_tx; i<nb_rx; ++i)
                rte_pktmbuf_free(bufs_rx[i]);
    }

    printf("[fw_ingress] stopped\n");
    return 0;
}

static int fw_egress(void *arg) {
    struct rte_mbuf *bufs_rx[BURST_SIZE];
    struct rte_ether_addr *tap_macaddr=(struct rte_ether_addr *) arg;

    printf("[fw_egress] lcore_id: %u\n", rte_lcore_id());

    while(running) {
        const unsigned nb_rx = rte_kni_rx_burst(kni, bufs_rx, BURST_SIZE);

        if(unlikely(nb_rx==0))
            continue;

        printf("[fw_egress] nb_rx: %u\n", nb_rx);

        if(tap_macaddr)
            for(unsigned i=0; i<nb_rx; ++i)
                rte_memcpy(&(rte_pktmbuf_mtod(bufs_rx[i], struct rte_ether_hdr *)->dst_addr), tap_macaddr, 6);

        const unsigned nb_tx = (unsigned) rte_eth_tx_burst(trunk_port_id, 0, bufs_rx, (uint16_t) nb_rx);

        if(unlikely(nb_tx<nb_rx))
            for(uint16_t i=nb_tx; i<nb_rx; ++i)
                rte_pktmbuf_free(bufs_rx[i]);
    }

    printf("[fw_egress] stopped\n");
    return 0;
}

static int poll_handle_requests(void *arg) {
    while(running) {
        rte_kni_handle_request(kni);
        rte_delay_ms(500);
    }
    return 0;
}

int main(int ac, char *as[]) {
    running=1;

    signal(SIGINT, exit_handler);
    signal(SIGKILL, exit_handler);

    int offset;

    if((offset=rte_eal_init(ac, as))<0)
        rte_exit(EXIT_FAILURE, "Error: could not init EAL.\n");
    ++offset;

    if(rte_kni_init(1)<0)
        rte_exit(EXIT_FAILURE, "Error: could not init kni.\n");

    uint16_t avail_eths;

    if((avail_eths=rte_eth_dev_count_avail())<1)
        rte_exit(EXIT_FAILURE, "Error: not enough devices available.\n");

    trunk_port_id=0;

    if(setup_memory(&ext_mem, &mpool_payload)) {
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }


#define RX_OC(X) RTE_ETH_RX_OFFLOAD_##X
#define TX_OC(X) RTE_ETH_TX_OFFLOAD_##X

    if(setup_port(trunk_port_id, &ext_mem, mpool_payload,
                  RX_OC(IPV4_CKSUM)|RX_OC(TCP_CKSUM)|RX_OC(UDP_CKSUM),
                  TX_OC(IPV4_CKSUM)|TX_OC(TCP_CKSUM)|TX_OC(UDP_CKSUM))) {
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

    puts("done setting up trunk port");
#undef RX_OC
#undef TX_OC

    puts("setup kni_port");
    kni=setup_kni_port(trunk_port_id, 8, mpool_payload);
    puts("done setup kni_port");
    if(kni==NULL)
        rte_exit(EXIT_FAILURE, "Error: could not init kni\n");

    rte_kni_update_link(kni, 1);

    unsigned int coreid=rte_get_next_lcore(rte_get_main_lcore(), 1, 1);
    rte_eal_remote_launch(fw_egress, NULL, coreid);

    coreid=rte_get_next_lcore(coreid, 1, 1);
    rte_eal_remote_launch(fw_ingress, NULL, coreid);

    coreid=rte_get_next_lcore(coreid, 1, 1);
    rte_eal_remote_launch(poll_handle_requests, NULL, coreid);

    RTE_LCORE_FOREACH_WORKER(coreid)
    rte_eal_wait_lcore(coreid);

    puts("HERE");

    rte_eal_cleanup();
    return EXIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
