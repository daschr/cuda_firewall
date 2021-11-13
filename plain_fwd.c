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

    for(int i=1; i<3; ++i) {
        printf("[exit_handler] waiting for lcore %d...\n", i);
        rte_eal_wait_lcore(i);
        printf("lcore %d stopped...\n", i);
    }

    sleep(5);
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

    while(running) {
        const uint16_t nb_rx = rte_eth_rx_burst(trunk_port_id, 0, bufs_rx, BURST_SIZE);

        if(unlikely(nb_rx==0))
            continue;

        const uint16_t nb_tx = rte_kni_tx_burst(kni, bufs_rx, nb_rx);

        if(unlikely(nb_tx<nb_rx))
            for(uint16_t i=nb_tx; i<nb_rx; ++i)
                rte_pktmbuf_free(bufs_rx[i]);
    }

    printf("[fw_ingress] stopped\n");
    return 0;
}

static int fw_egress(__rte_unused void *arg) {
    struct rte_mbuf *bufs_rx[BURST_SIZE];

    while(running) {
        const uint16_t nb_rx = rte_kni_rx_burst(kni, bufs_rx, BURST_SIZE);

        if(unlikely(nb_rx==0))
            continue;

        const uint16_t nb_tx = rte_eth_tx_burst(trunk_port_id, 0, bufs_rx, nb_rx);

        if(unlikely(nb_tx<nb_rx))
            for(uint16_t i=nb_tx; i<nb_rx; ++i)
                rte_pktmbuf_free(bufs_rx[i]);
    }

    printf("[fw_egress] stopped\n");
    return 0;
}

static __rte_noreturn void poll_handle_requests(void) {
    for(;;) {
        if(running)	rte_kni_handle_request(kni);
        usleep(500);
    }
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

    puts("setup kni_port");
    kni=setup_kni_port(trunk_port_id, 0, 0, mpool_payload);
    puts("done setup kni_port");
    if(kni==NULL) {
        rte_exit(EXIT_FAILURE, "Error: could not init kni\n");
    }

#define RX_OC(X) RTE_ETH_RX_OFFLOAD_##X
#define TX_OC(X) RTE_ETH_TX_OFFLOAD_##X

    if(setup_port(trunk_port_id, &ext_mem, mpool_payload,
                  TX_OC(IPV4_CKSUM)|TX_OC(TCP_CKSUM)|TX_OC(UDP_CKSUM),
                  TX_OC(IPV4_CKSUM)|TX_OC(TCP_CKSUM)|TX_OC(UDP_CKSUM))) {
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

#undef RX_OC
#undef TX_OC

    puts("done setting up trunk port");

    rte_eal_wait_lcore(1);
    rte_eal_remote_launch(fw_egress, NULL, 1);

    rte_eal_wait_lcore(2);
    rte_eal_remote_launch(fw_ingress, NULL, 2);

    for(;;) sleep(10);

    puts("HERE");
    rte_eal_cleanup();
    return EXIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
