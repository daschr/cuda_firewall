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

#include <cuda_runtime.h>

#include "rte_bv_classifier.h"
#include "parser.h"
#include "config.h"
#include "misc.h"

#define DEFAULT_NB_QUEUES 1
#define RTE_LOGTYPE_APP RTE_LOGTYPE_USER1

struct rte_pktmbuf_extmem ext_mem;
struct rte_mempool *mpool_payload;
struct rte_bv_classifier *classifier;
ruleset_t ruleset;
uint16_t tap_port_id, trunk_port_id;

volatile uint8_t running;

typedef struct {
    struct rte_ether_addr *tap_macaddr;
} callback_payload_t;

void exit_handler(int e) {
    running=0;

    rte_eal_mp_wait_lcore();

    free_ruleset(&ruleset);

    rte_bv_classifier_free(classifier);

    rte_eal_cleanup();

    exit(EXIT_SUCCESS);
}

void tx_callback(struct rte_mbuf **pkts, uint64_t pkts_mask, uint64_t lookup_hit_mask, void **actions_r, void *p_r) {
    uint8_t **actions=(uint8_t **) actions_r;
    callback_payload_t *p=(callback_payload_t *) p_r;
    const uint16_t nb_rx=__builtin_popcount(pkts_mask);
    uint16_t i=0, j=0;
    struct rte_mbuf *bufs_tx[BURST_SIZE];

    printf("[tx_callback] nb_rx: %hu lookup_hit_mask: %08lX\n", nb_rx, lookup_hit_mask);

    for(; i<nb_rx; ++i) {
        if(unlikely(!((lookup_hit_mask>>i)&1))) {
            bufs_tx[j++]=pkts[i];
            if(p->tap_macaddr)
                rte_memcpy(&(rte_pktmbuf_mtod(pkts[i], struct rte_ether_hdr *)->dst_addr), p->tap_macaddr, 6);
            continue;
        }

        if(unlikely(*(actions[i])==RULE_DROP))
            continue;

        bufs_tx[j++]=pkts[i];
        if(p->tap_macaddr)
            rte_memcpy(&(rte_pktmbuf_mtod(pkts[i], struct rte_ether_hdr *)->dst_addr), p->tap_macaddr, 6);
    }

    const uint16_t nb_tx = rte_eth_tx_burst(tap_port_id, 0, bufs_tx, j);

    if(unlikely(nb_tx<nb_rx)) {
        for(uint16_t b=nb_tx; b<nb_rx; ++b)
            rte_pktmbuf_free(pkts[b]);
    }
}

int trunk_rx(void *arg) {
    printf("[trunk_rx] launched\n");
    struct rte_bv_classifier *c=(struct rte_bv_classifier *) arg;
    struct rte_mbuf *bufs_rx[BURST_SIZE];
    uint64_t pkts_mask;

    while(running) {
        const uint16_t nb_rx = rte_eth_rx_burst(trunk_port_id, 0, bufs_rx, BURST_SIZE);

        if(unlikely(nb_rx==0))
            continue;

        pkts_mask=(1<<nb_rx)-1;

        printf("[trunk_rx] pkts_mask: %08lX\n", pkts_mask);
        rte_bv_classifier_enqueue_burst(c, bufs_rx, pkts_mask);
    }

    return 0;
}

static __rte_noreturn void trunk_tx(struct rte_bv_classifier *c, struct rte_ether_addr *tap_macaddr) {
    printf("[trunk_tx] launched\n");
    callback_payload_t payload= { .tap_macaddr=tap_macaddr};

    rte_bv_classifier_poll_lookups(c, tx_callback, (void *) &payload);
}

static int tap_tx(__rte_unused void *arg) {
    printf("[tap_tx] launched\n");
    struct rte_mbuf *bufs_rx[BURST_SIZE];

    while(running) {
        const uint16_t nb_rx = rte_eth_rx_burst(tap_port_id, 0, bufs_rx, BURST_SIZE);

        if(unlikely(nb_rx==0))
            continue;

        const uint16_t nb_tx = rte_eth_tx_burst(trunk_port_id, 0, bufs_rx, nb_rx);

        printf("[tap_tx] nb_tx: %hu\n", nb_tx);

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

    for(uint32_t id=0; id<avail_eths && found_ports!=3; ++id) {
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
    if(ac==1) {
        fprintf(stderr, "Usage: %s [rules]\n", as[0]);
        return EXIT_FAILURE;
    }

    signal(SIGINT, exit_handler);
    signal(SIGKILL, exit_handler);
//    signal(SIGSEGV, exit_handler);

    int offset;

    if((offset=rte_eal_init(ac, as))<0)
        rte_exit(EXIT_FAILURE, "Error: could not init EAL.\n");
    ++offset;

    if(offset>=ac) {
        rte_exit(EXIT_FAILURE, "Usage: %s [[rte  arguments]...] [rules]\n", as[0]);
    }

    ac-=offset;
    as=as+offset;

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

    memset(&ruleset, 0, sizeof(ruleset_t));

    if(!parse_ruleset(&ruleset, as[0])) {
        fprintf(stderr, "Error: could not parse ruleset \"%s\"\n", as[0]);
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

    printf("parsed ruleset \"%s\" with %lu rules\n", as[0], ruleset.num_rules);

    struct rte_bv_classifier_field_def fdefs[5];
    uint32_t fdefs_offsets[5]= {	offsetof(struct rte_ipv4_hdr, src_addr),
                                    offsetof(struct rte_ipv4_hdr, dst_addr),
                                    sizeof(struct rte_ipv4_hdr)+offsetof(struct rte_tcp_hdr, src_port),
                                    sizeof(struct rte_ipv4_hdr)+offsetof(struct rte_tcp_hdr, dst_port),
                                    offsetof(struct rte_ipv4_hdr, next_proto_id)
                               },
                               fdefs_sizes[5]= {4,4,2,2,1},
    ptype_masks[5] = {
        RTE_PTYPE_L2_MASK|RTE_PTYPE_L3_IPV4|RTE_PTYPE_L4_MASK,
        RTE_PTYPE_L2_MASK|RTE_PTYPE_L3_IPV4|RTE_PTYPE_L4_MASK,
        RTE_PTYPE_L2_MASK|RTE_PTYPE_L3_IPV4|RTE_PTYPE_L4_TCP|RTE_PTYPE_L4_UDP,
        RTE_PTYPE_L2_MASK|RTE_PTYPE_L3_IPV4|RTE_PTYPE_L4_TCP|RTE_PTYPE_L4_UDP,
        RTE_PTYPE_L2_ETHER|RTE_PTYPE_L3_MASK|RTE_PTYPE_L4_MASK
    };

    for(size_t i=0; i<5; ++i) {
        fdefs[i].offset=sizeof(struct rte_ether_hdr) + fdefs_offsets[i];
        fdefs[i].type=RTE_BV_CLASSIFIER_FIELD_TYPE_RANGE;
        fdefs[i].ptype_mask=ptype_masks[i];
        fdefs[i].size=fdefs_sizes[i];
    }

    struct rte_bv_classifier_params classifier_params = { .num_fields=5, .field_defs=fdefs, .num_rules=ruleset.num_rules };

    classifier=rte_bv_classifier_create(&classifier_params, rte_socket_id(), 1);

    if(classifier==NULL) {
        printf("classifier==NULL\n");
        goto err;
    }

    uint8_t **actions=rte_malloc("actions", ruleset.num_rules*sizeof(uint8_t *), sizeof(uint8_t *));
    for(uint32_t i=0; i<ruleset.num_rules; ++i)
        actions[i]=&ruleset.actions[i];

    if(rte_bv_classifier_entry_add_bulk(classifier, ruleset.rules, (void **) actions, ruleset.num_rules, NULL, NULL))
        goto err;

    free_ruleset_except_actions(&ruleset);

    uint16_t coreid=rte_get_next_lcore(rte_get_main_lcore(), 1, 1);
    rte_eal_remote_launch(tap_tx, NULL, coreid);

    coreid=rte_get_next_lcore(coreid, 1, 1);
    rte_eal_remote_launch(trunk_rx, (void *) classifier, 2);

    trunk_tx(classifier, &tap_macaddr);

    rte_eal_mp_wait_lcore();

    rte_bv_classifier_free(classifier);

    free(ruleset.actions);

    rte_eal_cleanup();
    return EXIT_SUCCESS;

err:
    free_ruleset(&ruleset);
    rte_eal_cleanup();
    return EXIT_FAILURE;
}

#ifdef __cplusplus
}
#endif
