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

#include "rte_table_bv.h"
#include "parser.h"
#include "config.h"
#include "misc.h"

#define RTE_LOGTYPE_APP RTE_LOGTYPE_USER1

struct rte_pktmbuf_extmem ext_mem;
struct rte_mempool *mpool_payload;
void *table;
ruleset_t ruleset;
uint16_t tap_port_id, trunk_port_id;

volatile uint8_t running;

void exit_handler(int e) {
    running=0;
    printf("[exit_handler] waiting for lcore 1...\n");
    rte_eal_wait_lcore(1);
    puts("lcore 1 stopped...");

    free_ruleset(&ruleset);

    rte_table_bv_ops.f_free(table);

    rte_eal_cleanup();

    exit(EXIT_SUCCESS);
}

static __rte_noreturn void firewall(void *table, uint8_t *actions, struct rte_ether_addr *tap_macaddr) {
    volatile uint64_t *lookup_hit_mask, *lookup_hit_mask_d, pkts_mask;
    volatile uint32_t *positions, *positions_d;

    cudaHostAlloc((void **) &positions, sizeof(uint32_t)*BURST_SIZE, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **) &positions_d, (uint32_t *) positions, 0);

    cudaHostAlloc((void **) &lookup_hit_mask, sizeof(uint64_t), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **) &lookup_hit_mask_d, (uint64_t *) lookup_hit_mask, 0);

    struct rte_mbuf **bufs_rx;
    struct rte_mbuf **bufs_rx_d;
    cudaHostAlloc((void **) &bufs_rx, sizeof(struct rte_mbuf*)*BURST_SIZE, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **) &bufs_rx_d, bufs_rx, 0);

    struct rte_mbuf *bufs_tx[BURST_SIZE];

    const int (*lookup) (void *, struct rte_mbuf **, uint64_t, uint64_t *, void **)=rte_table_bv_ops.f_lookup;

    int16_t i,j;

    *lookup_hit_mask=0;
    for(;; *lookup_hit_mask=0) {
        const uint16_t nb_rx = rte_eth_rx_burst(trunk_port_id, 0, bufs_rx, BURST_SIZE);

        if(unlikely(nb_rx==0))
            continue;

        pkts_mask=(1<<nb_rx)-1;

        lookup(table, bufs_rx_d, pkts_mask, (uint64_t *) lookup_hit_mask_d, (void **) positions_d);

        i=0;
        j=0;

        for(; i<nb_rx; ++i) {
            if(!(  (*lookup_hit_mask>>i)&1  &  (actions[positions[i]]==RULE_DROP) )) {
                bufs_tx[j++]=bufs_rx[i];
                if(tap_macaddr!=NULL)
                    rte_memcpy(&(rte_pktmbuf_mtod(bufs_rx[i], struct rte_ether_hdr *)->d_addr), tap_macaddr, 6);
            }
        }

        const uint16_t nb_tx = rte_eth_tx_burst(tap_port_id, 0, bufs_tx, j);

        if(unlikely(nb_tx<nb_rx)) {
            for(uint16_t b=nb_tx; b<nb_rx; ++b)
                rte_pktmbuf_free(bufs_rx[b]);
        }
    }
}

static int tap_tx(__rte_unused void *arg) {
    struct rte_mbuf *bufs_rx[BURST_SIZE];

    for(; running;) {
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

    for(uint32_t id=0; id<avail_eths&found_ports!=3; ++id) {
        rte_eth_dev_info_get(id, &dev_info);
        if(strcmp(dev_info.driver_name, "net_tap")==0&!(found_ports&1)) {
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
    signal(SIGSEGV, exit_handler);

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

    if(setup_port(trunk_port_id, &ext_mem, mpool_payload)|setup_port(tap_port_id, &ext_mem, mpool_payload)) {
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

    memset(&ruleset, 0, sizeof(ruleset_t));

    if(!parse_ruleset(&ruleset, as[0])) {
        fprintf(stderr, "Error: could not parse ruleset \"%s\"\n", as[0]);
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

    printf("parsed ruleset \"%s\" with %lu rules\n", as[0], ruleset.num_rules);

    struct rte_table_bv_field_def fdefs[5];
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
        fdefs[i].type=RTE_TABLE_BV_FIELD_TYPE_RANGE;
        fdefs[i].ptype_mask=ptype_masks[i];
        fdefs[i].size=fdefs_sizes[i];
    }

    struct rte_table_bv_params table_params = { .num_fields=5, .field_defs=fdefs };

    void *table=rte_table_bv_ops.f_create(&table_params, rte_socket_id(), 0);

    if(table==NULL)
        goto err;

    rte_table_bv_ops.f_add_bulk(table, (void **) ruleset.rules, NULL, ruleset.num_rules, NULL, NULL);

    free_ruleset_except_actions(&ruleset);

    rte_eal_wait_lcore(1);
    rte_eal_remote_launch(tap_tx, NULL, 1);

    firewall(table, ruleset.actions, &tap_macaddr);

    rte_table_bv_ops.f_free(table);

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
