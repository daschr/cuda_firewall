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

#include <cuda_runtime.h>

#include <rte_acl.h>
#include <rte_table.h>
#include <rte_table_acl.h>

#include "rte_table_bv.h"
#include "parser.h"
#include "acl_parser.h"
#include "config.h"
#include "misc.h"
#include "stats.h"

#define RTE_LOGTYPE_APP RTE_LOGTYPE_USER1
//#define MEASURE_TIME
//#define DO_NOT_TRANSMIT_TO_TAP

typedef struct {
    void *table_gpu;
    void *table_dpdk;
    uint32_t *position_data;
    struct rte_ether_addr *tap_macaddr;
    stats_t *stats;
} __rte_cache_aligned firewall_conf_t;

struct rte_pktmbuf_extmem ext_mem;
struct rte_mempool *mpool_payload;
void *table_gpu;
void *table_dpdk;
ruleset_t ruleset;
acl_ruleset_t ruleset_acl;
uint32_t *position_data;
uint16_t tap_port_id, trunk_port_id;
unsigned long timestamp;

stats_t *old_port_stats=NULL, *port_stats=NULL;

firewall_conf_t fw_conf;

volatile uint8_t running;
int *key_found=NULL;
uint8_t **entry_handles=NULL;

void exit_handler(int e) {
    running=0;

    rte_eal_mp_wait_lcore();

    free_ruleset(&ruleset);
    acl_free_ruleset(&ruleset_acl);

    rte_table_bv_ops.f_free(table_gpu);
    rte_table_acl_ops.f_free(table_dpdk);

    rte_free(port_stats);
    rte_free(old_port_stats);
    rte_free(key_found);
    rte_free(entry_handles);
    rte_free(position_data);

    rte_eal_cleanup();

    exit(EXIT_SUCCESS);
}

void print_stats(__rte_unused int e) {
    struct timeval t;
    gettimeofday(&t, NULL);

    unsigned long new_ts=1000000*t.tv_sec+t.tv_usec;
    double ts_d=(double) (new_ts-timestamp)/1000000.0;

#define PPS(X, P) (((double) (port_stats[P].X-old_port_stats[P].X))/ts_d)
    printf("[trunk] pkts_in: %.2lfpps pkts_out: %.2lfpps pkts_dropped: %.2lfpps pkts_accepted: %.2lfpps pkts_lookup_hit_miss: %.2lfpps\n",
           PPS(pkts_in, 0), PPS(pkts_out, 0), PPS(pkts_dropped, 0), PPS(pkts_accepted, 0), PPS(pkts_lookup_miss, 0));
    old_port_stats[0]=port_stats[0];

    printf("[fw-tap] pkts_in: %.2lfpps pkts_out: %.2lfpps pkts_dropped: %.2lfpps pkts_accepted: %.2lfpps pkts_lookup_miss: %.2lfpps\n",
           PPS(pkts_in, 1), PPS(pkts_out, 1), PPS(pkts_dropped, 1), PPS(pkts_accepted, 1), PPS(pkts_lookup_miss, 1));
    old_port_stats[1]=port_stats[1];
#undef PPS

    timestamp=new_ts;
}

static int firewall(void *arg) {
    firewall_conf_t *conf=(firewall_conf_t *) arg;

    const uint16_t queue_id=rte_lcore_id()>>1;

    stats_t *stats=conf->stats+((rte_lcore_id()&1)^1);

#ifdef MEASURE_TIME
    struct timeval t1,t2;
#endif

    if(rte_lcore_id()&1) {
        uint32_t **positions_gpu, **positions_gpu_d;
        uint32_t *positions_dpdk[BURST_SIZE];
        uint64_t lookup_hit_mask;

        cudaHostAlloc((void **) &positions_gpu, sizeof(uint32_t *)*BURST_SIZE, cudaHostAllocMapped);
        cudaHostGetDevicePointer((void **) &positions_gpu_d, (uint32_t **) positions_gpu, 0);

        struct rte_mbuf **bufs_rx;
        struct rte_mbuf **bufs_rx_d;
        cudaHostAlloc((void **) &bufs_rx, sizeof(struct rte_mbuf*)*BURST_SIZE, cudaHostAllocMapped);
        cudaHostGetDevicePointer((void **) &bufs_rx_d, bufs_rx, 0);

        uint8_t **pkts_data, *lookup_hit_vec;
        cudaHostAlloc((void **) &pkts_data, sizeof(uint8_t*)*BURST_SIZE, cudaHostAllocMapped|cudaHostAllocWriteCombined);
        cudaHostAlloc((void **) &lookup_hit_vec, sizeof(uint8_t*)*BURST_SIZE, cudaHostAllocMapped);

#ifndef DO_NOT_TRANSMIT_TO_TAP
        struct rte_mbuf *bufs_tx[BURST_SIZE];
#endif

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        int16_t i,j;
        uint64_t pkts_mask;

        while(likely(running)) {
            const uint16_t nb_rx = rte_eth_rx_burst(trunk_port_id, queue_id, bufs_rx, BURST_SIZE);

            if(unlikely(nb_rx==0))
                continue;

            pkts_mask=nb_rx<64?(1LU<<nb_rx)-1LU:64LU;
#ifdef MEASURE_TIME
            gettimeofday(&t1, NULL);
#endif

            rte_table_bv_lookup_stream(conf->table_gpu, stream, lookup_hit_vec, pkts_data, bufs_rx_d, nb_rx, (void **) positions_gpu_d);
            rte_table_acl_ops.f_lookup(conf->table_dpdk, bufs_rx, pkts_mask, &lookup_hit_mask, (void **) positions_dpdk);
            
			for(uint64_t pos=0; pos<nb_rx; ++pos) {
                if((lookup_hit_mask>>pos)&1) {
                    if(lookup_hit_vec[pos]){
                        if(*positions_gpu[pos]!=*positions_dpdk[pos]){
							printf("positions_gpu[%1$lu]=%2$u != %3$u=positions_dpdk[%1$lu]\n", pos, *positions_gpu[pos], *positions_dpdk[pos]);
						}
					}else{
                    	printf("positions_dpdk[%lu]=%u\n", pos, *positions_dpdk[pos]);
                        printf("positions_gpu[%lu]=%p\n", pos, positions_gpu[pos]);
                	}
				}
            }


#ifdef MEASURE_TIME
            gettimeofday(&t2, NULL);
            printf("LOOKUP took %luus\n", (t2.tv_sec*1000000+t2.tv_usec)-(t1.tv_sec*1000000+t1.tv_usec));
#endif

            i=0;
            j=0;

#ifdef DO_NOT_TRANSMIT_TO_TAP
            for(; i<nb_rx; ++i) {
                if(unlikely(!lookup_hit_vec[i])) {
                    ++j;
                    ++(stats->pkts_lookup_miss);
                    continue;
                }

                ++j;
            }

            rte_pktmbuf_free_bulk(bufs_rx, nb_rx);

            stats->pkts_in+=nb_rx;
            stats->pkts_out+=j;
            stats->pkts_accepted+=j;
            stats->pkts_dropped+=nb_rx-j;
#else
            for(; i<nb_rx; ++i) {
                if(unlikely(!lookup_hit_vec[i])) {
                    bufs_tx[j++]=bufs_rx[i];
                    if(conf->tap_macaddr!=NULL)
                        rte_memcpy(&(rte_pktmbuf_mtod(bufs_rx[i], struct rte_ether_hdr *)->dst_addr), conf->tap_macaddr, 6);

                    ++(stats->pkts_lookup_miss);
                    continue;
                }

                bufs_tx[j++]=bufs_rx[i];
                if(conf->tap_macaddr!=NULL)
                    rte_memcpy(&(rte_pktmbuf_mtod(bufs_rx[i], struct rte_ether_hdr *)->dst_addr), conf->tap_macaddr, 6);
            }

            const uint16_t nb_tx = rte_eth_tx_burst(tap_port_id, queue_id, bufs_tx, j);

            stats->pkts_in+=nb_rx;
            stats->pkts_out+=nb_tx;
            stats->pkts_accepted+=j;
            stats->pkts_dropped+=nb_rx-j;
#endif
        }
    } else {
        struct rte_mbuf *bufs_rx[BURST_SIZE];

        while(likely(running)) {
            const uint16_t nb_rx = rte_eth_rx_burst(tap_port_id, queue_id, bufs_rx, BURST_SIZE);

            if(unlikely(nb_rx==0))
                continue;

            const uint16_t nb_tx = rte_eth_tx_burst(trunk_port_id, queue_id, bufs_rx, nb_rx);

            stats->pkts_in+=nb_rx;
            stats->pkts_out+=nb_tx;

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

    struct timeval t;
    gettimeofday(&t, NULL);
    timestamp=1000000*t.tv_sec+t.tv_usec;

    signal(SIGINT, exit_handler);
    signal(SIGKILL, exit_handler);
    signal(SIGUSR1, print_stats);

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

    port_stats=rte_malloc("port_stats", sizeof(stats_t)*2, 0);
    old_port_stats=rte_malloc("old_port_stats", sizeof(stats_t)*2, 0);
    memset(port_stats, 0, sizeof(stats_t)*2);
    memset(old_port_stats, 0, sizeof(stats_t)*2);

    memset(&ruleset, 0, sizeof(ruleset_t));
    memset(&ruleset_acl, 0, sizeof(acl_ruleset_t));

    if(!parse_ruleset(&ruleset, as[0])) {
        fprintf(stderr, "Error: could not parse ruleset \"%s\" for gpu!\n", as[0]);
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

    if(!acl_parse_ruleset(&ruleset_acl, as[0])) {
        fprintf(stderr, "Error: could not parse ruleset \"%s\" for dpdk!\n", as[0]);
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

    printf("parsed ruleset \"%s\" with %lu rules\n", as[0], ruleset.num_rules);
    printf("parsed acl ruleset \"%s\" with %lu rules\n", as[0], ruleset_acl.num_rules);

    struct rte_table_bv_field_def fdefs[5];
    uint32_t fdefs_offsets[5]= {	offsetof(struct rte_ipv4_hdr, next_proto_id),
                                    offsetof(struct rte_ipv4_hdr, src_addr),
                                    offsetof(struct rte_ipv4_hdr, dst_addr),
                                    sizeof(struct rte_ipv4_hdr)+offsetof(struct rte_tcp_hdr, src_port),
                                    sizeof(struct rte_ipv4_hdr)+offsetof(struct rte_tcp_hdr, dst_port)
                               },
                               fdefs_sizes[5]= {1,4,4,2,2};

    for(size_t i=0; i<5; ++i) {
        fdefs[i].offset=sizeof(struct rte_ether_hdr) + fdefs_offsets[i];
        fdefs[i].type=RTE_TABLE_BV_FIELD_TYPE_RANGE;
        fdefs[i].size=fdefs_sizes[i];
    }

    struct rte_table_bv_params table_bv_params = { .num_fields=5, .field_defs=fdefs, .num_rules=ruleset.num_rules };

    table_gpu=rte_table_bv_ops.f_create(&table_bv_params, rte_socket_id(), sizeof(uint32_t));

    if(table_gpu==NULL) {
        fprintf(stderr, "could not create gpu table\n");
        goto err;
    }

    struct rte_table_acl_params table_dpdk_params;

    table_dpdk_params.name="5tuple";
    table_dpdk_params.n_rules=100001;
    table_dpdk_params.n_rule_fields=5;

    for(uint8_t i=0; i<5; ++i) {
        table_dpdk_params.field_format[i].offset=sizeof(struct rte_ether_hdr) + fdefs_offsets[i];
        table_dpdk_params.field_format[i].type=RTE_ACL_FIELD_TYPE_RANGE;
        table_dpdk_params.field_format[i].size=fdefs_sizes[i];
        table_dpdk_params.field_format[i].field_index=i;
        table_dpdk_params.field_format[i].input_index=i==4?3:i;
    }


    table_dpdk=rte_table_acl_ops.f_create(&table_dpdk_params, rte_socket_id(), sizeof(uint32_t));

    if(table_dpdk==NULL) {
        fprintf(stderr, "could not create dpdk table\n");
        goto err;
    }

    position_data=rte_malloc("position_data", ruleset.num_rules*sizeof(uint64_t), sizeof(uint64_t));
    uint32_t **positions=rte_malloc("positions", ruleset.num_rules*sizeof(uint32_t *), sizeof(uint32_t *));
    for(uint32_t i=0; i<ruleset.num_rules; ++i) {
        position_data[i]=i;
        positions[i]=&position_data[i];
    }

    if(rte_table_bv_ops.f_add_bulk(table_gpu, (void **) ruleset.rules, (void **) positions, ruleset.num_rules, NULL, NULL))
        goto err;

    key_found=rte_malloc("key_found", sizeof(int)*ruleset.num_rules, sizeof(int));
    entry_handles=rte_malloc("entry_handles", sizeof(uint8_t *)*ruleset.num_rules, sizeof(int8_t *));

    puts("add bulk");
    if(rte_table_acl_ops.f_add_bulk(table_dpdk, (void **) ruleset_acl.rules, (void **) positions, ruleset_acl.num_rules, key_found, (void **) entry_handles))
        goto err;

#define FIELD(I, X, B) (ruleset_acl.rules[I]->field_value[X].value.u##B)
#define MASK(I, X, B) (ruleset_acl.rules[I]->field_value[X].mask_range.u##B)
    for(uint32_t i=0; i<ruleset_acl.num_rules; ++i) {
        printf("key_found: %d entry_handles: %p entry_handles[%u]: %u\n", key_found[i], entry_handles[i], i, *((uint8_t *) entry_handles[i]));
        printf("%u: %02X-%02X %08X-%08X %08X-%08X %04X-%04X %04X-%04X\n",
               i,
               FIELD(i, 0, 8), MASK(i, 0, 8),
               FIELD(i, 1, 32), MASK(i, 1, 32),
               FIELD(i, 2, 32), MASK(i, 2, 32),
               FIELD(i, 3, 16), MASK(i, 3, 16),
               FIELD(i, 4, 16), MASK(i, 4, 16)
              );
    }
#undef FIELD
#undef MASK


    free_ruleset_except_actions(&ruleset);
    puts("free ruleset_acl");
    acl_free_ruleset_except_actions(&ruleset_acl);
    puts("done free ruleset_acl");

    fw_conf=(firewall_conf_t) {
        .table_gpu=table_gpu, .table_dpdk=table_dpdk, .position_data=position_data, .tap_macaddr=&tap_macaddr, .stats=port_stats
    };

    uint16_t coreid=rte_get_next_lcore(rte_get_main_lcore(), 1, 1);

    for(int16_t i=0; i<DEFAULT_NB_QUEUES-1; ++i) {
        coreid=rte_get_next_lcore(coreid, 1, 1);
        rte_eal_remote_launch(firewall, &fw_conf, coreid);
        coreid=rte_get_next_lcore(coreid, 1, 1);
        rte_eal_remote_launch(firewall, &fw_conf, coreid);
    }

    coreid=rte_get_next_lcore(rte_get_main_lcore(), 1, 1);
    rte_eal_remote_launch(firewall, &fw_conf, coreid);
    printf("FIREWALL READY\n");
    firewall(&fw_conf);

    rte_eal_mp_wait_lcore();

    rte_table_bv_ops.f_free(table_gpu);
    rte_table_acl_ops.f_free(table_dpdk);

    free(ruleset.actions);
    free(ruleset_acl.actions);

    rte_eal_cleanup();
    return EXIT_SUCCESS;

err:
    free_ruleset(&ruleset);
    acl_free_ruleset(&ruleset_acl);
    rte_eal_cleanup();
    free(port_stats);
    return EXIT_FAILURE;
}

#ifdef __cplusplus
}
#endif
