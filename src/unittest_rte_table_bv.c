#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>

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

struct rte_pktmbuf_extmem ext_mem;
struct rte_mempool *mpool_payload, *mpool_header;

static inline void check_error(cudaError_t e, const char *file, int line) {
    if(e != cudaSuccess) {
        fprintf(stderr, "[ERROR] %s in %s (line %d)\n", cudaGetErrorString(e), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CHECK(X) (check_error(X, __FILE__, __LINE__))


int setup_memory(int port_id) {
    ext_mem.elt_size= DEFAULT_MBUF_DATAROOM + RTE_PKTMBUF_HEADROOM;
    ext_mem.buf_len= RTE_ALIGN_CEIL(DEFAULT_NB_MBUF * ext_mem.elt_size, GPU_PAGE_SIZE);
    ext_mem.buf_ptr = rte_malloc("extmem", ext_mem.buf_len, 0);

    CHECK(cudaHostRegister(ext_mem.buf_ptr, ext_mem.buf_len, cudaHostRegisterMapped));
    void *buf_ptr_dev;
    CHECK(cudaHostGetDevicePointer(&buf_ptr_dev, ext_mem.buf_ptr, 0));
    if(ext_mem.buf_ptr != buf_ptr_dev)
        rte_exit(EXIT_FAILURE, "could not create external memory\next_mem.buf_ptr!=buf_ptr_dev\n");

    int r=rte_dev_dma_map(rte_eth_devices[port_id].device, ext_mem.buf_ptr, ext_mem.buf_iova, ext_mem.buf_len);
    if(r)
        rte_exit(EXIT_FAILURE, "Error: could not map external memory for DMA\n");

    mpool_payload=rte_pktmbuf_pool_create_extbuf("payload_mpool", DEFAULT_NB_MBUF,
                  0, 0, ext_mem.elt_size,
                  rte_socket_id(), &ext_mem, 1);
    
	if(!mpool_payload)
        rte_exit(EXIT_FAILURE, "Error: could not create mempool from external memory\n");

    return 0;
}


#define CHECK_R(X) if(X){fprintf(stderr, "Error: " #X  " (r=%d)\n", r); return 1;}
int setup_port(int port_id) {
    struct rte_eth_conf port_conf = {
        .rxmode = {
            .max_rx_pkt_len = RTE_ETHER_MAX_LEN,
        },
    };

    int r;

    struct rte_eth_dev_info dev_info;
    rte_eth_dev_info_get(0, &dev_info);
    printf("using device 0 with driver \"%s\"\n", dev_info.driver_name);

    if(dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_MBUF_FAST_FREE;

    CHECK_R((r=rte_eth_dev_configure(0, 1, 1, &port_conf))!=0);

    struct rte_eth_txconf txconf=dev_info.default_txconf;
    txconf.offloads=port_conf.txmode.offloads;

    CHECK_R((r=rte_eth_tx_queue_setup(port_id, 0, DEFAULT_NB_TX_DESC, rte_eth_dev_socket_id(0), &txconf))<0);

    CHECK_R((r=rte_eth_rx_queue_setup(port_id, 0, DEFAULT_NB_RX_DESC, rte_eth_dev_socket_id(0), NULL, mpool_payload))<0);
    
	CHECK_R((r=rte_eth_dev_start(port_id))<0);

    CHECK_R((r=rte_eth_promiscuous_enable(port_id))!=0);
    return 0;
}
#undef CHECK_R

static __rte_noreturn void firewall(void *table) {
    volatile uint64_t *lookup_hit_mask, *lookup_hit_mask_d, pkts_mask;
    volatile uint32_t *vals, *vals_d;
	cudaHostAlloc((void **) &vals, sizeof(uint32_t)*BURST_SIZE, cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **) &vals_d, (uint32_t *) vals, 0);

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
    for(;;*lookup_hit_mask=0) {
        const uint16_t nb_rx = rte_eth_rx_burst(0, 0, bufs_rx, BURST_SIZE);

        if(unlikely(nb_rx==0))
            continue;

        pkts_mask=(1<<nb_rx)-1;

        lookup(table, bufs_rx_d, pkts_mask, (uint64_t *) lookup_hit_mask_d, (void **) vals_d);
		printf("lookup_hit_mask: %016X\n", *lookup_hit_mask);

        for(i=0,j=0; i<nb_rx; ++i)
            if((*lookup_hit_mask>>i)&1&(vals[i]>127))
                bufs_tx[j++]=bufs_rx[i];

        const uint16_t nb_tx = rte_eth_tx_burst(0, 0, bufs_tx, j);
        if(unlikely(nb_tx<nb_rx)) {
            for(uint16_t b=nb_tx; b<nb_rx; ++b)
                rte_pktmbuf_free(bufs_rx[b]);
        }
    }
}


int main(int ac, char *as[]) {
    if(ac==1) {
        fprintf(stderr, "Usage: %s [rules]\n", as[0]);
        return EXIT_FAILURE;
    }

    int offset;

    if((offset=rte_eal_init(ac, as))<0)
        rte_exit(EXIT_FAILURE, "Error: could not init EAL.\n");
    ++offset;

    if(offset>=ac) {
        rte_exit(EXIT_FAILURE, "Usage: %s [[rte  arguments]...] [rules]\n", as[0]);
    }

    ac-=offset;
    as=as+offset;

    if(rte_eth_dev_count_avail()==0)
        rte_exit(EXIT_FAILURE, "Error: no eth devices available.\n");

    setup_memory(0);
    setup_port(0);


    ruleset_t ruleset;
    memset(&ruleset, 0, sizeof(ruleset_t));

    if(!parse_ruleset(&ruleset, as[0])) {
        fprintf(stderr, "Error: could not parse ruleset \"%s\"\n", as[0]);
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

    struct rte_table_bv_field_def fdefs[5];
    uint32_t fdefs_offsets[5]= {	offsetof(struct rte_ipv4_hdr, src_addr), offsetof(struct rte_ipv4_hdr, dst_addr),  
									sizeof(struct rte_ipv4_hdr)+offsetof(struct rte_tcp_hdr, src_port), 
									sizeof(struct rte_ipv4_hdr)+offsetof(struct rte_tcp_hdr, dst_port), offsetof(struct rte_ipv4_hdr, next_proto_id)
                               },
                               fdefs_sizes[5]= {4,4,2,2,1};

    for(size_t i=0; i<5; ++i) {
        fdefs[i].offset=sizeof(struct rte_ether_hdr) + fdefs_offsets[i];
        fdefs[i].type=RTE_TABLE_BV_FIELD_TYPE_RANGE;
        fdefs[i].size=fdefs_sizes[i];
    }

    struct rte_table_bv_params table_params = { .num_fields=5, .field_defs=fdefs };

    void *table=rte_table_bv_ops.f_create(&table_params, rte_socket_id(), 0);

    if(table==NULL)
        goto err;

    rte_table_bv_ops.f_add_bulk(table, (void **) ruleset.rules, NULL, ruleset.num_rules, NULL, NULL);
	free_ruleset(&ruleset);
    
	firewall(table);

    rte_table_bv_ops.f_free(table);

    rte_eal_cleanup();
    return EXIT_SUCCESS;

err:

    rte_eal_cleanup();
    return EXIT_FAILURE;
}

#ifdef __cplusplus
}
#endif
