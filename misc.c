#ifdef __cplusplus
extern "C" {
#endif

#include "misc.h"

#include <stdint.h>

#include <rte_common.h>
#include <rte_log.h>
#include <rte_malloc.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_eal.h>
#include <rte_launch.h>
#include <rte_per_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_kni.h>

#include <cuda_runtime.h>

#include "config.h"
#include "offload_capas.h"

static inline void check_error(cudaError_t e, const char *file, int line) {
    if(e != cudaSuccess) {
        fprintf(stderr, "[ERROR] %s in %s (line %d)\n", cudaGetErrorString(e), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CHECK(X) (check_error(X, __FILE__, __LINE__))


int setup_memory(struct rte_pktmbuf_extmem *ext_mem, struct rte_mempool **mpool_payload) {
    memset(ext_mem, 0, sizeof(struct rte_pktmbuf_extmem));

    ext_mem->elt_size= DEFAULT_MBUF_DATAROOM + RTE_PKTMBUF_HEADROOM;
    ext_mem->buf_len= RTE_ALIGN_CEIL(DEFAULT_NB_MBUF * ext_mem->elt_size, GPU_PAGE_SIZE);
    ext_mem->buf_ptr = rte_malloc("extmem", ext_mem->buf_len, 0);

    CHECK(cudaHostRegister(ext_mem->buf_ptr, ext_mem->buf_len, cudaHostRegisterMapped));
    void *buf_ptr_dev;
    CHECK(cudaHostGetDevicePointer(&buf_ptr_dev, ext_mem->buf_ptr, 0));
    if(ext_mem->buf_ptr != buf_ptr_dev) {
        fprintf(stderr, "could not create external memory\next_mem.buf_ptr!=buf_ptr_dev\n");
        return 1;
    }

    *mpool_payload=rte_pktmbuf_pool_create_extbuf("payload_mpool", DEFAULT_NB_MBUF,
                   0, 0, ext_mem->elt_size,
                   rte_socket_id(), ext_mem, 1);

    if(!*mpool_payload) {
        fprintf(stderr, "Error: could not create mempool from external memory\n");
        return 1;
    }

    return 0;
}

#define CHECK_R(X) if(X){fprintf(stderr, "Error: " #X  " (r=%d)\n", r); return 1;}
int setup_port(uint16_t port_id, struct rte_pktmbuf_extmem *ext_mem, struct rte_mempool *mpool_payload, uint64_t rx_offload_capas, uint64_t tx_offload_capas) {
    struct rte_eth_conf port_conf = {
        .rxmode = {
            .mq_mode=RTE_ETH_MQ_RX_NONE,
            .mtu = DEFAULT_MTU,
        },
        .txmode = {
            .mq_mode=ETH_MQ_TX_NONE,
        },
    };

    int r;

    struct rte_eth_dev_info dev_info;
    rte_eth_dev_info_get(port_id, &dev_info);
    printf("using device %d with driver \"%s\"\n", port_id, dev_info.driver_name);
    printf("RX OFFLOAD CAPABILITIES:\n");
    print_rx_offload_capas(dev_info.rx_offload_capa);
    printf("\nTX OFFLOAD CAPABILITIES:\n");
    print_tx_offload_capas(dev_info.tx_offload_capa);
    printf("\n");

#define RX_OC(X) RTE_ETH_RX_OFFLOAD_##X
#define TX_OC(X) RTE_ETH_TX_OFFLOAD_##X

    port_conf.rxmode.offloads=rx_offload_capas;
//RX_OC(IPV4_CKSUM)|RX_OC(TCP_CKSUM)|RX_OC(UDP_CKSUM); //dev_info.rx_offload_capa;
    port_conf.txmode.offloads=tx_offload_capas;
//TX_OC(IPV4_CKSUM)|TX_OC(TCP_CKSUM)|TX_OC(UDP_CKSUM);//|TX_OC(TCP_TSO); //dev_info.tx_offload_capa;

#undef RX_OC
#undef TX_OC

    if(dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE) {
        printf("enabling fast free\n");
        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_MBUF_FAST_FREE;
    }

    port_conf.rx_adv_conf.rss_conf.rss_hf&=dev_info.flow_type_rss_offloads;

    struct rte_flow_error flow_error;
    if(rte_flow_flush(port_id, &flow_error))
        rte_exit(EXIT_FAILURE, "Error: could not flush flow rules: %s\n", flow_error.message);

    CHECK_R((r=rte_eth_dev_configure(port_id, 1, 1, &port_conf))!=0);

    struct rte_eth_txconf txconf=dev_info.default_txconf;
    txconf.offloads=port_conf.txmode.offloads;

    CHECK_R((r=rte_eth_tx_queue_setup(port_id, 0, DEFAULT_NB_TX_DESC, rte_eth_dev_socket_id(0), &txconf))<0);

    CHECK_R((r=rte_eth_rx_queue_setup(port_id, 0, DEFAULT_NB_RX_DESC, rte_eth_dev_socket_id(0), NULL, mpool_payload))<0);

    rte_dev_dma_map(dev_info.device, ext_mem->buf_ptr, ext_mem->buf_iova, ext_mem->buf_len);

    CHECK_R((r=rte_eth_dev_start(port_id))<0);

    CHECK_R((r=rte_eth_promiscuous_enable(port_id))!=0);
    return 0;
}
#undef CHECK_R

#ifdef __cplusplus
}
#endif
