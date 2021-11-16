#ifdef __cplusplus
extern "C" {
#endif

#include "misc.h"

#include <stdint.h>
#include <string.h>
#include <time.h>

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

#define KNI_FIFO_COUNT_MAX 1024

static inline void check_error(cudaError_t e, const char *file, int line) {
    if(e != cudaSuccess) {
        fprintf(stderr, "[ERROR] %s in %s (line %d)\n", cudaGetErrorString(e), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CHECK(X) (check_error(X, __FILE__, __LINE__))


int setup_memory(struct rte_pktmbuf_extmem *ext_mem, struct rte_mempool **mpool) {
    if(DEFAULT_NB_MBUF<KNI_FIFO_COUNT_MAX*2) {
        fprintf(stderr, "DEFAULT_NB_MBUF=%u<%u=KNI_FIFO_COUNT_MAX*2\n", DEFAULT_NB_MBUF, KNI_FIFO_COUNT_MAX*2);
        return 1;
    }

    memset(ext_mem, 0, sizeof(struct rte_pktmbuf_extmem));

    ext_mem->elt_size= DEFAULT_MBUF_DATAROOM + RTE_PKTMBUF_HEADROOM;
    ext_mem->buf_len= RTE_ALIGN_CEIL(DEFAULT_NB_MBUF * ext_mem->elt_size, GPU_PAGE_SIZE);
    ext_mem->buf_iova=RTE_BAD_IOVA;
    ext_mem->buf_ptr = rte_malloc("extmem", ext_mem->buf_len, 0);
    rte_extmem_register(ext_mem->buf_ptr, ext_mem->buf_len, NULL, ext_mem->buf_iova, GPU_PAGE_SIZE);
    CHECK(cudaHostRegister(ext_mem->buf_ptr, ext_mem->buf_len, cudaHostRegisterMapped));
    void *buf_ptr_dev;
    CHECK(cudaHostGetDevicePointer(&buf_ptr_dev, ext_mem->buf_ptr, 0));
    if(ext_mem->buf_ptr != buf_ptr_dev) {
        fprintf(stderr, "could not create external memory\next_mem.buf_ptr!=buf_ptr_dev\n");
        return 1;
    }

    *mpool=rte_pktmbuf_pool_create_extbuf("payload_mpool", DEFAULT_NB_MBUF,
                                          0, 0, ext_mem->elt_size,
                                          rte_socket_id(), ext_mem, 1);

    if(!*mpool) {
        fprintf(stderr, "Error: could not create mempool from external memory\n");
        return 1;
    }

    return 0;
}

struct rte_kni *setup_kni_port(uint16_t port_id, uint32_t core_id, struct rte_mempool *mpool) {
    int r;
    struct rte_kni_conf conf;
    struct rte_kni_ops ops;

    memset(&ops, 0, sizeof(struct rte_kni_ops));
    memset(&conf, 0, sizeof(struct rte_kni_conf));

    ops.port_id=port_id;

    strcpy(conf.name, FW_IFACE_NAME);
    conf.core_id=core_id;
    conf.group_id=port_id;
    conf.force_bind=0;
    conf.mbuf_size=DEFAULT_MBUF_DATAROOM;

    struct rte_eth_dev_info dev_info;
    if((r=rte_eth_dev_info_get(port_id, &dev_info))!=0) {
        fprintf(stderr, "Failed to get device info (port %u): %s\n", port_id, rte_strerror(-r));
        return NULL;
    }

    conf.min_mtu=dev_info.min_mtu;
    conf.max_mtu=dev_info.max_mtu;

    rte_eth_dev_get_mtu(port_id, &conf.mtu);

    // copy mac from corresponding trunk port
    if((r=rte_eth_macaddr_get(port_id, (struct rte_ether_addr *) &conf.mac_addr))!=0) {
        fprintf(stderr, "Failed to get MAC address (port %u): %s\n", port_id, rte_strerror(-r));
        return NULL;
    }

    return rte_kni_alloc(mpool, &conf, &ops);
}

#define CHECK_R(X) if(X){fprintf(stderr, "Error: " #X  " (r=%d)\n", r); return 1;}
int setup_port(uint16_t port_id, struct rte_pktmbuf_extmem *ext_mem, struct rte_mempool *mpool, uint64_t rx_offload_capas, uint64_t tx_offload_capas) {
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

    port_conf.rxmode.offloads=rx_offload_capas&dev_info.rx_offload_capa;
    port_conf.txmode.offloads=tx_offload_capas&dev_info.tx_offload_capa;

    printf("ENABLED RX CAPABILITIES:\n");
    print_rx_offload_capas(port_conf.rxmode.offloads);

    printf("ENABLED TX CAPABILITIES:\n");
    print_tx_offload_capas(port_conf.txmode.offloads);

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

    CHECK_R((r=rte_eth_rx_queue_setup(port_id, 0, DEFAULT_NB_RX_DESC, rte_eth_dev_socket_id(0), NULL, mpool))<0);

    rte_dev_dma_map(dev_info.device, ext_mem->buf_ptr, ext_mem->buf_iova, ext_mem->buf_len);

    CHECK_R((r=rte_eth_dev_start(port_id))<0);

    CHECK_R((r=rte_eth_promiscuous_enable(port_id))!=0);
    return 0;
}
#undef CHECK_R

#ifdef __cplusplus
}
#endif
