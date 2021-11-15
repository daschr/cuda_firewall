#ifdef __cplusplus
extern "C" {
#endif

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

int setup_memory(struct rte_pktmbuf_extmem *ext_mem, struct rte_mempool **mpool_payload);
int setup_port(	uint16_t port_id, struct rte_pktmbuf_extmem *ext_mem, struct rte_mempool *mpool_payload,
                uint16_t nb_rx_queues, uint16_t nb_tx_queues, uint64_t rx_offload_capas, uint64_t tx_offload_capas);

#ifdef __cplusplus
}
#endif
