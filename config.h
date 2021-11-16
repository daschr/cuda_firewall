#ifndef __INCLUDE_CONFIG
#define __INCLUDE_CONFIG

#include <rte_kni.h>

#define FW_IFACE_NAME "fw0"

#define DEFAULT_NB_MBUF 16384
#define DEFAULT_MTU 1500
#define DEFAULT_MBUF_DATAROOM 2048
#define DEFAULT_NB_RX_DESC 2048
#define DEFAULT_NB_TX_DESC 2048
#define KNI_LCORE 7
#define BURST_SIZE 32

#define GPU_PAGE_SIZE (1U<<16)

#define RTE_TABLE_VAL_DROP 1

#endif
