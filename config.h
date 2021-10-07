#ifndef __INCLUDE_CONFIG
#define __INCLUDE_CONFIG

#include <rte_kni.h>

#define DEFAULT_NB_MBUF 8192
#define DEFAULT_MBUF_DATAROOM 2048
#define DEFAULT_NB_RX_DESC 1024
#define DEFAULT_NB_TX_DESC 1024

#define BURST_SIZE 64

#define GPU_PAGE_SIZE (1U<<16)

#define RTE_TABLE_VAL_DROP 1

#endif
