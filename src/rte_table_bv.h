#ifndef __INCLUDE_RTE_TABLE_BV__
#define __INCLUDE_RTE_TABLE_BV__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "rte_bv.h"
#include <rte_table.h>

#include <cuda_runtime.h>

#define RTE_TABLE_BV_MAX_RANGES ((size_t) (RTE_BV_MARKERS_MAX_ENTRIES>>1))
#define RTE_TABLE_BV_BS	((size_t) (RTE_TABLE_BV_MAX_RANGES>>6)+1)
#define RTE_TABLE_NON_ZERO_BV_BS ((size_t) (((RTE_TABLE_BV_MAX_RANGES>>6)+1)>>6)+1)
#define RTE_TABLE_BV_MAX_PKTS 1024
#define RTE_TABLE_BV_MAX_FIELDS 24

enum {
    RTE_TABLE_BV_FIELD_TYPE_RANGE,
    RTE_TABLE_BV_FIELD_TYPE_BITMASK
};

struct rte_table_bv_field_def {
    uint32_t offset; // offset from data start

    uint8_t type;
    uint8_t size; // in bytes
};

struct rte_table_bv_field {
    uint32_t value;
    uint32_t mask_range;
};

struct rte_table_bv_params {
    uint32_t num_fields;
    uint32_t num_rules; // max number of rules

    // size needs to be  >=num_fields
    const struct rte_table_bv_field_def *field_defs;
};

struct rte_table_bv_key {
    uint32_t *buf; // size = sum(rte_table_bv_params[*].num_fields*2)
    uint32_t pos;
};

extern struct rte_table_ops rte_table_bv_ops;

int rte_table_bv_lookup_stream(void *t_r, cudaStream_t stream, uint8_t *lookup_hit_vec, uint8_t **pkts_data,
                               struct rte_mbuf **pkts, uint64_t num_pkts, void **e);


#ifdef __cplusplus
}
#endif

#endif
