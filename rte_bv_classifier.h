#ifndef __INCLUDE_RTE_BV_CLASSIFIER__
#define __INCLUDE_RTE_BV_CLASSIFIER__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include <rte_table.h>
#include <pthread.h>
#include <cuda_runtime.h>

#include "rte_bv.h"

#define RTE_BV_CLASSIFIER_MAX_RANGES RTE_BV_MARKERS_MAX_ENTRIES>>1
#define RTE_BV_CLASSIFIER_BS (RTE_BV_CLASSIFIER_MAX_RANGES>>5)
#define RTE_BV_CLASSIFIER_MAX_PKTS 64

#define RTE_BV_CLASSIFIER_NUM_STREAMS 8
#define RTE_BV_CLASSIFIER_NUM_STREAMS_MASK 7

enum {
	RTE_BV_CLASSIFIER_FIELD_TYPE_RANGE,
	RTE_BV_CLASSIFIER_FIELD_TYPE_BITMASK
};

struct rte_bv_classifier_field_def {
	uint32_t offset; // offset from data start
	uint32_t ptype_mask; // packet type mask needed for matching
	
	uint8_t type;
	uint8_t size; // in bytes
};

struct rte_bv_classifier_field {
	uint32_t value;
	uint32_t mask_range;
};

struct rte_bv_classifier_params {
	uint32_t num_fields;
	// size needs to be  >=num_fields
	const struct rte_bv_classifier_field_def *field_defs;
};

struct rte_bv_classifier_key {
	uint32_t *buf; // size = sum(rte_bv_classifier_params[*].num_fields*2)
	uint32_t pos;
};

struct rte_bv_classifier {
    uint32_t num_fields;
    struct rte_table_stats stats;
    const struct rte_bv_classifier_field_def *field_defs;

    uint8_t *act_buf; // size==1, pointer for gpu
    uint8_t *act_buf_h; // host pointer
    
    size_t *num_ranges;
	uint32_t **ranges_db; // size==[2*num_fields][2*RTE_BV_CLASSIFIER_MAX_RANGES]
    uint32_t **ranges_db_dev;
    
	uint32_t **bvs_db; // size==[2*num_fields][RTE_BV_CLASSIFIER_BS*2*RTE_TABLE_BV_MAX_RANGES]
    uint32_t **bvs_db_dev;

    uint32_t *field_ptype_masks;
    uint32_t *field_offsets;
    uint8_t *field_sizes;

    uint8_t **pkts_data[RTE_BV_CLASSIFIER_NUM_STREAMS];
    uint8_t **pkts_data_h[RTE_BV_CLASSIFIER_NUM_STREAMS];

    uint32_t *packet_types[RTE_BV_CLASSIFIER_NUM_STREAMS];
    uint32_t *packet_types_h[RTE_BV_CLASSIFIER_NUM_STREAMS];

	uint32_t *positions[RTE_BV_CLASSIFIER_NUM_STREAMS];
	uint32_t *positions_h[RTE_BV_CLASSIFIER_NUM_STREAMS];

	uint64_t *lookup_hit_mask[RTE_BV_CLASSIFIER_NUM_STREAMS];
	uint64_t *lookup_hit_mask_h[RTE_BV_CLASSIFIER_NUM_STREAMS];
	
	struct rte_mbuf **pkts[RTE_BV_CLASSIFIER_NUM_STREAMS];
	uint64_t pkts_mask[RTE_BV_CLASSIFIER_NUM_STREAMS];

    rte_bv_markers_t *bv_markers; // size==num_fields
	
	size_t enqueue_pos;
	size_t dequeue_pos;

	uint8_t stream_running[RTE_BV_CLASSIFIER_NUM_STREAMS];
	pthread_mutex_t stream_running_mtx[RTE_BV_CLASSIFIER_NUM_STREAMS];
	cudaStream_t streams[RTE_BV_CLASSIFIER_NUM_STREAMS];
};

struct rte_bv_classifier *rte_bv_classifier_create(struct rte_bv_classifier_params *params, int socket_id);
int rte_bv_classifier_free(struct rte_bv_classifier *c);
int rte_bv_classifier_entry_add(struct rte_bv_classifier *c, struct rte_bv_classifier_key *k, uint32_t *pos, int *key_found);
int rte_bv_classifier_entry_delete(struct rte_bv_classifier *c, struct rte_bv_classifier_key *k);
int rte_bv_classifier_entry_add_bulk(struct rte_bv_classifier *c, struct rte_bv_classifier_key **ks, uint32_t n_keys);
int rte_bv_classifier_entry_delete_bulk(struct rte_bv_classifier *c, struct rte_bv_classifier_key **ks, uint32_t n_keys);

void rte_bv_classifier_enqueue_burst(struct rte_bv_classifier *c, struct rte_mbuf **pkts, uint64_t pkts_mask);
void __rte_noreturn rte_bv_classifier_poll_lookups(struct rte_bv_classifier *c, void (*callback) (struct rte_mbuf **, uint64_t,  uint64_t, uint32_t *, void *), void *p);

int rte_bv_classifier_stats_read(struct rte_bv_classifier *c, struct rte_table_stats *stats, int clear);

#ifdef __cplusplus
}
#endif

#endif
