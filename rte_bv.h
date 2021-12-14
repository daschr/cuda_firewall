#ifndef __INCLUDE_RTE_TABLE_BV_GEN__
#define __INCLUDE_RTE_TABLE_BV_GEN__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <rte_hash.h>

#define RTE_BV_MARKERS_MAX_ENTRIES 150000ULL
#define RTE_BV_MARKERS_LIST_STND_SIZE 24ULL

typedef struct {
    uint32_t value;
    uint8_t valid;
} rte_bv_marker_t;

typedef struct {
    size_t size[2];
    size_t num_markers[2];
    size_t num_valid_markers[2];
    rte_bv_marker_t *list[2];
} rte_bv_marker_list_t;

typedef struct {
    uint32_t max_value;
    size_t num_lists;
    struct rte_hash *table;
} rte_bv_markers_t;

// must be already allocated
typedef struct {
    size_t num_ranges; // initial: 0
    size_t bv_bs; // initial: >= number of ranges>>5
    uint32_t *ranges_from; // intial size: >= 2*(number of ranges)
    uint32_t *ranges_to; // intial size: >= 2*(number of ranges)
    uint32_t *bvs; // initial size: >= bv_bs*2*(number of ranges)
} rte_bv_ranges_t;

int rte_bv_markers_create(rte_bv_markers_t *markers);
int rte_bv_markers_range_add(rte_bv_markers_t *markers, const uint32_t *from_to, const uint32_t val);
void rte_bv_markers_range_del(rte_bv_markers_t *markers, const uint32_t *from_to, const uint32_t val);
int rte_bv_markers_to_ranges(rte_bv_markers_t *markers, const uint8_t gpu, const uint8_t cast_type, rte_bv_ranges_t *ranges);
void rte_bv_markers_free(rte_bv_markers_t *markers);

#ifdef __cplusplus
}
#endif

#endif
