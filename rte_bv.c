#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>
#include "rte_bv.h"
#include <rte_jhash.h>
#include <rte_lcore.h>
#include <rte_errno.h>
#include <stdio.h>
#include <stdlib.h>

int rte_bv_markers_create(rte_bv_markers_t *markers) {
    static size_t c=0;
    static char b[12];
    sprintf(b,"%lu",c);
    struct rte_hash_parameters params= {
        .name=b,
        .entries=RTE_BV_MARKERS_MAX_ENTRIES,
        .reserved=0,
        .key_len=sizeof(uint32_t),
        .hash_func=rte_jhash,
        .hash_func_init_val=0,
        .socket_id=(int) rte_socket_id(),
        .extra_flag=0
    };

    if((markers->table=rte_hash_create(&params))==NULL) {
        fprintf(stderr, "[rte_bv_markers_create|%d] error creating hash table: %s\n", __LINE__, rte_strerror(rte_errno));
        return 1;
    }

    markers->max_value=0;
    markers->num_lists=0;
    ++c;
    return 0;
}

int rte_bv_markers_range_add(rte_bv_markers_t *markers, const uint32_t *from_to, const uint32_t val) {
    rte_bv_marker_list_t *l;
    for(int i=0; i<2;) {
        l=NULL;
        if(rte_hash_lookup_data(markers->table, &from_to[i], (void **) &l)>=0) {
            if(l->num_markers[i] >= l->size[i]) {
                l->size[i]<<=1;
                l->list[i]=(rte_bv_marker_t *) realloc(l->list[i], sizeof(rte_bv_marker_t)*l->size[i]);
            }
        } else {
            ++markers->num_lists;
            l=(rte_bv_marker_list_t *) malloc(sizeof(rte_bv_marker_list_t));
            l->num_markers[0]=0;
            l->num_markers[1]=0;
            l->num_valid_markers[0]=0;
            l->num_valid_markers[1]=0;

            l->size[0]=RTE_BV_MARKERS_LIST_STND_SIZE;
            l->size[1]=RTE_BV_MARKERS_LIST_STND_SIZE;
            l->list[0]=(rte_bv_marker_t *) malloc(sizeof(rte_bv_marker_t)*l->size[0]);
            l->list[1]=(rte_bv_marker_t *) malloc(sizeof(rte_bv_marker_t)*l->size[1]);
            if(rte_hash_add_key_data(markers->table, &from_to[i], l)) {
                fprintf(stderr, "Error while adding entry to hash table\n");
            }
        }

        if(!l->list[i]) {
            fprintf(stderr, "l->list[i]==NULL\n");
            return 0;
        }

        for(size_t p=0; p<l->num_markers[i]; ++p) {
            if(!l->list[i][p].valid) {
                l->list[i][p].value=val;
                l->list[i][p].valid=1;
                goto next;
            }
        }

        l->list[i][l->num_markers[i]].value=val;
        l->list[i][l->num_markers[i]++].valid=1;
next:
        ++l->num_valid_markers[i];
        ++i;
    }

    if(val>markers->max_value)
        markers->max_value=val;
    return 1;
}

void rte_bv_markers_range_del(rte_bv_markers_t *markers, const uint32_t *from_to, const uint32_t val) {
    rte_bv_marker_list_t *l;
    for(int i=0; i<2; ++i) {
        if(rte_hash_lookup_data(markers->table, (void *) &from_to[i],(void **) &l)>0) {
            for(size_t p=0; p<l->num_markers[i]; ++p) {
                if(l->list[i][p].value==val) {
                    l->list[i][p].valid=0;
                    --l->num_valid_markers[i];
                }
            }
        }
    }
}

void rte_bv_markers_free(rte_bv_markers_t *markers) {
    uint32_t n=0, i;
    rte_bv_marker_list_t *l;

    while(rte_hash_iterate(markers->table, (const void **) &i, (void **) &l, &n)>=0) {
        free(l->list[0]);
        free(l->list[1]);
        free(l);
    }

    rte_hash_free(markers->table);
}

inline void bv_set(uint32_t *bv, uint32_t pos) {
    bv[pos>>5]|=1<<(pos&31);
}

inline void bv_unset(uint32_t *bv, uint32_t pos) {
    bv[pos>>5]&=~(1<<(pos&31));
}

inline void bv_set_list(uint32_t *bv, size_t num_markers, const rte_bv_marker_t *list) {
    for(size_t i=0; i<num_markers; ++i) {
        if(list[i].valid)
            bv_set(bv, list[i].value);
    }
}

inline void bv_unset_list(uint32_t *bv, size_t num_markers, const rte_bv_marker_t *list) {
    for(size_t i=0; i<num_markers; ++i) {
        if(list[i].valid)
            bv_unset(bv, list[i].value);
    }
}

void rte_bv_add_range_host(rte_bv_ranges_t *ranges, uint32_t from, uint32_t to, size_t bv_size, const uint32_t *bv) {
    ranges->ranges[ranges->num_ranges<<1]=from;
    ranges->ranges[(ranges->num_ranges<<1)+1]=to;
    memcpy(ranges->bvs+(ranges->num_ranges*ranges->bv_bs), bv, sizeof(uint32_t)*bv_size);
    ++ranges->num_ranges;
}

void rte_bv_add_range_gpu(rte_bv_ranges_t *ranges, uint32_t from, uint32_t to, size_t bv_size, const uint32_t *bv) {
	cudaMemcpy(ranges->ranges+(ranges->num_ranges<<1), &from, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ranges->ranges+((ranges->num_ranges<<1)+1), &to, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ranges->bvs+(ranges->num_ranges*ranges->bv_bs), bv, sizeof(uint32_t)*bv_size, cudaMemcpyHostToDevice);
    ++ranges->num_ranges;
}

typedef struct {
    uint32_t *v;
    rte_bv_marker_list_t *l;
} vp_t;

static int sort_vp_list(const void *a_r, const void *b_r) {
    uint *a=((vp_t *) a_r)->v, *b=((vp_t *) b_r)->v;

    return *a==*b?0:(*a<*b?-1:1);
}

int rte_bv_markers_to_ranges(rte_bv_markers_t *markers, const uint8_t gpu, const uint8_t cast_type, rte_bv_ranges_t *ranges) {
    void (*add_range)(rte_bv_ranges_t *, uint32_t, uint32_t, size_t, const uint32_t *)=gpu?&rte_bv_add_range_gpu:&rte_bv_add_range_host;

    const size_t bv_size=(markers->max_value>>5)+1;
    uint32_t *bv=(uint32_t *) malloc(sizeof(uint32_t)*bv_size);
    memset(bv, 0, sizeof(uint32_t)*bv_size);
    //create sorted array of marker lists
    vp_t *marker_lists=(vp_t *) malloc(sizeof(vp_t)*markers->num_lists);
    uint32_t n=0;
    for(vp_t *i=marker_lists; rte_hash_iterate(markers->table, (const void **) &i->v, (void **) &i->l, &n)>=0; ++i);
    qsort(marker_lists, markers->num_lists, sizeof(vp_t), sort_vp_list);

    uint32_t prev, cur=0;
    uint8_t first=1;
    for(size_t i=0; i<markers->num_lists; ++i) {
        prev=cur;
        cur=*marker_lists[i].v;
        if(first) {
            first=0;
            bv_set_list(bv, marker_lists[i].l->num_markers[0], marker_lists[i].l->list[0]);

            if(marker_lists[i].l->num_valid_markers[1]) {
                add_range(ranges, prev, cur, bv_size, bv);

                bv_unset_list(bv, marker_lists[i].l->num_markers[1], marker_lists[i].l->list[1]);
            }

            continue;
        }

        if(marker_lists[i].l->num_valid_markers[0]) {
            add_range(ranges, prev, cur-1, bv_size, bv);
            prev=cur;
            bv_set_list(bv, marker_lists[i].l->num_markers[0], marker_lists[i].l->list[0]);
        }

        if(marker_lists[i].l->num_valid_markers[1]) {
            add_range(ranges, prev, cur, bv_size, bv);
            ++cur;
            bv_unset_list(bv, marker_lists[i].l->num_markers[1], marker_lists[i].l->list[1]);
        }
    }

    free(bv);
    free(marker_lists);

    return 0;
}

#ifdef __cplusplus
}
#endif
