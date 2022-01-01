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

static inline int is_error(cudaError_t e, const char *file, int line) {
    if(e!=cudaSuccess) {
        rte_log(RTE_LOG_ERR, RTE_LOGTYPE_TABLE, "[rte_bv] error: %s in %s (line %d)\n", cudaGetErrorString(e), file, line);
        return 1;
    }
    return 0;
}
#define CHECK(X) is_error(X, __FILE__, __LINE__)

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

#define BLAME(X) fprintf(stderr, "[rte_bv_markers_range_add|%d] " X, __LINE__)
int rte_bv_markers_range_add(rte_bv_markers_t *markers, const uint32_t *from_to, const uint32_t val) {
    rte_bv_marker_list_t *l;
    for(int i=0; i<2;) {
        l=NULL;
        if(rte_hash_lookup_data(markers->table, &from_to[i], (void **) &l)>=0) {
            if(l->num_markers[i] >= l->size[i]) {
                rte_bv_marker_t *t=realloc(l->list[i], sizeof(rte_bv_marker_t)*(l->size[i]<<1));
                if(t==NULL) {
                    BLAME("Error increasing size of marker list\n");
                    return 0;
                }
                l->size[i]<<=1;
                l->list[i]=t;
            }
        } else {
            l=(rte_bv_marker_list_t *) malloc(sizeof(rte_bv_marker_list_t));
            if(!l) {
                BLAME("Error creating new marker list\n");
                return 0;
            }
            l->num_markers[0]=0;
            l->num_markers[1]=0;
            l->num_valid_markers[0]=0;
            l->num_valid_markers[1]=0;

            l->size[0]=RTE_BV_MARKERS_LIST_STND_SIZE;
            l->size[1]=RTE_BV_MARKERS_LIST_STND_SIZE;
            l->list[0]=(rte_bv_marker_t *) malloc(sizeof(rte_bv_marker_t)*l->size[0]);
            l->list[1]=(rte_bv_marker_t *) malloc(sizeof(rte_bv_marker_t)*l->size[1]);
            if(!l->list[0]||!l->list[1]) {
                BLAME("Error while allocating new marker list\n");
                return 0;
            }
            if(rte_hash_add_key_data(markers->table, &from_to[i], l)) {
                BLAME("Error while adding entry to hash table\n");
                return 0;
            }
            ++markers->num_lists;
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

typedef struct {
    size_t bv_size;
    size_t non_zero_bv_size;
    uint32_t *bv;
    uint32_t *non_zero_bv;
} bv_t;

inline void bv_set(bv_t *bv, uint32_t pos) {
    bv->bv[pos>>5]|=1<<(pos&31);
    bv->non_zero_bv[pos>>10]|=1<<((pos>>5)&31);
}

inline void bv_unset(bv_t *bv, uint32_t pos) {
    bv->bv[pos>>5]&=~(1<<(pos&31));
    if(!bv->bv[pos>>5])
        bv->non_zero_bv[pos>>10]&=~(1<<((pos>>5)&31));
}

inline void bv_set_list(bv_t *bv, size_t num_markers, const rte_bv_marker_t *list) {
    for(size_t i=0; i<num_markers; ++i) {
        if(list[i].valid)
            bv_set(bv, list[i].value);
    }
}

inline void bv_unset_list(bv_t *bv, size_t num_markers, const rte_bv_marker_t *list) {
    for(size_t i=0; i<num_markers; ++i) {
        if(list[i].valid)
            bv_unset(bv, list[i].value);
    }
}

uint8_t rte_bv_add_range_host(rte_bv_ranges_t *ranges, uint32_t from, uint32_t to, const bv_t *bv) {
    if(ranges->num_ranges>=ranges->max_num_ranges)
        return 1;
    ranges->ranges_from[ranges->num_ranges]=from;
    ranges->ranges_to[ranges->num_ranges]=to;
    memcpy(ranges->bvs+(ranges->num_ranges*ranges->bv_bs), bv->bv, sizeof(uint32_t)*bv->bv_size);
    memcpy(ranges->non_zero_bvs+(ranges->num_ranges*((ranges->bv_bs>>5)+1)), bv->non_zero_bv, sizeof(uint32_t)*bv->non_zero_bv_size);
    ++ranges->num_ranges;
    return 0;
}

uint8_t rte_bv_add_range_gpu(rte_bv_ranges_t *ranges, uint32_t from, uint32_t to, const bv_t *bv) {
    if(ranges->num_ranges>=ranges->max_num_ranges)
        return 1;
    CHECK(cudaMemcpy(ranges->ranges_from+ranges->num_ranges, &from, sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ranges->ranges_to+ranges->num_ranges, &to, sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ranges->bvs+(ranges->num_ranges*((size_t) ranges->bv_bs)), bv->bv, sizeof(uint32_t)*bv->bv_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ranges->non_zero_bvs+(ranges->num_ranges*((size_t) ((ranges->bv_bs>>5)+1))), bv->non_zero_bv, sizeof(uint32_t)*bv->non_zero_bv_size, cudaMemcpyHostToDevice));
    ++ranges->num_ranges;
    return 0;
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
    uint8_t (*add_range)(rte_bv_ranges_t *, uint32_t, uint32_t, const bv_t *)=gpu?&rte_bv_add_range_gpu:&rte_bv_add_range_host;

    bv_t bv;
    bv.bv_size=(markers->max_value>>5)+1;
    bv.non_zero_bv_size=(bv.bv_size>>5)+1;
    bv.bv=malloc(sizeof(uint32_t)*bv.bv_size);
    bv.non_zero_bv=malloc(sizeof(uint32_t)*bv.non_zero_bv_size);
    memset(bv.bv, 0, sizeof(uint32_t)*bv.bv_size);
    memset(bv.non_zero_bv, 0, sizeof(uint32_t)*bv.non_zero_bv_size);

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
            bv_set_list(&bv, marker_lists[i].l->num_markers[0], marker_lists[i].l->list[0]);

            if(marker_lists[i].l->num_valid_markers[1]) {
                if(add_range(ranges, prev, cur, &bv)) {
                    fprintf(stderr, "[rte_bv] error while adding range, reached limit\n");
                    return 1;
                }

                bv_unset_list(&bv, marker_lists[i].l->num_markers[1], marker_lists[i].l->list[1]);
            }

            continue;
        }

        if(marker_lists[i].l->num_valid_markers[0]) {
            if(add_range(ranges, prev, cur-1, &bv)) {
                fprintf(stderr, "[rte_bv] error while adding range, reached entry limit\n");
                return 1;
            }

            prev=cur;
            bv_set_list(&bv, marker_lists[i].l->num_markers[0], marker_lists[i].l->list[0]);
        }

        if(marker_lists[i].l->num_valid_markers[1]) {
            if(add_range(ranges, prev, cur, &bv)) {
                fprintf(stderr, "[rte_bv] error while adding range, reached entry limit\n");
                return 1;
            }

            ++cur;
            bv_unset_list(&bv, marker_lists[i].l->num_markers[1], marker_lists[i].l->list[1]);
        }
    }

    free(bv.bv);
    free(bv.non_zero_bv);
    free(marker_lists);

    return 0;
}

#ifdef __cplusplus
}
#endif
