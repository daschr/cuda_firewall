#include "rte_table_bv.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "rte_bv.h"
#include <rte_log.h>
#include <rte_malloc.h>
#include <stdlib.h>
#include <sys/time.h>

// #define MEASURE_TIME

#define WORKERS_PER_FIELD 32
#define PERSISTENT_WARPS_PER_THREAD 32
#define NUM_MULTIPROCESSORS 28

#ifdef RTE_TABLE_STATS_COLLECT
#define RTE_TABLE_BV_STATS_PKTS_IN_ADD(table, val) table->stats.n_pkts_in += val
#define RTE_TABLE_BV_STATS_PKTS_LOOKUP_MISS(table, val) table->stats.n_pkts_lookup_miss += val
#else
#define RTE_TABLE_BV_STATS_PKTS_IN_ADD(table, val)
#define RTE_TABLE_BV_STATS_PKTS_LOOKUP_MISS(table, val)
#endif

struct rte_table_bv {
    uint32_t num_fields;
    struct rte_table_stats stats;
    const struct rte_table_bv_field_def *field_defs;

    uint32_t packets_per_block;
    uint32_t num_blocks;

    uint32_t num_rules;
    uint32_t entry_size;

    uint32_t **ranges_from;
    uint32_t **ranges_to;
    uint64_t **bvs;
    uint64_t **non_zero_bvs;

    size_t *num_ranges;
    uint32_t *field_offsets;
    uint8_t *field_sizes;

    uint32_t **ranges_from_dev;
    uint32_t **ranges_to_dev;
    uint64_t **bvs_dev;
    uint64_t **non_zero_bvs_dev;

    uint8_t *entries;
    uint8_t *entries_h;

    volatile void *matched_entries;
    volatile void *matched_entries_h;

    volatile uint8_t **pkts_data;
    volatile uint8_t **pkts_data_h;

    volatile uint64_t *pkts_mask;
    volatile uint64_t *pkts_mask_h;

    volatile uint32_t *num_pkts;
    volatile uint32_t *num_pkts_h;

    volatile uint32_t *num_done_pkts;
    volatile uint32_t *num_done_pkts_h;

    volatile uint64_t *done_pkts_dev;

    volatile uint8_t *running;
    volatile uint8_t *running_h;

    uint8_t *lookup_hit_vec;
    uint8_t *lookup_hit_vec_h;
    rte_bv_markers_t *bv_markers; // size==num_fields
};

static inline int is_error(cudaError_t e, const char *file, int line) {
    if(e!=cudaSuccess) {
        rte_log(RTE_LOG_ERR, RTE_LOGTYPE_TABLE, "[rte_table_bv] error: %s in %s (line %d)\n", cudaGetErrorString(e), file, line);
        return 1;
    }
    return 0;
}

static int rte_table_bv_free(void *t_r) {
    if(t_r==NULL)
        return 0;

    struct rte_table_bv *t=(struct rte_table_bv *) t_r;

    for(size_t i=0; i<t->num_fields; ++i) {
        cudaFree(t->ranges_from[i]);
        cudaFree(t->ranges_to[i]);
        cudaFree(t->bvs[i]);
        cudaFree(t->non_zero_bvs[i]);
    }
    cudaFree(t->ranges_from_dev);
    cudaFree(t->ranges_to_dev);
    cudaFree(t->bvs_dev);

    cudaFree(t->num_ranges);
    cudaFree(t->field_offsets);
    cudaFree(t->field_sizes);

    cudaFreeHost(t->lookup_hit_vec_h);
    cudaFreeHost(t->entries);
    cudaFreeHost(t->pkts_data_h);

    cudaFreeHost((void *) t->pkts_data);
    cudaFreeHost((void *) t->num_pkts);
    cudaFreeHost((void *) t->num_done_pkts);
    cudaFreeHost((void *) t->running);
    cudaFreeHost((void *) t->entries);
    cudaFreeHost((void *) t->matched_entries);

    for(uint32_t i=0; i<t->num_fields; ++i)
        rte_bv_markers_free(t->bv_markers+i);

    rte_free(t->bv_markers);
    rte_free(t->ranges_from);
    rte_free(t->ranges_to);
    rte_free(t->bvs);

    rte_free(t);

    return 0;
}


static void *rte_table_bv_create(void *params, int socket_id, uint32_t entry_size) {
    struct rte_table_bv_params *p=(struct rte_table_bv_params *) params;
    struct rte_table_bv *t=(struct rte_table_bv *) rte_malloc("t", sizeof(struct rte_table_bv), 0);
    memset(t, 0, sizeof(struct rte_table_bv));

    t->num_fields=p->num_fields;
    t->packets_per_block=PERSISTENT_WARPS_PER_THREAD;
    t->num_blocks=ceil((double) RTE_TABLE_BV_MAX_PKTS/(double) PERSISTENT_WARPS_PER_THREAD)<28?ceil((double) RTE_TABLE_BV_MAX_PKTS/ (double) (double)  PERSISTENT_WARPS_PER_THREAD):NUM_MULTIPROCESSORS;
    printf("num_blocks: %u packets_per_block: %u\n", t->num_blocks, t->packets_per_block);

    t->field_defs=p->field_defs;
    t->num_rules=p->num_rules;
    t->entry_size=entry_size;

    t->ranges_from=(uint32_t **) rte_malloc("ranges_from", sizeof(uint32_t *)*t->num_fields, 0);
    t->ranges_to=(uint32_t **) rte_malloc("ranges_to", sizeof(uint32_t *)*t->num_fields, 0);
    t->bvs=(uint64_t **) rte_malloc("bvs", sizeof(uint64_t *)*t->num_fields, 0);
    t->non_zero_bvs=(uint64_t **) rte_malloc("non_zero_bvs", sizeof(uint64_t *)*t->num_fields, 0);

#define IS_ERROR(X) is_error(X, __FILE__, __LINE__)
#define CHECK(X) if(IS_ERROR(X)) return NULL

#define HOSTALLOC(DP, SIZE) CHECK(cudaHostAlloc((void **) &DP ## _h, SIZE, cudaHostAllocMapped));\
                            CHECK(cudaHostGetDevicePointer((void **) &DP, (void *) DP ## _h, 0));

    HOSTALLOC(t->pkts_mask, sizeof(uint64_t));
    HOSTALLOC(t->num_pkts, sizeof(uint32_t));
    HOSTALLOC(t->num_done_pkts, sizeof(uint32_t));
    HOSTALLOC(t->running, sizeof(uint8_t));
    HOSTALLOC(t->entries, t->entry_size*t->num_rules);
    HOSTALLOC(t->matched_entries, sizeof(void *)*RTE_TABLE_BV_MAX_PKTS);

#undef HOSTALLOC

    *t->num_pkts_h=0;
    *t->num_done_pkts_h=0;
    *t->running_h=1;

    CHECK(cudaHostAlloc((void **) &t->pkts_data_h, sizeof(uint8_t*)*RTE_TABLE_BV_MAX_PKTS, cudaHostAllocMapped|cudaHostAllocWriteCombined));
    CHECK(cudaHostGetDevicePointer((void **) &t->pkts_data, t->pkts_data_h, 0));

    CHECK(cudaHostAlloc((void **) &t->entries, t->entry_size*t->num_rules, cudaHostAllocMapped));

    CHECK(cudaHostAlloc((void **) &t->lookup_hit_vec_h, sizeof(uint8_t*)*RTE_TABLE_BV_MAX_PKTS, cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer((void **) &t->lookup_hit_vec, t->lookup_hit_vec_h, 0));

    CHECK(cudaMalloc((void **) &t->ranges_from_dev, sizeof(uint32_t *)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->ranges_to_dev, sizeof(uint32_t *)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->bvs_dev, sizeof(uint64_t *)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->non_zero_bvs_dev, sizeof(uint64_t *)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->field_offsets, sizeof(uint32_t)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->field_sizes, sizeof(uint32_t)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->num_ranges, sizeof(uint64_t)*t->num_fields));


    for(size_t i=0; i<t->num_fields; ++i) {
        CHECK(cudaMemcpy(t->field_offsets+i, &t->field_defs[i].offset, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(t->field_sizes+i, &t->field_defs[i].size, sizeof(uint32_t), cudaMemcpyHostToDevice));

#define RANGE_SIZE(DIV) ((sizeof(uint32_t)*((size_t) RTE_TABLE_BV_MAX_RANGES))/DIV+1LU)
        switch(p->field_defs[i].size) {
        case 4:
            CHECK(cudaMalloc((void **) &t->ranges_from[i], RANGE_SIZE(1LU)));
            CHECK(cudaMalloc((void **) &t->ranges_to[i], RANGE_SIZE(1LU)));
            printf("allocated %lu bytes for dimension %lu\n", RANGE_SIZE(1LU), i);
            break;
        case 2:
            CHECK(cudaMalloc((void **) &t->ranges_from[i], RANGE_SIZE(2LU)));
            CHECK(cudaMalloc((void **) &t->ranges_to[i], RANGE_SIZE(2LU)));
            printf("allocated %lu bytes for dimension %lu\n", RANGE_SIZE(2LU), i);
            break;
        case 1:
            CHECK(cudaMalloc((void **) &t->ranges_from[i], RANGE_SIZE(4LU)));
            CHECK(cudaMalloc((void **) &t->ranges_to[i], RANGE_SIZE(4LU)));
            printf("allocated %lu bytes for dimension %lu\n", RANGE_SIZE(4LU), i);
            break;
        default:
            printf("unkown field_def[%lu] size: %hhu\n", i, p->field_defs[i].size);
        }
#undef RANGE_SIZE

        printf("size: bvs[%lu] %lu bytes\n", i, sizeof(uint64_t)*((size_t) RTE_TABLE_BV_BS) * ((size_t ) RTE_TABLE_BV_MAX_RANGES));
        CHECK(cudaMalloc((void **) &t->bvs[i], sizeof(uint64_t)*((size_t) RTE_TABLE_BV_BS) * ((size_t ) RTE_TABLE_BV_MAX_RANGES)));
        printf("size: non_zero_bvs[%lu] %lu bytes\n", i, sizeof(uint64_t)*((size_t) RTE_TABLE_BV_BS>>5) * ((size_t ) RTE_TABLE_BV_MAX_RANGES));
        CHECK(cudaMalloc((void **) &t->non_zero_bvs[i], sizeof(uint64_t)*((size_t) RTE_TABLE_BV_BS>>5) * ((size_t ) RTE_TABLE_BV_MAX_RANGES)));
    }

    CHECK(cudaMemcpy(t->ranges_from_dev, t->ranges_from, sizeof(uint32_t *)*t->num_fields, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(t->ranges_to_dev, t->ranges_to, sizeof(uint32_t *)*t->num_fields, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(t->bvs_dev, t->bvs, sizeof(uint64_t *)*t->num_fields, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(t->non_zero_bvs_dev, t->non_zero_bvs, sizeof(uint64_t *)*t->num_fields, cudaMemcpyHostToDevice));
#undef CHECK
#undef IS_ERROR

    t->bv_markers=(rte_bv_markers_t *) rte_malloc("bv_markers", sizeof(rte_bv_markers_t)*t->num_fields, 0);

    for(size_t i=0; i<t->num_fields; ++i) {
        if(rte_bv_markers_create(&t->bv_markers[i])) {
            rte_table_bv_free(t);
            rte_log(RTE_LOG_ERR, RTE_LOGTYPE_TABLE, "Error creating marker!\n");
            return NULL;
        }
    }

    return (void *) t;
}
#undef IS_ERROR

static inline void cal_from_to(uint32_t *from_to, uint32_t *v, uint8_t type, uint8_t size) {
    if(type==RTE_TABLE_BV_FIELD_TYPE_RANGE) {
        from_to[0]=*v;
        from_to[1]=v[1];
    } else {
        from_to[0]=(*v)&v[1];
        switch(size) {
        case 1:
            from_to[1]=(*v)|((uint8_t) (~v[1]));
            break;
        case 2:
            from_to[1]=(*v)|((uint16_t) (~v[1]));
            break;
        case 4:
            from_to[1]=(*v)|((uint32_t) (~v[1]));
            break;
        default:
#ifdef DEBUG
            fprintf(stderr, "[cal_from_to] error: unknown size: %d bits\n", size);
#endif
            break;
        }
    }
}

static int rte_table_bv_entry_add(void *t_r, void *k_r, void *e_r, int *key_found, __rte_unused void **e_ptr) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;
    struct rte_table_bv_key *k=(struct rte_table_bv_key *) k_r;

    if(key_found)
        *key_found=0;

    uint32_t from_to[2];
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<t->num_fields; ++f) {
        cal_from_to(from_to, k->buf +(f<<1), t->field_defs[f].type, t->field_defs[f].size);
        rte_bv_markers_range_add(t->bv_markers+f, from_to, k->pos);

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_TABLE_BV_BS;
        ranges.max_num_ranges=RTE_TABLE_BV_MAX_RANGES;
        ranges.ranges_from=t->ranges_from[f];
        ranges.ranges_to=t->ranges_to[f];
        ranges.bvs=t->bvs[f];
        ranges.non_zero_bvs=t->non_zero_bvs[f];
        if(rte_bv_markers_to_ranges(t->bv_markers+f, 1, t->field_defs[f].size, &ranges))
            return 1;
        cudaMemcpy(t->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(&t->entries[t->entry_size*k->pos], e_r, t->entry_size, cudaMemcpyHostToDevice);

    if(e_ptr)
        *e_ptr=&t->entries[t->entry_size*k->pos];

    return 0;
}

static int rte_table_bv_entry_delete(void  *t_r, void *k_r, int *key_found, __rte_unused void *e) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;
    struct rte_table_bv_key *k=(struct rte_table_bv_key *) k_r;

    if(key_found)
        *key_found=0;

    uint32_t from_to[2];
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<t->num_fields; ++f) {
        cal_from_to(from_to, k->buf+(f<<1), t->field_defs[f].type, t->field_defs[f].size);
        rte_bv_markers_range_del(t->bv_markers+f, from_to, k->pos);

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_TABLE_BV_BS;
        ranges.max_num_ranges=RTE_TABLE_BV_MAX_RANGES;
        ranges.ranges_from=t->ranges_from[f];
        ranges.ranges_to=t->ranges_to[f];
        ranges.bvs=t->bvs[f];
        ranges.non_zero_bvs=t->non_zero_bvs[f];
        if(rte_bv_markers_to_ranges(t->bv_markers+f, 1, t->field_defs[f].size, &ranges))
            return 1;
        cudaMemcpy(t->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    return 0;
}

static int rte_table_bv_entry_add_bulk(void *t_r, void **ks_r, void **es_r, uint32_t n_keys, int *key_found, __rte_unused void **e_ptr) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;
    struct rte_table_bv_key **ks=(struct rte_table_bv_key **) ks_r;

    uint32_t from_to[2];
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<t->num_fields; ++f) {
        for(uint32_t k=0; k<n_keys; ++k) {
            cal_from_to(from_to, ks[k]->buf+(f<<1), t->field_defs[f].type, t->field_defs[f].size);
            rte_bv_markers_range_add(t->bv_markers+f, from_to, ks[k]->pos);
        }

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_TABLE_BV_BS;
        ranges.max_num_ranges=RTE_TABLE_BV_MAX_RANGES;
        ranges.ranges_from=t->ranges_from[f];
        ranges.ranges_to=t->ranges_to[f];
        ranges.bvs=t->bvs[f];
        ranges.non_zero_bvs=t->non_zero_bvs[f];
        if(rte_bv_markers_to_ranges(t->bv_markers+f, 1, t->field_defs[f].size, &ranges))
            return 1;
        cudaMemcpy(t->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }


    for(uint32_t k=0; k<n_keys; ++k) {
        if(key_found)
            key_found[k]=0;

        cudaMemcpy(&t->entries[t->entry_size*ks[k]->pos], es_r[ks[k]->pos], t->entry_size, cudaMemcpyHostToDevice);

        if(e_ptr)
            e_ptr[k]=&t->entries[t->entry_size*ks[k]->pos];
    }

    return 0;
}

static int rte_table_bv_entry_delete_bulk(void  *t_r, void **ks_r, uint32_t n_keys, int *key_found, __rte_unused void **es_r) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;
    struct rte_table_bv_key **ks=(struct rte_table_bv_key **) ks_r;

    uint32_t from_to[2];
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<t->num_fields; ++f) {
        for(uint32_t k=0; k<n_keys; ++k) {
            cal_from_to(from_to, ks[k]->buf+(f<<1), t->field_defs[f].type, t->field_defs[f].size);
            rte_bv_markers_range_del(t->bv_markers+f, from_to, ks[k]->pos);
        }

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_TABLE_BV_BS;
        ranges.max_num_ranges=RTE_TABLE_BV_MAX_RANGES;
        ranges.ranges_from=t->ranges_from[f];
        ranges.ranges_to=t->ranges_to[f];
        ranges.bvs=t->bvs[f];
        ranges.non_zero_bvs=t->non_zero_bvs[f];
        if(rte_bv_markers_to_ranges(t->bv_markers+f, 1, t->field_defs[f].size, &ranges))
            return 1;
        cudaMemcpy(t->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    if(key_found)
        for(uint32_t k=0; k<n_keys; ++k)
            key_found[k]=0;

    return 0;
}

__device__ __inline__ uint32_t leu(uint32_t a, uint32_t b, uint8_t size) {
    switch(size) {
    case 4:
        return a<=b;
        break;
    case 2:
        return __vcmpleu2(a, b);
        break;
    case 1:
        return __vcmpleu4(a, b);
        break;
    default:
        __builtin_unreachable();
    }
}

__device__ __inline__ uint32_t leu_offset(const uint32_t x, const uint8_t size) {
    switch(size) {
    case 1:
        return (__ffs(x)>>3);
    case 2:
        return (__ffs(x)>>4);
    case 4:
        return 0;
    default:
        __builtin_unreachable();
    }
}


__device__ __constant__ uint8_t compression_levels[5]= {0,2,1,0,0};

__global__ void bv_search(	uint32_t *__restrict__ *__restrict__ ranges_from,
                            uint32_t *__restrict__ *__restrict__ ranges_to,
                            uint64_t *__restrict__ num_ranges,
                            uint32_t *__restrict__ offsets, uint8_t *__restrict__ sizes,
                            uint64_t *__restrict__ *__restrict__ bvs, uint64_t *__restrict__ *__restrict__ non_zero_bvs,
                            const uint32_t num_fields,
                            const uint32_t entry_size, uint8_t *__restrict__ entries,
                            volatile uint *__restrict__ num_pkts, volatile uint8_t *__restrict__ *__restrict__ pkts,
                            void *__restrict__ *matched_entries, uint8_t *__restrict__ lookup_hit_vec,
                            volatile uint *__restrict__ num_done_pkts, volatile uint8_t *__restrict__ running) {

    __shared__ uint64_t *__restrict__ bv[32][32];
    __shared__ uint64_t *__restrict__ non_zero_bv[32][RTE_TABLE_BV_MAX_FIELDS];
    __shared__ uint8_t field_sizes[RTE_TABLE_BV_MAX_FIELDS];
    __shared__ uint8_t comp_levels[RTE_TABLE_BV_MAX_FIELDS];
    if(!threadIdx.y&&threadIdx.x<num_fields) {
        field_sizes[threadIdx.x]=sizes[threadIdx.x];
        comp_levels[threadIdx.x]=compression_levels[sizes[threadIdx.x]];
    }

    __shared__ uint c_num_pkts;
    __shared__ uint c_num_done_pkts;
    __shared__ uint8_t stop;

    if(!(threadIdx.x|threadIdx.y))
        stop=0;

    while(1) {
        if(!(threadIdx.x|threadIdx.y)) {
            while(*running & ((blockDim.y*blockIdx.x+threadIdx.y)>=*num_pkts));
            if(!*running) {
                stop=1;
            }

            c_num_pkts=*num_pkts;
            c_num_done_pkts=0;
        }

        __syncthreads();

        if(stop)
            break;

        for(int pkt_id=blockDim.y*blockIdx.x+threadIdx.y; pkt_id<c_num_pkts; pkt_id+=blockDim.y*gridDim.x) {
            for(int field_id=0; field_id<num_fields;) {
                uint v;
                if(!threadIdx.x) {
                    bv[threadIdx.y][field_id]=NULL;
                    uint8_t *pkt=(uint8_t * ) pkts[pkt_id]+offsets[field_id];
                    switch(sizes[field_id]) {
                    case 1:
                        v=(*pkt<<24)|(*pkt<<16)|(*pkt<<8)|*pkt;
                        break;
                    case 2:
                        v=(pkt[0]<<8)|(pkt[1]);
                        v|=v<<16;
                        break;
                    case 4:
                        v=(pkt[0]<<24)|(pkt[1]<<16)|(pkt[2]<<8)|(pkt[3]);
                        break;
                    default:
                        printf("[%d|%d] unknown size: %u byte\n", blockIdx.x, threadIdx.y, field_sizes[field_id]);
                    }
                }

                __syncwarp();
                v=__shfl_sync(UINT32_MAX, v, 0);
                long size=num_ranges[field_id]>>(5+comp_levels[field_id]);
                long start=0, offset;
                uint32_t l,r; //left, right
                uint32_t lres, rres;
                while(size) {
                    offset=start+((long) threadIdx.x)*size;
                    lres=leu(ranges_from[field_id][offset],v,field_sizes[field_id]);
                    rres=leu(v,ranges_to[field_id][offset],field_sizes[field_id]);
                    l=__ballot_sync(UINT32_MAX, lres);
                    r=__ballot_sync(UINT32_MAX, rres);
                    if(l&r) {
                        if((__ffs(l&r)-1)==threadIdx.x) {
                            if(!(lres&rres))
                                goto found_bv;
                            const long pos=(offset<<comp_levels[field_id])|leu_offset(lres&rres, field_sizes[field_id]);
                            bv[threadIdx.y][field_id]=bvs[field_id]+pos*RTE_TABLE_BV_BS;
                            non_zero_bv[threadIdx.y][field_id]=non_zero_bvs[field_id]+pos*RTE_TABLE_NON_ZERO_BV_BS;
                        }
                        __syncwarp();
                        goto found_bv;
                    }
                    if(!l)
                        goto found_bv;

                    r=__popc(l)-1;
                    start=__shfl_sync(UINT32_MAX, offset+1, r);
                    size=r==31?((num_ranges[field_id]>>comp_levels[field_id])-start)>>5:(size-1LU)>>5;

                    __syncwarp();
                }

                offset=start+threadIdx.x;

                lres=offset<num_ranges[field_id]?leu(ranges_from[field_id][offset],v,field_sizes[field_id]):0;
                rres=offset<num_ranges[field_id]?leu(v,ranges_to[field_id][offset],field_sizes[field_id]):0;

                if(lres&rres) {
                    const long pos=(offset<<comp_levels[field_id])|leu_offset(lres&rres, field_sizes[field_id]);
                    bv[threadIdx.y][field_id]=bvs[field_id]+pos*RTE_TABLE_BV_BS;
                    non_zero_bv[threadIdx.y][field_id]=non_zero_bvs[field_id]+pos*RTE_TABLE_NON_ZERO_BV_BS;
                }
                __syncwarp();
found_bv:
                ++field_id;
                __syncwarp();
            }

            if(__ballot_sync(UINT32_MAX, threadIdx.x<num_fields&&!bv[threadIdx.y][threadIdx.x])) {
                if(!threadIdx.x)
                    lookup_hit_vec[pkt_id]=0;
                atomicAdd(&c_num_done_pkts, 1);
                goto wait_for_other_warps;
            }

            // all bitvectors found, now getting highest-priority rule


            int nz_bv_b=threadIdx.x;
            uint32_t in_loop=__ballot_sync(UINT32_MAX, nz_bv_b<RTE_TABLE_NON_ZERO_BV_BS);

            while(nz_bv_b<RTE_TABLE_NON_ZERO_BV_BS) {
                uint64_t x=non_zero_bv[threadIdx.y][0][nz_bv_b];
                for(int field_id=1; field_id<num_fields; ++field_id)
                    x&=non_zero_bv[threadIdx.y][field_id][nz_bv_b];

                int pos;
                uint64_t y=0;
                while((pos=__ffsll(x))) {
                    const int p=(nz_bv_b<<6)|(pos-1);

                    y=bv[threadIdx.y][0][p];
                    for(int field_id=1; field_id<num_fields; ++field_id)
                        y&=bv[threadIdx.y][field_id][p];

                    if(y)
                        break;

                    x>>=pos;
                    x<<=pos;
                }

                const uint32_t tm=__ballot_sync(in_loop, __ffsll(y));
                if(tm) {
                    if((__ffs(tm)-1)==threadIdx.x) {
                        matched_entries[pkt_id]=(void *) &entries[entry_size*((nz_bv_b<<12)+((pos-1)<<6)+__ffsll(y)-1LU)];
                        lookup_hit_vec[pkt_id]=1;
                        atomicAdd(&c_num_done_pkts, 1);
                    }
                    goto found_rule;
                }

                nz_bv_b+=blockDim.x;
                in_loop=__ballot_sync(in_loop, nz_bv_b<RTE_TABLE_NON_ZERO_BV_BS);
            }

            if(!threadIdx.x) {
                lookup_hit_vec[pkt_id]=0;
                atomicAdd(&c_num_done_pkts, 1);
            }

found_rule:
            __syncwarp();
        }

wait_for_other_warps:

        __syncthreads();
        if(!(threadIdx.x|threadIdx.y)) {
            atomicSub((uint *) num_pkts, c_num_done_pkts);
            while(*num_pkts);
            __threadfence_system();
            atomicAdd((uint *) num_done_pkts, c_num_done_pkts);
            __threadfence_system();
        }
        __syncthreads();
    }

    if(!(threadIdx.x|threadIdx.y|blockIdx.x))
        printf("CUDA kernel stopped\n");
}

int rte_table_bv_start_kernel(void *t_r) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;
    *t->running_h=1;

    bv_search<<<t->num_blocks, dim3{32, t->packets_per_block}>>>(
        t->ranges_from_dev, t->ranges_to_dev, t->num_ranges,
        t->field_offsets, t->field_sizes, t->bvs_dev, t->non_zero_bvs_dev,
        t->num_fields, t->entry_size, t->entries,
        t->num_pkts, (volatile uint8_t **) t->pkts_data,
        (void **) t->matched_entries,
        t->lookup_hit_vec,
        (uint32_t *) t->num_done_pkts,
        t->running);
    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess) {
        printf("[rte_table_start_kernel|%d] error: %s\n", __LINE__, cudaGetErrorString(err));
        return 1;
    }
    printf("DONE LAUNCHING...\n");
    return 0;
}

int rte_table_bv_stop_kernel(void *t_r) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;

    printf("stopping kernel\n");
    *t->running_h=0;
    cudaStreamSynchronize(0);
    printf("kernel stopped\n");

    return 0;
}

#define  ONCE(X) (*(volatile typeof(X) *) &(X))
#define IS_ERROR(X) is_error(X, __FILE__, __LINE__)
int rte_table_bv_lookup_burst(void *t_r, uint8_t *lookup_hit_vec,
                              struct rte_mbuf **pkts, uint32_t num_pkts, void **e) {
#ifdef MEASURE_TIME
    struct timeval k_t1,k_t2,l_t1,l_t2;
    gettimeofday(&l_t1, NULL);
#endif

    struct rte_table_bv *t=(struct rte_table_bv *) t_r;


#ifdef MEASURE_TIME
    gettimeofday(&k_t1, NULL);
#endif

    for(uint64_t i=0; i<num_pkts; ++i)
        t->pkts_data[i]=rte_pktmbuf_mtod(pkts[i], uint8_t *);

    ONCE(*t->num_done_pkts)=0;
    ONCE(*t->num_pkts)=num_pkts;

    while(ONCE(*t->num_done_pkts)!=num_pkts);

    memcpy(e, (const void *) t->matched_entries_h, sizeof(void *)*num_pkts);
    memcpy(lookup_hit_vec, t->lookup_hit_vec_h, sizeof(uint8_t)*num_pkts);

#ifdef MEASURE_TIME
    gettimeofday(&k_t2, NULL);
    printf("KERNEL took %luus\n", (k_t2.tv_sec*1000000+k_t2.tv_usec)-(k_t1.tv_sec*1000000+k_t1.tv_usec));

    gettimeofday(&k_t2, NULL);
    gettimeofday(&l_t2, NULL);
    printf("LOOKUP took %luus\n", (l_t2.tv_sec*1000000+l_t2.tv_usec)-(l_t1.tv_sec*1000000+l_t1.tv_usec));
#endif
    return 0;
}

static int rte_table_bv_lookup(void *t_r, struct rte_mbuf **pkts, uint64_t pkts_mask, uint64_t *lookup_hit_mask, void **e) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;

    const uint32_t num_pkts=__builtin_popcountll(pkts_mask);
    RTE_TABLE_BV_STATS_PKTS_IN_ADD(t, n_pkts_in);

    for(uint64_t i=0; i<num_pkts; ++i)
        t->pkts_data[i]=rte_pktmbuf_mtod(pkts[i], uint8_t *);

    ONCE(*t->num_done_pkts)=0;
    ONCE(*t->num_pkts)=(uint32_t) num_pkts;

    while(ONCE(*t->num_done_pkts)!=num_pkts);

    memcpy(e, (const void *) t->matched_entries_h, sizeof(void *)*num_pkts);

    uint64_t lhm=0;
    for(uint32_t i=0; i<num_pkts; ++i) {
        if(t->lookup_hit_vec_h[i]) {
            t->lookup_hit_vec_h[i]=0;
        }
    }
    *lookup_hit_mask=lhm;


    RTE_TABLE_BV_STATS_PKTS_LOOKUP_MISS(t, n_pkts_in-__builtin_popcountll(*lookup_hit_mask));

    return 0;
}

static int rte_table_bv_stats_read(void *t_r, struct rte_table_stats *stats, int clear) {
    struct rte_table_bv *t = (struct rte_table_bv *) t_r;

    if (stats != NULL)
        memcpy(stats, &t->stats, sizeof(t->stats));

    if (clear)
        memset(&t->stats, 0, sizeof(t->stats));

    return 0;
}

struct rte_table_ops rte_table_bv_ops = {
    .f_create = rte_table_bv_create,
    .f_free = rte_table_bv_free,
    .f_add = rte_table_bv_entry_add,
    .f_delete = rte_table_bv_entry_delete,
    .f_add_bulk = rte_table_bv_entry_add_bulk,
    .f_delete_bulk = rte_table_bv_entry_delete_bulk,
    .f_lookup = rte_table_bv_lookup,
    .f_stats = rte_table_bv_stats_read
};

#ifdef __cplusplus
}
#endif
