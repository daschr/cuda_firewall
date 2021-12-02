#include "rte_table_bv.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "rte_bv.h"
#include <rte_log.h>
#include <rte_malloc.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef RTE_TABLE_STATS_COLLECT

#define RTE_TABLE_BV_STATS_PKTS_IN_ADD(table, val) table->stats.n_pkts_in += val
#define RTE_TABLE_BV_STATS_PKTS_LOOKUP_MISS(table, val) table->stats.n_pkts_lookup_miss += val

#else

#define RTE_TABLE_BV_STATS_PKTS_IN_ADD(table, val)
#define RTE_TABLE_BV_STATS_PKTS_LOOKUP_MISS(table, val)

#endif

#define ONCE(X) (*(volatile typeof(X) *) &(X))

struct rte_table_bv {
    uint32_t num_fields;
    struct rte_table_stats stats;
    const struct rte_table_bv_field_def *field_defs;

    uint8_t act_buf; // which double buffer is currently active
    uint32_t **ranges; // size==[num_fields][2*RTE_TABLE_BV_MAX_RANGES]
    uint32_t **bvs; // size==[num_fields][RTE_TABLE_BV_BS*2*RTE_TABLE_BV_MAX_RANGES]

    size_t *num_ranges;
    uint32_t *field_ptype_masks;
    uint32_t *field_offsets;
    uint8_t *field_sizes;

    uint32_t **ranges_dev;
    uint32_t **bvs_dev;

    volatile uint8_t **pkts_data;
    volatile uint8_t **pkts_data_h;

    volatile uint32_t *packet_types;
    volatile uint32_t *packet_types_h;

    volatile uint64_t *pkts_mask;
    volatile uint64_t *pkts_mask_h;

    volatile uint32_t *positions;
    volatile uint32_t *positions_h;

    volatile uint64_t *lookup_hit_mask;
    volatile uint64_t *lookup_hit_mask_h;

    volatile uint64_t *done_pkts;
    volatile uint64_t *done_pkts_h;

    volatile uint64_t *done_pkts_dev;

    volatile uint8_t *running;
    volatile uint8_t *running_h;

    rte_bv_markers_t *bv_markers; // size==num_fields
};

static inline int is_error(cudaError_t e, const char *file, int line) {
    if(e!=cudaSuccess) {
        fprintf(stderr, "[rte_table_bv] error: %s in %s (line %d)\n", cudaGetErrorString(e), file, line);
        return 1;
    }
    return 0;
}

static int rte_table_bv_free(void *t_r) {
    if(t_r==NULL)
        return 0;

    struct rte_table_bv *t=(struct rte_table_bv *) t_r;

    for(size_t i=0; i<t->num_fields; ++i) {
        cudaFree(t->ranges[i]);
        cudaFree(t->bvs[i]);
    }
    cudaFree(t->ranges_dev);
    cudaFree(t->bvs_dev);

    cudaFree(t->num_ranges);
    cudaFree(t->field_offsets);
    cudaFree(t->field_sizes);
    cudaFree((void *) t->done_pkts_dev);

    cudaFreeHost((void *) t->pkts_data);
    cudaFreeHost((void *) t->packet_types);
    cudaFreeHost((void *) t->positions);
    cudaFreeHost((void *) t->lookup_hit_mask);
    cudaFreeHost((void *) t->done_pkts);
    cudaFreeHost((void *) t->running);

    for(uint32_t i=0; i<t->num_fields; ++i)
        rte_bv_markers_free(t->bv_markers+i);

    rte_free(t->bv_markers);
    rte_free(t->ranges);
    rte_free(t->bvs);

    rte_free(t);

    return 0;
}

#define IS_ERROR(X) is_error(X, __FILE__, __LINE__)

static void *rte_table_bv_create(void *params, int socket_id, uint32_t entry_size) {
    struct rte_table_bv_params *p=(struct rte_table_bv_params *) params;
    struct rte_table_bv *t=(struct rte_table_bv *) rte_malloc("t", sizeof(struct rte_table_bv), 0);
    memset(t, 0, sizeof(struct rte_table_bv));

    t->num_fields=p->num_fields;
    t->field_defs=p->field_defs;
    t->act_buf=0;

    t->ranges=(uint32_t **) rte_malloc("ranges_db", sizeof(uint32_t *)*t->num_fields, 0);
    t->bvs=(uint32_t **) rte_malloc("bvs_db", sizeof(uint32_t *)*t->num_fields, 0);

#define CHECK(X) if(IS_ERROR(X)) return NULL

    CHECK(cudaHostAlloc((void **) &t->pkts_data_h, sizeof(uint8_t*)*RTE_TABLE_BV_MAX_PKTS, cudaHostAllocMapped|cudaHostAllocWriteCombined));
    CHECK(cudaHostGetDevicePointer((void **) &t->pkts_data, t->pkts_data_h, 0));

    CHECK(cudaHostAlloc((void **) &t->packet_types_h, sizeof(uint32_t)*RTE_TABLE_BV_MAX_PKTS, cudaHostAllocMapped|cudaHostAllocWriteCombined));
    CHECK(cudaHostGetDevicePointer((void **) &t->packet_types, (void *) t->packet_types_h, 0));

#define HOSTALLOC(DP, SIZE) CHECK(cudaHostAlloc((void **) &DP ## _h, SIZE, cudaHostAllocMapped));\
                            CHECK(cudaHostGetDevicePointer((void **) &DP, (void *) DP ## _h, 0));

    HOSTALLOC(t->pkts_mask, sizeof(uint64_t));
    HOSTALLOC(t->positions, sizeof(uint32_t)*RTE_TABLE_BV_MAX_PKTS);
    HOSTALLOC(t->lookup_hit_mask, sizeof(uint64_t));
    HOSTALLOC(t->done_pkts, sizeof(uint64_t));
    HOSTALLOC(t->running, sizeof(uint8_t));

#undef HOSTALLOC

    *t->pkts_mask_h=0;
    *t->running_h=1;

    CHECK(cudaMalloc((void **) &t->ranges_dev, sizeof(uint32_t *)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->bvs_dev, sizeof(uint32_t *)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->field_offsets, sizeof(uint32_t)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->field_ptype_masks, sizeof(uint32_t)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->field_sizes, sizeof(uint32_t)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->num_ranges, sizeof(uint64_t)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->done_pkts_dev, sizeof(uint64_t)));

    for(size_t i=0; i<t->num_fields; ++i) {
        CHECK(cudaMemcpy(t->field_offsets+i, &t->field_defs[i].offset, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(t->field_sizes+i, &t->field_defs[i].size, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(t->field_ptype_masks+i, &t->field_defs[i].ptype_mask, sizeof(uint32_t), cudaMemcpyHostToDevice));

        CHECK(cudaMalloc((void **) &t->ranges[i], sizeof(uint32_t)*((size_t) RTE_TABLE_BV_MAX_RANGES) *2));
        CHECK(cudaMalloc((void **) &t->bvs[i], sizeof(uint32_t)*((size_t) RTE_TABLE_BV_BS) * ((size_t ) RTE_TABLE_BV_MAX_RANGES) *2));
    }

    CHECK(cudaMemcpy(t->ranges_dev, t->ranges, sizeof(uint32_t *)*t->num_fields, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(t->bvs_dev, t->bvs, sizeof(uint32_t *)*t->num_fields, cudaMemcpyHostToDevice));

    int mp_count;
    CHECK(cudaDeviceGetAttribute(&mp_count, cudaDevAttrMultiProcessorCount, 0));
    printf("mp_count: %d\n", mp_count);

#undef CHECK

    t->bv_markers=(rte_bv_markers_t *) rte_malloc("bv_markers", sizeof(rte_bv_markers_t)*t->num_fields, 0);

    for(size_t i=0; i<t->num_fields; ++i) {
        if(rte_bv_markers_create(&t->bv_markers[i])) {
            rte_table_bv_free(t);
            rte_log(RTE_LOG_ERR, RTE_LOGTYPE_HASH, "Error creating marker!\n");
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
            fprintf(stderr, "[cal_from_to] error: unkown size: %d bits\n", size);
#endif
            break;
        }
    }
}

static int rte_table_bv_entry_add(void *t_r, void *k_r, void *e_r, int *key_found, void **e_ptr) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;
    struct rte_table_bv_key *k=(struct rte_table_bv_key *) k_r;
    uint32_t *pos=(uint32_t *) e_r;
    *key_found=0;

    uint32_t from_to[2];
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<t->num_fields; ++f) {
        cal_from_to(from_to, k->buf +(f<<1), t->field_defs[f].type, t->field_defs[f].size);
        rte_bv_markers_range_add(t->bv_markers+f, from_to, *pos);

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_TABLE_BV_BS;
        ranges.ranges=t->ranges[t->num_fields+f];
        ranges.bvs=t->bvs[t->num_fields+f];
        rte_bv_markers_to_ranges(t->bv_markers+f, 1, sizeof(uint32_t), &ranges);
        cudaMemcpy(t->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    return 0;
}

static int rte_table_bv_entry_delete(void  *t_r, void *k_r, int *key_found, void *e) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;
    struct rte_table_bv_key *k=(struct rte_table_bv_key *) k_r;
    *key_found=0;

    uint32_t from_to[2];
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<t->num_fields; ++f) {
        cal_from_to(from_to, k->buf+(f<<1), t->field_defs[f].type, t->field_defs[f].size);
        rte_bv_markers_range_del(t->bv_markers+f, from_to, k->pos);

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_TABLE_BV_BS;
        ranges.ranges=t->ranges[f];
        ranges.bvs=t->bvs[f];
        rte_bv_markers_to_ranges(t->bv_markers+f, 1, sizeof(uint32_t), &ranges);
        cudaMemcpy(t->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    return 0;
}

static int rte_table_bv_entry_add_bulk(void *t_r, void **ks_r, void **es_r, uint32_t n_keys, int *key_found, void **e_ptr) {
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
        ranges.ranges=t->ranges[f];
        ranges.bvs=t->bvs[f];
        rte_bv_markers_to_ranges(t->bv_markers+f, 1, sizeof(uint32_t), &ranges);
        cudaMemcpy(t->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    return 0;
}

static int rte_table_bv_entry_delete_bulk(void  *t_r, void **ks_r, uint32_t n_keys, int *key_found, void **es_r) {
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
        ranges.ranges=t->ranges[f];
        ranges.bvs=t->bvs[f];
        rte_bv_markers_to_ranges(t->bv_markers+f, 1, sizeof(uint32_t), &ranges);
        cudaMemcpy(t->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    return 0;
}

__global__ void bv_search(	uint32_t **__restrict__ ranges, const uint64_t *__restrict__ num_ranges,
                            const uint32_t *__restrict__ offsets, const uint8_t *__restrict__ sizes,
                            const uint32_t *__restrict__ ptype_mask,  uint32_t **__restrict__ bvs,
                            const uint32_t bv_bs, volatile uint64_t *__restrict__ pkts_mask,
                            volatile uint8_t **__restrict__ pkts, volatile uint32_t *__restrict__ pkts_type,
                            volatile uint *__restrict__ positions, volatile uint64_t *__restrict__ lookup_hit_mask,
                            volatile uint64_t *__restrict__ done_pkts, volatile uint64_t *__restrict__ done_pkts_dev,
                            volatile uint8_t *__restrict__ running) {

    __shared__ uint *bv[16][RTE_TABLE_BV_MAX_FIELDS];
    __shared__ bool field_found[16][RTE_TABLE_BV_MAX_FIELDS];

    volatile __shared__ uint64_t c_pkts_mask;
    __shared__ uint64_t c_done_pkts;
    __shared__ uint64_t c_lookup_hit_mask;
    volatile __shared__ uint8_t stop;

    if(!(threadIdx.x|threadIdx.y))
        stop=0;

    const int pkt_id=blockIdx.x*blockDim.y+threadIdx.y;
    const uint64_t reset_block_mask=~(0xffffLU<<pkt_id);

    __threadfence_block();
    __syncthreads();

    while(1) {
        if(!(threadIdx.x|threadIdx.y)) {
            while(*running&(((*pkts_mask>>pkt_id)&1LU)==0));
            if(!*running)
                stop=1;

            c_pkts_mask=*pkts_mask;
            c_lookup_hit_mask=0;
            c_done_pkts=0;
            atomicAnd((unsigned long long int *) done_pkts, reset_block_mask);
        }

        __syncthreads();

        if(stop)
            goto exit;

        field_found[threadIdx.y][threadIdx.x]=false;

        const uint32_t ptype_a=pkts_type[pkt_id]&ptype_mask[threadIdx.x];
        const bool ptype_matches=  (ptype_a&RTE_PTYPE_L2_MASK)!=0
                                   & (ptype_a&RTE_PTYPE_L3_MASK)!=0
                                   & (ptype_a&RTE_PTYPE_L4_MASK)!=0;


        if((c_pkts_mask>>pkt_id)&1LU& ptype_matches) {
            uint v;
            const uint8_t *pkt=(uint8_t * ) pkts[pkt_id]+offsets[threadIdx.x];
            switch(sizes[threadIdx.x]) {
            case 1:
                v=*pkt;
                break;
            case 2:
                v=pkt[1]+(pkt[0]<<8);
                break;
            case 4:
                v=pkt[3]+(pkt[2]<<8)+(pkt[1]<<16)+(pkt[0]<<24);
                break;
            default:
                printf("[%d|%d] unknown size: %u byte\n", blockIdx.x, threadIdx.x, sizes[threadIdx.x]);
                break;
            }

            const uint *range_dim=ranges[threadIdx.x];
            long se[]= {0, (long) num_ranges[threadIdx.x]};
            uint8_t l,r;
            bv[threadIdx.y][threadIdx.x]=NULL;
            for(long i=se[1]>>1; se[0]<=se[1]; i=(se[0]+se[1])>>1) {
                l=v>=range_dim[i<<1];
                r=v<=range_dim[(i<<1)+1];
                if(l&r) {
                    bv[threadIdx.y][threadIdx.x]=bvs[threadIdx.x]+i*RTE_TABLE_BV_BS;
                    field_found[threadIdx.y][threadIdx.x]=true;
                    break;
                }

                se[!l]=!l?i-1:i+1;
            }
        }

        __syncthreads();
        if((c_pkts_mask>>pkt_id)&1 & (!threadIdx.x)) {
            uint x, pos;
            for(int i=0; i<bv_bs; ++i) {
                x=0xffffffff;
                for(int b=0; b<blockDim.x; ++b) {
                    if(!field_found[threadIdx.y][b])
                        goto end;
                    x&=bv[threadIdx.y][b][i];
                }

                if((pos=__ffs(x))!=0) {
                    positions[pkt_id]=(i<<5)+pos-1;
                    atomicOr((unsigned long long int *)&c_lookup_hit_mask, 1LU<<pkt_id);
                    break;
                }
            }

end:
            atomicOr((unsigned long long int *) &c_done_pkts, 1LU<<pkt_id);
        }

        __syncthreads();
        if(!(threadIdx.x|threadIdx.y)) {
            atomicAnd((unsigned long long int *) pkts_mask, reset_block_mask);
            atomicOr((unsigned long long int *) lookup_hit_mask, c_lookup_hit_mask);

            __threadfence_system(); // sync global writes on host, make sure everything other is written except done_pkts
            atomicOr((unsigned long long int *) done_pkts, c_done_pkts);
            __threadfence_system(); // sync global writes on host
        }

        __syncthreads();
    }

exit:
    if(!(threadIdx.x|threadIdx.y|blockIdx.x))
        printf("CUDA kernel stopped\n");
}

int rte_table_bv_start_kernel(void *t_r) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;
    *t->running_h=1;

    bv_search<<<4, dim3{t->num_fields,16}>>>(   t->ranges_dev, t->num_ranges,
            t->field_offsets, t->field_sizes, t->field_ptype_masks,
            t->bvs_dev, RTE_TABLE_BV_BS,
            t->pkts_mask, (volatile uint8_t **) t->pkts_data, t->packet_types,
            t->positions, t->lookup_hit_mask, t->done_pkts, t->done_pkts_dev, t->running);
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

static int rte_table_bv_lookup(void *t_r, struct rte_mbuf **pkts, uint64_t pkts_mask, uint64_t *lookup_hit_mask, void **e) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;

    const uint32_t n_pkts_in=__builtin_popcountll(pkts_mask);
    RTE_TABLE_BV_STATS_PKTS_IN_ADD(t, n_pkts_in);

    for(uint32_t i=0; i<n_pkts_in; ++i)
        if((pkts_mask>>i)&1) {
            t->pkts_data_h[i]=rte_pktmbuf_mtod(pkts[i], uint8_t *);
            t->packet_types_h[i]=pkts[i]->packet_type;
        }

    ONCE(*t->lookup_hit_mask_h)=0LU;
    ONCE(*t->done_pkts_h)=0LU;
    ONCE(*t->pkts_mask_h)=pkts_mask;

    while(*ONCE(t->done_pkts_h)!=pkts_mask);

    *lookup_hit_mask=ONCE(*t->lookup_hit_mask_h);
    memcpy(e, (const void *) t->positions_h, sizeof(uint32_t)*__builtin_popcountll(*ONCE(t->lookup_hit_mask_h)));

    RTE_TABLE_BV_STATS_PKTS_LOOKUP_MISS(t, n_pkts_in-__builtin_popcountll(*ONCE(t->lookup_hit_mask_h)));

    /*
        cudaError_t err = cudaGetLastError();
        if(err!=cudaSuccess)
            printf("[bv_search] error: %s\n", cudaGetErrorString(err));
    */
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
