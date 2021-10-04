#include "rte_table_bv.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "rte_bv.h"
#include <rte_log.h>
#include <stdlib.h>

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

    uint8_t *act_buf; // size==1, pointer for gpu
    uint8_t *act_buf_h; // host pointer
    uint32_t **ranges_db; // size==[2*num_fields][2*RTE_TABLE_BV_MAX_RANGES]
    uint32_t **bvs_db; // size==[2*num_fields][RTE_TABLE_BV_BS*2*RTE_TABLE_BV_MAX_RANGES]

    size_t *num_ranges;
    uint32_t *field_offsets;
    uint8_t *field_sizes;
    uint32_t **ranges_db_dev;
    uint32_t **bvs_db_dev;

    uint8_t **pkts_data;
    uint8_t **pkts_data_h;

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

    cudaFreeHost(t->act_buf);
    for(size_t i=0; i<t->num_fields<<1; ++i) {
        cudaFree(t->ranges_db[i]);
        cudaFree(t->bvs_db[i]);
    }
    cudaFree(t->ranges_db_dev);
    cudaFree(t->bvs_db_dev);

    cudaFree(t->num_ranges);
    cudaFree(t->field_offsets);
    cudaFree(t->field_sizes);

    for(uint32_t i=0; i<t->num_fields; ++i)
        rte_bv_markers_free(t->bv_markers+i);

    free(t->bv_markers);
    free(t);

    return 0;
}

#define IS_ERROR(X) is_error(X, __FILE__, __LINE__)

static void *rte_table_bv_create(void *params, int socket_id, uint32_t entry_size) {
    struct rte_table_bv_params *p=(struct rte_table_bv_params *) params;
    struct rte_table_bv *t=(struct rte_table_bv *) malloc(sizeof(struct rte_table_bv));
    memset(t, 0, sizeof(struct rte_table_bv));

    t->num_fields=p->num_fields;
    t->field_defs=p->field_defs;

    t->ranges_db=(uint32_t **) malloc(sizeof(uint32_t *)*(t->num_fields<<1));
    t->bvs_db=(uint32_t **) malloc(sizeof(uint32_t *)*(t->num_fields<<1));
#define CHECK(X) if(IS_ERROR(X)) return NULL
    CHECK(cudaHostAlloc((void **) &t->act_buf_h, sizeof(uint8_t), cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer((void **) &t->act_buf, t->act_buf_h, 0));

    CHECK(cudaHostAlloc((void **) &t->pkts_data_h, sizeof(uint8_t*)*64, cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer((void **) &t->pkts_data, t->pkts_data_h, 0));

    CHECK(cudaMalloc((void **) &t->ranges_db_dev, sizeof(uint32_t *)*t->num_fields*2));
    CHECK(cudaMalloc((void **) &t->bvs_db_dev, sizeof(uint32_t *)*t->num_fields*2));
    CHECK(cudaMalloc((void **) &t->field_offsets, sizeof(uint32_t)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->field_sizes, sizeof(uint32_t)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->num_ranges, sizeof(size_t)*t->num_fields));

    for(size_t i=0; i<t->num_fields; ++i) {
        CHECK(cudaMemcpy(t->field_offsets+i, &t->field_defs[i].offset, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(t->field_sizes+i, &t->field_defs[i].size, sizeof(uint32_t), cudaMemcpyHostToDevice));

        CHECK(cudaMalloc((void **) &t->ranges_db[i], sizeof(uint32_t)*RTE_TABLE_BV_MAX_RANGES*2));
        CHECK(cudaMalloc((void **) &t->ranges_db[t->num_fields+i], sizeof(uint32_t)*RTE_TABLE_BV_MAX_RANGES*2));
        CHECK(cudaMalloc((void **) &t->bvs_db[i], sizeof(uint32_t)*RTE_TABLE_BV_BS*RTE_TABLE_BV_MAX_RANGES*2));
        CHECK(cudaMalloc((void **) &t->bvs_db[t->num_fields+i], sizeof(uint32_t)*RTE_TABLE_BV_BS*RTE_TABLE_BV_MAX_RANGES*2));
    }

    CHECK(cudaMemcpy(t->ranges_db_dev, t->ranges_db, sizeof(uint32_t *)*t->num_fields*2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(t->bvs_db_dev, t->bvs_db, sizeof(uint32_t *)*t->num_fields*2, cudaMemcpyHostToDevice));
#undef CHECK

    t->bv_markers=(rte_bv_markers_t *) malloc(sizeof(rte_bv_markers_t)*t->num_fields);

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
    uint32_t *val=(uint32_t *) e_r;
    *key_found=0;

    uint32_t from_to[2];
    uint8_t next_act_buf=*t->act_buf_h^1;
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<t->num_fields; ++f) {
        cal_from_to(from_to, k->buf +(f<<1), t->field_defs[f].type, t->field_defs[f].size);
        rte_bv_markers_range_add(t->bv_markers+f, from_to, *val);

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_TABLE_BV_BS;
        ranges.ranges=t->ranges_db[(next_act_buf*t->num_fields)+f];
        ranges.bvs=t->bvs_db[(next_act_buf*t->num_fields)+f];
        rte_bv_markers_to_ranges(t->bv_markers+f, 1, sizeof(uint32_t), &ranges);
        cudaMemcpy(t->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    *t->act_buf_h=next_act_buf;

    return 0;
}

static int rte_table_bv_entry_delete(void  *t_r, void *k_r, int *key_found, void *e) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;
    struct rte_table_bv_key *k=(struct rte_table_bv_key *) k_r;
    *key_found=0;

    uint32_t from_to[2];
    uint8_t next_act_buf=*t->act_buf_h^1;
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<t->num_fields; ++f) {
        cal_from_to(from_to, k->buf+(f<<1), t->field_defs[f].type, t->field_defs[f].size);
        rte_bv_markers_range_del(t->bv_markers+f, from_to, k->val);

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_TABLE_BV_BS;
        ranges.ranges=t->ranges_db[(next_act_buf*t->num_fields)+f];
        ranges.bvs=t->bvs_db[(next_act_buf*t->num_fields)+f];
        rte_bv_markers_to_ranges(t->bv_markers+f, 1, sizeof(uint32_t), &ranges);
        cudaMemcpy(t->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    return 0;
}

static int rte_table_bv_entry_add_bulk(void *t_r, void **ks_r, void **es_r, uint32_t n_keys, int *key_found, void **e_ptr) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;
    struct rte_table_bv_key **ks=(struct rte_table_bv_key **) ks_r;

    uint32_t from_to[2];
    uint8_t next_act_buf=*t->act_buf_h^1;
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<t->num_fields; ++f) {
        for(uint32_t k=0; k<n_keys; ++k) {
            cal_from_to(from_to, ks[k]->buf+(f<<1), t->field_defs[f].type, t->field_defs[f].size);
            rte_bv_markers_range_add(t->bv_markers+f, from_to, ks[k]->pos);
        }

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_TABLE_BV_BS;
        ranges.ranges=t->ranges_db[(next_act_buf*t->num_fields)+f];
        ranges.bvs=t->bvs_db[(next_act_buf*t->num_fields)+f];
        rte_bv_markers_to_ranges(t->bv_markers+f, 1, sizeof(uint32_t), &ranges);
        cudaMemcpy(t->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    *t->act_buf_h=next_act_buf;

    return 0;
}

static int rte_table_bv_entry_delete_bulk(void  *t_r, void **ks_r, uint32_t n_keys, int *key_found, void **es_r) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;
    struct rte_table_bv_key **ks=(struct rte_table_bv_key **) ks_r;

    uint32_t from_to[2];
    uint8_t next_act_buf=*t->act_buf_h^1;
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<t->num_fields; ++f) {
        for(uint32_t k=0; k<n_keys; ++k) {
            cal_from_to(from_to, ks[k]->buf+(f<<1), t->field_defs[f].type, t->field_defs[f].size);
            rte_bv_markers_range_del(t->bv_markers+f, from_to, ks[k]->pos);
        }

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_TABLE_BV_BS;
        ranges.ranges=t->ranges_db[(next_act_buf*t->num_fields)+f];
        ranges.bvs=t->bvs_db[(next_act_buf*t->num_fields)+f];
        rte_bv_markers_to_ranges(t->bv_markers+f, 1, sizeof(uint32_t), &ranges);
        cudaMemcpy(t->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    *t->act_buf_h=next_act_buf;

    return 0;
}

__global__ void bv_search(	uint32_t **ranges, uint64_t *num_ranges, uint32_t *offsets, uint8_t *sizes,
                            uint32_t **bvs, uint32_t bv_bs,
                            ulong pkts_mask, uint8_t **pkts,
                            volatile uint *__restrict__ vals, volatile ulong *lookup_hit_mask) {

    if(!((pkts_mask>>blockIdx.x)&1))
        return;

    uint8_t *pkt=pkts[blockIdx.x]+offsets[threadIdx.x];

    __shared__ uint *bv[24];
    uint v=0;
    switch(sizes[threadIdx.x]) {
    case 1:
        v=*pkt;
        printf("[%d|%d] 8bit: %u\n", blockIdx.x, threadIdx.x, v);
        break;
    case 2:
        v=pkt[1]+(pkt[0]<<8);
        printf("[%d|%d] 16bit: %u\n", blockIdx.x, threadIdx.x, v);
        break;
    case 4:
        printf("[%d|%d] IP: %u.%u.%u.%u\n", blockIdx.x, threadIdx.x, pkt[0], pkt[1], pkt[2], pkt[3]);
        v=pkt[3]+(pkt[2]<<8)+(pkt[1]<<16)+(pkt[0]<<24);
        break;
    default:
        printf("[%d|%d] unknown size: %ubit\n", blockIdx.x, threadIdx.x, sizes[threadIdx.x]);
        break;
    }

    uint *range_dim=ranges[threadIdx.x];
    long se[]= {0, (long) num_ranges[threadIdx.x]};
    uint8_t l,r;
    bv[threadIdx.x]=NULL;
    for(long i=se[1]>>1; se[0]<=se[1]; i=(se[0]+se[1])>>1) {
        l=v>=range_dim[i<<1];
        r=v<=range_dim[(i<<1)+1];

        if(l&r) {
            bv[threadIdx.x]=bvs[threadIdx.x]+i*RTE_TABLE_BV_BS;
            break;
        }

        se[!l]=!l?i-1:i+1;
    }

    __syncthreads();
    if(!threadIdx.x) {
        for(uint i=0; i<blockDim.x; ++i)
            if(bv[i]==NULL)
                goto end;
        uint x, pos;
        for(uint i=0; i<bv_bs; ++i) {
            x=bv[0][i];
            for(uint b=0; b<blockDim.x; ++b)
                x&=bv[b][i];

            if((pos=__ffs(x))!=0) {
                vals[blockIdx.x]=(i<<5)+pos;
                atomicOr((unsigned long long *)lookup_hit_mask, 1<<blockIdx.x);
                break;
            }
        }
    }
end:
    __syncthreads();
}

static int rte_table_bv_lookup(void *t_r, struct rte_mbuf **pkts, uint64_t pkts_mask, uint64_t *lookup_hit_mask, void **e) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;

    uint32_t n_pkts_in=__builtin_popcountll(pkts_mask);
    RTE_TABLE_BV_STATS_PKTS_IN_ADD(t, n_pkts_in);

    for(uint32_t i=0; i<n_pkts_in; ++i)
        if((pkts_mask>>i)&1)
            t->pkts_data_h[i]=rte_pktmbuf_mtod(pkts[i], uint8_t *);

    bv_search<<<64, t->num_fields>>>(	t->ranges_db_dev+(t->num_fields*(*t->act_buf_h)), t->num_ranges,
                                        t->field_offsets, t->field_sizes,
                                        t->bvs_db_dev+(t->num_fields*(*t->act_buf_h)), RTE_TABLE_BV_BS,
                                        pkts_mask, t->pkts_data,
                                        (uint32_t *) e, lookup_hit_mask);
    cudaStreamSynchronize(0);

    RTE_TABLE_BV_STATS_PKTS_LOOKUP_MISS(t, n_pkts_in-__builtin_popcountll(*lookup_hit_mask));

    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess)
        printf("[bv_search] error: %s\n", cudaGetErrorString(err));
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
