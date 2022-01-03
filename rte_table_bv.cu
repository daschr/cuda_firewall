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
/*
#define NUM_BLOCKS 2
#define WORKERS_PER_PACKET 32
#define PACKETS_PER_BLOCK 32
*/
#define NUM_BLOCKS 64
#define WORKERS_PER_FIELD 32

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

    uint32_t num_blocks;
    uint32_t packets_per_block;

    uint32_t ptype_mask;
    uint32_t num_rules;
    uint32_t entry_size;
    uint8_t *entries;

    uint32_t **ranges_from;
    uint32_t **ranges_to;
    uint32_t **bvs;
    uint32_t **non_zero_bvs;

    size_t *num_ranges;
    uint32_t *field_offsets;
    uint8_t *field_sizes;

    uint32_t **ranges_from_dev;
    uint32_t **ranges_to_dev;
    uint32_t **bvs_dev;
    uint32_t **non_zero_bvs_dev;

    uint8_t **pkts_data;
    uint8_t **pkts_data_h;

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
    t->packets_per_block=32/p->num_fields;
    t->num_blocks=ceil(64.0f/((double) t->packets_per_block));
    printf("packets_per_block: %u num_blocks: %u\n", t->packets_per_block, t->num_blocks);

    t->field_defs=p->field_defs;
    t->num_rules=p->num_rules;
    t->entry_size=entry_size;

    t->ranges_from=(uint32_t **) rte_malloc("ranges_from", sizeof(uint32_t *)*t->num_fields, 0);
    t->ranges_to=(uint32_t **) rte_malloc("ranges_to", sizeof(uint32_t *)*t->num_fields, 0);
    t->bvs=(uint32_t **) rte_malloc("bvs", sizeof(uint32_t *)*t->num_fields, 0);
    t->non_zero_bvs=(uint32_t **) rte_malloc("non_zero_bvs", sizeof(uint32_t *)*t->num_fields, 0);

#define IS_ERROR(X) is_error(X, __FILE__, __LINE__)
#define CHECK(X) if(IS_ERROR(X)) return NULL

    CHECK(cudaHostAlloc((void **) &t->pkts_data_h, sizeof(uint8_t*)*RTE_TABLE_BV_MAX_PKTS, cudaHostAllocMapped|cudaHostAllocWriteCombined));
    CHECK(cudaHostGetDevicePointer((void **) &t->pkts_data, t->pkts_data_h, 0));

    CHECK(cudaHostAlloc((void **) &t->entries, t->entry_size*t->num_rules, cudaHostAllocMapped));

    CHECK(cudaHostAlloc((void **) &t->lookup_hit_vec_h, sizeof(uint8_t*)*RTE_TABLE_BV_MAX_PKTS, cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer((void **) &t->lookup_hit_vec, t->lookup_hit_vec_h, 0));

    CHECK(cudaMalloc((void **) &t->ranges_from_dev, sizeof(uint32_t *)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->ranges_to_dev, sizeof(uint32_t *)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->bvs_dev, sizeof(uint32_t *)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->non_zero_bvs_dev, sizeof(uint32_t *)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->field_offsets, sizeof(uint32_t)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->field_sizes, sizeof(uint32_t)*t->num_fields));
    CHECK(cudaMalloc((void **) &t->num_ranges, sizeof(uint64_t)*t->num_fields));

    t->ptype_mask=UINT32_MAX;

    for(size_t i=0; i<t->num_fields; ++i) {
        CHECK(cudaMemcpy(t->field_offsets+i, &t->field_defs[i].offset, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(t->field_sizes+i, &t->field_defs[i].size, sizeof(uint32_t), cudaMemcpyHostToDevice));
        t->ptype_mask&=t->field_defs[i].ptype_mask;

        CHECK(cudaMalloc((void **) &t->ranges_from[i], sizeof(uint32_t)*((size_t) RTE_TABLE_BV_MAX_RANGES)));
        CHECK(cudaMalloc((void **) &t->ranges_to[i], sizeof(uint32_t)*((size_t) RTE_TABLE_BV_MAX_RANGES)));
        printf("size: bvs[%ld] %ld bytes\n", i, sizeof(uint32_t)*((size_t) RTE_TABLE_BV_BS) * ((size_t ) RTE_TABLE_BV_MAX_RANGES));
        CHECK(cudaMalloc((void **) &t->bvs[i], sizeof(uint32_t)*((size_t) RTE_TABLE_BV_BS) * ((size_t ) RTE_TABLE_BV_MAX_RANGES)));
        printf("size: non_zero_bvs[%ld] %ld bytes\n", i, sizeof(uint32_t)*((size_t) RTE_TABLE_BV_BS>>5) * ((size_t ) RTE_TABLE_BV_MAX_RANGES));
        CHECK(cudaMalloc((void **) &t->non_zero_bvs[i], sizeof(uint32_t)*((size_t) RTE_TABLE_BV_BS>>5) * ((size_t ) RTE_TABLE_BV_MAX_RANGES)));
    }

    CHECK(cudaMemcpy(t->ranges_from_dev, t->ranges_from, sizeof(uint32_t *)*t->num_fields, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(t->ranges_to_dev, t->ranges_to, sizeof(uint32_t *)*t->num_fields, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(t->bvs_dev, t->bvs, sizeof(uint32_t *)*t->num_fields, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(t->non_zero_bvs_dev, t->non_zero_bvs, sizeof(uint32_t *)*t->num_fields, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

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
        if(rte_bv_markers_to_ranges(t->bv_markers+f, 1, sizeof(uint32_t), &ranges))
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
        if(rte_bv_markers_to_ranges(t->bv_markers+f, 1, sizeof(uint32_t), &ranges))
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
        if(rte_bv_markers_to_ranges(t->bv_markers+f, 1, sizeof(uint32_t), &ranges))
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
        if(rte_bv_markers_to_ranges(t->bv_markers+f, 1, sizeof(uint32_t), &ranges))
            return 1;
        cudaMemcpy(t->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    if(key_found)
        for(uint32_t k=0; k<n_keys; ++k)
            key_found[k]=0;

    return 0;
}

__global__ void bv_search(	uint32_t *__restrict__ *__restrict__ ranges_from,
                            uint32_t *__restrict__ *__restrict__ ranges_to,
                            const uint64_t *__restrict__ num_ranges,
                            const uint32_t *__restrict__ offsets, const uint8_t *__restrict__ sizes,
                            uint32_t *__restrict__ *__restrict__ bvs, uint32_t *__restrict__ *__restrict__ non_zero_bvs,
                            const uint32_t num_fields,
                            const uint32_t entry_size, uint8_t *__restrict__ entries,
                            const uint64_t pkts_mask, uint8_t *__restrict__ *__restrict__ pkts,
                            void *__restrict__ *matched_entries, uint8_t *__restrict__ lookup_hit_vec) {

#define field_id threadIdx.z

    int pkt_id=blockDim.y*blockIdx.x+threadIdx.y;
    __shared__ uint *__restrict__ bv[32][RTE_TABLE_BV_MAX_FIELDS];
    __shared__ uint *__restrict__ non_zero_bv[32][RTE_TABLE_BV_MAX_FIELDS];
    __shared__ uint32_t bv_not_found[32];

    if(!((pkts_mask>>pkt_id)&1LU))
        return;

    uint v;
    if(!threadIdx.x) {
        bv_not_found[threadIdx.y]=0;
        bv[threadIdx.y][field_id]=NULL;
		const uint8_t *pkt=(uint8_t * ) pkts[pkt_id]+offsets[field_id];
		switch(sizes[field_id]) {
        case 1:
            v=*pkt;
            break;
        case 2:
            v=pkt[1]|(pkt[0]<<8);
            break;
        case 4:
            v=pkt[3]|(pkt[2]<<8)|(pkt[1]<<16)|(pkt[0]<<24);
            break;
        default:
            __builtin_unreachable();
            printf("[%d|%d] unknown size: %u byte\n", blockIdx.x, threadIdx.y, sizes[field_id]);
            break;
        }
    }
    __syncwarp();
    v=__shfl_sync(UINT32_MAX, v, 0);
    long size=num_ranges[field_id]>>5;
    long start=0, offset;
    uint32_t l,r; //left, right

    while(size) {
        offset=start+((long) threadIdx.x)*size;
        l=__ballot_sync(UINT32_MAX, v>=ranges_from[field_id][offset]);
        r=__ballot_sync(UINT32_MAX, v<=ranges_to[field_id][offset]);
        if(l&r) {
            if((__ffs(l&r)-1)==threadIdx.x) {
//				printf("[%d|%d] %08X <= %08X <= %08X, %ld\n", pkt_id, field_id, ranges_from[field_id][offset], v, ranges_to[field_id][offset], offset);
                bv[threadIdx.y][field_id]=bvs[field_id]+offset*RTE_TABLE_BV_BS;
                non_zero_bv[threadIdx.y][field_id]=non_zero_bvs[field_id]+offset*RTE_TABLE_NON_ZERO_BV_BS;
            }
            goto found_bv;
        }
        if(!l)
            goto found_bv;

        //reuse r to save one register per thread
        r=__popc(l)-1;
        start=__shfl_sync(UINT32_MAX, offset+1, r);
        size=r==31?(num_ranges[field_id]-start)>>5:(size-1)>>5;

        __syncwarp();
    }
    offset=start+threadIdx.x;
    l=__ballot_sync(UINT32_MAX, offset<num_ranges[field_id]?v>=ranges_from[field_id][offset]:0);
    r=__ballot_sync(UINT32_MAX, offset<num_ranges[field_id]?v<=ranges_to[field_id][offset]:0);
    if(l&r) {
        if((__ffs(l&r)-1)==threadIdx.x) {
            //printf("[%d|%d] %08X <= %08X <= %08X, %ld\n", pkt_id, field_id, ranges_from[field_id][offset], v, ranges_to[field_id][offset], offset);
            bv[threadIdx.y][field_id]=bvs[field_id]+offset*RTE_TABLE_BV_BS;
            non_zero_bv[threadIdx.y][field_id]=non_zero_bvs[field_id]+offset*RTE_TABLE_NON_ZERO_BV_BS;
        }
    }

found_bv:

    if(!threadIdx.x && !bv[threadIdx.y][field_id]) {
        bv_not_found[threadIdx.y]=1;
    }

    __syncthreads();

    if(bv_not_found[threadIdx.y])
        return;

    if(threadIdx.z!=0)
        return;

#undef field_id

    __syncwarp();

    uint32_t in_loop;
    // all bitvectors found, now getting highest-priority rule
    uint x, y, tm;
    for(int nz_bv_b=threadIdx.x; nz_bv_b<RTE_TABLE_NON_ZERO_BV_BS; nz_bv_b+=blockDim.x) { // TODO maybe use WORKERS_PER_PACKET
        in_loop=__activemask();
        x=UINT32_MAX;

        for(int field_id=0; field_id<num_fields; ++field_id)
            x&=non_zero_bv[threadIdx.y][field_id][nz_bv_b];

        int pos;
        while((pos=__ffs(x))) {
            y=UINT32_MAX;
            for(int field_id=0; field_id<num_fields; ++field_id)
                y&=bv[threadIdx.y][field_id][(nz_bv_b<<5)|(pos-1)];
            if(y)
                break;
            x=(x>>pos)<<pos;
        }

        __syncwarp(in_loop); //TODO maybe remove
        if((tm=__ballot_sync(in_loop, __ffs(y)))) {
            if((__ffs(tm)-1)==threadIdx.x) {
                matched_entries[pkt_id]=(void *) &entries[entry_size*((nz_bv_b<<5)+__ffs(y)-1LU)];
                lookup_hit_vec[pkt_id]=1;
                //atomicOr((unsigned long long int *)lookup_hit_mask, 1LU<<pkt_id);
                //__threadfence_system();
            }
            break;
        }
    }

}


#define IS_ERROR(X) is_error(X, __FILE__, __LINE__)
int rte_table_bv_lookup_stream(void *t_r, cudaStream_t stream, struct rte_mbuf **pkts, uint64_t pkts_mask,
                               uint64_t *lookup_hit_mask, void **e) {
#ifdef MEASURE_TIME
    struct timeval k_t1,k_t2,l_t1,l_t2;
    gettimeofday(&l_t1, NULL);
#endif

    struct rte_table_bv *t=(struct rte_table_bv *) t_r;

    const uint32_t n_pkts_in=__builtin_popcountll(pkts_mask);
    RTE_TABLE_BV_STATS_PKTS_IN_ADD(t, n_pkts_in);

    uint64_t real_pkts_mask=0;
    for(uint32_t i=0; i<n_pkts_in; ++i) {
        const uint32_t mp=pkts[i]->packet_type&t->ptype_mask;
        if((pkts_mask>>i)&1&((mp&RTE_PTYPE_L2_MASK)!=0)&((mp&RTE_PTYPE_L3_MASK)!=0)&((mp&RTE_PTYPE_L4_MASK)!=0)) {
            t->pkts_data_h[i]=rte_pktmbuf_mtod(pkts[i], uint8_t *);
            real_pkts_mask|=1LU<<i;
        }
    }

#ifdef MEASURE_TIME
    gettimeofday(&k_t1, NULL);
#endif

    bv_search<<<t->num_blocks, dim3{WORKERS_PER_FIELD, t->packets_per_block, t->num_fields}, 0, stream>>>(	t->ranges_from_dev, t->ranges_to_dev, t->num_ranges,
            t->field_offsets, t->field_sizes,
            t->bvs_dev, t->non_zero_bvs_dev, t->num_fields, t->entry_size, t->entries,
            real_pkts_mask, t->pkts_data,
            e, t->lookup_hit_vec);

    cudaStreamSynchronize(stream);

#ifdef MEASURE_TIME
    gettimeofday(&k_t2, NULL);
    printf("KERNEL took %luus\n", (k_t2.tv_sec*1000000+k_t2.tv_usec)-(k_t1.tv_sec*1000000+k_t1.tv_usec));
#endif
    uint64_t lhm=0;
    for(uint32_t i=0; i<n_pkts_in; ++i) {
        if(t->lookup_hit_vec_h[i]) {
            lhm|=1LU<<i;
            t->lookup_hit_vec_h[i]=0;
        }
    }
    *lookup_hit_mask=lhm;

    RTE_TABLE_BV_STATS_PKTS_LOOKUP_MISS(t, n_pkts_in-__builtin_popcountll(*lookup_hit_mask));

#ifdef MEASURE_TIME
    gettimeofday(&k_t2, NULL);
    gettimeofday(&l_t2, NULL);
    printf("LOOKUP took %luus\n", (l_t2.tv_sec*1000000+l_t2.tv_usec)-(l_t1.tv_sec*1000000+l_t1.tv_usec));
#endif
    return 0;
}

static int rte_table_bv_lookup(void *t_r, struct rte_mbuf **pkts, uint64_t pkts_mask, uint64_t *lookup_hit_mask, void **e) {
    struct rte_table_bv *t=(struct rte_table_bv *) t_r;

    const uint32_t n_pkts_in=__builtin_popcountll(pkts_mask);
    RTE_TABLE_BV_STATS_PKTS_IN_ADD(t, n_pkts_in);

    uint64_t real_pkts_mask=0;
    for(uint32_t i=0; i<n_pkts_in; ++i) {
        const uint32_t mp=pkts[i]->packet_type&t->ptype_mask;
        if((pkts_mask>>i)&1&((mp&RTE_PTYPE_L2_MASK)!=0)&((mp&RTE_PTYPE_L3_MASK)!=0)&((mp&RTE_PTYPE_L4_MASK)!=0)) {
            t->pkts_data_h[i]=rte_pktmbuf_mtod(pkts[i], uint8_t *);
            real_pkts_mask|=1LU<<i;
        }
    }

    bv_search<<<NUM_BLOCKS, dim3{WORKERS_PER_FIELD, t->num_fields}>>>(	t->ranges_from_dev, t->ranges_to_dev, t->num_ranges,
            t->field_offsets, t->field_sizes,
            t->bvs_dev, t->non_zero_bvs_dev, t->num_fields, t->entry_size, t->entries,
            real_pkts_mask, t->pkts_data,
            e, t->lookup_hit_vec);

    cudaStreamSynchronize(0);

    uint64_t lhm=0;
    for(uint32_t i=0; i<n_pkts_in; ++i) {
        if(t->lookup_hit_vec_h[i]) {
            lhm|=1LU<<i;
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
