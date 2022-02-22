#include "rte_bv_classifier.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "rte_bv.h"
#include <rte_log.h>
#include <rte_malloc.h>
#include <stdlib.h>

#define WORKERS_PER_FIELD 32

#ifdef RTE_TABLE_STATS_COLLECT
#define RTE_BV_CLASSIFIER_STATS_PKTS_IN_ADD(table, val) table->stats.n_pkts_in += val
#define RTE_BV_CLASSIFIER_STATS_PKTS_LOOKUP_MISS(table, val) table->stats.n_pkts_lookup_miss += val
#else
#define RTE_BV_CLASSIFIER_STATS_PKTS_IN_ADD(table, val)
#define RTE_BV_CLASSIFIER_STATS_PKTS_LOOKUP_MISS(table, val)
#endif

static inline int is_error(cudaError_t e, const char *file, int line) {
    if(e!=cudaSuccess) {
        rte_log(RTE_LOG_ERR, RTE_LOGTYPE_TABLE, "[rte_bv_classifier] error: %s in %s (line %d)\n", cudaGetErrorString(e), file, line);
        return 1;
    }
    return 0;
}

int rte_bv_classifier_free(struct rte_bv_classifier *c) {
    if(c==NULL)
        return 0;

    for(size_t i=0; i<c->num_fields; ++i) {
        cudaFree(c->ranges_from[i]);
        cudaFree(c->ranges_to[i]);
        cudaFree(c->bvs[i]);
        cudaFree(c->non_zero_bvs[i]);
    }
    cudaFree(c->ranges_from_dev);
    cudaFree(c->ranges_to_dev);
    cudaFree(c->bvs_dev);
    cudaFree(c->non_zero_bvs_dev);

    cudaFree(c->num_ranges);
    cudaFree(c->field_offsets);
    cudaFree(c->field_sizes);

    cudaFreeHost(c->entries);

    for(size_t i=0; i<RTE_BV_CLASSIFIER_NUM_STREAMS; ++i) {
        cudaFreeHost(c->matched_entries_h[i]);
        cudaFreeHost(c->pkts_data_h[i]);
        cudaStreamDestroy(c->streams[i]);
        cudaFreeHost(c->lookup_hit_vec_h[i]);
        rte_free(c->pkts[i]);
    }

    for(uint32_t i=0; i<c->num_fields; ++i)
        rte_bv_markers_free(c->bv_markers+i);

    rte_free(c->bv_markers);
    rte_free(c->ranges_from);
    rte_free(c->ranges_to);
    rte_free(c->bvs);
    rte_free(c->non_zero_bvs);

    rte_free(c);

    return 0;
}

#define IS_ERROR(X) is_error(X, __FILE__, __LINE__)

struct rte_bv_classifier *rte_bv_classifier_create(struct rte_bv_classifier_params *p, int socket_id, uint32_t entry_size) {
    struct rte_bv_classifier *c=(struct rte_bv_classifier *) rte_malloc("rte_bv_classifier", sizeof(struct rte_bv_classifier), 0);
    memset(c, 0, sizeof(struct rte_bv_classifier));

    c->num_fields=p->num_fields;
    c->packets_per_block=32/p->num_fields;
    c->bvs_size=sizeof(uint64_t *)*c->packets_per_block*RTE_BV_CLASSIFIER_MAX_FIELDS*2;

    c->field_defs=p->field_defs;
    c->num_rules=p->num_rules;
    c->entry_size=entry_size;

    c->enqueue_pos=0;
    c->dequeue_pos=0;

    c->ranges_from=(uint32_t **) rte_malloc("ranges_from", sizeof(uint32_t *)*c->num_fields, 0);
    c->ranges_to=(uint32_t **) rte_malloc("ranges_to", sizeof(uint32_t *)*c->num_fields, 0);
    c->bvs=(uint64_t **) rte_malloc("bvs", sizeof(uint64_t *)*c->num_fields, 0);
    c->non_zero_bvs=(uint64_t **) rte_malloc("non_zero_bvs", sizeof(uint64_t *)*c->num_fields, 0);

#define CHECK(X) if(IS_ERROR(X)) return NULL

    for(size_t i=0; i<RTE_BV_CLASSIFIER_NUM_STREAMS; ++i) {
        CHECK(cudaHostAlloc((void **) &c->pkts_data_h[i], sizeof(uint8_t*)*RTE_BV_CLASSIFIER_MAX_PKTS, cudaHostAllocMapped));
        CHECK(cudaHostGetDevicePointer((void **) &c->pkts_data[i], c->pkts_data_h[i], 0));
        CHECK(cudaHostAlloc((void **) &c->matched_entries_h[i], sizeof(void *)*RTE_BV_CLASSIFIER_MAX_PKTS, cudaHostAllocMapped));
        CHECK(cudaHostGetDevicePointer((void **) &c->matched_entries[i], c->matched_entries_h[i], 0));

        CHECK(cudaHostAlloc((void **) &c->lookup_hit_vec_h[i], sizeof(uint8_t)*RTE_BV_CLASSIFIER_MAX_PKTS, cudaHostAllocMapped));
        CHECK(cudaHostGetDevicePointer((void **) &c->lookup_hit_vec[i], c->lookup_hit_vec_h[i], 0));

        CHECK(cudaStreamCreateWithFlags(&c->streams[i], cudaStreamNonBlocking));
        c->pkts[i]=(struct rte_mbuf **) rte_malloc("pkts", sizeof(struct rte_mbuf *)*RTE_BV_CLASSIFIER_MAX_PKTS, 0);
    }

    CHECK(cudaHostAlloc((void **) &c->entries, c->entry_size*c->num_rules, cudaHostAllocMapped));

    CHECK(cudaMalloc((void **) &c->ranges_from_dev, sizeof(uint32_t *)*c->num_fields));
    CHECK(cudaMalloc((void **) &c->ranges_to_dev, sizeof(uint32_t *)*c->num_fields));
    CHECK(cudaMalloc((void **) &c->bvs_dev, sizeof(uint64_t *)*c->num_fields));
    CHECK(cudaMalloc((void **) &c->non_zero_bvs_dev, sizeof(uint64_t *)*c->num_fields));
    CHECK(cudaMalloc((void **) &c->field_offsets, sizeof(uint32_t)*c->num_fields));
    CHECK(cudaMalloc((void **) &c->field_sizes, sizeof(uint32_t)*c->num_fields));
    CHECK(cudaMalloc((void **) &c->num_ranges, sizeof(uint64_t)*c->num_fields));


    for(size_t i=0; i<c->num_fields; ++i) {
        CHECK(cudaMemcpy(c->field_offsets+i, &c->field_defs[i].offset, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(c->field_sizes+i, &c->field_defs[i].size, sizeof(uint32_t), cudaMemcpyHostToDevice));

#define RANGE_SIZE(DIV) ((sizeof(uint32_t)*((size_t) RTE_BV_CLASSIFIER_MAX_RANGES))/DIV+1LU)
        switch(p->field_defs[i].size) {
        case 4:
            CHECK(cudaMalloc((void **) &c->ranges_from[i], RANGE_SIZE(1LU)));
            CHECK(cudaMalloc((void **) &c->ranges_to[i], RANGE_SIZE(1LU)));
            printf("allocated %lu bytes for dimension %lu\n", RANGE_SIZE(1LU), i);
            break;
        case 2:
            CHECK(cudaMalloc((void **) &c->ranges_from[i], RANGE_SIZE(2LU)));
            CHECK(cudaMalloc((void **) &c->ranges_to[i], RANGE_SIZE(2LU)));
            printf("allocated %lu bytes for dimension %lu\n", RANGE_SIZE(2LU), i);
            break;
        case 1:
            CHECK(cudaMalloc((void **) &c->ranges_from[i], RANGE_SIZE(4LU)));
            CHECK(cudaMalloc((void **) &c->ranges_to[i], RANGE_SIZE(4LU)));
            printf("allocated %lu bytes for dimension %lu\n", RANGE_SIZE(4LU), i);
            break;
        default:
            printf("unkown field_def[%lu] size: %hhu\n", i, p->field_defs[i].size);
        }
#undef RANGE_SIZE

        printf("size: bvs[%lu] %lu bytes\n", i, sizeof(uint64_t)*((size_t) RTE_BV_CLASSIFIER_BV_BS) * ((size_t ) RTE_BV_CLASSIFIER_MAX_RANGES));
        CHECK(cudaMalloc((void **) &c->bvs[i], sizeof(uint64_t)*((size_t) RTE_BV_CLASSIFIER_BV_BS) * ((size_t ) RTE_BV_CLASSIFIER_MAX_RANGES)));
        printf("size: non_zero_bvs[%lu] %lu bytes\n", i, sizeof(uint64_t)*((size_t) RTE_BV_CLASSIFIER_BV_BS>>5) * ((size_t ) RTE_BV_CLASSIFIER_MAX_RANGES));
        CHECK(cudaMalloc((void **) &c->non_zero_bvs[i], sizeof(uint64_t)*((size_t) RTE_BV_CLASSIFIER_BV_BS>>5) * ((size_t ) RTE_BV_CLASSIFIER_MAX_RANGES)));
    }

    CHECK(cudaMemcpy(c->ranges_from_dev, c->ranges_from, sizeof(uint32_t *)*c->num_fields, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(c->ranges_to_dev, c->ranges_to, sizeof(uint32_t *)*c->num_fields, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(c->bvs_dev, c->bvs, sizeof(uint64_t *)*c->num_fields, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(c->non_zero_bvs_dev, c->non_zero_bvs, sizeof(uint64_t *)*c->num_fields, cudaMemcpyHostToDevice));
#undef CHECK

    c->bv_markers=(rte_bv_markers_t *) rte_malloc("bv_markers", sizeof(rte_bv_markers_t)*c->num_fields, 0);

    for(size_t i=0; i<c->num_fields; ++i) {
        if(rte_bv_markers_create(&c->bv_markers[i])) {
            rte_bv_classifier_free(c);
            rte_log(RTE_LOG_ERR, RTE_LOGTYPE_TABLE, "Error creating marker!\n");
            return NULL;
        }
    }

    return c;
}
//#undef IS_ERROR

inline void cal_from_to(uint32_t *from_to, uint32_t *v, uint8_t type, uint8_t size) {
    if(type==RTE_BV_CLASSIFIER_FIELD_TYPE_RANGE) {
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

int rte_bv_classifier_entry_add(struct rte_bv_classifier *c, struct rte_bv_classifier_key *k, void *e_r, int *key_found, void **e_ptr) {
    if(key_found)
        *key_found=0;

    uint32_t from_to[2];
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<c->num_fields; ++f) {
        cal_from_to(from_to, k->buf +(f<<1), c->field_defs[f].type, c->field_defs[f].size);
        rte_bv_markers_range_add(c->bv_markers+f, from_to, k->pos);

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_BV_CLASSIFIER_BV_BS;
        ranges.max_num_ranges=RTE_BV_CLASSIFIER_MAX_RANGES;
        ranges.ranges_from=c->ranges_from[f];
        ranges.ranges_to=c->ranges_to[f];
        ranges.bvs=c->bvs[f];
        ranges.non_zero_bvs=c->non_zero_bvs[f];
        if(rte_bv_markers_to_ranges(c->bv_markers+f, 1, c->field_defs[f].size, &ranges))
            return 1;
        cudaMemcpy(c->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(&c->entries[c->entry_size*k->pos], e_r, c->entry_size, cudaMemcpyHostToDevice);

    if(e_ptr)
        *e_ptr=&c->entries[c->entry_size*k->pos];

    return 0;
}

int rte_bv_classifier_entry_delete(struct rte_bv_classifier *c, struct rte_bv_classifier_key *k, int *key_found, __rte_unused void *e) {

    if(key_found)
        *key_found=0;

    uint32_t from_to[2];
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<c->num_fields; ++f) {
        cal_from_to(from_to, k->buf+(f<<1), c->field_defs[f].type, c->field_defs[f].size);
        rte_bv_markers_range_del(c->bv_markers+f, from_to, k->pos);

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_BV_CLASSIFIER_BV_BS;
        ranges.max_num_ranges=RTE_BV_CLASSIFIER_MAX_RANGES;
        ranges.ranges_from=c->ranges_from[f];
        ranges.ranges_to=c->ranges_to[f];
        ranges.bvs=c->bvs[f];
        ranges.non_zero_bvs=c->non_zero_bvs[f];
        if(rte_bv_markers_to_ranges(c->bv_markers+f, 1, c->field_defs[f].size, &ranges))
            return 1;
        cudaMemcpy(c->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    return 0;
}

int rte_bv_classifier_entry_add_bulk(struct rte_bv_classifier *c, struct rte_bv_classifier_key **ks, void **es_r, uint32_t n_keys, int *key_found, __rte_unused void **e_ptr) {

    uint32_t from_to[2];
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<c->num_fields; ++f) {
        for(uint32_t k=0; k<n_keys; ++k) {
            cal_from_to(from_to, ks[k]->buf+(f<<1), c->field_defs[f].type, c->field_defs[f].size);
            rte_bv_markers_range_add(c->bv_markers+f, from_to, ks[k]->pos);
        }

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_BV_CLASSIFIER_BV_BS;
        ranges.max_num_ranges=RTE_BV_CLASSIFIER_MAX_RANGES;
        ranges.ranges_from=c->ranges_from[f];
        ranges.ranges_to=c->ranges_to[f];
        ranges.bvs=c->bvs[f];
        ranges.non_zero_bvs=c->non_zero_bvs[f];
        if(rte_bv_markers_to_ranges(c->bv_markers+f, 1, c->field_defs[f].size, &ranges))
            return 1;
        cudaMemcpy(c->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }


    for(uint32_t k=0; k<n_keys; ++k) {
        if(key_found)
            key_found[k]=0;

        cudaMemcpy(&c->entries[c->entry_size*ks[k]->pos], es_r[ks[k]->pos], c->entry_size, cudaMemcpyHostToDevice);

        if(e_ptr)
            e_ptr[k]=&c->entries[c->entry_size*ks[k]->pos];

    }

    return 0;
}

int rte_bv_classifier_entry_delete_bulk(struct rte_bv_classifier *c, struct rte_bv_classifier_key **ks, uint32_t n_keys, int *key_found, __rte_unused void **es_r) {
    uint32_t from_to[2];
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<c->num_fields; ++f) {
        for(uint32_t k=0; k<n_keys; ++k) {
            cal_from_to(from_to, ks[k]->buf+(f<<1), c->field_defs[f].type, c->field_defs[f].size);
            rte_bv_markers_range_del(c->bv_markers+f, from_to, ks[k]->pos);
        }

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_BV_CLASSIFIER_BV_BS;
        ranges.max_num_ranges=RTE_BV_CLASSIFIER_MAX_RANGES;
        ranges.ranges_from=c->ranges_from[f];
        ranges.ranges_to=c->ranges_to[f];
        ranges.bvs=c->bvs[f];
        ranges.non_zero_bvs=c->non_zero_bvs[f];
        if(rte_bv_markers_to_ranges(c->bv_markers+f, 1, c->field_defs[f].size, &ranges))
            return 1;
        cudaMemcpy(c->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint64_t), cudaMemcpyHostToDevice);
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


__device__ __constant__ uint8_t compression_level[5]= {0,2,1,0,0};


__global__ void bv_search(	uint32_t *__restrict__ *__restrict__ ranges_from,
                            uint32_t *__restrict__ *__restrict__ ranges_to,
                            uint64_t *__restrict__ num_ranges,
                            uint32_t *__restrict__ offsets, uint8_t *__restrict__ sizes,
                            uint64_t *__restrict__ *__restrict__ bvs, uint64_t *__restrict__ *__restrict__ non_zero_bvs,
                            const uint32_t num_fields,
                            const uint32_t entry_size, uint8_t *__restrict__ entries,
                            const uint16_t num_pkts, uint8_t *__restrict__ *__restrict__ pkts,
                            void *__restrict__ *matched_entries, uint8_t *__restrict__ lookup_hit_vec) {

    extern __shared__ uint8_t mem[];

#define field_id threadIdx.z

    const int pkt_id=blockDim.y*blockIdx.x+threadIdx.y;
    if(pkt_id>=num_pkts)
        return;

    uint64_t *(*bv)[RTE_BV_CLASSIFIER_MAX_FIELDS]=(uint64_t *(*)[RTE_BV_CLASSIFIER_MAX_FIELDS]) mem;
    uint64_t *(*non_zero_bv)[RTE_BV_CLASSIFIER_MAX_FIELDS]=(uint64_t *(*)[RTE_BV_CLASSIFIER_MAX_FIELDS]) (mem+(sizeof(uint64_t *)*blockDim.y*RTE_BV_CLASSIFIER_MAX_FIELDS));
    const uint8_t field_size=sizes[field_id];
    const uint8_t comp_level=compression_level[field_size];

    uint v;
    if(!threadIdx.x) {
        bv[threadIdx.y][field_id]=NULL;
        uint8_t *pkt=(uint8_t * ) pkts[pkt_id]+offsets[field_id];
        switch(field_size) {
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
            __builtin_unreachable();
        }
    }

    __syncwarp();
    v=__shfl_sync(UINT32_MAX, v, 0);
    long size=num_ranges[field_id]>>(5+comp_level);
    long start=0, offset;
    uint32_t l,r; //left, right
    uint32_t lres, rres;
    while(size) {
        offset=start+((long) threadIdx.x)*size;
        lres=leu(ranges_from[field_id][offset],v,field_size);
        rres=leu(v,ranges_to[field_id][offset],field_size);
        l=__ballot_sync(UINT32_MAX, lres);
        r=__ballot_sync(UINT32_MAX, rres);
        if(l&r) {
            if((__ffs(l&r)-1)==threadIdx.x) {
                if(!(lres&rres))
                    goto found_bv;

                const long pos=(offset<<comp_level)|leu_offset(lres&rres, field_size);
                bv[threadIdx.y][field_id]=bvs[field_id]+pos*RTE_BV_CLASSIFIER_BV_BS;
                non_zero_bv[threadIdx.y][field_id]=non_zero_bvs[field_id]+pos*RTE_BV_CLASSIFIER_NON_ZERO_BV_BS;
            }
            __syncwarp();
            goto found_bv;
        }
        if(!l)
            goto found_bv;

        r=__popc(l)-1;
        start=__shfl_sync(UINT32_MAX, offset+1, r);
        size=r==31?((num_ranges[field_id]>>comp_level)-start)>>5:(size-1LU)>>5;

        __syncwarp();
    }
    offset=start+threadIdx.x;

    lres=offset<num_ranges[field_id]?leu(ranges_from[field_id][offset],v,field_size):0;
    rres=offset<num_ranges[field_id]?leu(v,ranges_to[field_id][offset],field_size):0;
    if(lres&rres) {
        const long pos=(offset<<comp_level)|leu_offset(lres&rres, field_size);
        bv[threadIdx.y][field_id]=bvs[field_id]+pos*RTE_BV_CLASSIFIER_BV_BS;
        non_zero_bv[threadIdx.y][field_id]=non_zero_bvs[field_id]+pos*RTE_BV_CLASSIFIER_NON_ZERO_BV_BS;
    }

    __syncwarp();

found_bv:

    __syncthreads();
    if(field_id)
        return;

#undef field_id

    if(__ballot_sync(UINT32_MAX, threadIdx.x<num_fields&&!bv[threadIdx.y][threadIdx.x])) {
        if(!threadIdx.x)
            lookup_hit_vec[pkt_id]=0;

        return;
    }

    // all bitvectors found, now getting highest-priority rule

    int nz_bv_b=threadIdx.x;
    uint32_t in_loop=__ballot_sync(UINT32_MAX, nz_bv_b<RTE_BV_CLASSIFIER_NON_ZERO_BV_BS);

    while(nz_bv_b<RTE_BV_CLASSIFIER_NON_ZERO_BV_BS) {
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

// __syncwarp(in_loop); //TODO maybe remove
        const uint32_t tm=__ballot_sync(in_loop, __ffsll(y));
        if(tm) {
            if((__ffs(tm)-1)==threadIdx.x) {
                matched_entries[pkt_id]=(void *) &entries[entry_size*((nz_bv_b<<6)+__ffsll(y)-1LU)];
                lookup_hit_vec[pkt_id]=1;
            }
            return;
        }

        nz_bv_b+=blockDim.x;
        in_loop=__ballot_sync(in_loop, nz_bv_b<RTE_BV_CLASSIFIER_NON_ZERO_BV_BS);
    }

    if(!threadIdx.x)
        lookup_hit_vec[pkt_id]=0;
}

void rte_bv_classifier_enqueue_burst(struct rte_bv_classifier *c, struct rte_mbuf **pkts, uint16_t n_pkts_in) {
    const size_t epos=c->enqueue_pos;
    while(__atomic_load_n(&c->stream_running[epos], __ATOMIC_RELAXED));

    c->n_pkts_in[epos]=n_pkts_in;
    for(uint16_t i=0; i<n_pkts_in; ++i) {
        c->pkts[epos][i]=pkts[i];
        c->pkts_data_h[epos][i]=rte_pktmbuf_mtod(pkts[i], uint8_t *);
    }

    bv_search<<<(n_pkts_in/c->packets_per_block)+1, dim3{WORKERS_PER_FIELD, c->packets_per_block, c->num_fields}, c->bvs_size, c->streams[epos]>>>(
        c->ranges_from_dev, c->ranges_to_dev, c->num_ranges,
        c->field_offsets, c->field_sizes,
        c->bvs_dev, c->non_zero_bvs_dev, c->num_fields, c->entry_size, c->entries,
        n_pkts_in, c->pkts_data[epos],
        c->matched_entries[epos], c->lookup_hit_vec[epos]);

    c->enqueue_pos=(epos+1)&RTE_BV_CLASSIFIER_NUM_STREAMS_MASK;

    __atomic_store_n(&c->stream_running[epos], 1, __ATOMIC_RELAXED);
}

void __rte_noreturn rte_bv_classifier_poll_lookups(struct rte_bv_classifier *c, void (*callback) (struct rte_mbuf **, uint16_t,  uint8_t *, void **, void *), void *p) {
    size_t dpos=0;

    for(;;) {
        while(!__atomic_load_n(&c->stream_running[dpos], __ATOMIC_RELAXED));

        IS_ERROR(cudaStreamSynchronize(c->streams[dpos]));

        callback(c->pkts[dpos], *((volatile uint16_t *) &c->n_pkts_in[dpos]), c->lookup_hit_vec_h[dpos], c->matched_entries[dpos], p);

        __atomic_store_n(&c->stream_running[dpos], 0, __ATOMIC_RELAXED);

        dpos=(dpos+1)&RTE_BV_CLASSIFIER_NUM_STREAMS_MASK;
    }
}

int rte_bv_classifier_stats_read(struct rte_bv_classifier *c, struct rte_table_stats *stats, int clear) {
    if (stats != NULL)
        memcpy(stats, &c->stats, sizeof(c->stats));

    if (clear)
        memset(&c->stats, 0, sizeof(c->stats));

    return 0;
}


#ifdef __cplusplus
}
#endif
