#include "rte_bv_classifier.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "rte_bv.h"
#include <rte_log.h>
#include <rte_malloc.h>
#include <stdlib.h>

#define NUM_BLOCKS 2
#define WORKERS_PER_PACKET 32
#define PACKETS_PER_BLOCK 32

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
    }
    cudaFree(c->ranges_from_dev);
    cudaFree(c->ranges_to_dev);
    cudaFree(c->bvs_dev);

    cudaFree(c->num_ranges);
    cudaFree(c->field_offsets);
    cudaFree(c->field_sizes);

    cudaFreeHost(c->entries);
    cudaFreeHost(c->lookup_hit_mask);

    for(size_t i=0; i<RTE_BV_CLASSIFIER_NUM_STREAMS; ++i) {
        cudaFreeHost(c->matched_entries_h[i]);
        cudaFreeHost(c->pkts_data_h[i]);
    }

    for(uint32_t i=0; i<c->num_fields; ++i)
        rte_bv_markers_free(c->bv_markers+i);

    rte_free(c->bv_markers);
    rte_free(c->ranges_from);
    rte_free(c->ranges_to);
    rte_free(c->bvs);

    rte_free(c);

    return 0;
}

#define IS_ERROR(X) is_error(X, __FILE__, __LINE__)

struct rte_bv_classifier *rte_bv_classifier_create(struct rte_bv_classifier_params *p, int socket_id, uint32_t entry_size) {
    struct rte_bv_classifier *c=(struct rte_bv_classifier *) rte_malloc("rte_bv_classifier", sizeof(struct rte_bv_classifier), 0);
    memset(c, 0, sizeof(struct rte_bv_classifier));

    c->num_fields=p->num_fields;
    c->field_defs=p->field_defs;
    c->num_rules=p->num_rules;
    c->entry_size=entry_size;

    c->ranges_from=(uint32_t **) rte_malloc("ranges_from", sizeof(uint32_t *)*c->num_fields, 0);
    c->ranges_to=(uint32_t **) rte_malloc("ranges_to", sizeof(uint32_t *)*c->num_fields, 0);
    c->bvs=(uint32_t **) rte_malloc("bvs_db", sizeof(uint32_t *)*c->num_fields, 0);

#define CHECK(X) if(IS_ERROR(X)) return NULL

    for(size_t i=0; i<RTE_BV_CLASSIFIER_NUM_STREAMS; ++i) {
        CHECK(cudaHostAlloc((void **) &c->pkts_data_h[i], sizeof(uint8_t*)*RTE_BV_CLASSIFIER_MAX_PKTS, cudaHostAllocMapped|cudaHostAllocWriteCombined));
        CHECK(cudaHostGetDevicePointer((void **) &c->pkts_data[i], c->pkts_data_h[i], 0));
        CHECK(cudaHostAlloc((void **) &c->matched_entries_h[i], sizeof(void *)*RTE_BV_CLASSIFIER_MAX_PKTS, cudaHostAllocMapped));
        CHECK(cudaHostGetDevicePointer((void **) &c->matched_entries[i], c->matched_entries_h[i], 0));
    }

    CHECK(cudaHostAlloc((void **) &c->entries, c->entry_size*c->num_rules, cudaHostAllocMapped));

    CHECK(cudaMalloc((void **) &c->ranges_from_dev, sizeof(uint32_t *)*c->num_fields));
    CHECK(cudaMalloc((void **) &c->ranges_to_dev, sizeof(uint32_t *)*c->num_fields));
    CHECK(cudaMalloc((void **) &c->bvs_dev, sizeof(uint32_t *)*c->num_fields));
    CHECK(cudaMalloc((void **) &c->field_offsets, sizeof(uint32_t)*c->num_fields));
    CHECK(cudaMalloc((void **) &c->field_sizes, sizeof(uint32_t)*c->num_fields));
    CHECK(cudaMalloc((void **) &c->num_ranges, sizeof(uint64_t)*c->num_fields));

    CHECK(cudaHostAlloc((void **) &c->lookup_hit_mask_h, sizeof(uint64_t)*RTE_BV_CLASSIFIER_NUM_STREAMS, cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer((void **) &c->lookup_hit_mask, c->lookup_hit_mask_h, 0));

    c->ptype_mask=UINT32_MAX;

    for(size_t i=0; i<c->num_fields; ++i) {
        CHECK(cudaMemcpy(c->field_offsets+i, &c->field_defs[i].offset, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(c->field_sizes+i, &c->field_defs[i].size, sizeof(uint32_t), cudaMemcpyHostToDevice));
        c->ptype_mask&=c->field_defs[i].ptype_mask;

        CHECK(cudaMalloc((void **) &c->ranges_from[i], sizeof(uint32_t)*((size_t) RTE_BV_CLASSIFIER_MAX_RANGES)));
        CHECK(cudaMalloc((void **) &c->ranges_to[i], sizeof(uint32_t)*((size_t) RTE_BV_CLASSIFIER_MAX_RANGES)));
        CHECK(cudaMalloc((void **) &c->bvs[i], sizeof(uint32_t)*((size_t) RTE_BV_CLASSIFIER_BS) * ((size_t ) RTE_BV_CLASSIFIER_MAX_RANGES)));


    }



    CHECK(cudaMemcpy(c->ranges_from_dev, c->ranges_from, sizeof(uint32_t *)*c->num_fields, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(c->ranges_to_dev, c->ranges_to, sizeof(uint32_t *)*c->num_fields, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(c->bvs_dev, c->bvs, sizeof(uint32_t *)*c->num_fields, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
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
#undef IS_ERROR

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
        ranges.bv_bs=RTE_BV_CLASSIFIER_BS;
        ranges.max_num_ranges=RTE_BV_CLASSIFIER_MAX_RANGES;
        ranges.ranges_from=c->ranges_from[f];
        ranges.ranges_to=c->ranges_to[f];
        ranges.bvs=c->bvs[c->num_fields+f];
        if(rte_bv_markers_to_ranges(c->bv_markers+f, 1, sizeof(uint32_t), &ranges))
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
        ranges.bv_bs=RTE_BV_CLASSIFIER_BS;
        ranges.max_num_ranges=RTE_BV_CLASSIFIER_MAX_RANGES;
        ranges.ranges_from=c->ranges_from[f];
        ranges.ranges_to=c->ranges_to[f];
        ranges.bvs=c->bvs[f];
        if(rte_bv_markers_to_ranges(c->bv_markers+f, 1, sizeof(uint32_t), &ranges))
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
        ranges.bv_bs=RTE_BV_CLASSIFIER_BS;
        ranges.max_num_ranges=RTE_BV_CLASSIFIER_MAX_RANGES;
        ranges.ranges_from=c->ranges_from[f];
        ranges.ranges_to=c->ranges_to[f];
        ranges.bvs=c->bvs[f];
        if(rte_bv_markers_to_ranges(c->bv_markers+f, 1, sizeof(uint32_t), &ranges))
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
        ranges.bv_bs=RTE_BV_CLASSIFIER_BS;
        ranges.max_num_ranges=RTE_BV_CLASSIFIER_MAX_RANGES;
        ranges.ranges_from=c->ranges_from[f];
        ranges.ranges_to=c->ranges_to[f];
        ranges.bvs=c->bvs[f];
        if(rte_bv_markers_to_ranges(c->bv_markers+f, 1, sizeof(uint32_t), &ranges))
            return 1;
        cudaMemcpy(c->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    if(key_found)
        for(uint32_t k=0; k<n_keys; ++k)
            key_found[k]=0;

    return 0;
}

__global__ void bv_search(	uint32_t *__restrict__ *__restrict__ ranges_from, uint32_t *__restrict__ *__restrict__ ranges_to,
                            const uint64_t *__restrict__ num_ranges, const uint32_t *__restrict__ offsets,  uint8_t *__restrict__ sizes,
                            uint32_t *__restrict__ *__restrict__ bvs, const uint32_t bv_bs,
                            const uint32_t num_fields, const uint32_t entry_size, const uint8_t *__restrict__ entries,
                            const ulong pkts_mask, uint8_t *__restrict__ *__restrict__ pkts,
                            void *__restrict__ *matched_entries, ulong *__restrict__ lookup_hit_mask) {

    __shared__ uint *bv[RTE_BV_CLASSIFIER_MAX_PKTS][RTE_BV_CLASSIFIER_MAX_FIELDS];
    __shared__ uint64_t c_lookup_hit_mask;

    if(!(threadIdx.x|threadIdx.y|threadIdx.z)) {
        c_lookup_hit_mask=0;
        __threadfence_block();
    }

    const int pkt_id=(blockDim.y*blockIdx.x+threadIdx.y);
    if(!((pkts_mask>>pkt_id)&1))
        return;

    for(int field_id=0; field_id<num_fields; ++field_id) {
        uint v;
        if(!threadIdx.x) {
            bv[pkt_id][field_id]=NULL;
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
                printf("[%d|%d] unknown size: %u byte\n", blockIdx.x, threadIdx.y, sizes[field_id]);
                break;
            }
        }
        v=__shfl_sync(UINT32_MAX, v, 0);


        long size=num_ranges[field_id]>>5;
        long start=0, offset;
        uint32_t l,r; //left, right
        __syncwarp();

        while(size) {
            offset=start+((long) threadIdx.x)*size;
            l=__ballot_sync(UINT32_MAX, v>=ranges_from[field_id][offset]);
            r=__ballot_sync(UINT32_MAX, v<=ranges_to[field_id][offset]);
            if(l&r) {
                if((__ffs(l&r)-1)==threadIdx.x)
                    bv[pkt_id][field_id]=bvs[field_id]+offset*RTE_BV_CLASSIFIER_BS;
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
            if((__ffs(l&r)-1)==threadIdx.x)
                bv[pkt_id][field_id]=bvs[field_id]+offset*RTE_BV_CLASSIFIER_BS;
        }

found_bv:
        if(!bv[pkt_id][field_id])
            goto end;
    }
    {
        __syncwarp();
        // all bitvectors found, now getting highest-priority rule
        uint x, tm;
        for(int bv_block=threadIdx.x; bv_block<bv_bs; bv_block+=blockDim.x) { // TODO maybe use WORKERS_PER_PACKET
            x=UINT32_MAX;
            for(int field_id=0; field_id<num_fields; ++field_id)
                x&=bv[pkt_id][field_id][bv_block];

            __syncwarp(__activemask()); //TODO maybe remove
            if((tm=__ballot_sync(__activemask(), __ffs(x)))) {
                if((__ffs(tm)-1)==threadIdx.x) {
                    matched_entries[pkt_id]=(void *) &entries[entry_size*((bv_block<<5)+__ffs(x)-1)];
                    //positions[pkt_id]=(bv_block<<5)+__ffs(x)-1;
                    atomicOr((unsigned long long int *)&c_lookup_hit_mask, 1LU<<pkt_id);
                }
                break;
            }
        }
    }
end:

    __syncthreads();

    if(!(threadIdx.x|threadIdx.y|threadIdx.z)) {
        atomicOr((unsigned long long int *) lookup_hit_mask, c_lookup_hit_mask);
        __threadfence_system();
    }
}

void rte_bv_classifier_enqueue_burst(struct rte_bv_classifier *c, struct rte_mbuf **pkts, uint64_t pkts_mask) {
    const uint32_t n_pkts_in=__builtin_popcountll(pkts_mask);
    const size_t epos=c->enqueue_pos;
    for(;;) {
        pthread_mutex_lock(&c->stream_running_mtx[epos]);
        if(!c->stream_running[epos]) {
            uint64_t real_pkts_mask=0;
            c->pkts_mask[epos]=pkts_mask;
            c->pkts[epos]=pkts;

            for(uint32_t i=0; i<n_pkts_in; ++i) {
                const uint32_t mp=pkts[i]->packet_type&c->ptype_mask;
                if((pkts_mask>>i)&1&((mp&RTE_PTYPE_L2_MASK)!=0)&((mp&RTE_PTYPE_L3_MASK)!=0)&((mp&RTE_PTYPE_L4_MASK)!=0)) {
                    c->pkts_data_h[epos][i]=rte_pktmbuf_mtod(pkts[i], uint8_t *);
                    real_pkts_mask|=1LU<<i;
                }
            }

            bv_search<<<NUM_BLOCKS, dim3{WORKERS_PER_PACKET, PACKETS_PER_BLOCK}, 0, c->streams[epos]>>>(
                c->ranges_from_dev, c->ranges_to_dev, c->num_ranges,
                c->field_offsets, c->field_sizes,
                c->bvs_dev, RTE_BV_CLASSIFIER_BS, c->num_fields, c->entry_size, c->entries,
                real_pkts_mask, c->pkts_data[epos],
                c->matched_entries[epos], &c->lookup_hit_mask[epos]);

            c->stream_running[epos]=1;
            c->enqueue_pos=(epos+1)&RTE_BV_CLASSIFIER_NUM_STREAMS_MASK;
            pthread_mutex_unlock(&c->stream_running_mtx[epos]);
            break;
        }
        pthread_mutex_unlock(&c->stream_running_mtx[epos]);
    }

}

void __rte_noreturn rte_bv_classifier_poll_lookups(struct rte_bv_classifier *c, void (*callback) (struct rte_mbuf **, uint64_t,  uint64_t, void **, void *), void *p) {
    size_t dpos=0;
    uint8_t stream_running;

    for(;;) {
        do {
            pthread_mutex_lock(&c->stream_running_mtx[dpos]);
            stream_running=c->stream_running[dpos];
            pthread_mutex_unlock(&c->stream_running_mtx[dpos]);
        } while(!stream_running);

        cudaStreamSynchronize(c->streams[dpos]);

        callback(c->pkts[dpos], c->pkts_mask[dpos], c->lookup_hit_mask_h[dpos], c->matched_entries[dpos], p);

        pthread_mutex_lock(&c->stream_running_mtx[dpos]);
        c->stream_running[dpos]=0;
        pthread_mutex_unlock(&c->stream_running_mtx[dpos]);

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
