#include "rte_bv_classifier.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "rte_bv.h"
#include <rte_log.h>
#include <rte_malloc.h>
#include <stdlib.h>


static inline int is_error(cudaError_t e, const char *file, int line) {
    if(e!=cudaSuccess) {
        fprintf(stderr, "[rte_bv_classifier] error: %s in %s (line %d)\n", cudaGetErrorString(e), file, line);
        return 1;
    }
    return 0;
}

int rte_bv_classifier_free(struct rte_bv_classifier *c) {
    if(c==NULL)
        return 0;

    cudaFreeHost(c->act_buf);
    for(size_t i=0; i<c->num_fields<<1; ++i) {
        cudaFree(c->ranges_db[i]);
        cudaFree(c->bvs_db[i]);
    }

    cudaFree(c->ranges_db_dev);
    cudaFree(c->bvs_db_dev);

    cudaFree(c->num_ranges);
    cudaFree(c->field_offsets);
    cudaFree(c->field_sizes);

    for(uint32_t i=0; i<c->num_fields; ++i)
        rte_bv_markers_free(c->bv_markers+i);

    rte_free(c->bv_markers);
    rte_free(c->ranges_db);
    rte_free(c->bvs_db);

    rte_free(c);

    return 0;
}

#define IS_ERROR(X) is_error(X, __FILE__, __LINE__)

struct rte_bv_classifier *rte_bv_classifier_create(struct rte_bv_classifier_params *params, int socket_id) {
    struct rte_bv_classifier_params *p=params;
    struct rte_bv_classifier *c=(struct rte_bv_classifier *) rte_malloc("c", sizeof(struct rte_bv_classifier), 0);
    memset(c, 0, sizeof(struct rte_bv_classifier));

    c->num_fields=p->num_fields;
    c->field_defs=p->field_defs;

    c->ranges_db=(uint32_t **) rte_malloc("ranges_db", sizeof(uint32_t *)*(c->num_fields<<1), 0);
    c->bvs_db=(uint32_t **) rte_malloc("bvs_db", sizeof(uint32_t *)*(c->num_fields<<1), 0);
#define CHECK(X) if(IS_ERROR(X)) return NULL
#define HOSTALLOC(DP, HP, SIZE, MODE)	CHECK(cudaHostAlloc((void **) &HP, SIZE, cudaHostAllocMapped | MODE)); \
    								CHECK(cudaHostGetDevicePointer((void **) &DP, (void *) HP, 0));

    HOSTALLOC(c->act_buf, c->act_buf_h, sizeof(uint8_t), 0);

    for(size_t stream=0; stream<RTE_BV_CLASSIFIER_NUM_STREAMS; ++stream) {
        CHECK(cudaStreamCreateWithFlags(c->streams+stream, 0));

        HOSTALLOC(c->pkts_data[stream], c->pkts_data_h[stream], sizeof(uint8_t*)*RTE_BV_CLASSIFIER_MAX_PKTS, cudaHostAllocWriteCombined);
        HOSTALLOC(c->packet_types[stream], c->packet_types_h[stream], sizeof(uint32_t)*RTE_BV_CLASSIFIER_MAX_PKTS, cudaHostAllocWriteCombined);
        HOSTALLOC(c->positions[stream], c->positions_h[stream], sizeof(uint32_t)*RTE_BV_CLASSIFIER_MAX_PKTS, 0);
        HOSTALLOC(c->lookup_hit_mask[stream], c->lookup_hit_mask_h[stream], sizeof(uint64_t), 0);

        c->stream_running[stream]=0;
        c->stream_running_mtx[stream]=PTHREAD_MUTEX_INITIALIZER;
    }

    c->enqueue_pos=0;
    c->dequeue_pos=0;

    CHECK(cudaMalloc((void **) &c->ranges_db_dev, 		sizeof(uint32_t *)*	c->num_fields*2));
    CHECK(cudaMalloc((void **) &c->bvs_db_dev, 			sizeof(uint32_t *)*	c->num_fields*2));
    CHECK(cudaMalloc((void **) &c->field_offsets, 		sizeof(uint32_t)  *	c->num_fields));
    CHECK(cudaMalloc((void **) &c->field_ptype_masks, 	sizeof(uint32_t)  *	c->num_fields));
    CHECK(cudaMalloc((void **) &c->field_sizes, 		sizeof(uint32_t)  *	c->num_fields));
    CHECK(cudaMalloc((void **) &c->num_ranges, 			sizeof(size_t)    *	c->num_fields));

    for(size_t i=0; i<c->num_fields; ++i) {
        CHECK(cudaMemcpy(c->field_offsets+i, 		&c->field_defs[i].offset, 		sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(c->field_sizes+i, 			&c->field_defs[i].size, 		sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(c->field_ptype_masks+i, 	&c->field_defs[i].ptype_mask, 	sizeof(uint32_t), cudaMemcpyHostToDevice));

        CHECK(cudaMalloc((void **) &c->ranges_db[i], 				sizeof(uint32_t)*RTE_BV_CLASSIFIER_MAX_RANGES*2));
        CHECK(cudaMalloc((void **) &c->ranges_db[c->num_fields+i], 	sizeof(uint32_t)*RTE_BV_CLASSIFIER_MAX_RANGES*2));
        CHECK(cudaMalloc((void **) &c->bvs_db[i], 					sizeof(uint32_t)*RTE_BV_CLASSIFIER_BS*RTE_BV_CLASSIFIER_MAX_RANGES*2));
        CHECK(cudaMalloc((void **) &c->bvs_db[c->num_fields+i], 	sizeof(uint32_t)*RTE_BV_CLASSIFIER_BS*RTE_BV_CLASSIFIER_MAX_RANGES*2));
    }

    CHECK(cudaMemcpy(c->ranges_db_dev, 	c->ranges_db, 	sizeof(uint32_t *)*c->num_fields*2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(c->bvs_db_dev, 	c->bvs_db, 		sizeof(uint32_t *)*c->num_fields*2, cudaMemcpyHostToDevice));

#undef HOSTALLOC
#undef CHECK

    c->bv_markers=(rte_bv_markers_t *) rte_malloc("bv_markers", sizeof(rte_bv_markers_t)*c->num_fields, 0);

    for(size_t i=0; i<c->num_fields; ++i) {
        if(rte_bv_markers_create(&c->bv_markers[i])) {
            rte_bv_classifier_free(c);
            rte_log(RTE_LOG_ERR, RTE_LOGTYPE_HASH, "Error creating marker!\n");
            return NULL;
        }
    }

    return c;
}
#undef IS_ERROR

static inline void cal_from_to(uint32_t *from_to, uint32_t *v, uint8_t type, uint8_t size) {
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
            fprintf(stderr, "[cal_from_to] error: unkown size: %d bits\n", size);
#endif
            break;
        }
    }
}

int rte_bv_classifier_entry_add(struct rte_bv_classifier *c, struct rte_bv_classifier_key *k, uint32_t *pos, int *key_found) {
    *key_found=0;

    uint32_t from_to[2];
    uint8_t next_act_buf=*c->act_buf_h^1;
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<c->num_fields; ++f) {
        cal_from_to(from_to, k->buf +(f<<1), c->field_defs[f].type, c->field_defs[f].size);
        rte_bv_markers_range_add(c->bv_markers+f, from_to, *pos);

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_BV_CLASSIFIER_BS;
        ranges.ranges=c->ranges_db[(next_act_buf*c->num_fields)+f];
        ranges.bvs=c->bvs_db[(next_act_buf*c->num_fields)+f];
        rte_bv_markers_to_ranges(c->bv_markers+f, 1, sizeof(uint32_t), &ranges);
        cudaMemcpy(c->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    *c->act_buf_h=next_act_buf;

    return 0;
}

int rte_bv_classifier_entry_delete(struct rte_bv_classifier *c, struct rte_bv_classifier_key *k) {
    uint32_t from_to[2];
    uint8_t next_act_buf=*c->act_buf_h^1;
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<c->num_fields; ++f) {
        cal_from_to(from_to, k->buf+(f<<1), c->field_defs[f].type, c->field_defs[f].size);
        rte_bv_markers_range_del(c->bv_markers+f, from_to, k->pos);

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_BV_CLASSIFIER_BS;
        ranges.ranges=c->ranges_db[(next_act_buf*c->num_fields)+f];
        ranges.bvs=c->bvs_db[(next_act_buf*c->num_fields)+f];
        rte_bv_markers_to_ranges(c->bv_markers+f, 1, sizeof(uint32_t), &ranges);
        cudaMemcpy(c->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    return 0;
}

int rte_bv_classifier_entry_add_bulk(struct rte_bv_classifier *c, struct rte_bv_classifier_key **ks, uint32_t n_keys) {
    uint32_t from_to[2];
    uint8_t next_act_buf=*c->act_buf_h^1;
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<c->num_fields; ++f) {
        for(uint32_t k=0; k<n_keys; ++k) {
            cal_from_to(from_to, ks[k]->buf+(f<<1), c->field_defs[f].type, c->field_defs[f].size);
            rte_bv_markers_range_add(c->bv_markers+f, from_to, ks[k]->pos);
        }

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_BV_CLASSIFIER_BS;
        ranges.ranges=c->ranges_db[(next_act_buf*c->num_fields)+f];
        ranges.bvs=c->bvs_db[(next_act_buf*c->num_fields)+f];
        rte_bv_markers_to_ranges(c->bv_markers+f, 1, sizeof(uint32_t), &ranges);
        cudaMemcpy(c->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    *c->act_buf_h=next_act_buf;

    return 0;
}

int rte_bv_classifier_entry_delete_bulk(struct rte_bv_classifier *c, struct rte_bv_classifier_key **ks, uint32_t n_keys) {
    uint32_t from_to[2];
    uint8_t next_act_buf=*c->act_buf_h^1;
    rte_bv_ranges_t ranges;

    for(uint32_t f=0; f<c->num_fields; ++f) {
        for(uint32_t k=0; k<n_keys; ++k) {
            cal_from_to(from_to, ks[k]->buf+(f<<1), c->field_defs[f].type, c->field_defs[f].size);
            rte_bv_markers_range_del(c->bv_markers+f, from_to, ks[k]->pos);
        }

        memset(&ranges, 0, sizeof(rte_bv_ranges_t));
        ranges.bv_bs=RTE_BV_CLASSIFIER_BS;
        ranges.ranges=c->ranges_db[(next_act_buf*c->num_fields)+f];
        ranges.bvs=c->bvs_db[(next_act_buf*c->num_fields)+f];
        rte_bv_markers_to_ranges(c->bv_markers+f, 1, sizeof(uint32_t), &ranges);
        cudaMemcpy(c->num_ranges+f, (void *) &ranges.num_ranges, sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    *c->act_buf_h=next_act_buf;

    return 0;
}

__global__ void bv_search(	 uint32_t **ranges,  uint64_t *num_ranges,  uint32_t *offsets,  uint8_t *sizes,
                             uint32_t *ptype_mask,  uint32_t **bvs, const uint32_t bv_bs,
                             const ulong pkts_mask, uint8_t **pkts, uint32_t *__restrict__ pkts_type,
                             volatile uint *__restrict__ positions, volatile ulong *__restrict__ lookup_hit_mask) {

    if(!((pkts_mask>>blockIdx.x)&1))
        return;

    uint8_t *pkt;
    __shared__ uint *bv[24];
    __shared__ bool field_found[24];
    uint v=0;

    field_found[threadIdx.x]=false;

    const uint32_t ptype_a=pkts_type[blockIdx.x]&ptype_mask[threadIdx.x];
    const bool ptype_matches=  (ptype_a&RTE_PTYPE_L2_MASK)!=0
                               & (ptype_a&RTE_PTYPE_L3_MASK)!=0
                               & (ptype_a&RTE_PTYPE_L4_MASK)!=0;

    if(ptype_matches) {
        pkt=pkts[blockIdx.x]+offsets[threadIdx.x];

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
                bv[threadIdx.x]=bvs[threadIdx.x]+i*RTE_BV_CLASSIFIER_BS;
                field_found[threadIdx.x]=true;
                break;
            }

            se[!l]=!l?i-1:i+1;
        }
    }

    __syncthreads();
    if(!threadIdx.x) {
        uint x, pos;
        for(uint i=0; i<bv_bs; ++i) {
            x=0xffffffff;
            for(uint b=0; b<blockDim.x; ++b) {
                if(!field_found[b])
                    goto end;
                x&=bv[b][i];
            }

            if((pos=__ffs(x))!=0) {
                positions[blockIdx.x]=(i<<5)+pos-1;
                atomicOr((unsigned long long *)lookup_hit_mask, 1<<blockIdx.x);
                break;
            }
        }
    }
end:
    __syncthreads();
}

void rte_bv_classifier_enqueue_burst(struct rte_bv_classifier *c, struct rte_mbuf **pkts, uint64_t pkts_mask) {
    const uint32_t n_pkts_in=__builtin_popcountll(pkts_mask);
    const size_t epos=c->enqueue_pos;

    for(;;) {
        pthread_mutex_lock(&c->stream_running_mtx[epos]);;
        if(!c->stream_running[epos]) {

            c->pkts_mask[epos]=pkts_mask;
            c->pkts[epos]=pkts;

            for(uint32_t i=0; i<n_pkts_in; ++i)
                if((pkts_mask>>i)&1) {
                    c->pkts_data_h[epos][i]=rte_pktmbuf_mtod(pkts[i], uint8_t *);
                    c->packet_types_h[epos][i]=pkts[i]->packet_type;
                }

            bv_search<<<64, c->num_fields, 0, c->streams[epos]>>>(c->ranges_db_dev+(c->num_fields*(*c->act_buf_h)), c->num_ranges,
                    c->field_offsets, c->field_sizes, c->field_ptype_masks,
                    c->bvs_db_dev+(c->num_fields*(*c->act_buf_h)), RTE_BV_CLASSIFIER_BS,
                    pkts_mask, c->pkts_data[epos], c->packet_types[epos],
                    c->positions[epos], c->lookup_hit_mask[epos]);
            c->stream_running[epos]=1;
            c->enqueue_pos=(epos+1)&RTE_BV_CLASSIFIER_NUM_STREAMS_MASK;
            break;
        }
        pthread_mutex_unlock(&c->stream_running_mtx[epos]);;
    }

}

void __rte_noreturn rte_bv_classifier_poll_lookups(struct rte_bv_classifier *c, void (*callback) (struct rte_mbuf **, uint64_t,  uint64_t, uint32_t *, void *), void *p) {
    size_t dpos;
    uint8_t stream_running;

    for(;;) {
        do {
            pthread_mutex_lock(&c->stream_running_mtx[dpos]);
            stream_running=c->stream_running[dpos];
            pthread_mutex_unlock(&c->stream_running_mtx[dpos]);
        } while(stream_running);

        cudaStreamSynchronize(c->streams[dpos]);

        callback(c->pkts[dpos], c->pkts_mask[dpos], *(c->lookup_hit_mask_h[dpos]), c->positions_h[dpos], p);

        pthread_mutex_lock(&c->stream_running_mtx[dpos]);
        stream_running=0;
        pthread_mutex_unlock(&c->stream_running_mtx[dpos]);

        dpos=(++dpos)&RTE_BV_CLASSIFIER_NUM_STREAMS_MASK;
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
