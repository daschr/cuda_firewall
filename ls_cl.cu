extern "C" {
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "ls_cl.h"
#include "parser.h"
}

static inline void check_error(cudaError_t e, const char *file, int line) {
    if(e != cudaSuccess) {
        fprintf(stderr, "[ERROR] %s in %s (line %d)\n", cudaGetErrorString(e), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CHECK(X) (check_error(X, __FILE__, __LINE__))

static inline void cpy_rules(const ruleset_t *rules, uint32_t *buffer, uint8_t upper) {
    size_t bp;
    for(size_t i=0; i<rules->num_rules; ++i) {
        bp=i<<2;
        buffer[bp++]=rules->rules[i].c1[upper];
        buffer[bp++]=rules->rules[i].c2[upper];
        buffer[bp++]=(uint32_t) (rules->rules[i].c3[upper]<<16) | (uint32_t) rules->rules[i].c4[upper];
        buffer[bp]=rules->rules[i].c5[upper];
    }
}

__global__ void ls(	const uint *__restrict__ lower,  const uint *__restrict__ upper, const ulong rules_size,
                    volatile uint *__restrict__ header, volatile uint *__restrict__ pos,
                    volatile uint8_t *new_pkt, volatile uint8_t *done_pkt, volatile uint8_t *running) {

    ulong start=blockDim.x*blockIdx.x+threadIdx.x, step=(gridDim.x*blockDim.x)<<2;
    __shared__ uint h[4];
    __shared__ uint8_t found;
    __shared__ uint8_t run;
    ulong i;
    uint8_t r;

    if(!threadIdx.x) {
        run=*running;
    }

    __syncthreads();

    while(run) {
        if(!threadIdx.x) {
            while(*new_pkt==0);

            for(int i=0; i<4; ++i)
                h[i]=header[i];
            found=0;
        }

        __syncthreads();

        i=start<<2;
        while(!found) {
            r=i<rules_size?lower[i]<=h[0] & h[0]<=upper[i]
              & lower[i+1]<=h[1] & h[1]<=upper[i+1]
              & (__vcmpleu2(lower[i+2], h[2]) & __vcmpgeu2(upper[i+2], h[2]))==0xffffffff
              & lower[i+3]<=h[3] & h[3]<=upper[i+3]:0;

            if(r) {
                atomicMin((uint *) pos, i>>2);
                found=1;
                __threadfence_system();
            }

            if((!threadIdx.x) & (i>rules_size))
                found=1;
            
			i+=step;
            __syncthreads();
        }

        __syncthreads();

        if(!start) {
            *new_pkt=0;
            *done_pkt=1;
        }

        if(!threadIdx.x) {
            run=*running;
        }

        __syncthreads();
    }
}

bool ls_cl_new(ls_cl_t *lscl, const ruleset_t *rules) {
    // lower upper buffer

    size_t bufsize=(sizeof(uint32_t)<<2)*rules->num_rules;
    uint32_t *buffer=(uint32_t *) malloc(bufsize);
    memset(buffer, 0, bufsize);

    CHECK(cudaMalloc((void **) &lscl->lower, bufsize));
    CHECK(cudaMalloc((void **) &lscl->upper, bufsize));

    cpy_rules(rules, buffer, 0);
    CHECK(cudaMemcpy(lscl->lower, buffer, bufsize, cudaMemcpyHostToDevice));

    cpy_rules(rules, buffer, 1);
    CHECK(cudaMemcpy(lscl->upper, buffer, bufsize, cudaMemcpyHostToDevice));

    free(buffer);

    CHECK(cudaHostAlloc((void **) &lscl->header_h, (sizeof(uint32_t)<<2), cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer((void **) &lscl->header, lscl->header_h, 0));
    CHECK(cudaHostAlloc((void **) &lscl->pos_h, sizeof(uint32_t), cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer((void **) &lscl->pos, lscl->pos_h, 0));

    CHECK(cudaHostAlloc((void **) &lscl->new_pkt_h, sizeof(uint8_t), cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer((void **) &lscl->new_pkt, (uint8_t *) lscl->new_pkt_h, 0));
    CHECK(cudaHostAlloc((void **) &lscl->done_pkt_h, sizeof(uint8_t), cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer((void **) &lscl->done_pkt, (uint8_t *) lscl->done_pkt_h, 0));
    CHECK(cudaHostAlloc((void **) &lscl->running_h, sizeof(uint8_t), cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer((void **) &lscl->running, (uint8_t *) lscl->running_h, 0));

    *lscl->new_pkt_h=0;
    *lscl->done_pkt_h=0;
    *lscl->running_h=1;

    CHECK(cudaStreamCreateWithFlags(&lscl->kernel_stream, 0));

    int mp_count;
    CHECK(cudaDeviceGetAttribute(&mp_count, cudaDevAttrMultiProcessorCount, 0));
    ls<<<1, 128, 0,lscl->kernel_stream>>>(lscl->lower, lscl->upper, (uint64_t) rules->num_rules<<2,
                                          lscl->header, lscl->pos, lscl->new_pkt, lscl->done_pkt, lscl->running);

    return true;
}

uint8_t ls_cl_get(ls_cl_t *lscl, const header_t *header, const ruleset_t *rules) {
#define H(X) lscl->header_h[X-1]=header->h ## X
    lscl->header_h[0]=header->h1;
    lscl->header_h[1]=header->h2;
    lscl->header_h[2]=(uint32_t) (header->h3<<16) | (uint32_t) header->h4;
    lscl->header_h[3]=header->h5;
#undef H

    *lscl->pos_h=UINT_MAX;
    *lscl->new_pkt_h=1;
    while(!(*lscl->done_pkt_h));
    *lscl->done_pkt_h=0;

    return *lscl->pos_h==UINT_MAX?0xff:rules->rules[*lscl->pos_h].val;
}

void ls_cl_free(ls_cl_t *lscl) {
    *lscl->running_h=0;
    *lscl->new_pkt_h=1;
    cudaStreamSynchronize(lscl->kernel_stream);
    CHECK(cudaFree(lscl->lower));
    CHECK(cudaFree(lscl->upper));
    CHECK(cudaFreeHost(lscl->header_h));
    CHECK(cudaFreeHost(lscl->pos_h));
    CHECK(cudaFreeHost((void *) lscl->done_pkt_h));
    CHECK(cudaFreeHost((void *) lscl->new_pkt_h));
    CHECK(cudaFreeHost((void *) lscl->running_h));
}

int main(int ac, char *as[]) {
    if(ac<3) {
        fprintf(stderr, "Usage: %s [ruleset] [headers] [?result file]\n", as[0]);
        return EXIT_FAILURE;
    }

    FILE *res_file=stdout;
    uint8_t *results=NULL;
    if(ac>3) {
        if((res_file=fopen(as[3], "w"))==NULL) {
            fprintf(stderr, "could not open \"%s\" for writing!\n", as[3]);
            return EXIT_FAILURE;
        }
    }

    ruleset_t rules= {.num_rules=0, .rules_size=0, .rules=NULL};
    headers_t headers= {.num_headers=0, .headers_size=0, .headers=NULL};
    if(!parse_ruleset(&rules, as[1]) || !parse_headers(&headers, as[2]))
        goto fail;

    if((results=(uint8_t *)malloc(sizeof(uint8_t)*(headers.num_headers)))==NULL)
        goto fail;

    struct timeval tv1, tv2;
    ls_cl_t lscl;

    gettimeofday(&tv1, NULL);
    if(!ls_cl_new(&lscl, &rules)) {
        fputs("could not initiate ls_cl!\n", stderr);
        goto fail;
    }
    gettimeofday(&tv2, NULL);
    printf("PREPROCESSING  took %12lu us\n", 1000000*(tv2.tv_sec-tv1.tv_sec)+(tv2.tv_usec-tv1.tv_usec));

    gettimeofday(&tv1, NULL);
    for(size_t i=0; i<headers.num_headers; ++i)
        results[i]=ls_cl_get(&lscl, headers.headers+i, &rules);
    gettimeofday(&tv2, NULL);
    printf("CLASSIFICATION took %12lu us\n", 1000000*(tv2.tv_sec-tv1.tv_sec)+(tv2.tv_usec-tv1.tv_usec));

    for(size_t i=0; i<headers.num_headers; ++i)
        fprintf(res_file, "%02X\n", results[i]);

    ls_cl_free(&lscl);

    return EXIT_SUCCESS;
fail:
    free(rules.rules);
    free(headers.headers);
    free(results);

    return EXIT_FAILURE;
}
