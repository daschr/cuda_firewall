#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <rte_eal.h>
#include "rte_bv.h"

#ifdef __cplusplus
}
#endif

typedef struct {
    uint32_t from;
    uint32_t to;
    uint32_t val;
} range_t;

typedef struct {
    size_t ranges_size;
    size_t num_ranges;
    range_t *ranges;
} ranges_t;

int parse_borders(const char *filename, ranges_t *ranges) {
    size_t *i=&ranges->num_ranges;
    *i=0;
    if(ranges->ranges_size==0) {
        ranges->ranges_size=1024;
        ranges->ranges=(range_t *) malloc(sizeof(range_t)*ranges->ranges_size);
    }

    FILE *f=fopen(filename, "r");
    if(f==NULL) return 1;

    while(fscanf(f, "%X %X %X\n", &ranges->ranges[*i].from, &ranges->ranges[*i].to, &ranges->ranges[*i].val)==3) {
        ++(*i);
        if(*i==ranges->ranges_size) {
            ranges->ranges_size<<=1;
            ranges->ranges=(range_t *) realloc(ranges->ranges, sizeof(range_t)*ranges->ranges_size);
        }
    }

    fclose(f);

    return 0;
}

int main(int ac, char *as[]) {
    if(rte_eal_init(ac, as)<0)
        rte_exit(EXIT_FAILURE, "Error: could not initialize EAL.\n");

    ranges_t r;
    memset(&r, 0, sizeof(range_t));

    if(parse_borders(as[1], &r))
        goto fail;

    rte_bv_markers_t m;

    if(rte_bv_markers_create(&m))
        goto fail;

    for(size_t i=0; i<r.num_ranges; ++i) {
        printf("[ADD] %08X %08X %04X\n", r.ranges[i].from, r.ranges[i].to, r.ranges[i].val);
        rte_bv_markers_range_add(&m, (uint32_t *) &r.ranges[i], r.ranges[i].val);
    }

    free(r.ranges);

    rte_bv_ranges_t bv_r;
    bv_r.num_ranges=0;
    bv_r.bv_bs=(256>>5)+1;
    bv_r.ranges=(uint32_t *) malloc(sizeof(uint32_t)*4*r.num_ranges);
    bv_r.bvs=(uint32_t *) malloc(sizeof(uint32_t)*2*bv_r.bv_bs*r.num_ranges);
    memset(bv_r.bvs, 0, sizeof(uint32_t)*2*bv_r.bv_bs*r.num_ranges);

    rte_bv_markers_to_ranges(&m, 0, 1, &bv_r);
    rte_bv_markers_free(&m);

    printf("num_ranges: %lu\n", bv_r.num_ranges);
    for(size_t i=0; i<bv_r.num_ranges; ++i) {
        printf("%08X %08X ", bv_r.ranges[i<<1], bv_r.ranges[(i<<1)+1]);
        for(size_t j=0; j<bv_r.bv_bs; ++j)
            if(bv_r.bvs[bv_r.bv_bs*i+j]) {
                for(size_t p=0; p<32; ++p)
                    if((bv_r.bvs[bv_r.bv_bs*i+j]>>p)&1)
                        printf("%lu, ", 32*j+p);
            }
        puts("");
    }

    free(bv_r.ranges);
    free(bv_r.bvs);
    rte_eal_cleanup();

    return EXIT_SUCCESS;
fail:
    rte_eal_cleanup();
    return EXIT_FAILURE;
}
