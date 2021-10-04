#ifndef _in_ls_cl
#define _in_ls_cl

#include <cuda_runtime.h>
#include <pthread.h>
#include "parser.h"
#include "stdbool.h"

typedef struct{
	uint32_t *lower;
	uint32_t *upper;
	uint32_t *header;
	uint32_t *pos;
	uint32_t *pos_h;
	uint32_t *header_h;
	uint8_t *done_pkt;
	volatile uint8_t *done_pkt_h;
	uint8_t *new_pkt;
	volatile uint8_t *new_pkt_h;
	uint8_t *running;
	volatile uint8_t *running_h;
	cudaStream_t kernel_stream;
}ls_cl_t;

bool ls_cl_new(ls_cl_t *lscl, const ruleset_t *rules);
uint8_t ls_cl_get(ls_cl_t *lscl, const header_t *header, const ruleset_t *rules);
void ls_cl_free(ls_cl_t *lscl);

#endif
