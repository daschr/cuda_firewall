#ifndef __INCLUDE_STATS
#define __INCLUDE_STATS

#include <rte_memory.h>

typedef struct {
	size_t pkts_in;
	size_t pkts_out;
	size_t pkts_dropped;
	size_t pkts_accepted;
	size_t pkts_lookup_miss;
} stats_t;

#endif
