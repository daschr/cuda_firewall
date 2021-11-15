#ifndef __INCLUDE_STATS
#define __INCLUDE_STATS

typedef struct {
	size_t pkts_in;
	size_t pkts_out;
	size_t pkts_dropped;
	size_t pkts_accepted;
} stats_t;

#endif
