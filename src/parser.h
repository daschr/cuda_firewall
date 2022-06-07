#ifndef __INCLUDE_PARSER
#define __INCLUDE_PARSER

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include "rte_bv.h"

#define INITIAL_BUFSIZE 16

enum {
	RULE_DROP = 1,
	RULE_ACCEPT
};

typedef struct {
    size_t num_rules;
    size_t rules_size;
    struct rte_bv_classifier_key **rules;
	uint8_t *actions; 
} ruleset_t;

bool parse_ruleset(ruleset_t *ruleset, const char *file);
void free_ruleset(ruleset_t *ruleset);
void free_ruleset_except_actions(ruleset_t *ruleset);
#endif
