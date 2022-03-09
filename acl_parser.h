#ifndef __INCLUDE_ACL_PARSER
#define __INCLUDE_ACL_PARSER

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#define INITIAL_BUFSIZE 16

#include <rte_acl.h>
#include <rte_table.h>
#include <rte_table_acl.h>

enum {
	ACL_RULE_DROP = 1,
	ACL_RULE_ACCEPT
};

typedef struct {
    size_t num_rules;
    size_t rules_size;
    struct rte_table_acl_rule_add_params **rules;
	uint8_t *actions; 
} acl_ruleset_t;

bool acl_parse_ruleset(acl_ruleset_t *ruleset, const char *file);
void acl_free_ruleset(acl_ruleset_t *ruleset);
void acl_free_ruleset_except_actions(acl_ruleset_t *ruleset);
#endif
