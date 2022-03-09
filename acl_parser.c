#ifdef __cplusplus
extern "C" {
#endif

#include "acl_parser.h"

#include <rte_acl.h>
#include <rte_table.h>
#include <rte_table_acl.h>

#include <stdlib.h>
#include <stdio.h>

#define PARSER_LINESIZE 1024

static int acl_parse_range_u16(const char *s, struct rte_acl_field *out);
static int acl_parse_range_u8(const char *s, struct rte_acl_field *out);

bool acl_parse_ruleset(acl_ruleset_t *ruleset, const char *file) {
    FILE *fd=NULL;
    if((fd=fopen(file, "r"))==NULL) {
        fprintf(stderr, "ERROR: could not open file \"%s\" for ruleset!\n", file);
        return false;
    }

    char *inp_line;
    if((inp_line=malloc(sizeof(char)*PARSER_LINESIZE))==NULL) {
        fprintf(stderr, "ERROR: could not allocate %u bytes for line buffer\n", PARSER_LINESIZE);
        return false;
    }

    uint8_t ip_buf[10];
    char v_buf[3][13], command[13];

    if(ruleset->rules==NULL) {
        if((ruleset->rules=malloc(sizeof(struct rte_table_acl_rule_add_params *)*INITIAL_BUFSIZE))==NULL) {
            fprintf(stderr, "ERROR: could not allocate memory for rules!\n");
            goto failure;
        }

        if((ruleset->actions=malloc(sizeof(uint8_t)*INITIAL_BUFSIZE))==NULL) {
            fprintf(stderr, "ERROR: could not allocate memory for actions!\n");
            goto failure;
        }

        for(size_t i=0; i<INITIAL_BUFSIZE; ++i) {
            if((ruleset->rules[i]=malloc(sizeof(struct rte_table_acl_rule_add_params)))==NULL) {
                fprintf(stderr, "ERROR: could not allocate memory for %luth rule!\n", i);
                goto failure;
            }
        }

        ruleset->rules_size=INITIAL_BUFSIZE;
    }

    ruleset->num_rules=0;

    while(fgets(inp_line, PARSER_LINESIZE, fd)!=NULL) {
        if(*inp_line=='#') continue;

        if(sscanf(inp_line, "%hhu.%hhu.%hhu.%hhu/%hhu %hhu.%hhu.%hhu.%hhu/%hhu %12s %12s %12s %6s",
                  &ip_buf[0], &ip_buf[1], &ip_buf[2], &ip_buf[3], &ip_buf[4],
                  &ip_buf[5], &ip_buf[6], &ip_buf[7], &ip_buf[8], &ip_buf[9],
                  v_buf[0], v_buf[1], v_buf[2], command)==14) {

			if(acl_parse_range_u8(v_buf[2], ruleset->rules[ruleset->num_rules]->field_value)) {
                fprintf(stderr, "ERROR: could not parse range: \"%s\"\n", v_buf[2]);
                goto failure;
            }

            // src ipv4 range
            ruleset->rules[ruleset->num_rules]->field_value[1].value.u32=(((ip_buf[0]<<24)+(ip_buf[1]<<16)+(ip_buf[2]<<8)+ip_buf[3])>>(32-ip_buf[4])) <<(32-ip_buf[4]);
            ruleset->rules[ruleset->num_rules]->field_value[1].mask_range.u32=ruleset->rules[ruleset->num_rules]->field_value[1].value.u32|(UINT32_MAX>>ip_buf[4]);

            // dst ipv4 range
            ruleset->rules[ruleset->num_rules]->field_value[2].value.u32=(((ip_buf[5]<<24)+(ip_buf[6]<<16)+(ip_buf[7]<<8)+ip_buf[8])>>(32-ip_buf[9]))<<(32-ip_buf[9]);
            ruleset->rules[ruleset->num_rules]->field_value[2].mask_range.u32=ruleset->rules[ruleset->num_rules]->field_value[2].value.u32|(UINT32_MAX>>ip_buf[9]);

            for(int i=0; i<2; ++i) {
                if(acl_parse_range_u16(v_buf[i], ruleset->rules[ruleset->num_rules]->field_value+3+i)) {
                    fprintf(stderr, "ERROR: could not parse range: \"%s\"\n", v_buf[i]);
                    goto failure;
                }
            }
            
            if(strcmp(command, "DROP")==0) {
                ruleset->actions[ruleset->num_rules]=ACL_RULE_DROP;
            } else if(strcmp(command, "ACCEPT")==0) {
                ruleset->actions[ruleset->num_rules]=ACL_RULE_ACCEPT;
            } else {
                fprintf(stderr, "ERROR: could not parse command \"%s\"\n", command);
                goto failure;
            }

            ruleset->rules[ruleset->num_rules]->priority=ruleset->num_rules;

            if(++(ruleset->num_rules)==ruleset->rules_size) {
                ruleset->rules_size<<=1;
                if((ruleset->rules=realloc(ruleset->rules, sizeof(struct rte_table_acl_rule_add_params *)*ruleset->rules_size))==NULL) {
                    fprintf(stderr, "ERROR: could not realloc memory for rules!\n");
                    goto failure;
                }

                if((ruleset->actions=realloc(ruleset->actions, sizeof(uint8_t)*ruleset->rules_size))==NULL) {
                    fprintf(stderr, "ERROR: could not realloc memory for actions!\n");
                    goto failure;
                }

                for(size_t i=ruleset->rules_size>>1; i<ruleset->rules_size; ++i) {
                    if((ruleset->rules[i]=malloc(sizeof(struct rte_table_acl_rule_add_params)))==NULL) {
                        fprintf(stderr, "ERROR: could not allocate memory for %luth rule!\n", i);
                        goto failure;
                    }
                }
            }
        }
    }

    free(inp_line);
    fclose(fd);
    return true;

failure:
    fprintf(stderr, "ERROR: parse exiting...\n");
    free(inp_line);
    if(ruleset->rules) {
        for(size_t i=0; i<ruleset->rules_size; ++i) {
            free(ruleset->rules[i]);
        }

        free(ruleset->rules);
        free(ruleset->actions);
    }

    ruleset->rules=NULL;
    ruleset->actions=NULL;

    fclose(fd);
    return false;
}

void acl_free_ruleset(acl_ruleset_t *ruleset) {
    if(ruleset->rules) {
        for(size_t i=0; i<ruleset->rules_size; ++i) {
            free(ruleset->rules[i]);
        }

        free(ruleset->rules);
        free(ruleset->actions);
    }

    ruleset->rules=NULL;
    ruleset->actions=NULL;
}

void acl_free_ruleset_except_actions(acl_ruleset_t *ruleset) {
    if(ruleset->rules) {
        for(size_t i=0; i<ruleset->rules_size; ++i) {
            free(ruleset->rules[i]);
        }

        free(ruleset->rules);
    }

    ruleset->rules=NULL;
}

static int acl_parse_range_u16(const char *s, struct rte_acl_field  *out) {
    if(sscanf(s, "%hu-%hu", &(out->value.u16), &(out->mask_range.u16))==2)
        return 0;
	
    if(sscanf(s, "%hu", &(out->value.u16))==1) {
		switch(out->value.u16) {
        case 0:
            out->mask_range.u16=UINT16_MAX;
            break;
        default:
            out->mask_range.u16=out->value.u16;
            break;
        }

        return 0;
    }

    return 1;
}

static int acl_parse_range_u8(const char *s, struct rte_acl_field  *out) {
    if(sscanf(s, "%hhu-%hhu", &(out->value.u8), &(out->mask_range.u8))==2)
        return 0;

    if(sscanf(s, "%hhu", &(out->value.u8))==1) {
        switch(out->value.u8) {
        case 0:
            out->mask_range.u8=UINT8_MAX;
            break;
        default:
            out->mask_range.u8=out->value.u8;
            break;
        }

        return 0;
    }

    return 1;
}

#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
}
#endif
