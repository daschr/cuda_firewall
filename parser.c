#include "parser.h"

#include "rte_table_bv.h"
#include <stdlib.h>
#include <stdio.h>

#define PARSER_LINESIZE 1024

static int parse_range(const char *s, uint32_t *out);

bool parse_ruleset(ruleset_t *ruleset, const char *file) {
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
        if((ruleset->rules=malloc(sizeof(struct rte_table_bv_key *)*INITIAL_BUFSIZE))==NULL) {
            fprintf(stderr, "ERROR: could not allocate memory for rules!\n");
            goto failure;
        }

        if((ruleset->actions=malloc(sizeof(uint8_t)*INITIAL_BUFSIZE))==NULL) {
            fprintf(stderr, "ERROR: could not allocate memory for actions!\n");
            goto failure;
        }

        for(size_t i=0; i<INITIAL_BUFSIZE; ++i) {
            if((ruleset->rules[i]=malloc(sizeof(struct rte_table_bv_key)))==NULL) {
                fprintf(stderr, "ERROR: could not allocate memory for %luth rule!\n", i);
                goto failure;
            }

            if((ruleset->rules[i]->buf=malloc(sizeof(uint32_t)*10))==NULL) {
                fprintf(stderr, "ERROR: could not  allocate memory for %luth rule!\n", i);
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

            // src ipv4 range
            ruleset->rules[ruleset->num_rules]->buf[0]=(((ip_buf[0]<<24)+(ip_buf[1]<<16)+(ip_buf[2]<<8)+ip_buf[3])>>(32-ip_buf[4]))
                    <<(32-ip_buf[4]);
            ruleset->rules[ruleset->num_rules]->buf[1]=ruleset->rules[ruleset->num_rules]->buf[0]|(UINT32_MAX>>ip_buf[4]);

            // dst ipv4 range
            ruleset->rules[ruleset->num_rules]->buf[2]=(((ip_buf[5]<<24)+(ip_buf[6]<<16)+(ip_buf[7]<<8)+ip_buf[8])>>(32-ip_buf[9]))
                    <<(32-ip_buf[9]);
            ruleset->rules[ruleset->num_rules]->buf[3]=ruleset->rules[ruleset->num_rules]->buf[2]|(UINT32_MAX>>ip_buf[9]);

            for(int i=0; i<3; ++i) {
                if(parse_range(v_buf[i], ruleset->rules[ruleset->num_rules]->buf+4+(i<<1))) {
                    fprintf(stderr, "ERROR: could not parse range: \"%s\"\n", v_buf[i]);
                    goto failure;
                }
            }

            if(strcmp(command, "DROP")==0) {
                ruleset->actions[ruleset->num_rules]=RULE_DROP;
            } else if(strcmp(command, "ACCEPT")==0) {
                ruleset->actions[ruleset->num_rules]=RULE_ACCEPT;
            } else {
                fprintf(stderr, "ERROR: could not parse command \"%s\"\n", command);
                goto failure;
            }

            ruleset->rules[ruleset->num_rules]->pos=ruleset->num_rules;

            if(++(ruleset->num_rules)==ruleset->rules_size) {
                ruleset->rules_size<<=1;
                if((ruleset->rules=realloc(ruleset->rules, sizeof(struct rte_table_bv_key *)*ruleset->rules_size))==NULL) {
                    fprintf(stderr, "ERROR: could not realloc memory for rules!\n");
                    goto failure;
                }

                if((ruleset->actions=realloc(ruleset->actions, sizeof(uint8_t)*ruleset->rules_size))==NULL) {
                    fprintf(stderr, "ERROR: could not realloc memory for actions!\n");
                    goto failure;
                }

                for(size_t i=ruleset->rules_size>>1; i<ruleset->rules_size; ++i) {
                    if((ruleset->rules[i]=malloc(sizeof(struct rte_table_bv_key)))==NULL) {
                        fprintf(stderr, "ERROR: could not allocate memory for %luth rule!\n", i);
                        goto failure;
                    }

                    if((ruleset->rules[i]->buf=malloc(sizeof(uint32_t)*10))==NULL) {
                        fprintf(stderr, "ERORR: could not allocate memory for %luth rule!\n", i);
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
            free(ruleset->rules[i]->buf);
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

void free_ruleset(ruleset_t *ruleset) {
    if(ruleset->rules) {
        for(size_t i=0; i<ruleset->rules_size; ++i) {
            free(ruleset->rules[i]->buf);
            free(ruleset->rules[i]);
        }

        free(ruleset->rules);
        free(ruleset->actions);
    }

    ruleset->rules=NULL;
    ruleset->actions=NULL;
}

void free_ruleset_except_actions(ruleset_t *ruleset) {
    if(ruleset->rules) {
        for(size_t i=0; i<ruleset->rules_size; ++i) {
            free(ruleset->rules[i]->buf);
            free(ruleset->rules[i]);
        }

        free(ruleset->rules);
    }

    ruleset->rules=NULL;
}

static int parse_range(const char *s, uint32_t *out) {
    if(sscanf(s, "%u-%u", out, out+1)==2)
        return 0;

    if(sscanf(s, "%u", out)==1) {
        switch(*out) {
        case 0:
            out[1]=UINT16_MAX;
            break;
        default:
            out[1]=*out;
            break;
        }

        return 0;
    }

    return 1;
}
