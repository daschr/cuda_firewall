CFLAGS=$(shell pkg-config --cflags libdpdk)
CLIBS=$(shell pkg-config --libs libdpdk)
NVCC_FLAGS=--gpu-architecture compute_75

firewall: rte_bv rte_table_bv
	nvcc $(NVCC_FLAGS) -forward-unknown-to-host-compiler -O3 -mssse3 -march=native -Wall -Werror  cross-execution-space-call,default-stream-launch,ext-lambda-captures-this,reorder  $(CFLAGS) -o firewall firewall.c rte_table_bv.o rte_bv.o parser.c misc.c $(CLIBS) 

rte_bv:
	nvcc $(NVCC_FLAGS) -forward-unknown-to-host-compiler -O3 -march=native -Wall -pedantic -Werror all-warnings $(CFLAGS) -O3 -c rte_bv.c

rte_table_bv:
	nvcc $(NVCC_FLAGS) -forward-unknown-to-host-compiler -O3 -march=native -Wall -Werror all-warnings $(CFLAGS) -O3 -c rte_table_bv.cu

unittest_rte_bv: rte_bv
	nvcc $(NVCC_FLAGS) -forward-unknown-to-host-compiler -O3 -march=native -Wall -Werror all-warnings  $(CFLAGS) -o unittest_rte_bv unittest_rte_bv.c rte_bv.o $(CLIBS)

clean:
	rm *.o firewall

all: firewall
