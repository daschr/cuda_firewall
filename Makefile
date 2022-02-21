CFLAGS=$(shell pkg-config --cflags libdpdk)
CLIBS=$(shell pkg-config --libs libdpdk)
NVCC_FLAGS=-Xptxas -v --gpu-architecture sm_86 -lineinfo

firewall: rte_bv rte_bv_classifier
		nvcc $(NVCC_FLAGS) -forward-unknown-to-host-compiler -O3 -mssse3 -march=native -Wall -Werror  cross-execution-space-call,default-stream-launch,ext-lambda-captures-this,reorder  $(CFLAGS) -o firewall firewall.c rte_bv_classifier.o rte_bv.o parser.c misc.c $(CLIBS) 

rte_bv:
		nvcc $(NVCC_FLAGS) -forward-unknown-to-host-compiler -O3 -march=native -Wall -pedantic -Werror all-warnings $(CFLAGS) -c rte_bv.c

rte_bv_classifier:
		nvcc $(NVCC_FLAGS) -forward-unknown-to-host-compiler -O3 -march=native -Wall -Werror all-warnings $(CFLAGS) -c rte_bv_classifier.cu

clean:
		rm *.o firewall

all: firewall

