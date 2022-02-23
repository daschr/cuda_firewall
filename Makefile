CFLAGS=$(shell pkg-config --cflags libdpdk)
CLIBS=$(shell pkg-config --libs libdpdk)
NVCC_FLAGS=--gpu-architecture sm_86

all:
	nvcc $(NVCC_FLAGS) -forward-unknown-to-host-compiler -O3 -mssse3 -march=native -Wall -Werror  cross-execution-space-call,default-stream-launch,ext-lambda-captures-this,reorder $(CFLAGS) $(CFLAGS) -o firewall firewall.c parser.c misc.c $(CLIBS) 

clean:
	rm firewall
