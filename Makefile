CFLAGS=$(shell pkg-config --cflags libdpdk)
CLIBS=$(shell pkg-config --libs libdpdk)

plain_fwd: plain_fwd
	nvcc -forward-unknown-to-host-compiler -O3 -mssse3 -march=native -Wall -Werror  cross-execution-space-call,default-stream-launch,ext-lambda-captures-this,reorder  $(CFLAGS) -o plain_fwd plain_fwd.c misc.c $(CLIBS) 

clean:
	rm *.o plain_fwd

all: plain_fwd
