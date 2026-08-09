CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -Werror -Wconversion
LDLIBS = -lm

CORE = core/rs_proto.c core/rs_ring.c core/rs_sha256.c

all: test device-sim

build:
	mkdir -p build

build/test_core: tests/test_core.c $(CORE) | build
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test: build/test_core
	./build/test_core

device-sim: build/device-sim
build/device-sim: ports/posix/sim_main.c $(CORE) | build
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -rf build

.PHONY: all test device-sim clean
