# Host simulator. For target builds use your MCU toolchain and drop sim_main.c.
CFLAGS = -O2 -std=c11 -Wall -Wextra -ffast-math
sim: edge_gate.c edge_policy.c sim_main.c
	$(CC) $(CFLAGS) $^ -o $@ -lm
# Footprint check (arm-none-eabi-gcc if available)
size: edge_gate.c edge_policy.c
	$(CC) $(CFLAGS) -c edge_gate.c -o /tmp/g.o && $(CC) $(CFLAGS) -c edge_policy.c -o /tmp/p.o && size -t /tmp/g.o /tmp/p.o
clean:
	rm -f sim *.o
