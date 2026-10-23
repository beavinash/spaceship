CC ?= cc
WERROR ?= -Werror
OPT ?= -O2
CFLAGS ?= -std=c11 $(OPT) -Wall -Wextra -Wpedantic $(WERROR)
CPPFLAGS += -Isrc
LDLIBS += -lm -lpthread

BUILD := build
HEADERS := $(wildcard src/*.h tests/*.h tests/*.def)
CORE_SRC := $(filter-out src/main.c src/viewer.c,$(wildcard src/*.c))
TEST_SRC := $(wildcard tests/*.c)

.PHONY: all test smoke clean

all: $(BUILD)/test_runner $(BUILD)/ppo

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_runner: $(TEST_SRC) $(CORE_SRC) $(HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(TEST_SRC) $(CORE_SRC) -o $@ $(LDLIBS)

$(BUILD)/ppo: src/main.c $(CORE_SRC) $(HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) src/main.c $(CORE_SRC) -o $@ $(LDLIBS)

test: $(BUILD)/test_runner
	./$(BUILD)/test_runner $(T)

# Two tiny updates through the real command, with 1 s episodes so some finish.
smoke: $(BUILD)/ppo
	rm -rf runs/smoke
	./$(BUILD)/ppo train --run-dir runs/smoke --updates 2 --envs 4 --steps 32 --hidden 32 --episode-seconds 1
	head -c 400 runs/smoke/metrics.csv

clean:
	rm -rf $(BUILD)
