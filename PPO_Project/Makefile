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

.PHONY: all test clean

all: $(BUILD)/test_runner

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_runner: $(TEST_SRC) $(CORE_SRC) $(HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(TEST_SRC) $(CORE_SRC) -o $@ $(LDLIBS)

test: $(BUILD)/test_runner
	./$(BUILD)/test_runner $(T)

clean:
	rm -rf $(BUILD)
