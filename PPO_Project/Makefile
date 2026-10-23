CC ?= cc
WERROR ?= -Werror
OPT ?= -O2
CFLAGS ?= -std=c11 $(OPT) -Wall -Wextra -Wpedantic $(WERROR)
CPPFLAGS += -Isrc
LDLIBS += -lm -lpthread
# cJSON is optional: it only enables the legacy baseline (reading upstream Game_Data tables).
CJSON_LIBS := $(shell pkg-config --libs libcjson 2>/dev/null)
ifneq ($(CJSON_LIBS),)
CPPFLAGS += -DHAVE_CJSON $(shell pkg-config --cflags libcjson)
LDLIBS += $(CJSON_LIBS)
endif
GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null || echo unknown)$(shell test -n "$$(git status --porcelain -- . 2>/dev/null)" && echo -dirty)

BUILD := build
HEADERS := $(wildcard src/*.h tests/*.h tests/*.def)
CORE_SRC := $(filter-out src/main.c src/viewer.c,$(wildcard src/*.c))
TEST_SRC := $(wildcard tests/*.c)

.PHONY: all test smoke sanitize analyze clean

all: $(BUILD)/test_runner $(BUILD)/ppo

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_runner: $(TEST_SRC) $(CORE_SRC) $(HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(TEST_SRC) $(CORE_SRC) -o $@ $(LDLIBS)

$(BUILD)/ppo: src/main.c $(CORE_SRC) $(HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -DPPO_GIT_COMMIT='"$(GIT_COMMIT)"' $(CFLAGS) src/main.c $(CORE_SRC) -o $@ $(LDLIBS)

test: $(BUILD)/test_runner
	./$(BUILD)/test_runner $(T)

# Two tiny updates through the real command, with 1 s episodes so some finish.
smoke: $(BUILD)/ppo
	rm -rf runs/smoke
	./$(BUILD)/ppo train --run-dir runs/smoke --updates 2 --envs 4 --steps 32 --hidden 32 --episode-seconds 1
	head -c 400 runs/smoke/metrics.csv

# AddressSanitizer + UndefinedBehaviorSanitizer over the tests and one million random decisions.
SAN_FLAGS := -std=c11 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	-Wall -Wextra -Wpedantic $(WERROR)
sanitize: | $(BUILD)
	mkdir -p $(BUILD)/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(TEST_SRC) $(CORE_SRC) -o $(BUILD)/sanitize/test_runner $(LDLIBS)
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) src/main.c $(CORE_SRC) -o $(BUILD)/sanitize/ppo $(LDLIBS)
	./$(BUILD)/sanitize/test_runner
	./$(BUILD)/sanitize/ppo stress --decisions 1000000

# Clang static analyzer over every source and test file; fails on any finding.
analyze:
	@status=0; for f in $(CORE_SRC) src/main.c $(TEST_SRC); do \
		out=$$(clang --analyze $(CPPFLAGS) -std=c11 -Xanalyzer -analyzer-output=text $$f -o /dev/null 2>&1); \
		if printf '%s' "$$out" | grep -q 'warning:'; then printf '%s\n' "$$out"; status=1; fi; \
	done; \
	if [ $$status -eq 0 ]; then echo "analyze: 0 findings"; fi; exit $$status

clean:
	rm -rf $(BUILD)
