#!/bin/sh
# Build, test, smoke-train, train, select, and archive one seed. Runs the same way on this Mac
# and inside a Modal container, from the PPO_Project directory.
#
# usage: scripts/container_train.sh RUN_DIR SEED TOTAL_DECISIONS MAX_SECONDS [extra train flags...]
# Writes RUN_DIR (logs, manifest, metrics, checkpoints, selection) and RUN_DIR.tar.gz.
set -eu

if [ $# -lt 4 ]; then
    echo "usage: $0 RUN_DIR SEED TOTAL_DECISIONS MAX_SECONDS [extra train flags...]" >&2
    exit 2
fi
run_dir=$1 seed=$2 decisions=$3 seconds=$4
shift 4

rm -rf "$run_dir" "$run_dir.tar.gz"
mkdir -p "$run_dir"
{
    uname -a
    cc --version | head -1
    echo "cpus: $(getconf _NPROCESSORS_ONLN)"
    echo "git commit: ${GIT_COMMIT:-from make}"
} > "$run_dir/host.txt"

# Compiler warnings are logged, not fatal, so a different compiler cannot block a paid run.
make clean > /dev/null
make WERROR= ${GIT_COMMIT:+GIT_COMMIT=$GIT_COMMIT} build/test_runner build/ppo > "$run_dir/build.log" 2>&1
./build/test_runner > "$run_dir/test.log" 2>&1 || { tail -20 "$run_dir/test.log" >&2; exit 1; }
tail -1 "$run_dir/test.log"
make WERROR= smoke > "$run_dir/smoke.log" 2>&1 || { tail -20 "$run_dir/smoke.log" >&2; exit 1; }
echo "smoke: ok"

./build/ppo train --run-dir "$run_dir" --seed "$seed" --total-decisions "$decisions" \
    --max-seconds "$seconds" "$@"
./build/ppo select --run-dir "$run_dir" | tee "$run_dir/select.log"
grep -E '"status"|"updates"|"decisions"|"seconds"' "$run_dir/manifest.json" | tr -d ' ,'

tar -czf "$run_dir.tar.gz" -C "$(dirname "$run_dir")" "$(basename "$run_dir")"
echo "archive: $run_dir.tar.gz ($(du -h "$run_dir.tar.gz" | cut -f1))"
