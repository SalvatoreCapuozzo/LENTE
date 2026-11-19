#!/usr/bin/env bash
# Progress of the on-device suite (scripts/jetson_suite.sh).
#   bash scripts/jetson_progress.sh            # one snapshot
#   bash scripts/jetson_progress.sh --watch    # refresh every 15 s (Ctrl-C to leave; the suite keeps running)
cd "$(dirname "$0")/.."
OUT=${OUT:-runs/e6_edge_640/jetson}
show() {
  echo "== trashscribe-jetson unit: $(systemctl --user is-active trashscribe-jetson 2>/dev/null)   ($(date '+%F %T'))"
  if [ ! -f "$OUT/status.log" ]; then echo "(suite not started)"; return; fi
  # only the latest run of the suite
  cur_run=$(awk '/ suite start /{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$OUT/status.log" | grep -v "^|")
  echo "-- steps of the current run (full history: $OUT/status.log)"
  echo "$cur_run" | tail -n 25
  last=$(echo "$cur_run" | grep -vE "^$" | tail -1)
  cur=$(echo "$cur_run" | grep -E " START " | tail -1 | awk '{print $4}')
  if echo "$last" | grep -q " WAIT "; then
    echo "-- WAITING FOR YOU: $(echo "$last" | cut -d' ' -f4-)"
  elif [ -n "$cur" ] && ! echo "$cur_run" | grep -qE " (DONE|FAIL) +$cur( |$)"; then
    since=$(echo "$cur_run" | grep -E " START +$cur$" | tail -1 | cut -c1-19)
    echo "-- running: $cur since $since ($(( ($(date +%s) - $(date -d "$since" +%s)) / 60 )) min); last lines of $OUT/$cur.log:"
    tail -n 3 "$OUT/$cur.log" 2>/dev/null | cut -c1-200
  fi
  echo "-- clocks: GPU $(( $(cat /sys/class/devfreq/17000000.ga10b/cur_freq) / 1000000 )) MHz," \
       "locked=$([ "$(cat /sys/class/devfreq/17000000.ga10b/min_freq)" = "$(cat /sys/class/devfreq/17000000.ga10b/max_freq)" ] && echo yes || echo 'NO (sudo jetson_clocks)')" \
       "| memory available: $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MB"
  if [ -f "$OUT/tegrastats.log" ]; then echo "-- power log: $(wc -l < "$OUT/tegrastats.log") tegrastats samples"
  else echo "-- power log: none yet (sudo tegrastats --interval 500 --logfile $PWD/$OUT/tegrastats.log)"; fi
  echo "-- device now: $(timeout 3 tegrastats --interval 1000 2>/dev/null | head -1 | grep -oE 'RAM [0-9/]+MB|GR3D_FREQ [0-9]+%|GPU@[0-9.]+C' | tr '\n' ' ')"
  if [ -f "$OUT/summary.md" ]; then echo "-- summary: $OUT/summary.md"; fi
}
if [ "${1:-}" = "--watch" ]; then
  while true; do clear; show; sleep "${2:-15}"; done
else
  show
fi
