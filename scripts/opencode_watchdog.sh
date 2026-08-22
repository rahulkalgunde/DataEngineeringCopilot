#!/usr/bin/env bash
# opencode_watchdog.sh — belt-and-suspenders companion to .opencode/plugin/auto-resume.ts
#
# The plugin (preferred mechanism) reacts to session.error bus events inside
# opencode. This watchdog covers the residual cases from OUTSIDE: it tails the
# opencode log and, when a retryable provider failure is followed by an idle
# window, sends one nudge keystroke ("?") into the tmux window running the
# opencode TUI.
#
# Guards:
#   - fires ONLY on the specific "stream error ... Service Unavailable" signature
#   - requires IDLE_SECONDS of log silence afterwards (run really ended)
#   - refuses to type while a permission prompt is visible in the pane
#   - cooldown between nudges; hourly cap; never loops on a hard-down endpoint
#
# Usage:
#   tmux new-session -d -s opencode 'opencode'          # run TUI in tmux
#   scripts/opencode_watchdog.sh opencode               # watch that session
#
# Env overrides: SESSION (positional), IDLE_SECONDS=45, COOLDOWN_SECONDS=120,
# MAX_PER_HOUR=10.

set -u

SESSION="${1:-${SESSION:-opencode}}"
IDLE_SECONDS="${IDLE_SECONDS:-45}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-120}"
MAX_PER_HOUR="${MAX_PER_HOUR:-10}"
LOG="${OPENCODE_LOG:-$HOME/.local/share/opencode/log/opencode.log}"

last_nudge=0
hour_stamps=()

now_ms() { date +%s%3N; }
log() { echo "[watchdog $(date +%H:%M:%S)] $*"; }

pane_has_permission_prompt() {
  # Any of these markers in the visible pane means a human must decide.
  local pane
  pane=$(tmux capture-pane -p -t "$SESSION" 2>/dev/null) || return 0
  echo "$pane" | grep -qiE "allow|deny|permission|trust this folder|y/n|yes/no" && return 0
  return 1
}

send_nudge() {
  local reason="$1"
  local now; now=$(now_ms)
  # hourly cap
  local fresh=()
  local ts
  for ts in "${hour_stamps[@]:-}"; do
    [[ -n "$ts" ]] && (( now - ts <= 3600000 )) && fresh+=("$ts")
  done
  hour_stamps=("${fresh[@]:-}")
  if (( ${#hour_stamps[@]} >= MAX_PER_HOUR )); then
    log "hourly cap ${MAX_PER_HOUR} reached; not nudging ($reason)"
    return 1
  fi
  if (( now - last_nudge < COOLDOWN_SECONDS * 1000 )); then
    return 1
  fi
  if pane_has_permission_prompt; then
    log "permission prompt visible in pane; NOT typing"
    return 1
  fi
  last_nudge=$now
  hour_stamps+=("$now")
  tmux send-keys -t "$SESSION" "?" Enter
  log "nudged '$SESSION' ($reason)"
}

log "watching tmux session '$SESSION' for provider stream failures (log: $LOG)"
log "idle=${IDLE_SECONDS}s cooldown=${COOLDOWN_SECONDS}s cap=${MAX_PER_HOUR}/h"

while true; do
  # Newest failure timestamp in the log.
  err_epoch=$(grep "stream error" "$LOG" 2>/dev/null | tail -1 | cut -d'T' -f2 | cut -d'.' -f1)
  if [[ -n "$err_epoch" ]]; then
    err_ts=$(date -d "$err_epoch" +%s 2>/dev/null || echo 0)
    log_mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
    now_epoch=$(date +%s)
    # Failure is recent AND the log has been silent since => run died.
    if (( err_ts > 0 )) \
       && (( now_epoch - err_ts < 600 )) \
       && (( log_mtime > 0 )) \
       && (( now_epoch - log_mtime >= IDLE_SECONDS )); then
      send_nudge "stream error at ${err_epoch} + ${IDLE_SECONDS}s silence"
      # consume the failure so we don't re-fire on it
      sleep 2
    fi
  fi
  sleep 5
done
