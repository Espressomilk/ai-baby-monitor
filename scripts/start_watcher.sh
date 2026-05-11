#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper around `uv run scripts/run_watcher.py`.
# Handles config selection, motion-gating tuning, and optional detached mode.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PIDFILE="$PROJECT_ROOT/.watcher.pid"
LOGFILE="$PROJECT_ROOT/.watcher.log"

# --- defaults (override via flags) -------------------------------------------
CONFIG="bedroom"
QUIET=0
DETACH=0
NO_MOTION_GATE=0
MOTION_THRESHOLD=2.0
MOTION_COOLDOWN=5.0
IDLE_SLEEP=1.0
# No default mask. Pass --motion-ignore X0,Y0,X1,Y1 to add one (e.g. to mask
# a top-right camera-overlay timestamp: --motion-ignore 0.75,0,1.0,0.10).
MOTION_IGNORE_DEFAULT=()
MOTION_IGNORE=()
# Audio gating
AUDIO=0
AUDIO_THRESHOLD=800
AUDIO_COOLDOWN=8.0
AUDIO_SUSTAINED=2
EXTRA_ARGS=()
# -----------------------------------------------------------------------------

usage() {
  cat <<'EOF'
Usage:
  ./scripts/start_watcher.sh [config] [options] [-- extra-uv-args...]

Positional:
  config                       Room config name in configs/<config>.yaml.
                               Default: bedroom

Watcher tuning:
  --motion-threshold FLOAT     Mean abs pixel diff (160x90 grayscale) above
                               which a frame batch counts as "motion".
                               Lower = more sensitive. Default: 2.0
  --motion-cooldown FLOAT      Keep analyzing this many seconds after the
                               last detected motion. Default: 5.0
  --idle-sleep FLOAT           Sleep between motion checks when idle, seconds.
                               Default: 1.0
  --no-motion-gate             Disable motion gating; run LLM on every batch.
  --motion-ignore X0,Y0,X1,Y1  Mask a region from motion detection (normalized
                               [0,1] coords). Repeat for multiple regions.
                               No regions are masked by default.
                               E.g. top-right timestamp: --motion-ignore 0.75,0,1,0.1

Audio gating (off by default; consumes RMS values pushed to Redis by the
streamer when it is run with --audio):
  --audio                      Enable audio gating. Triggers VLM analysis AND
                               an immediate alert sound on sustained loud noise
                               (cry, scream, etc.). Requires the streamer
                               container to be running with --audio so the
                               <room>:audio_rms Redis stream is populated.
  --audio-threshold INT        Int16-RMS loudness threshold per 0.25s window.
                               Default: 800. Tune by watching audio_rms in
                               logs while quiet vs during a real cry.
  --audio-cooldown FLOAT       Seconds to keep VLM analysis active after the
                               last loud event. Default: 8.0
  --audio-sustained N          How many consecutive loud 0.25s windows count
                               as a real event (filters claps/door-slams).
                               Default: 2 (= 0.5s of sustained loudness)

Runtime:
  -q, --quiet                  Suppress INFO/WARNING logs (errors + alerts only).
  -d, --detach                 Run in background; pidfile -> .watcher.pid,
                               logs -> .watcher.log. (Implies --quiet unless
                               --no-quiet was already supplied.)
  -h, --help                   Show this help.

Examples:
  ./scripts/start_watcher.sh
  ./scripts/start_watcher.sh bedroom -q
  ./scripts/start_watcher.sh bedroom --motion-threshold 1.5 --motion-cooldown 8
  ./scripts/start_watcher.sh living_room -d
  ./scripts/start_watcher.sh -- --foo --bar    # forward extra args to run_watcher.py

Stop a detached watcher:
  kill "$(cat .watcher.pid)"

Tuning tip:
  Run in foreground first and watch motion_score values in the logs. Pick a
  --motion-threshold between the quiet-room baseline and real-motion peaks.
EOF
}

# --- parse args --------------------------------------------------------------
while (( $# )); do
  case "$1" in
    -h|--help)              usage; exit 0 ;;
    -q|--quiet)             QUIET=1 ;;
    -d|--detach)            DETACH=1 ;;
    --no-motion-gate)       NO_MOTION_GATE=1 ;;
    --motion-threshold)     MOTION_THRESHOLD="$2"; shift ;;
    --motion-threshold=*)   MOTION_THRESHOLD="${1#*=}" ;;
    --motion-cooldown)      MOTION_COOLDOWN="$2"; shift ;;
    --motion-cooldown=*)    MOTION_COOLDOWN="${1#*=}" ;;
    --idle-sleep)           IDLE_SLEEP="$2"; shift ;;
    --idle-sleep=*)         IDLE_SLEEP="${1#*=}" ;;
    --motion-ignore)        MOTION_IGNORE+=("$2"); shift ;;
    --motion-ignore=*)      MOTION_IGNORE+=("${1#*=}") ;;
    --audio)                AUDIO=1 ;;
    --audio-threshold)      AUDIO_THRESHOLD="$2"; shift ;;
    --audio-threshold=*)    AUDIO_THRESHOLD="${1#*=}" ;;
    --audio-cooldown)       AUDIO_COOLDOWN="$2"; shift ;;
    --audio-cooldown=*)     AUDIO_COOLDOWN="${1#*=}" ;;
    --audio-sustained)      AUDIO_SUSTAINED="$2"; shift ;;
    --audio-sustained=*)    AUDIO_SUSTAINED="${1#*=}" ;;
    --)                     shift; EXTRA_ARGS+=("$@"); break ;;
    -*)                     echo "Unknown flag: $1 (try --help)" >&2; exit 1 ;;
    *)                      CONFIG="$1" ;;
  esac
  shift
done
# -----------------------------------------------------------------------------

CONFIG_PATH="configs/${CONFIG}.yaml"
[[ -f "$CONFIG_PATH" ]] || { echo "Config not found: $CONFIG_PATH" >&2; exit 1; }

# Build uv args.
UV_ARGS=(run scripts/run_watcher.py --config-file "$CONFIG_PATH")
if (( NO_MOTION_GATE )); then
  UV_ARGS+=(--no-motion-gate)
else
  UV_ARGS+=(
    --motion-threshold "$MOTION_THRESHOLD"
    --motion-cooldown  "$MOTION_COOLDOWN"
    --idle-sleep       "$IDLE_SLEEP"
  )
  # Apply user-supplied --motion-ignore regions (none by default).
  regions_to_use=("${MOTION_IGNORE[@]:-${MOTION_IGNORE_DEFAULT[@]:-}}")
  for region in "${regions_to_use[@]}"; do
    [[ -n "$region" ]] && UV_ARGS+=(--motion-ignore "$region")
  done
fi
if (( AUDIO )); then
  UV_ARGS+=(
    --audio
    --audio-threshold "$AUDIO_THRESHOLD"
    --audio-cooldown  "$AUDIO_COOLDOWN"
    --audio-sustained-windows "$AUDIO_SUSTAINED"
  )
fi
# Detached mode is always quiet unless explicitly disabled (we don't expose
# --no-quiet, but `--` extra args could re-add INFO logging if the user wants).
if (( QUIET )) || (( DETACH )); then
  UV_ARGS+=(-q)
fi
if (( ${#EXTRA_ARGS[@]} )); then
  UV_ARGS+=("${EXTRA_ARGS[@]}")
fi

if (( DETACH )); then
  # Refuse to clobber a running watcher.
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Watcher already running (pid $(cat "$PIDFILE"))" >&2
    exit 1
  fi
  echo "==> Starting watcher in background: uv ${UV_ARGS[*]}"
  nohup uv "${UV_ARGS[@]}" >"$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  sleep 1
  if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "==> Watcher started (pid $(cat "$PIDFILE")). Tail: tail -f $LOGFILE"
  else
    echo "Watcher failed to start. See $LOGFILE" >&2
    rm -f "$PIDFILE"
    exit 1
  fi
else
  echo "==> Running watcher: uv ${UV_ARGS[*]}"
  exec uv "${UV_ARGS[@]}"
fi
