#!/usr/bin/env bash
set -euo pipefail

# Manages the docker stack only (redis, vllm, streamlit_viewer, stream_to_redis).
# The host-side watcher is managed by scripts/start_watcher.sh -- this script
# never invokes uv directly, but `status` and `stop` will surface / clean up
# a detached watcher (.watcher.pid) if one is running.
#
# Usage:
#   ./scripts/service.sh start
#   ./scripts/service.sh stop                # docker down; also kills detached watcher if any
#   ./scripts/service.sh restart
#   ./scripts/service.sh status              # docker services + watcher state + GPU
#   ./scripts/service.sh logs [service]      # default: all docker services;
#                                            #   "watcher" tails .watcher.log
#
# To start the watcher, use:
#   ./scripts/start_watcher.sh --help

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Watcher pidfile/log live here so we can read them for status/stop/logs.
# They are written by scripts/start_watcher.sh, not by this script.
PIDFILE="$PROJECT_ROOT/.watcher.pid"
LOGFILE="$PROJECT_ROOT/.watcher.log"

_watcher_pid() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid=$(cat "$PIDFILE")
  kill -0 "$pid" 2>/dev/null && echo "$pid"
}

_vllm_healthy() {
  [[ "$(docker inspect -f '{{.State.Health.Status}}' ai-baby-monitor-vllm-1 2>/dev/null)" == "healthy" ]]
}

cmd_start() {
  if (( $# )); then
    echo "Note: 'start' takes no arguments now. To start the watcher, use ./scripts/start_watcher.sh"
    echo "      Continuing with docker stack..."
  fi
  echo "==> Bringing up docker stack..."
  docker compose up -d

  echo "==> Waiting for vLLM to become healthy (this can take a few minutes)..."
  local waited=0
  until _vllm_healthy; do
    sleep 5
    waited=$((waited + 5))
    if (( waited % 30 == 0 )); then
      echo "    still waiting... ${waited}s"
    fi
    if (( waited >= 1800 )); then
      echo "vLLM did not become healthy within 30 minutes. Check 'docker logs ai-baby-monitor-vllm-1'." >&2
      exit 1
    fi
  done
  echo "==> vLLM is healthy. Stack is up."
  echo "==> To start the watcher: ./scripts/start_watcher.sh --help"
}

cmd_stop() {
  local pid
  if pid=$(_watcher_pid 2>/dev/null) && [[ -n "$pid" ]]; then
    echo "==> Stopping watcher (pid $pid)"
    kill "$pid" || true
    sleep 1
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" || true
    rm -f "$PIDFILE"
  else
    echo "==> Watcher not running"
    rm -f "$PIDFILE" 2>/dev/null || true
  fi

  echo "==> Bringing down docker stack..."
  docker compose down
}

cmd_restart() {
  cmd_stop
  cmd_start
}

cmd_status() {
  # Colors (only when output is a terminal)
  if [[ -t 1 ]]; then
    local C_RESET="\033[0m" C_BOLD="\033[1m" C_DIM="\033[2m"
    local C_GREEN="\033[32m" C_RED="\033[31m" C_YELLOW="\033[33m" C_CYAN="\033[36m"
  else
    local C_RESET="" C_BOLD="" C_DIM="" C_GREEN="" C_RED="" C_YELLOW="" C_CYAN=""
  fi

  _row() {
    # name, state, detail
    local glyph color
    case "$2" in
      up|healthy|running) glyph="●"; color="$C_GREEN" ;;
      starting|restarting) glyph="◐"; color="$C_YELLOW" ;;
      *) glyph="○"; color="$C_RED" ;;
    esac
    printf "  ${color}%s${C_RESET} ${C_BOLD}%-20s${C_RESET} ${color}%-12s${C_RESET} ${C_DIM}%s${C_RESET}\n" \
      "$glyph" "$1" "$2" "$3"
  }

  echo
  echo -e "${C_BOLD}${C_CYAN}Baby Monitor Service Status${C_RESET}"
  echo -e "${C_DIM}$(date '+%Y-%m-%d %H:%M:%S')${C_RESET}"
  echo

  echo -e "${C_BOLD}Docker services${C_RESET}"
  local services
  services=$(docker compose config --services 2>/dev/null || echo "")
  if [[ -z "$services" ]]; then
    echo -e "  ${C_DIM}(no compose project found)${C_RESET}"
  else
    for svc in $services; do
      local container_id state health detail=""
      container_id=$(docker compose ps -q "$svc" 2>/dev/null || true)
      if [[ -z "$container_id" ]]; then
        _row "$svc" "stopped" ""
        continue
      fi
      state=$(docker inspect -f '{{.State.Status}}' "$container_id" 2>/dev/null || echo "unknown")
      health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$container_id" 2>/dev/null || echo "")
      local display_state="$state"
      [[ -n "$health" ]] && display_state="$health"
      # Uptime
      local started
      started=$(docker inspect -f '{{.State.StartedAt}}' "$container_id" 2>/dev/null || echo "")
      if [[ -n "$started" ]]; then
        local elapsed
        elapsed=$(( $(date +%s) - $(date -d "$started" +%s 2>/dev/null || echo 0) ))
        if (( elapsed > 0 )); then
          if   (( elapsed < 60   )); then detail="up ${elapsed}s"
          elif (( elapsed < 3600 )); then detail="up $(( elapsed / 60 ))m"
          elif (( elapsed < 86400)); then detail="up $(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m"
          else                            detail="up $(( elapsed / 86400 ))d $(( (elapsed % 86400) / 3600 ))h"
          fi
        fi
      fi
      _row "$svc" "$display_state" "$detail"
    done
  fi

  echo
  echo -e "${C_BOLD}Host watcher${C_RESET}"
  local pid
  if pid=$(_watcher_pid); then
    local started_epoch elapsed detail
    started_epoch=$(stat -c %Y "$PIDFILE" 2>/dev/null || echo 0)
    elapsed=$(( $(date +%s) - started_epoch ))
    if   (( elapsed < 60   )); then detail="up ${elapsed}s, pid $pid"
    elif (( elapsed < 3600 )); then detail="up $(( elapsed / 60 ))m, pid $pid"
    elif (( elapsed < 86400)); then detail="up $(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m, pid $pid"
    else                            detail="up $(( elapsed / 86400 ))d, pid $pid"
    fi
    _row "watcher" "running" "$detail"
    if [[ -f "$LOGFILE" ]]; then
      local size_h
      size_h=$(du -h "$LOGFILE" 2>/dev/null | cut -f1)
      echo -e "  ${C_DIM}log: $LOGFILE (${size_h})${C_RESET}"
    fi
  else
    _row "watcher" "stopped" ""
  fi

  # GPU power if available
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_line
    gpu_line=$(nvidia-smi --query-gpu=name,power.draw,utilization.gpu,memory.used,memory.total \
                          --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ -n "$gpu_line" ]]; then
      IFS=',' read -r gname gpower gutil gmem_used gmem_total <<<"$gpu_line"
      echo
      echo -e "${C_BOLD}GPU${C_RESET}"
      printf "  ${C_DIM}%s${C_RESET}  ${C_BOLD}%sW${C_RESET}  util ${C_BOLD}%s%%${C_RESET}  vram ${C_BOLD}%s/%s MiB${C_RESET}\n" \
        "$(echo "$gname" | xargs)" "$(echo "$gpower" | xargs)" \
        "$(echo "$gutil" | xargs)" "$(echo "$gmem_used" | xargs)" "$(echo "$gmem_total" | xargs)"
    fi
  fi

  echo
}

cmd_logs() {
  local target="${1:-}"
  if [[ "$target" == "watcher" ]]; then
    [[ -f "$LOGFILE" ]] || { echo "No watcher log at $LOGFILE" >&2; exit 1; }
    tail -f "$LOGFILE"
  elif [[ -n "$target" ]]; then
    docker compose logs -f "$target"
  else
    docker compose logs -f
  fi
}

case "${1:-}" in
  start)   shift; cmd_start "$@" ;;
  stop)    cmd_stop ;;
  restart) cmd_restart ;;
  status)  cmd_status ;;
  logs)    shift; cmd_logs "$@" ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs [service|watcher]}" >&2
    echo "       Watcher is managed separately via ./scripts/start_watcher.sh" >&2
    exit 1
    ;;
esac
