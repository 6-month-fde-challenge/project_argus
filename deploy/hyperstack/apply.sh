#!/usr/bin/env bash
# Run ON the VM. Apply changed settings (.env): recreate the given services so they pick up the new values.
# The UI's Settings page runs this through the operations agent after you save; by hand it is the same:
#
#   bash deploy/hyperstack/apply.sh scoring stream platform
#
# The core services (frontend, platform, mlflow) are always recreated. Serving and stream services (scoring,
# explainer, stream, kinesis) only when they run now: stopping serving to free the GPU for training stays a
# deliberate choice, and a stopped service reads the new .env when it next starts anyway.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
[ $# -gt 0 ] || die "usage: bash deploy/hyperstack/apply.sh <service>...  (frontend platform mlflow scoring explainer stream kinesis)"

running=" $(compose --profile serve --profile stream ps --services --status running 2>/dev/null | tr '\n' ' ') "
todo=()
for s in "$@"; do
  case "$s" in
    frontend|platform|mlflow) todo+=("$s") ;;
    scoring|explainer|stream|kinesis)
      if [[ "$running" == *" $s "* ]]; then todo+=("$s")
      else info "$s is not running: it reads the new settings when it next starts"; fi ;;
    *) die "unknown service '$s' (frontend platform mlflow scoring explainer stream kinesis)" ;;
  esac
done
[ ${#todo[@]} -gt 0 ] || { ok "nothing running needs a restart"; exit 0; }

# `up -d` recreates a container only when its configuration (environment, ports) changed. The platform restarts
# last: it is what the UI talks to, so the page reconnects after a few seconds.
others=(); platform=()
for s in "${todo[@]}"; do [ "$s" = "platform" ] && platform=(platform) || others+=("$s"); done
[ ${#others[@]} -eq 0 ] || compose --profile serve --profile stream up -d --no-deps "${others[@]}"
[ ${#platform[@]} -eq 0 ] || compose up -d --no-deps platform
ok "applied: ${todo[*]}"
compose --profile serve --profile stream ps --format 'table {{.Service}}\t{{.Status}}\t{{.Ports}}'
