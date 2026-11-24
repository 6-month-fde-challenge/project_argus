#!/usr/bin/env bash
# Run ON the VM. Starts the ARGUS stack and serves a finished run:
#   UI :3000, platform :8090, MLflow :5000 (127.0.0.1 only: SSH tunnel)  +  Fraud Model API :8080 (public, bearer
#   token) with the vLLM explainer :8000 (127.0.0.1); defaults, ARGUS_*_PORT in .env changes them. Containers restart
#   by themselves (restart: unless-stopped).
#
#   bash deploy/hyperstack/serve.sh argus_prod_a100     # a run under training/outputs/
#   bash deploy/hyperstack/serve.sh                     # the newest run that has a servable scorer
#   bash deploy/hyperstack/serve.sh stop                # stop serving (frees the GPU for training)
#   FORCE=1 bash deploy/hyperstack/serve.sh <run>       # serve even while a training session (tmux 'train') runs
# Hyperstack firewall: inbound TCP 22 and 8080 only.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

if [ "${1:-}" = "stop" ]; then
  compose --profile serve stop scoring explainer
  exit 0
fi

servable() {  # a run the scoring API can activate (scoring_api/argus_api/deployments.py)
  local d="training/outputs/$1"
  [ -f "$d/pilot/selected.json" ] || return 1
  { [ -d "$d/scorer_student/fp8" ] && [ -f "$d/calibration/scorer_student_fp8.json" ]; } \
    || { [ -f "$d/xgboost/model.json" ] && [ -f "$d/calibration/xgboost.json" ]; }
}
RUN="${1:-}"
if [ -z "$RUN" ]; then
  # shellcheck disable=SC2045  # newest first by mtime; run names are [a-z0-9_-]
  for d in $(ls -1dt training/outputs/*/ 2>/dev/null); do
    n=$(basename "$d"); case "$n" in _*|mlruns) continue ;; esac
    servable "$n" && { RUN="$n"; break; }
  done
fi
[ -n "$RUN" ] || die "no finished run with a servable scorer in training/outputs (train first: deploy/hyperstack/train.sh)"
servable "$RUN" || die "run '$RUN' has no servable scorer (needs pilot, calibration and an FP8 student or XGBoost)"
SERVICES="scoring explainer"
if [ ! -d "training/outputs/$RUN/explainer_student/awq" ]; then
  # vLLM cannot start without the model: it would restart forever and hold GPU memory. Templates explain instead.
  info "run '$RUN' has no AWQ explainer: serving the scorer only, explanations come from templates"
  SERVICES="scoring"
fi
if tmux has-session -t train 2>/dev/null && [ "${FORCE:-0}" != "1" ]; then
  die "training is using the GPU (tmux 'train'): wait for it, or FORCE=1 to serve anyway (both may run out of memory)"
fi

# In .env, not the environment: compose reads .env itself (sudo would drop exported variables), and after a VM
# reboot the restarting containers keep serving the same run.
env_set RUN_NAME "$RUN"
env_set EXPLAINER_RUN "$RUN"
compose up -d
[ "$SERVICES" = "scoring" ] && compose --profile serve stop explainer 2>/dev/null || true
# The scorer first and on its own: payments need it. The explainer is optional (the API explains with templates
# without it), so a vLLM image that cannot be pulled or a model that does not fit never blocks scoring.
compose --profile serve up -d --force-recreate scoring
if [ "$SERVICES" != "scoring" ]; then
  compose --profile serve up -d --force-recreate explainer \
    || info "the vLLM explainer did not start (image download / GPU memory?): explanations come from templates; run serve.sh $RUN again to retry"
fi
HEALTH="http://localhost:$SCORING_PORT/v1/health"
echo -n "Loading $RUN on the GPU"
for _ in $(seq 1 180); do
  curl -fs --max-time 2 "$HEALTH" 2>/dev/null | grep -q '"scorer":"ok"' && break
  echo -n "."; sleep 2
done
echo
curl -fs "$HEALTH" | grep -q '"scorer":"ok"' \
  || { compose --profile serve logs --tail 40 scoring; die "the Fraud Model API did not come up with $RUN"; }

IP="${PUBLIC_IP:-$(public_ip)}"
echo "Serving $RUN -> $(curl -s "$HEALTH")"
echo "  Fraud Model API : http://$IP:$SCORING_PORT/docs   (Authorization: Bearer \$ARGUS_SCORING_TOKEN from .env)"
echo "  UI / platform   : bash deploy/hyperstack/remote.sh tunnel   (on your PC), then http://localhost:$UI_PORT"
echo "  Stop            : bash deploy/hyperstack/serve.sh stop"
