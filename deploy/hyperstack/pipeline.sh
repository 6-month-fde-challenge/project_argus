#!/usr/bin/env bash
# Run ON the Hyperstack VM. The whole model side in ONE command, unattended (tmux session "pipeline", survives SSH):
#   setup -> health check -> train the profile's runs -> reports -> serve the result -> API scenarios + load test
#   -> training/outputs/_deploy/aws.env (the settings `python deploy/ops.py up` needs on your PC)
#
#   bash deploy/hyperstack/pipeline.sh                    # PROFILE=demo: demo_hybrid, then demo_llm_scorer (reuses it)
#   PROFILE=prod  bash deploy/hyperstack/pipeline.sh      # prod_a100 on one A100 80GB (HF_TOKEN for gated models)
#   PROFILE=smoke bash deploy/hyperstack/pipeline.sh      # every stage, tiny sizes: is the VM healthy? (~20 min)
#   SKIP_TRAIN=1  bash deploy/hyperstack/pipeline.sh      # VM restarted: models already trained, just serve again
#   SERVE_RUN=<run> ...                                   # serve another run than the profile's default
#
# Follow: tail -f training/outputs/_logs/pipeline.log     Result: training/outputs/_deploy/aws.env
# Hyperstack firewall: inbound TCP 22 and 8080.
set -uo pipefail
source "$(dirname "$0")/_common.sh"
PROFILE="${PROFILE:-demo}"
LOG="$LOGS/pipeline.log"

if [ "${ARGUS_PIPELINE_INSIDE:-0}" != "1" ]; then
  command -v tmux >/dev/null || { sudo apt-get update -y && sudo apt-get install -y tmux; }
  tmux has-session -t pipeline 2>/dev/null && die "the pipeline is already running: tail -f $LOG  (stop: tmux kill-session -t pipeline)"
  : > "$LOG"
  tmux new-session -d -s pipeline "ARGUS_PIPELINE_INSIDE=1 PROFILE=$PROFILE SKIP_TRAIN=${SKIP_TRAIN:-0} SERVE_RUN=${SERVE_RUN:-} bash deploy/hyperstack/pipeline.sh"
  echo "Pipeline ($PROFILE) started in tmux session 'pipeline' - safe to disconnect."
  echo "  follow : tail -f $LOG"
  echo "  done   : the log ends with 'PIPELINE DONE'; settings for your PC in $DEPLOY_OUT/aws.env"
  exit 0
fi

exec > >(tee -a "$LOG") 2>&1
fail() { echo "PIPELINE FAILED at: $*"; exit 1; }
SERVE="${SERVE_RUN:-$(serve_run_for_profile "$PROFILE")}"

step "1/6 setup (Docker, NVIDIA toolkit, .env secrets, images, data, model weights)"
PROFILE="$PROFILE" bash deploy/hyperstack/setup_vm.sh || fail "setup_vm.sh (is .env filled with SUPABASE_URL / SUPABASE_KEY?)"
source deploy/hyperstack/_common.sh  # docker may only work without sudo after setup

step "2/6 health check"
bash deploy/hyperstack/check.sh || echo "(problems above; services not running yet is expected)"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  compose --profile serve stop scoring explainer 2>/dev/null || true   # one GPU owner at a time
  step "3/6 train: $(runs_for_profile "$PROFILE")"
  for cfg in $(runs_for_profile "$PROFILE"); do
    echo "--- configs/$cfg.yaml"
    compose run --rm -T trainer python -m argus_ft.pipeline --config "configs/$cfg.yaml" --no-sync || fail "training $cfg"
  done
  step "4/6 reports"
  runs=$(for cfg in $(runs_for_profile "$PROFILE"); do grep -E '^run_name:' "training/configs/$cfg.yaml" | awk '{print $2}'; done)
  compose run --rm --no-deps -T trainer python -m argus_ft.compare $runs || echo "(report failed; the runs are fine)"
else
  step "3-4/6 training skipped (SKIP_TRAIN=1)"
fi

step "5/6 serve $SERVE (Fraud Model API :$SCORING_PORT, vLLM explainer, UI)"
bash deploy/hyperstack/serve.sh "$SERVE" || fail "serve.sh $SERVE"

step "6/6 API scenarios (CRD 10 / LLD 6) and a short load test (OBJ-10)"
# both clients run INSIDE the scoring container, so they use its internal port 8080 whatever ARGUS_SCORING_PORT is
SCORING_CFG="$(env_get SCORING_CONFIG)"; SCORING_CFG="${SCORING_CFG:-configs/local_demo.yaml}"
if [ "$SCORING_CFG" = "configs/local_demo.yaml" ]; then  # the scenarios replay 2025 transactions: no +-15 min window
  compose --profile serve exec -T scoring python -m argus_api.demo --config "$SCORING_CFG" --run "$SERVE" \
    --url http://localhost:8080 >/dev/null && echo "scenarios -> training/outputs/$SERVE/api/demo_results.json" \
    || echo "(demo scenarios reported a problem: see the scoring logs)"
else
  echo "(CRD scenarios skipped: $SCORING_CFG enforces the +-15 min transaction_time window)"
fi
compose --profile serve exec -T scoring python -m argus_api.loadtest --config "$SCORING_CFG" --run "$SERVE" \
  --url http://localhost:8080 --rps 5 --duration 30 --fresh-time | tail -25 || echo "(load test failed)"

IP="${PUBLIC_IP:-$(public_ip)}"
case "$IP" in
  ""|"<"*) die "could not find this VM's public IP (api.ipify.org): rerun with PUBLIC_IP=<the VM's IP> to write aws.env" ;;
esac
printf 'TF_VAR_scoring_host=%s\nTF_VAR_scoring_port=%s\nTF_VAR_scoring_token=%s\n' "$IP" "$SCORING_PORT" \
  "$(env_get ARGUS_SCORING_TOKEN)" > "$DEPLOY_OUT/aws.env"
chmod 600 "$DEPLOY_OUT/aws.env"
echo
echo "PIPELINE DONE. $SERVE is served on http://$IP:$SCORING_PORT and restarts by itself if it crashes."
echo "  results : training/outputs/$SERVE/report.md, training/outputs/$SERVE/api/{demo_results,loadtest}.json"
echo "  AWS     : copy the 3 lines of $DEPLOY_OUT/aws.env into .env on your PC, then: python deploy/ops.py up"
echo "            (or from the PC: bash deploy/hyperstack/remote.sh awsenv >> .env)"
