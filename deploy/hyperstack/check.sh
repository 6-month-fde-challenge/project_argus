#!/usr/bin/env bash
# Run ON the VM any time: one-screen health check of the ARGUS deployment.
#   bash deploy/hyperstack/check.sh           # quick: host, .env, data, images, runs, services
#   bash deploy/hyperstack/check.sh --full    # + CUDA inside the training image, gated-model access, test suites
source "$(dirname "$0")/_common.sh"
FAILS=0
fail() { bad "$*"; FAILS=$((FAILS + 1)); }

echo "ARGUS VM check  ($(hostname), $(date '+%F %T'), $ROOT)"
echo "[host]"
if nvidia-smi >/dev/null 2>&1; then ok "GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | head -1)"
else fail "nvidia-smi failed (no GPU / driver)"; fi
FREE_GB=$(df -Pk . | awk 'NR==2{print int($4/1024/1024)}')
[ "$FREE_GB" -ge 20 ] && ok "disk free: ${FREE_GB} GB" || fail "disk free only ${FREE_GB} GB"
$DOCKER info >/dev/null 2>&1 && ok "docker: $($DOCKER --version | cut -d, -f1)" || fail "docker not running (sudo systemctl start docker)"
command -v tmux >/dev/null && ok "tmux installed" || fail "tmux missing (setup_vm.sh)"

echo "[.env]"
for k in SUPABASE_URL SUPABASE_KEY ARGUS_SCORING_TOKEN ARGUS_SCORING_ADMIN_TOKEN ARGUS_STREAM_CONTROL_KEY; do
  [ -n "$(env_get "$k")" ] && ok "$k set" || fail "$k missing (setup_vm.sh)"
done
case "$(env_get ARGUS_BIND)" in
  ""|127.0.0.1) ok "UI / platform / MLflow bound to 127.0.0.1 (SSH tunnel)" ;;
  *) info "ARGUS_BIND=$(env_get ARGUS_BIND): the UI and platform (no auth) listen publicly" ;;
esac
[ "$(env_get ARGUS_SCORING_BIND)" = "0.0.0.0" ] && ok "Fraud Model API reachable from AWS (ARGUS_SCORING_BIND=0.0.0.0)" \
  || info "ARGUS_SCORING_BIND is not 0.0.0.0: the AWS stream cannot reach the Fraud Model API"
info "SCORING_CONFIG=$(env_get SCORING_CONFIG)  HF_TOKEN $([ -n "$(env_get HF_TOKEN)" ] && echo set || echo unset)"

echo "[data + images]"
for s in train validation holdout; do
  n=$(ls training/data/raw/training/sft/$s/*.jsonl.gz 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && ok "sft/$s: $n files" || fail "sft/$s missing (setup_vm.sh step 5)"
done
for t in features labels transactions customers; do
  [ -f "training/data/db/$t.parquet" ] || fail "table snapshot $t.parquet missing"
done
for img in argus-training argus-scoring argus-platform argus-frontend argus-stream; do
  $DOCKER image inspect "$img:latest" >/dev/null 2>&1 && ok "image $img" || fail "image $img missing (setup_vm.sh step 4)"
done

echo "[runs]"
tmux has-session -t train 2>/dev/null && info "training RUNNING -> tail -f $LOGS/train.log" || info "no CLI training running"
tmux has-session -t pipeline 2>/dev/null && info "pipeline RUNNING -> tail -f $LOGS/pipeline.log"
# shellcheck disable=SC2045  # newest first by mtime; run names are [a-z0-9_-]
for d in $(ls -1dt training/outputs/*/ 2>/dev/null); do
  name=$(basename "$d"); case "$name" in _*|mlruns) continue ;; esac
  [ -f "${d}report.json" ] || continue
  python3 - "${d}report.json" "$name" <<'EOF' 2>/dev/null || info "$name: report.json unreadable"
import json, sys
r = json.load(open(sys.argv[1])); st = r.get("stages", {})
failed = [k for k, v in st.items() if v.get("status") == "failed"]
acc = st.get("acceptance", {})
print(f"  ..   {sys.argv[2]}: {len(st)} stages, " + (f"FAILED at {failed[0]}" if failed else "ok")
      + (f", holdout {acc.get('verdict', acc.get('status', '?'))}"
         + (f" ({', '.join(acc['failing'])})" if acc.get('failing') else "") if acc else ""))
EOF
done

echo "[services]"
probe() { curl -fs --max-time 3 "$2" >/dev/null 2>&1 && ok "$1 up ($2)" || info "$1 not running ($3)"; }
probe "platform" "http://localhost:$API_PORT/api/v1/health" "bash deploy/hyperstack/serve.sh"
probe "MLflow" "http://localhost:$MLFLOW_PORT/health" "bash deploy/hyperstack/serve.sh"
if curl -fs --max-time 3 "http://localhost:$SCORING_PORT/v1/health" -o /tmp/argus_health 2>/dev/null; then
  ok "Fraud Model API: $(python3 -c "import json;h=json.load(open('/tmp/argus_health'));print(h['status'],'|',h.get('active_run'),'|',h.get('model_version'))")"
else info "Fraud Model API not running (bash deploy/hyperstack/serve.sh <run>)"; fi
probe "vLLM explainer" "http://localhost:$EXPLAINER_PORT/health" "starts with serve.sh"
probe "stream" "http://localhost:$STREAM_PORT/live/health" "optional locally: docker compose --profile stream up -d"

if [ "${1:-}" = "--full" ]; then
  echo "[full]"
  compose run --rm --no-deps trainer python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0), 'bf16' if torch.cuda.is_bf16_supported() else 'no bf16')" \
    >/tmp/argus_torch 2>&1 && ok "training image sees the GPU: $(tail -1 /tmp/argus_torch)" || { fail "CUDA in the training image:"; tail -5 /tmp/argus_torch; }
  for cfg in $(runs_for_profile "${PROFILE:-demo}"); do
    compose run --rm --no-deps trainer python -m argus_ft.prefetch --config "configs/$cfg.yaml" --check-only >/tmp/argus_pf 2>&1 \
      && ok "model access for $cfg" || { fail "model access for $cfg:"; grep FAIL /tmp/argus_pf; }
  done
  $DOCKER build -q --build-arg WITH_TESTS=true -t argus-platform:test ./platform >/dev/null \
    && $DOCKER run --rm argus-platform:test python -m pytest -q >/tmp/argus_pt 2>&1 \
    && ok "platform tests: $(tail -1 /tmp/argus_pt)" || { fail "platform tests:"; tail -15 /tmp/argus_pt; }
  $DOCKER build -q --build-arg WITH_TESTS=true -t argus-stream:test ./stream >/dev/null \
    && $DOCKER run --rm argus-stream:test python -m pytest -q >/tmp/argus_st 2>&1 \
    && ok "stream tests: $(tail -1 /tmp/argus_st)" || { fail "stream tests:"; tail -15 /tmp/argus_st; }
fi

echo
[ "$FAILS" -eq 0 ] && echo -e "\033[32mALL GOOD\033[0m" || { echo -e "\033[31m$FAILS problem(s)\033[0m: fix the FAIL lines above"; exit 1; }
