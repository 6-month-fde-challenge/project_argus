#!/usr/bin/env bash
# Run ON the VM. One ladder run of a profile in tmux session "train" (survives SSH drops). Same command as
# `.\argus.ps1 train` / the UI's Launch page, so the run shows up live in the UI too.
#
#   bash deploy/hyperstack/train.sh prod_a100                       # the A100 profile
#   bash deploy/hyperstack/train.sh demo_hybrid                     # any training/configs/<profile>.yaml
#   bash deploy/hyperstack/train.sh smoke --stages calibrate,acceptance,registry
#   bash deploy/hyperstack/train.sh prod_a100 --set pilot.max_steps=30
#   FORCE=1 bash deploy/hyperstack/train.sh ...                     # replace a run still in progress
#   SYNC=1  ...                                                     # download data changes first (default: cached)
# Follow:  tail -f training/outputs/_logs/train.log      or  tmux attach -t train   (detach: Ctrl-B then D)
set -euo pipefail
source "$(dirname "$0")/_common.sh"
PROFILE="${1:?usage: bash deploy/hyperstack/train.sh <profile> [pipeline args...]}"; shift
[ -f "training/configs/$PROFILE.yaml" ] || die "no profile training/configs/$PROFILE.yaml"

if tmux has-session -t train 2>/dev/null; then
  [ "${FORCE:-0}" = "1" ] || die "a training run is in progress (tmux 'train'): tail -f $LOGS/train.log, or FORCE=1 to replace it"
  tmux kill-session -t train
fi
if curl -fs --max-time 2 "http://localhost:$SCORING_PORT/v1/health" >/dev/null 2>&1; then
  info "stopping serving to free the GPU (start it again with serve.sh)"
  compose --profile serve stop scoring explainer
fi

# shellcheck disable=SC2206  # $DOCKER is "docker" or "sudo docker"
CMD=($DOCKER compose run --rm trainer python -m argus_ft.pipeline --config "configs/$PROFILE.yaml")
[ "${SYNC:-0}" = "1" ] || CMD+=(--no-sync)
CMD+=("$@")
LOG="$ROOT/$LOGS/train.log"
RUNNER="$ROOT/$LOGS/.train_runner.sh"  # what tmux runs: exact argv, any quoting, under bash
{
  echo "#!/usr/bin/env bash"
  printf 'cd %q\n' "$ROOT"
  printf '{ echo %q; %s; } 2>&1 | tee -a %q\n' "> ${CMD[*]}" "$(printf '%q ' "${CMD[@]}")" "$LOG"
  printf 'echo "[train.sh] finished with exit code ${PIPESTATUS[0]}" | tee -a %q\n' "$LOG"
} > "$RUNNER"
: > "$LOG"
tmux new-session -d -s train "bash '$RUNNER'"
sleep 10
if ! tmux has-session -t train 2>/dev/null && ! grep -q "finished with exit code 0" "$LOGS/train.log"; then
  tail -n 30 "$LOGS/train.log"; die "training failed to start"
fi
echo "Training $PROFILE started (tmux 'train')."
echo "  follow : tail -f $LOGS/train.log        (done at '=== done in N min')"
echo "  UI     : Runs page (through the SSH tunnel: bash deploy/hyperstack/remote.sh tunnel)"
