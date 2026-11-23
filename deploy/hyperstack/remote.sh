#!/usr/bin/env bash
# Drive the Hyperstack GPU VM from your PC over SSH (Git Bash on Windows, or any Linux / macOS shell).
#
#   export HS_HOST=<vm public ip>        # Hyperstack console
#   export HS_USER=ubuntu                # default
#   export HS_KEY=~/.ssh/id_ed25519      # the key pair attached to the VM
#
#   bash deploy/hyperstack/remote.sh push              # copy this repo (code, configs, .env) to ~/argus_solution
#   bash deploy/hyperstack/remote.sh setup [profile]   # setup_vm.sh on the VM (demo | prod | smoke)
#   bash deploy/hyperstack/remote.sh pipeline [profile] # the whole model side, unattended (tmux)
#   bash deploy/hyperstack/remote.sh train <profile> [args]  # one ladder run (tmux)
#   bash deploy/hyperstack/remote.sh logs [train|pipeline]   # follow a log
#   bash deploy/hyperstack/remote.sh status            # check.sh on the VM
#   bash deploy/hyperstack/remote.sh tunnel            # the VM's UI, platform, MLflow, stream on localhost (Ctrl-C ends)
#   bash deploy/hyperstack/remote.sh pull              # copy training/outputs back (models, reports, MLflow)
#   bash deploy/hyperstack/remote.sh awsenv            # print the VM's aws.env (append it to your .env)
#   bash deploy/hyperstack/remote.sh ssh               # interactive shell
#
# The VM bills by the hour: hibernate it in the Hyperstack console when done.
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/../.."

: "${HS_HOST:?set HS_HOST to the VM public IP}"
HS_USER="${HS_USER:-ubuntu}"
HS_KEY="${HS_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-argus_solution}"
SSH_OPTS=(-i "$HS_KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
TARGET="$HS_USER@$HS_HOST"
rsh() { ssh "${SSH_OPTS[@]}" "$TARGET" "$@"; }

push() {
  echo ">> pushing code to $TARGET:~/$REMOTE_DIR"
  rsh "mkdir -p ~/$REMOTE_DIR"
  tar czf - --exclude=.git --exclude=node_modules --exclude=dist --exclude='__pycache__' --exclude='.pytest_cache' \
      --exclude=training/data --exclude=training/outputs --exclude=training/.hf_cache --exclude=demo \
      --exclude=demo_walkthrough --exclude='*.mp4' --exclude=.terraform --exclude='*.tfstate*' --exclude=.env . \
    | rsh "tar xzf - -C ~/$REMOTE_DIR && find ~/$REMOTE_DIR/deploy -name '*.sh' -exec sed -i 's/\r$//' {} +"
  # .env only the first time: afterwards the VM's copy holds the tokens setup_vm.sh generated (AWS depends on them)
  if [ -f .env ] && ! rsh "test -f ~/$REMOTE_DIR/.env"; then
    rsh "umask 077 && cat > ~/$REMOTE_DIR/.env" < .env
    echo ">> copied .env (first push)"
  fi
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  push)     push ;;
  setup)    rsh "cd ~/$REMOTE_DIR && PROFILE=${1:-demo} bash deploy/hyperstack/setup_vm.sh" ;;
  pipeline) rsh "cd ~/$REMOTE_DIR && PROFILE=${1:-demo} bash deploy/hyperstack/pipeline.sh" ;;
  train)    rsh "cd ~/$REMOTE_DIR && bash deploy/hyperstack/train.sh $(printf '%q ' "$@")" ;;
  logs)     rsh "tail -n 80 -f ~/$REMOTE_DIR/training/outputs/_logs/${1:-pipeline}.log" ;;
  status)   rsh "cd ~/$REMOTE_DIR && bash deploy/hyperstack/check.sh" ;;
  tunnel)
    # the same port numbers on both ends, read from the VM's .env (ARGUS_*_PORT), defaults otherwise
    remote_env=$(rsh "cat ~/$REMOTE_DIR/.env 2>/dev/null" || true)
    port() { local v; v=$(printf '%s\n' "$remote_env" | grep -E "^ARGUS_$1_PORT=" | tail -1 | cut -d= -f2 | tr -d '\r'); echo "${v:-$2}"; }
    ui=$(port UI 3000); api=$(port API 8090); mlflow=$(port MLFLOW 5000); stream=$(port STREAM 8003)
    echo ">> http://localhost:$ui (UI)  :$api (platform)  :$mlflow (MLflow)  :$stream (stream, if running) - Ctrl-C to close"
    ssh "${SSH_OPTS[@]}" -N -L "$ui:localhost:$ui" -L "$api:localhost:$api" -L "$mlflow:localhost:$mlflow" \
      -L "$stream:localhost:$stream" "$TARGET" ;;
  pull)
    mkdir -p training/outputs
    rsh "cd ~/$REMOTE_DIR && tar czf - --exclude='*/trainer' --exclude='_logs' training/outputs" | tar xzf - -C .
    echo ">> runs now in training/outputs:"; ls -1 training/outputs ;;
  awsenv)   rsh "cat ~/$REMOTE_DIR/training/outputs/_deploy/aws.env" ;;
  ssh)      ssh "${SSH_OPTS[@]}" "$TARGET" ;;
  all)      push; bash "$SELF" pipeline "$@" ;;
  *)        sed -n '2,21p' "$SELF" ;;
esac
