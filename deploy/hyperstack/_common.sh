#!/usr/bin/env bash
# Shared by every deploy/hyperstack script: run from the repository root (argus_solution/), print helpers, .env access.
# Sourced, never run on its own.

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
# shellcheck disable=SC2034  # ROOT is used by the scripts that source this file
ROOT="$(pwd)"
LOGS="training/outputs/_logs"            # same place as the UI-era console logs (README section 17)
DEPLOY_OUT="training/outputs/_deploy"    # what the VM hands to the PC (aws.env)
mkdir -p "$LOGS" "$DEPLOY_OUT"

ok()   { echo -e "  \033[32mOK\033[0m   $*"; }
bad()  { echo -e "  \033[31mFAIL\033[0m $*"; }
info() { echo -e "  \033[33m..\033[0m   $*"; }
step() { echo; echo -e "\033[1;34m=== [$(date '+%H:%M:%S')] $*\033[0m"; }
die()  { echo -e "\033[31m$*\033[0m" >&2; exit 1; }

# docker without sudo once the user is in the docker group; sudo docker right after setup_vm.sh added it
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi
compose() { $DOCKER compose "$@"; }

env_get() {  # value of KEY in .env (last one wins), empty if absent
  [ -f .env ] && grep -E "^$1=" .env | tail -1 | cut -d= -f2- | tr -d '\r' || true
}

env_set() {  # set KEY=VALUE in .env (replace or append), keeps the file private
  touch .env && chmod 600 .env
  if grep -qE "^$1=" .env; then sed -i "s|^$1=.*|$1=$2|" .env; else echo "$1=$2" >> .env; fi
}

env_ensure_secret() {  # generate a random value for KEY once
  [ -n "$(env_get "$1")" ] || { env_set "$1" "$(head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 40)"; info "generated $1 in .env"; }
}

# host ports (docker-compose.yml): .env overrides, defaults otherwise. Containers keep their internal ports.
UI_PORT="$(env_get ARGUS_UI_PORT)";               UI_PORT="${UI_PORT:-3000}"
API_PORT="$(env_get ARGUS_API_PORT)";             API_PORT="${API_PORT:-8090}"
MLFLOW_PORT="$(env_get ARGUS_MLFLOW_PORT)";       MLFLOW_PORT="${MLFLOW_PORT:-5000}"
SCORING_PORT="$(env_get ARGUS_SCORING_PORT)";     SCORING_PORT="${SCORING_PORT:-8080}"
EXPLAINER_PORT="$(env_get ARGUS_EXPLAINER_PORT)"; EXPLAINER_PORT="${EXPLAINER_PORT:-8000}"
STREAM_PORT="$(env_get ARGUS_STREAM_PORT)";       STREAM_PORT="${STREAM_PORT:-8003}"

public_ip() { curl -fs --max-time 3 https://api.ipify.org || echo "<VM_PUBLIC_IP>"; }

# profile -> the runs it trains, and the run that gets served
runs_for_profile() {
  case "$1" in
    demo)  echo "demo_hybrid demo_llm_scorer" ;;
    prod)  echo "prod_a100" ;;
    smoke) echo "smoke" ;;
    *)     echo "$1" ;;  # any configs/<name>.yaml
  esac
}
serve_run_for_profile() {
  case "$1" in
    demo)  echo "argus_demo_llm_scorer" ;;
    prod)  echo "argus_prod_a100" ;;
    smoke) echo "smoke" ;;
    *)     grep -E '^run_name:' "training/configs/$1.yaml" 2>/dev/null | awk '{print $2}' ;;
  esac
}
scoring_config_for_profile() {  # the api: budgets the Fraud Model API runs with
  case "$1" in prod) echo "configs/prod_a100.yaml" ;; *) echo "configs/local_demo.yaml" ;; esac
}
