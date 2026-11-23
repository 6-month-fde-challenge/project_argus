#!/usr/bin/env bash
# Run ON the Hyperstack GPU VM (Ubuntu 22.04 / 24.04 image with the NVIDIA driver). Safe to re-run.
#
#   bash deploy/hyperstack/setup_vm.sh               # demo profile (models of demo_hybrid / demo_llm_scorer)
#   PROFILE=prod bash deploy/hyperstack/setup_vm.sh  # prod_a100 (Qwen2.5-7B, Llama 3.2, Gemma 2: needs HF_TOKEN)
#
#  1. checks: NVIDIA GPU, free disk, Ubuntu
#  2. Docker Engine + compose plugin + NVIDIA Container Toolkit (skipped when present), tmux, curl
#  3. .env: Supabase settings must be filled; secrets generated once (scoring tokens, stream control key);
#     ARGUS_BIND=127.0.0.1 so only the token-protected Fraud Model API listens publicly
#  4. images: training (+ MLflow), scoring, platform, frontend, stream
#  5. data from Supabase, and every model of the profile pre-downloaded (gated access checked first),
#     so paid GPU time is never spent downloading
set -euo pipefail
source "$(dirname "$0")/_common.sh"
PROFILE="${PROFILE:-demo}"

step "1/5 checks"
command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null || die "no NVIDIA driver / GPU visible: pick a GPU flavour with an NVIDIA image"
ok "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
free_gb() { df -Pk "$1" | awk 'NR==2{print int($4/1024/1024)}'; }
case "$ROOT" in /ephemeral*)
  # Hyperstack's ephemeral disk is wiped when the VM is hibernated: models, runs and MLflow would be lost
  [ "${ALLOW_EPHEMERAL:-0}" = "1" ] || die "$ROOT is on the ephemeral disk (lost on hibernate). Clone into ~ or a \
persistent volume, or ALLOW_EPHEMERAL=1 to accept that" ;;
esac
FREE_GB=$(free_gb .)
[ "$FREE_GB" -ge 40 ] || die "only ${FREE_GB} GB free for $ROOT; need >= 40 GB (data, model cache, runs, MLflow)"
ok "disk for the repo ($ROOT): ${FREE_GB} GB free"
grep -qi ubuntu /etc/os-release || info "not Ubuntu: install Docker + NVIDIA Container Toolkit yourself if the next step fails"

step "2/5 Docker, NVIDIA Container Toolkit, tmux"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
APT="$SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600"
need=(); for p in tmux curl ca-certificates gnupg; do dpkg -s "$p" >/dev/null 2>&1 || need+=("$p"); done
[ ${#need[@]} -eq 0 ] || { $APT update -y && $APT install -y "${need[@]}"; }
if ! command -v docker >/dev/null || ! docker compose version >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | $SUDO sh       # Docker Engine + compose plugin, official convenience script
fi
if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  $APT update -y && $APT install -y nvidia-container-toolkit
  $SUDO nvidia-ctk runtime configure --runtime=docker && $SUDO systemctl restart docker
fi
id -nG "$USER" | grep -qw docker || { $SUDO usermod -aG docker "$USER"; info "added $USER to the docker group (new SSH sessions use docker without sudo)"; }
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi
DOCKER_ROOT=$($DOCKER info --format '{{.DockerRootDir}}')
DOCKER_FREE=$(free_gb "$DOCKER_ROOT")
[ "$DOCKER_FREE" -ge 60 ] || $DOCKER image inspect argus-training:latest >/dev/null 2>&1 || die "only ${DOCKER_FREE} GB \
free for Docker images in $DOCKER_ROOT; need >= 60 GB (training 13 GB, vLLM 35 GB, the rest). Use a large-root-disk \
flavour, or point Docker's data-root at a persistent volume (/etc/docker/daemon.json), then re-run"
ok "disk for Docker images ($DOCKER_ROOT): ${DOCKER_FREE} GB free"
$DOCKER run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi -L >/dev/null || die "containers cannot see the GPU (NVIDIA Container Toolkit)"
ok "$($DOCKER --version | cut -d, -f1), $($DOCKER compose version --short) and GPU inside containers"

step "3/5 .env"
[ -f .env ] || { cp .env.example .env; chmod 600 .env; }
if [ -z "$(env_get SUPABASE_URL)" ] || env_get SUPABASE_URL | grep -q "<project-ref>"; then
  die "fill SUPABASE_URL and SUPABASE_KEY in $ROOT/.env (nano .env), then re-run"
fi
env_ensure_secret ARGUS_SCORING_TOKEN          # Fraud Model API service token (replaces demo-service-token)
env_ensure_secret ARGUS_SCORING_ADMIN_TOKEN    # admin API token (replaces demo-admin-token)
env_ensure_secret ARGUS_STREAM_CONTROL_KEY     # stream traffic start / stop
[ -n "$(env_get ARGUS_BIND)" ] || env_set ARGUS_BIND 127.0.0.1   # UI / platform / MLflow / vLLM: SSH tunnel only
[ -n "$(env_get ARGUS_SCORING_BIND)" ] || env_set ARGUS_SCORING_BIND 0.0.0.0   # the token-protected API: AWS calls it
[ -n "$(env_get SCORING_CONFIG)" ] || env_set SCORING_CONFIG "$(scoring_config_for_profile "$PROFILE")"
if [ "$PROFILE" = "prod" ] && [ -z "$(env_get HF_TOKEN)" ]; then
  die "PROFILE=prod pilots Llama 3.2 and Gemma 2 (gated): set HF_TOKEN in .env for an account that accepted both licences"
fi
ok ".env ready (secrets generated once; UI bound to 127.0.0.1, Fraud Model API public on :$SCORING_PORT)"

step "4/5 images (first build 20-30 min)"
compose build trainer
compose --profile serve build scoring
compose build platform frontend
compose --profile stream build stream
compose --profile serve pull explainer   # the vLLM image (~10 GB): pulled now, not on the first serve
ok "images: $($DOCKER images --filter reference='argus-*' --format '{{.Repository}}' | sort | tr '\n' ' ')"

step "5/5 data and model weights"
compose run --rm --no-deps trainer python -c "from argus_ft.supabase_data import fetch_tables, sync; sync('data/raw'); fetch_tables('data/db')"
for cfg in $(runs_for_profile "$PROFILE"); do
  compose run --rm --no-deps trainer python -m argus_ft.prefetch --config "configs/$cfg.yaml"
done
ok "data in training/data, models in training/.hf_cache"

echo -e "\n\033[32mVM ready.\033[0m  Next: bash deploy/hyperstack/check.sh   then   bash deploy/hyperstack/pipeline.sh"
