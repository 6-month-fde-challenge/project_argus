"""The GPU VM scripts (deploy/hyperstack/*.sh) run for real under bash, with stand-ins for the host's tools.

docker, tmux, curl, ssh and sudo are replaced by small stubs on PATH that record every call, so the scripts' own logic
(profile choice, .env handling, which services start or restart, what goes over SSH) is checked on every CI run,
without a VM, a GPU or Docker. Linux / macOS only (CI runs on Ubuntu; locally: the python Docker image)."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="bash scripts: Linux / macOS")
SCRIPTS = Path(__file__).resolve().parents[1] / "hyperstack"

DOCKER = r"""#!/usr/bin/env bash
echo "docker $*" >> "$STUB_LOG"
case "$*" in
  info*) exit 0 ;;
  *"ps --services --status running"*) printf '%s\n' $RUNNING ; exit 0 ;;
  *"up -d"*explainer*) [ "${EXPLAINER_FAILS:-0}" = "1" ] && exit 1 ; exit 0 ;;
esac
exit 0
"""
LOGGER = """#!/usr/bin/env bash
echo "{name} $*" >> "$STUB_LOG"
{body}
"""


@pytest.fixture
def vm(tmp_path):
    """A repository on a 'VM': the scripts, profiles, a .env, and stub tools first on PATH."""
    repo = tmp_path / "argus_solution"
    shutil.copytree(SCRIPTS, repo / "deploy" / "hyperstack")
    configs = repo / "training" / "configs"
    configs.mkdir(parents=True)
    for name, run in (("prod_a100", "argus_prod_a100"), ("demo_hybrid", "argus_demo_hybrid"),
                      ("demo_llm_scorer", "argus_demo_llm_scorer"), ("smoke", "smoke"), ("local_demo", "argus_local_demo")):
        (configs / f"{name}.yaml").write_text(f"extends: base.yaml\nrun_name: {run}\n")
    (repo / ".env").write_text("SUPABASE_URL=https://x.supabase.co\n")
    stubs = tmp_path / "bin"
    stubs.mkdir()
    tools = {
        "docker": DOCKER,
        "tmux": LOGGER.format(name="tmux", body='[ "$1" = "has-session" ] && exit "${TMUX_SESSION:-1}"; exit 0'),
        "curl": LOGGER.format(name="curl", body='case "$*" in *v1/health*) echo \'{"scorer":"ok","status":"ok"}\' ;; '
                                               '*ipify*) echo 203.0.113.9 ;; esac; exit 0'),
        "ssh": LOGGER.format(name="ssh", body='case "$*" in *"cat ~/argus_solution/.env"*) printf "ARGUS_UI_PORT=3005\\n" ;; esac; exit 0'),
        "sudo": '#!/usr/bin/env bash\nexec "$@"\n',
        "sleep": "#!/usr/bin/env bash\nexit 0\n",
    }
    for name, body in tools.items():
        f = stubs / name
        f.write_text(body)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.log"
    log.write_text("")

    def run(script, *args, env=None, input_=None):
        e = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "STUB_LOG": str(log), "HOME": str(tmp_path),
             "USER": os.environ.get("USER", "ubuntu"), **(env or {})}
        for k in ("PROFILE", "ARGUS_PROFILE"):
            if k not in (env or {}):
                e.pop(k, None)
        r = subprocess.run(["bash", str(repo / "deploy" / "hyperstack" / script), *args], cwd=repo, env=e,
                           capture_output=True, text=True, input=input_, timeout=60)
        return r, log.read_text()

    return repo, run


# ------------------------------------------------------------------ the profile is never assumed
def resolve(repo, env):
    e = {**os.environ, **env}
    for k in ("PROFILE", "ARGUS_PROFILE"):
        if k not in env:
            e.pop(k, None)
    script = f'source deploy/hyperstack/_common.sh >/dev/null 2>&1 || true; resolve_profile'
    return subprocess.run(["bash", "-c", script], cwd=repo, env=e, capture_output=True, text=True)


def test_no_profile_stops_and_explains_the_choices(vm):
    repo, _ = vm
    r = resolve(repo, {})
    assert r.returncode == 1 and "Choose a profile" in r.stderr and "prod" in r.stderr and "never behind live traffic" in r.stderr


def test_profile_from_command_line_then_env_file_then_custom_profiles(vm):
    repo, _ = vm
    assert resolve(repo, {"PROFILE": "demo"}).stdout.strip() == "demo"
    (repo / ".env").write_text("ARGUS_PROFILE=prod\n")
    assert resolve(repo, {}).stdout.strip() == "prod"                       # the saved choice
    assert resolve(repo, {"PROFILE": "smoke"}).stdout.strip() == "smoke"     # a one-off wins
    assert resolve(repo, {"PROFILE": "local_demo"}).stdout.strip() == "local_demo"
    assert resolve(repo, {"PROFILE": "nope"}).returncode == 1


def test_setup_and_pipeline_refuse_to_guess_a_profile(vm):
    _, run = vm
    for script in ("setup_vm.sh", "pipeline.sh"):
        r, _ = run(script)
        assert r.returncode != 0 and "Choose a profile" in r.stderr, script


# ------------------------------------------------------------------ .env helpers
def test_env_set_replaces_appends_and_keeps_the_file_private(vm):
    repo, _ = vm
    script = "source deploy/hyperstack/_common.sh; env_set A 1; env_set B 2; env_set A 3; echo \"$(env_get A)/$(env_get B)\""
    r = subprocess.run(["bash", "-c", script], cwd=repo, capture_output=True, text=True,
                       env={**os.environ, "PATH": os.environ["PATH"]})
    assert r.stdout.strip().endswith("3/2")
    text = (repo / ".env").read_text()
    assert text.count("A=") == 1 and stat.S_IMODE((repo / ".env").stat().st_mode) == 0o600


# ------------------------------------------------------------------ apply.sh: what restarts after a settings change
def test_apply_restarts_core_services_and_only_running_optional_ones(vm):
    _, run = vm
    r, calls = run("apply.sh", "scoring", "stream", "platform", env={"RUNNING": "scoring platform frontend"})
    assert r.returncode == 0, r.stderr
    started = [line.split("--no-deps", 1)[1].split() for line in calls.splitlines() if " up -d --no-deps" in line]
    assert started == [["scoring"], ["platform"]]                        # stream is not running: left alone;
    #                                                                      the platform last (the UI talks to it)
    assert "stream is not running" in r.stdout


def test_apply_rejects_unknown_services(vm):
    _, run = vm
    r, _ = run("apply.sh", "postgres")
    assert r.returncode != 0 and "unknown service" in r.stderr


# ------------------------------------------------------------------ serve.sh: scoring never waits for the explainer
def make_run(repo, name, awq: bool):
    d = repo / "training" / "outputs" / name
    for sub in ("pilot", "calibration", "scorer_student/fp8"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    (d / "pilot" / "selected.json").write_text("{}")
    (d / "calibration" / "scorer_student_fp8.json").write_text("{}")
    if awq:
        (d / "explainer_student" / "awq").mkdir(parents=True)


def test_serve_starts_scoring_first_and_survives_an_explainer_that_cannot_start(vm):
    repo, run = vm
    make_run(repo, "argus_prod_a100", awq=True)
    r, calls = run("serve.sh", "argus_prod_a100", env={"EXPLAINER_FAILS": "1"})
    assert r.returncode == 0, r.stderr + r.stdout
    ups = [line for line in calls.splitlines() if "--force-recreate" in line]
    assert ups[0].endswith("up -d --force-recreate scoring") and ups[1].endswith("up -d --force-recreate explainer")
    assert "explanations come from templates" in r.stdout
    env = (repo / ".env").read_text()
    assert "RUN_NAME=argus_prod_a100" in env and "EXPLAINER_RUN=argus_prod_a100" in env


def test_serve_without_an_awq_explainer_serves_the_scorer_only(vm):
    repo, run = vm
    make_run(repo, "argus_local_demo", awq=False)
    r, calls = run("serve.sh", "argus_local_demo")
    assert r.returncode == 0, r.stderr + r.stdout
    assert "stop explainer" in calls and not any(line.endswith("explainer") and "up -d" in line for line in calls.splitlines())


def test_serve_refuses_while_training_holds_the_gpu_unless_forced(vm):
    repo, run = vm
    make_run(repo, "argus_prod_a100", awq=False)
    r, _ = run("serve.sh", "argus_prod_a100", env={"TMUX_SESSION": "0"})       # tmux 'train' exists
    assert r.returncode != 0 and "training is using the GPU" in r.stderr
    r, _ = run("serve.sh", "argus_prod_a100", env={"TMUX_SESSION": "0", "FORCE": "1"})
    assert r.returncode == 0, r.stderr


def test_serve_refuses_a_run_that_cannot_serve(vm):
    repo, run = vm
    (repo / "training" / "outputs" / "half_done").mkdir(parents=True)
    r, _ = run("serve.sh", "half_done")
    assert r.returncode != 0 and "no servable scorer" in r.stderr


# ------------------------------------------------------------------ pipeline.sh: unattended in tmux with the chosen profile
def test_pipeline_starts_in_tmux_with_the_profile(vm):
    _, run = vm
    r, calls = run("pipeline.sh", env={"PROFILE": "prod"})
    assert r.returncode == 0, r.stderr
    tmux = [line for line in calls.splitlines() if line.startswith("tmux new-session")]
    assert tmux and "PROFILE=prod" in tmux[0] and "ARGUS_PIPELINE_INSIDE=1" in tmux[0]


# ------------------------------------------------------------------ remote.sh: what goes over SSH
def test_remote_passes_the_profile_quoted_and_never_assumes_one(vm):
    _, run = vm
    env = {"HS_HOST": "203.0.113.9", "HS_KEY": "/dev/null"}
    _, calls = run("remote.sh", "setup", "prod", env=env)
    assert "PROFILE=prod bash deploy/hyperstack/setup_vm.sh" in calls
    _, calls = run("remote.sh", "pipeline", env=env)
    assert "bash deploy/hyperstack/pipeline.sh" in calls and "PROFILE=" not in calls.split("pipeline.sh")[0].splitlines()[-1]
    _, calls = run("remote.sh", "pipeline", "x; reboot", env=env)
    assert "PROFILE=x\\;\\ reboot" in calls                                    # quoted: never runs `reboot`


def test_remote_tunnel_uses_the_vm_ports_on_127_0_0_1(vm):
    _, run = vm
    _, calls = run("remote.sh", "tunnel", env={"HS_HOST": "203.0.113.9", "HS_KEY": "/dev/null"})
    tunnel = [line for line in calls.splitlines() if " -N " in line][0]
    assert "-L 3005:127.0.0.1:3005" in tunnel and "-L 8090:127.0.0.1:8090" in tunnel   # port from the VM's .env
