"""ARGUS operations agent: runs on the host what the UI's Operations / Settings pages ask for.

The platform runs in a container and cannot run the host's scripts (deploy/hyperstack/*.sh need the host's Docker,
tmux, sudo, GPU). So it writes a request file; this agent, on the host, re-validates it against the same catalogue
(deploy/ops_actions.json + deploy/ops_catalog.py), runs exactly that command, and writes its log and result back.
Only the catalogue's actions can run, with checked parameters, without a shell. Every action stays runnable by hand:
the UI shows the command.

    python3 deploy/agent.py            run (GPU VM: the systemd service argus-agent that setup_vm.sh installs;
                                       Windows: .\\argus.ps1 agent)
    python3 deploy/agent.py --status   is it running, and what is it doing

Files (training/outputs/_platform/ops/, shared with the platform):
    agent.json          heartbeat every few seconds: pid, host, os, the action running
    actions/<id>.json   one request: written once by the platform (queued), then only by this agent
    actions/<id>.log    its output, appended live (the UI tails it)
    actions/<id>.cancel the platform asks to cancel; the agent stops the action (and everything it started)
One action at a time, oldest first. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import ops_catalog as catalog  # noqa: E402

OS_NAME = "windows" if os.name == "nt" else "linux"
BEAT_S = 3


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, data: dict) -> None:
    """Atomic: readers (the platform) never see a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Agent:
    def __init__(self, ops_dir: Path, repo: Path = REPO):
        self.repo, self.ops = repo, ops_dir
        self.actions_dir = ops_dir / "actions"
        self.actions_dir.mkdir(parents=True, exist_ok=True)
        self.started = now()
        self.running: str | None = None
        self.source_mtime = Path(__file__).stat().st_mtime

    # ------------------------------------------------------------ heartbeat
    def beat(self) -> None:
        write_json(self.ops / "agent.json", {
            "pid": os.getpid(), "host": socket.gethostname(), "os": OS_NAME, "platform": platform.platform(),
            "repo": str(self.repo), "started": self.started, "beat": time.time(), "running": self.running,
            "python": sys.version.split()[0]})

    # ------------------------------------------------------------ queue
    def _requests(self) -> list[tuple[Path, dict]]:
        out = []
        for f in self.actions_dir.glob("*.json"):
            data = read_json(f)
            if data and data.get("id") == f.stem:
                out.append((f, data))
        return sorted(out, key=lambda x: x[1].get("created", ""))

    def recover(self) -> None:
        """An action still 'running' when the agent starts was cut off (reboot, crash): say so, never re-run it."""
        for f, data in self._requests():
            if data.get("status") == "running":
                data.update(status="failed", finished=now(), message="interrupted: the agent restarted while it ran")
                write_json(f, data)

    def next_queued(self) -> tuple[Path, dict] | None:
        return next(((f, d) for f, d in self._requests() if d.get("status") == "queued"), None)

    # ------------------------------------------------------------ one action
    def execute(self, f: Path, req: dict) -> None:
        log = f.with_suffix(".log")
        cancel = f.with_suffix(".cancel")
        if cancel.exists():
            req.update(status="cancelled", finished=now(), message="cancelled before it started")
            write_json(f, req)
            return
        try:
            actions = catalog.load(self.repo)  # re-read: an updated catalogue applies without a restart
            params = catalog.validate(actions, req["action"], req.get("params"), self.repo)
            argv, extra_env = catalog.render(actions, req["action"], params, OS_NAME)
            timeout = actions[req["action"]].get("timeout_min")
        except (catalog.ActionError, KeyError, OSError, ValueError) as e:
            req.update(status="failed", finished=now(), message=f"refused by the agent: {e}")
            write_json(f, req)
            return

        req.update(status="running", started=now(), agent_os=OS_NAME, command=catalog.preview(argv, extra_env, OS_NAME))
        write_json(f, req)
        self.running = req["id"]
        self.beat()
        with log.open("a", encoding="utf-8") as out:
            out.write(f"$ {req['command']}\n[agent] {now()} on {socket.gethostname()} ({OS_NAME})\n")
            out.flush()
            env = {**os.environ, **extra_env, "PYTHONUNBUFFERED": "1"}
            kw = {"start_new_session": True} if OS_NAME == "linux" else \
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
            try:
                proc = subprocess.Popen(argv, cwd=self.repo, env=env, stdout=out, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, **kw)
            except OSError as e:
                out.write(f"[agent] could not start: {e}\n")
                req.update(status="failed", finished=now(), exit_code=None, message=f"could not start: {e}")
                write_json(f, req)
                self.running = None
                return
            deadline = time.monotonic() + timeout * 60 if timeout else None
            why = None
            while proc.poll() is None:
                time.sleep(1)
                self.beat()
                if cancel.exists():
                    why = "cancelled from the UI"
                elif deadline and time.monotonic() > deadline:
                    why = f"stopped after its {timeout}-minute limit"
                if why:
                    out.write(f"\n[agent] {why}: stopping it\n")
                    out.flush()
                    stop_tree(proc)
                    break
            code = proc.wait()
            out.write(f"\n[agent] {now()} finished with exit code {code}\n")
        status = "cancelled" if why == "cancelled from the UI" else "succeeded" if code == 0 and not why else "failed"
        req.update(status=status, finished=now(), exit_code=code, message=why)
        write_json(f, req)
        self.running = None

    # ------------------------------------------------------------ loop
    def reload_if_updated(self) -> None:
        """After `git pull`, pick up the new agent code between actions (systemd keeps the same process)."""
        try:
            changed = Path(__file__).stat().st_mtime != self.source_mtime
        except OSError:
            return
        if changed and self.running is None:
            print(f"[agent] {now()} agent.py changed: reloading", flush=True)
            os.execv(sys.executable, [sys.executable, *sys.argv])

    def serve_forever(self) -> None:
        self.recover()
        print(f"[agent] {now()} watching {self.actions_dir} ({OS_NAME}, repo {self.repo})", flush=True)
        while True:
            self.beat()
            item = self.next_queued()
            if item:
                print(f"[agent] {now()} {item[1]['id']}: {item[1]['action']}", flush=True)
                self.execute(*item)
                continue
            self.reload_if_updated()
            time.sleep(1)


def stop_tree(proc: subprocess.Popen) -> None:
    """The action and everything it started (docker compose, python, ...)."""
    if OS_NAME == "windows":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def status(ops_dir: Path) -> int:
    beat = read_json(ops_dir / "agent.json")
    if not beat:
        print("agent: never started on this machine")
        return 1
    age = time.time() - beat.get("beat", 0)
    alive = age < 15
    print(f"agent: {'running' if alive else 'NOT running'} (pid {beat.get('pid')} on {beat.get('host')}, "
          f"last heartbeat {age:.0f} s ago){', running action ' + beat['running'] if alive and beat.get('running') else ''}")
    return 0 if alive else 1


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ops-dir", type=Path, default=REPO / "training" / "outputs" / "_platform" / "ops")
    ap.add_argument("--status", action="store_true", help="is the agent running (exit code 0) or not (1)")
    ap.add_argument("--once", action="store_true", help="run the queued actions, then exit (tests, cron)")
    a = ap.parse_args(argv)
    if a.status:
        sys.exit(status(a.ops_dir))
    agent = Agent(a.ops_dir)
    if a.once:
        agent.recover()
        while item := agent.next_queued():
            agent.execute(*item)
        agent.beat()
        return
    try:
        agent.serve_forever()
    except KeyboardInterrupt:
        print("[agent] stopped", flush=True)


if __name__ == "__main__":
    main()
