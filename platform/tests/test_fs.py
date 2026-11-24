"""Files the platform (root in its container) writes into the host repository stay owned by the host user.

On the GPU VM a root-owned .env (mode 600) would lock the VM user, the scripts and docker compose out of it."""
from __future__ import annotations

import json
import os
import stat

import pytest

from app.infra import fs
from app.infra.change_log import ChangeLog
from app.infra.env_file import EnvFile
from app.infra.job_store import JobStore

VM_USER = (4321, 4321)  # stands in for the VM user (ubuntu)
as_root = pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() != 0,
                             reason="needs root, like the platform container (the Docker test image runs as root)")


def owned(path):
    st = os.stat(path)
    return st.st_uid, st.st_gid


@as_root
def test_everything_the_platform_writes_stays_with_the_vm_user(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    os.chown(repo, *VM_USER)
    env = repo / ".env"
    env.write_text("SUPABASE_URL=x\n")
    os.chown(env, *VM_USER)
    os.chmod(env, 0o600)

    EnvFile(env).update({"PRODUCER_SOURCE": "holdout"})          # Settings page
    assert owned(env) == VM_USER and stat.S_IMODE(os.stat(env).st_mode) == 0o600

    ops = repo / "training" / "outputs" / "_platform" / "ops" / "actions"
    fs.mkdirs(ops)                                                 # Operations queue
    for p in (ops, ops.parent, ops.parent.parent, ops.parent.parent.parent):
        assert owned(p) == VM_USER, p
    fs.write_atomic(ops / "a1.json", b"{}")
    assert owned(ops / "a1.json") == VM_USER

    log = ChangeLog(repo / "training" / "outputs" / "_platform" / "changes.jsonl")
    log.add("settings", "x")                                       # change log
    assert owned(log.path) == VM_USER

    jobs = JobStore(repo / "training" / "outputs" / "_platform" / "jobs.json")
    jobs.create(run_name="r", profile="smoke", command=[])         # job registry
    assert owned(jobs.path) == VM_USER and json.loads(jobs.path.read_text())[0]["run_name"] == "r"


def test_new_files_take_the_folder_owner_and_replaced_files_keep_theirs(tmp_path, monkeypatch):
    """Runs anywhere (no root needed): records the chown calls instead of making them."""
    calls = []
    folder_ids, file_ids = (1000, 1000), (1001, 1001)
    real_stat = os.stat

    class FakeStat:
        def __init__(self, st, ids):
            self.st_mode, self.st_uid, self.st_gid = st.st_mode, *ids

    def fake_stat(path, *a, **kw):
        st = real_stat(path, *a, **kw)
        name = str(path)
        if name.endswith(".tmp"):
            return FakeStat(st, (0, 0))          # just written by the platform: root's
        return FakeStat(st, file_ids if name.endswith("kept.txt") else folder_ids)

    monkeypatch.setattr(os, "stat", fake_stat)
    monkeypatch.setattr(fs.Path, "stat", lambda self, *a, **kw: fake_stat(self, *a, **kw))
    monkeypatch.setattr(os, "chown", lambda p, uid, gid: calls.append((os.path.basename(p), uid, gid)), raising=False)
    (tmp_path / "kept.txt").write_text("old")
    fs.write_atomic(tmp_path / "kept.txt", b"new")
    fs.write_atomic(tmp_path / "fresh.txt", b"new")
    assert (".kept.txt.tmp", 1001, 1001) in calls      # the replaced file's owner is kept
    assert (".fresh.txt.tmp", 1000, 1000) in calls     # a new file takes its folder's owner
    assert (tmp_path / "kept.txt").read_bytes() == b"new"
