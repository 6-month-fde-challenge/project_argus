"""Writing into the host's repository from a container without taking files away from the host user.

The platform container runs as root. On a Linux host (the GPU VM) every file it creates in a bind mount would then be
owned by root: after one save on the Settings page, `.env` (mode 600) could no longer be read by the VM user, so
deploy/hyperstack/*.sh and `docker compose` itself would fail; a profile saved in the UI could not be edited or
updated by git; the operations agent (running as that user) could not write its results. So every file and folder
the platform creates or replaces in a shared folder gets the owner of what it replaces, or of the folder it lands in.
On Docker Desktop (Windows / macOS) ownership is not enforced and chown is a no-op or unsupported: ignored.
"""
from __future__ import annotations

import os
from pathlib import Path


def owner(path: Path) -> tuple[int, int] | None:
    """(uid, gid) of `path`, or of its nearest existing parent."""
    p = Path(path)
    for candidate in (p, *p.parents):
        try:
            st = candidate.stat()
            return st.st_uid, st.st_gid
        except OSError:
            continue
    return None


def adopt(path: Path, ids: tuple[int, int] | None) -> None:
    """chown `path` to `ids`, when it differs and the platform may (root in its container)."""
    if ids is None or not hasattr(os, "chown"):
        return
    try:
        st = os.stat(path)
        if (st.st_uid, st.st_gid) != ids:
            os.chown(path, *ids)
    except OSError:
        pass  # not root, or a file system without POSIX owners (Docker Desktop bind mounts)


def mkdirs(path: Path) -> Path:
    """mkdir -p, every newly created folder owned like its nearest existing parent."""
    path = Path(path)
    missing = []
    for p in (path, *path.parents):
        if p.exists():
            ids = owner(p)
            break
        missing.append(p)
    else:
        ids = None
    path.mkdir(parents=True, exist_ok=True)
    for p in reversed(missing):
        adopt(p, ids)
    return path


def write_atomic(path: Path, data: bytes, default_mode: int = 0o644) -> None:
    """Replace `path` atomically (temp file + rename in the same folder): readers never see half a file. Keeps the
    replaced file's owner and permissions; a new file gets the folder's owner and `default_mode`."""
    path = Path(path)
    mkdirs(path.parent)
    existing = path.exists()
    ids = owner(path) if existing else owner(path.parent)
    mode = (path.stat().st_mode & 0o777) if existing else default_mode
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    try:
        os.chmod(tmp, mode)
    except OSError:
        pass
    adopt(tmp, ids)
    os.replace(tmp, path)
