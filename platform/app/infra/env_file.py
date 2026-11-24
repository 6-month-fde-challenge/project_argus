"""The repository's .env: read like docker compose reads it, updated in place without losing comments or order.

The same file feeds docker compose, deploy/hyperstack/*.sh, argus.ps1 and deploy/ops.py, so the Settings page and
anyone editing it by hand always see the same values. Writes are atomic (temp file + rename in the same folder), LF,
UTF-8 without BOM, and keep the file's permissions (setup_vm.sh makes it 600).
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

from .fs import write_atomic

LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def _decode(raw: bytes) -> str:
    """Tolerates what Windows tools leave behind: a UTF-8 BOM, or UTF-16 from PowerShell's `>>`."""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace").replace("\ufeff", "").replace("\x00", "")


def _unquote(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    if " #" in v:  # an inline comment after an unquoted value (compose's rule)
        v = v.split(" #", 1)[0].rstrip()
    return v


class EnvFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def exists(self) -> bool:
        return self.path.is_file()

    def _lines(self) -> list[str]:
        if not self.path.exists():
            return []
        return _decode(self.path.read_bytes()).splitlines()

    def values(self) -> dict[str, str]:
        """KEY -> value; a key written twice keeps its last value (docker compose and deploy/ops.py agree)."""
        out: dict[str, str] = {}
        for line in self._lines():
            if line.lstrip().startswith("#"):
                continue
            m = LINE.match(line)
            if m:
                out[m.group(1)] = _unquote(m.group(2))
        return out

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.values().get(key, default)

    def update(self, changes: dict[str, str | None]) -> None:
        """Set (str) or remove (None) keys. An existing key is changed where it is (later duplicates dropped); a new
        one is appended. Comments and the order of everything else are kept."""
        with self._lock:
            lines, done = self._lines(), set()
            out = []
            for line in lines:
                m = None if line.lstrip().startswith("#") else LINE.match(line)
                key = m.group(1) if m else None
                if key in changes:
                    if key in done or changes[key] is None:
                        continue  # removed, or a duplicate of a key already written
                    out.append(f"{key}={changes[key]}")
                    done.add(key)
                else:
                    out.append(line)
            new = [k for k, v in changes.items() if k not in done and v is not None]
            if new:
                if out and out[-1].strip():
                    out.append("")
                out += [f"{k}={changes[k]}" for k in new]
            self._write("\n".join(out) + "\n")

    def _write(self, text: str) -> None:
        # keeps the file's owner and mode (600 on the VM): the VM user, the scripts and compose must still read it
        write_atomic(self.path, text.encode("utf-8"), default_mode=0o600)
