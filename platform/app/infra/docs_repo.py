"""Client documents and solution diagrams, read-only (01_argus/client_docs, day_wise/part_03/diagrams).

.docx files are converted to HTML on demand (mammoth) and cached by modification time, so the UI can show the
client's own documents in place.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..core.errors import NotFound

DIAGRAM_TYPES = {".svg": "image/svg+xml", ".png": "image/png"}


@lru_cache(maxsize=32)
def _docx_html(path: str, mtime: float) -> str:  # mtime is part of the cache key
    import mammoth
    with open(path, "rb") as f:
        return mammoth.convert_to_html(f).value


class DocsRepository:
    def __init__(self, docs_dir: Path, diagrams_dir: Path):
        self.docs, self.diagrams = docs_dir, diagrams_dir

    @staticmethod
    def _title(stem: str) -> str:
        return stem.replace("_", " ").replace("Project ARGUS", "").strip(" -")

    def documents(self) -> list[dict]:
        if not self.docs.exists():
            return []
        return [{"name": p.name, "title": self._title(p.stem), "size_kb": round(p.stat().st_size / 1024)}
                for p in sorted(self.docs.glob("*.docx"))]

    def document_html(self, name: str) -> str:
        p = self.docs / name
        if p.suffix != ".docx" or p.name not in {d["name"] for d in self.documents()}:
            raise NotFound(f"document '{name}' not found")
        return _docx_html(str(p), p.stat().st_mtime)

    def diagram_list(self) -> list[dict]:
        """One entry per diagram (prefers SVG, skips the docx-fit PNG copies)."""
        if not self.diagrams.exists():
            return []
        stems: dict[str, Path] = {}
        for p in sorted(self.diagrams.iterdir()):
            if p.suffix in DIAGRAM_TYPES and ".fit-docx" not in p.name:
                if p.stem not in stems or p.suffix == ".svg":
                    stems[p.stem] = p
        return [{"name": p.name, "title": self._title(p.stem.split("_", 1)[-1]).title(), "media_type": DIAGRAM_TYPES[p.suffix]}
                for p in stems.values()]

    def diagram(self, name: str) -> tuple[bytes, str]:
        p = self.diagrams / name
        if p.name not in {d["name"] for d in self.diagram_list()}:
            raise NotFound(f"diagram '{name}' not found")
        return p.read_bytes(), DIAGRAM_TYPES[p.suffix]
