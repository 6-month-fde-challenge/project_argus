"""/api/v1/data and /api/v1/docs - the dataset behind the runs, and the client documents + diagrams."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, Response

from ...core.container import Container, get_container
from ...infra.runs_repo import load_json

router = APIRouter(tags=["knowledge"])


@router.get("/data/overview")
def data_overview(c: Container = Depends(get_container)):
    raw = c.settings.data_dir / "raw"
    card = raw / "DATA_CARD.md"
    quality, manifest = load_json(raw / "quality_report.json") or {}, load_json(raw / "manifest.json") or {}
    return {
        "data_card": card.read_text(encoding="utf-8") if card.exists() else None,
        "dataset_id": manifest.get("dataset_id"), "version": manifest.get("version"),
        "label_snapshot": manifest.get("label_snapshot"), "counts": manifest.get("counts"),
        "observed": quality.get("observed"), "fraud_rate": quality.get("main_fraud_rate"),
        "corpus_tokens": quality.get("estimated_tokens_by_corpus"), "pii_audit": quality.get("pii_pattern_audit"),
        "rules": load_json(raw / "rules_config.json"),
    }


@router.get("/docs/documents")
def documents(c: Container = Depends(get_container)):
    return c.docs.documents()


@router.get("/docs/documents/{name}", response_class=HTMLResponse)
def document(name: str, c: Container = Depends(get_container)):
    return c.docs.document_html(name)


@router.get("/docs/diagrams")
def diagrams(c: Container = Depends(get_container)):
    return c.docs.diagram_list()


@router.get("/docs/diagrams/{name}")
def diagram(name: str, c: Container = Depends(get_container)):
    content, media_type = c.docs.diagram(name)
    return Response(content=content, media_type=media_type, headers={"Cache-Control": "max-age=3600"})
