"""Explanation guardrails (TR-LLM-07, FR-ML-07): schema check, grounding check, template fallback.

An explanation is accepted only if it is valid JSON matching the schema AND every number it mentions
appears in the transaction prompt or is a known rule constant (e.g. "500 km", "05:00"). Otherwise the
answer is rebuilt from per-reason-code template sentences, which are factually correct by construction.
"""
from __future__ import annotations

import json
import re

from .data import REASON_CODES, answer_codes

NUMBER = re.compile(r"\d+(?:\.\d+)?")
NO_SIGNAL = "No elevated risk signals were observed in the transaction context."


def _numbers(text: str) -> set[float]:
    return {float(n) for n in NUMBER.findall(text.replace(",", ""))}


class ExplanationGuard:
    def __init__(self, gold_train_rows: list[dict]):
        """Learn template sentences and rule constants from the gold training explanations."""
        self.templates: dict[str, str] = {}
        self.constants: set[float] = set()
        for r in gold_train_rows:
            codes, text = r["target"]["reason_codes"], r["target"]["explanation"]
            self.constants |= _numbers(text)
            phrases = [p.strip().rstrip(".") for p in text.split(";")]
            if len(phrases) == len(codes):  # one phrase per code, in order
                for code, phrase in zip(codes, phrases):
                    self.templates.setdefault(code, phrase)

    def check(self, text: str, prompt: str) -> tuple[dict | None, str]:
        """Returns (parsed answer, "ok") or (None, failure reason)."""
        try:
            obj = json.loads(text.strip())
        except json.JSONDecodeError:
            return None, "invalid_json"
        codes = answer_codes(obj)
        if codes is None or not set(codes) <= set(REASON_CODES) or not isinstance(obj.get("explanation"), str):
            return None, "schema"
        if not _numbers(obj["explanation"]) <= _numbers(prompt) | self.constants:
            return None, "ungrounded_number"
        return obj, "ok"

    def template(self, codes: list[str]) -> dict:
        phrases = [self.templates[c] for c in codes if c in self.templates]
        return {"reason_codes": codes, "explanation": "; ".join(phrases) + "." if phrases else NO_SIGNAL}

    def guard(self, text: str, prompt: str, fallback_codes: list[str]) -> tuple[dict, str]:
        """Final answer and its source: "model" or "template:<failure reason>".
        In production `fallback_codes` come from the scorer's reason-code heads."""
        obj, status = self.check(text, prompt)
        return (obj, "model") if obj else (self.template(fallback_codes), f"template:{status}")
