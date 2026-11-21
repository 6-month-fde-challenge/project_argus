"""Request / response contract: exactly CRD Section 10.2-10.3, validated as in LLD Section 4.1."""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

MCC_GROUPS = {"education", "electronics", "food", "gaming", "gift_cards", "grocery", "healthcare", "other",
              "quick_commerce", "travel"}
ID_PATTERN = r"^[A-Za-z0-9_-]+$"


def sanitise(text: str) -> str:
    """LLD 6.2: NFKC-normalise, keep [A-Za-z0-9 .,-], cap at 32 characters (free text never reaches instructions)."""
    return re.sub(r"[^A-Za-z0-9 .,-]", "", unicodedata.normalize("NFKC", text))[:32].strip()


class Location(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    city: str = Field(max_length=64)

    @field_validator("city")
    @classmethod
    def _clean_city(cls, v: str) -> str:
        return sanitise(v) or "unk"


class TransactionIn(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=64, pattern=ID_PATTERN)
    customer_id: str = Field(min_length=1, max_length=64, pattern=ID_PATTERN)
    transaction_amount: float = Field(gt=0, le=10_000_000)
    merchant_category: str
    transaction_time: datetime
    device_id: str | None = Field(default=None, max_length=64)
    customer_age: int = Field(ge=18, le=120)
    customer_location: Location | None = None
    merchant_location: Location
    account_age: int = Field(ge=0)

    @field_validator("transaction_amount")
    @classmethod
    def _two_decimals(cls, v: float) -> float:
        if round(v, 2) != v:
            raise ValueError("must have at most 2 decimals")
        return v

    @field_validator("merchant_category")
    @classmethod
    def _known_mcc(cls, v: str) -> str:
        return v if v in MCC_GROUPS else "other"  # LLD: unknown -> other (+ warning in the audit record)

    @field_validator("transaction_time")
    @classmethod
    def _with_offset(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("must be RFC 3339 with a UTC offset")
        return v


class ScoreResponse(BaseModel):
    """Field for field the CRD 10.3 response; `explanation` is null when produced asynchronously (LLD 4.1)."""
    transaction_id: str
    fraud_probability: float | None
    decision: str
    risk_level: str
    reason_codes: list[str]
    explanation: str | None
    decision_source: str
    model_version: str
    latency_ms: int
