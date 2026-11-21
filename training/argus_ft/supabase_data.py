"""Pull the ARGUS dataset from Supabase: training files from the `argus-data` storage bucket,
engineered features + labels from the Postgres tables (for the XGBoost baseline).

Usage:  python -m argus_ft.supabase_data [--prefix training/] [--out data/raw] [--force]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BUCKET = "argus-data"
# What the training pipeline needs; pass --prefix "" to mirror the whole bucket.
DEFAULT_PREFIXES = (
    "training/sft/", "training/gold/", "training/dpo/", "training/cpt/", "training/format_comparison/",
    "parquet/transactions/",  # full request fields, used by the API demo client to send CRD 10.2 requests
    "DATA_CARD.md", "manifest.json", "quality_report.json", "rules_config.json",
)
PAGE = 1000  # PostgREST max rows per request on this project


class SupabaseClient:
    """Read-only access to the project's storage bucket (files) and REST API (tables)."""

    def __init__(self, url: str | None = None, key: str | None = None, bucket: str = BUCKET):
        load_dotenv()
        self.url = (url or os.environ["SUPABASE_URL"]).rstrip("/")
        key = key or os.environ["SUPABASE_KEY"]
        self.bucket = bucket
        self.session = requests.Session()
        self.session.headers.update({"apikey": key, "Authorization": f"Bearer {key}"})

    def _list(self, prefix: str) -> list[dict]:
        resp = self.session.post(
            f"{self.url}/storage/v1/object/list/{self.bucket}",
            json={"prefix": prefix, "limit": 1000},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def walk(self, prefix: str = ""):
        """Yield (path, size) for every object under `prefix` (folders have id=None)."""
        folder = prefix.rpartition("/")[0]
        folder = f"{folder}/" if folder else ""
        for obj in self._list(folder):
            path = f"{folder}{obj['name']}"
            if obj.get("id") is None:
                if f"{path}/".startswith(prefix) or prefix.startswith(f"{path}/"):
                    yield from self.walk(f"{path}/")
            elif path.startswith(prefix):
                yield path, obj["metadata"]["size"]

    def download(self, path: str, dst: Path) -> None:
        resp = self.session.get(f"{self.url}/storage/v1/object/{self.bucket}/{path}", timeout=300)
        resp.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(resp.content)

    def fetch_table(self, table: str, select: str = "*", order: str = "transaction_id") -> pd.DataFrame:
        """Page through a Postgres table via the REST API."""
        rows, start = [], 0
        while True:
            resp = self.session.get(
                f"{self.url}/rest/v1/{table}",
                params={"select": select, "order": order},
                headers={"Range": f"{start}-{start + PAGE - 1}"},
                timeout=120,
            )
            resp.raise_for_status()
            page = resp.json()
            rows += page
            if len(page) < PAGE:
                return pd.DataFrame(rows)
            start += PAGE


# table -> (columns, sort key). XGBoost features, labels/splits, and the fields evaluation needs:
# amounts (rupee-weighted recall), timestamps (leakage checks), customer age + region (fairness report).
TABLES = {
    "features": ("*", "transaction_id"),
    "labels": ("transaction_id,fraud_label,split", "transaction_id"),
    "transactions": ("transaction_id,customer_id,transaction_amount,transaction_time", "transaction_id"),
    "customers": ("customer_id,customer_age,region", "customer_id"),
}


def fetch_tables(out_dir: str | Path = "data/db") -> dict[str, Path]:
    """Snapshot DB tables to parquet once; later runs reuse the snapshot."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store, paths = None, {}
    for table, (select, order) in TABLES.items():
        path = paths[table] = out_dir / f"{table}.parquet"
        if path.exists():
            continue
        store = store or SupabaseClient()
        df = store.fetch_table(table, select, order)
        df.to_parquet(path, index=False)
        print(f"  fetched table {table}: {len(df)} rows")
    return paths


def sync(out_dir: str | Path = "data/raw", prefixes=DEFAULT_PREFIXES, force: bool = False) -> Path:
    """Download objects not already present locally with the same size."""
    out_dir = Path(out_dir)
    store = SupabaseClient()
    fetched = skipped = 0
    for prefix in prefixes:
        for path, size in store.walk(prefix):
            dst = out_dir / path
            if not force and dst.exists() and dst.stat().st_size == size:
                skipped += 1
                continue
            store.download(path, dst)
            fetched += 1
            print(f"  fetched {path} ({size / 1e6:.2f} MB)")
    print(f"Supabase sync -> {out_dir}: {fetched} downloaded, {skipped} up to date")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--prefix", action="append", help="bucket prefix to fetch (repeatable); '' = everything")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sync(args.out, tuple(args.prefix) if args.prefix is not None else DEFAULT_PREFIXES, args.force)


if __name__ == "__main__":
    main()
