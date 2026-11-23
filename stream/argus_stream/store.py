"""SQLite store of the scored stream and the analytics behind the UI's Live stream page.

One writer (the processor thread), many readers (API requests); WAL mode keeps reads from blocking. Rows older than
RETENTION_HOURS are pruned. Inserts ignore a repeated event_id, so at-least-once delivery never double counts.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path

REAL = {"ts", "scored_ts", "amount", "probability", "api_latency_ms", "shadow_probability", "shadow_latency_ms"}
INTEGER = {"hour", "fraud_label", "http_status"}
COLUMNS = ["event_id", "ts", "scored_ts", "source", "transaction_id", "customer_id", "device_id", "merchant_category",
           "amount", "city", "merchant_city", "hour", "fraud_label", "fraud_type", "ring_id", "http_status", "decision",
           "risk_level", "probability", "decision_source", "reason_codes", "explanation", "model_version",
           "api_latency_ms", "action", "source_shard", "source_seq", "source_epoch",  # reason_codes: comma-separated
           "shadow_run", "shadow_decision", "shadow_probability", "shadow_model_version", "shadow_latency_ms"]


def _type(column: str) -> str:
    return "TEXT UNIQUE" if column == "event_id" else "REAL" if column in REAL else "INTEGER" if column in INTEGER else "TEXT"


SCHEMA = (f"CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
          f"{', '.join(f'{c} {_type(c)}' for c in COLUMNS)});\n"
          "CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts);\n"
          # last Kinesis sequence number processed per transactions-stream shard: where the consumer resumes
          "CREATE TABLE IF NOT EXISTS checkpoints (shard TEXT PRIMARY KEY, seq TEXT NOT NULL);\n"
          # which stream the checkpoints belong to (stream_epoch: its creation time)
          "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);")


class Store:
    def __init__(self, path: str = "/data/argus_stream.db", retention_hours: float = 72):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.retention_s = retention_hours * 3600
        self._local = threading.local()
        self._lock = threading.Lock()
        self._shared = sqlite3.connect(self.path, check_same_thread=False) if self.path == ":memory:" else None
        conn = self._conn()
        conn.executescript(SCHEMA)
        have = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
        for column in (c for c in COLUMNS if c not in have):  # a store written by an older version: add new columns
            conn.execute(f"ALTER TABLE events ADD COLUMN {column} {_type(column).replace(' UNIQUE', '')}")
        conn.commit()
        if self._shared is None:
            conn.execute("PRAGMA journal_mode=WAL")
        self._pruned = 0.0

    def _conn(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = sqlite3.connect(self.path, timeout=10)
        return conn

    def _rows(self, sql: str, args=()) -> list[dict]:
        """Reads use their own thread-local connection and, with WAL, never wait for the processor's writes (nor
        block them). Only the in-memory store (tests) shares one connection and needs the lock."""
        if self._shared is not None:
            with self._lock:
                cur = self._shared.execute(sql, args)
                return [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]
        cur = self._conn().execute(sql, args)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]

    # ------------------------------------------------------------ write
    def insert(self, records: list[dict]) -> None:
        if not records:
            return
        rows = [tuple(",".join(r[c]) if c == "reason_codes" and isinstance(r.get(c), list) else r.get(c) for c in COLUMNS)
                for r in records]
        sql = f"INSERT OR IGNORE INTO events ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})"
        with self._lock:
            conn = self._conn()
            conn.executemany(sql, rows)
            if time.time() - self._pruned > 600:
                conn.execute("DELETE FROM events WHERE ts < ?", (time.time() - self.retention_s,))
                self._pruned = time.time()
            conn.commit()

    def save_checkpoints(self, positions: dict[str, str]) -> None:
        """Advance per-shard checkpoints, only forward (Kinesis sequence numbers compare as integers)."""
        if not positions:
            return
        with self._lock:
            conn = self._conn()
            have = dict(conn.execute("SELECT shard, seq FROM checkpoints").fetchall())
            for shard, seq in positions.items():
                if shard not in have or int(seq) > int(have[shard]):
                    conn.execute("INSERT OR REPLACE INTO checkpoints (shard, seq) VALUES (?, ?)", (shard, str(seq)))
            conn.commit()

    def checkpoints(self) -> dict[str, str]:
        return {r["shard"]: r["seq"] for r in self._rows("SELECT shard, seq FROM checkpoints")}

    def get_meta(self, key: str) -> str | None:
        rows = self._rows("SELECT value FROM meta WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    def restore_epoch(self, epoch: str) -> None:
        """Rebuild: the restored checkpoints were taken on this stream (their decisions carry its epoch)."""
        with self._lock:
            conn = self._conn()
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('tx_stream_epoch', ?)", (epoch,))
            conn.commit()

    def bind_checkpoints(self, epoch: str) -> str:
        """Tie the checkpoints to the stream they were taken on. -> "same" (resume after them), "unknown" (taken
        before the stream was recorded: dropped, resume by time) or "new" (another stream: dropped, read it all)."""
        with self._lock:
            conn = self._conn()
            row = conn.execute("SELECT value FROM meta WHERE key = 'tx_stream_epoch'").fetchone()
            verdict = "same" if row and row[0] == epoch else "new" if row else "unknown"
            if verdict != "same":
                conn.execute("DELETE FROM checkpoints")
                conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('tx_stream_epoch', ?)", (epoch,))
            conn.commit()
            return verdict

    SHADOW = ("shadow_run", "shadow_decision", "shadow_probability", "shadow_model_version", "shadow_latency_ms")

    def set_shadow(self, rows: list[tuple[str, dict]]) -> None:
        """Attach shadow-run results to decisions already stored: [(event_id, {shadow_*: value})]."""
        if not rows:
            return
        sql = f"UPDATE events SET {', '.join(f'{c} = ?' for c in self.SHADOW)} WHERE event_id = ?"
        with self._lock:
            conn = self._conn()
            conn.executemany(sql, [(*(fields.get(c) for c in self.SHADOW), event_id) for event_id, fields in rows])
            conn.commit()

    # ------------------------------------------------------------ read
    def head(self) -> int:
        return self._rows("SELECT COALESCE(MAX(id), 0) AS m FROM events")[0]["m"]

    def last_event_ts(self) -> float:
        """Newest event time: where the processor resumes after a restart (0 when empty)."""
        return float(self._rows("SELECT COALESCE(MAX(ts), 0) AS m FROM events")[0]["m"] or 0)

    def recent(self, limit: int = 50, since_id: int = 0) -> list[dict]:
        rows = self._rows("SELECT * FROM events WHERE id > ? ORDER BY id DESC LIMIT ?", (since_id, limit))
        for r in rows:
            r["reason_codes"] = [c for c in (r.get("reason_codes") or "").split(",") if c]
        return rows

    def summary(self, minutes: int = 60, top: int = 10) -> dict:
        since = (time.time() - minutes * 60,)
        w = "WHERE ts >= ?"
        t = self._rows(f"""SELECT COUNT(*) n, SUM(decision='ALLOW') allow, SUM(decision='REVIEW') review,
            SUM(decision='BLOCK') block, SUM(decision_source='FAIL_OPEN') fail_open, SUM(decision_source='HARD_RULE') hard_rule,
            SUM(fraud_label) fraud, SUM(amount) amount,
            SUM(CASE WHEN decision='BLOCK' AND fraud_label=1 THEN amount ELSE 0 END) fraud_amount_blocked,
            SUM(CASE WHEN decision='ALLOW' AND fraud_label=1 THEN amount ELSE 0 END) fraud_amount_missed,
            SUM(decision!='ALLOW' AND fraud_label=1) tp, SUM(decision!='ALLOW' AND fraud_label=0) fp,
            SUM(decision='ALLOW' AND fraud_label=1) fn, SUM(decision='BLOCK' AND fraud_label=1) block_tp,
            SUM(decision='BLOCK' AND fraud_label=0) block_fp, SUM(fraud_label=0) genuine,
            AVG(scored_ts - ts) * 1000 end_to_end_ms FROM events {w}""", since)[0]
        t = {k: (v or 0) for k, v in t.items()}
        latency = [r["api_latency_ms"] for r in self._rows(
            f"SELECT api_latency_ms FROM events {w} AND api_latency_ms IS NOT NULL ORDER BY api_latency_ms", since)]
        pct = lambda q: latency[min(len(latency) - 1, int(q * len(latency)))] if latency else None  # noqa: E731
        bucket = 60 if minutes <= 180 else 300 if minutes <= 1440 else 3600

        def group(col: str, extra: str = "", n: int = top) -> list[dict]:
            return self._rows(f"""SELECT {col} key, COUNT(*) n, SUM(decision='REVIEW') review, SUM(decision='BLOCK') block,
                SUM(fraud_label) fraud, ROUND(SUM(amount), 2) amount, ROUND(AVG(probability), 4) avg_probability
                FROM events {w} {extra} GROUP BY {col}
                ORDER BY (SUM(decision='BLOCK') * 2 + SUM(decision='REVIEW')) DESC, n DESC LIMIT {int(n)}""", since)

        shadow = []  # live model vs shadow run, on the transactions both scored (MLOPS-06)
        for r in self._rows(f"""SELECT shadow_run, MAX(shadow_model_version) model_version, COUNT(*) n,
                SUM(shadow_decision = decision) agree, AVG(shadow_latency_ms) avg_latency_ms,
                SUM(decision != 'ALLOW' AND fraud_label = 1) live_tp, SUM(decision = 'ALLOW' AND fraud_label = 1) live_fn,
                SUM(decision = 'BLOCK' AND fraud_label = 1) live_btp, SUM(decision = 'BLOCK') live_blocks,
                SUM(shadow_decision != 'ALLOW' AND fraud_label = 1) sh_tp, SUM(shadow_decision = 'ALLOW' AND fraud_label = 1) sh_fn,
                SUM(shadow_decision = 'BLOCK' AND fraud_label = 1) sh_btp, SUM(shadow_decision = 'BLOCK') sh_blocks
                FROM events {w} AND shadow_decision IS NOT NULL GROUP BY shadow_run""", since):
            r = {k: 0 if v is None and k not in ("shadow_run", "model_version", "avg_latency_ms") else v for k, v in r.items()}
            ratio = lambda a, b: r[a] / r[b] if r[b] else None  # noqa: E731
            r["live_recall"] = r["live_tp"] / (r["live_tp"] + r["live_fn"]) if r["live_tp"] + r["live_fn"] else None
            r["shadow_recall"] = r["sh_tp"] / (r["sh_tp"] + r["sh_fn"]) if r["sh_tp"] + r["sh_fn"] else None
            r.update(agreement=ratio("agree", "n"), live_precision_at_block=ratio("live_btp", "live_blocks"),
                     shadow_precision_at_block=ratio("sh_btp", "sh_blocks"))
            shadow.append(r)
        reasons: Counter = Counter()
        for r in self._rows(f"SELECT reason_codes FROM events {w} AND decision != 'ALLOW'", since):
            reasons.update(c for c in (r["reason_codes"] or "").split(",") if c)
        risky = "AND decision != 'ALLOW'"
        return {
            "minutes": minutes, "bucket_s": bucket,
            "totals": {k: t[k] for k in ("n", "allow", "review", "block", "fail_open", "hard_rule", "fraud", "amount",
                                         "fraud_amount_blocked", "fraud_amount_missed", "end_to_end_ms")},
            "live_accuracy": {  # against the ground truth the producer attaches (synthetic patterns / holdout labels)
                "recall": t["tp"] / (t["tp"] + t["fn"]) if t["tp"] + t["fn"] else None,
                "precision": t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else None,
                "precision_at_block": t["block_tp"] / (t["block_tp"] + t["block_fp"]) if t["block_tp"] + t["block_fp"] else None,
                "false_decline_rate": t["block_fp"] / t["genuine"] if t["genuine"] else None,
            },
            "api_latency": {"n": len(latency), "p50_ms": pct(0.5), "p95_ms": pct(0.95), "p99_ms": pct(0.99)},
            "timeseries": self._rows(f"""SELECT CAST(ts / {bucket} AS INTEGER) * {bucket} t, decision, COUNT(*) n
                FROM events {w} GROUP BY t, decision ORDER BY t""", since),
            "reasons": dict(reasons.most_common()), "shadow": shadow,
            "by_category": group("merchant_category", n=12), "by_city": group("city", n=12),
            "by_fraud_type": group("fraud_type"), "by_source": group("source", n=5),
            "by_decision_source": group("decision_source", n=5), "by_model": group("model_version", n=5),
            "by_device": group("device_id", risky), "by_customer": group("customer_id", risky),
            "by_ring": group("ring_id", "AND ring_id IS NOT NULL"),
        }
