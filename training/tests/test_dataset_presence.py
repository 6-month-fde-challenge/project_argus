"""A fresh machine has no dataset: training must download it even when started with --no-sync (the UI default)."""
from argus_ft.supabase_data import TABLES, have_dataset


def test_have_dataset_needs_the_manifest_and_every_table(tmp_path):
    raw, db = tmp_path / "raw", tmp_path / "db"
    assert not have_dataset(raw, db)                      # fresh machine
    raw.mkdir()
    (raw / "manifest.json").write_text("{}")
    db.mkdir()
    tables = list(TABLES)
    for t in tables[:-1]:
        (db / f"{t}.parquet").write_bytes(b"x")
    assert not have_dataset(raw, db)                      # an interrupted first sync
    (db / f"{tables[-1]}.parquet").write_bytes(b"x")
    assert have_dataset(raw, db)
