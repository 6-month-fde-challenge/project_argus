"""Download every model a profile will use into the Hugging Face cache before training starts.

    python -m argus_ft.prefetch --config configs/prod_a100.yaml
    docker compose run --rm trainer python -m argus_ft.prefetch --config configs/demo_hybrid.yaml

A long GPU run should never stop at stage 2 because a gated model (Llama 3.2, Gemma 2) needs a licence that the
HF_TOKEN account has not accepted, or spend paid GPU-VM time downloading weights. This reads the profile (pilot
candidates, fallback teacher, both students), checks access to each model first, then downloads only the files
training needs. Already-cached files are skipped, so it is safe to re-run.
"""
from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from .config import load_config

WEIGHTS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.jinja"]  # config, weights, tokenizer


def profile_models(cfg) -> list[str]:
    """Every base model a run of this profile can load, in ladder order, without duplicates."""
    names = [*cfg.pilot.candidates, cfg.models.teacher, cfg.models.student_scorer, cfg.models.student_explainer]
    return list(dict.fromkeys(names))


def check_access(api: HfApi, name: str, token: str | None) -> str | None:
    """None if the model can be downloaded, else the reason it cannot."""
    try:
        api.auth_check(name, token=token)
        return None
    except GatedRepoError:
        return "gated: accept the licence on huggingface.co with the HF_TOKEN account" + ("" if token else " (HF_TOKEN unset)")
    except RepositoryNotFoundError:
        return "not found (typo, or private without access)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="profile YAML, e.g. configs/prod_a100.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--check-only", action="store_true", help="only verify access, download nothing")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    token = os.environ.get("HF_TOKEN") or None
    api, names = HfApi(), profile_models(cfg)
    print(f"=== {len(names)} models for {args.config}")
    problems = {n: why for n in names if (why := check_access(api, n, token))}
    for n in names:
        print(f"  {'FAIL' if n in problems else 'ok  '} {n}" + (f"  - {problems[n]}" if n in problems else ""))
    if problems:
        sys.exit(f"{len(problems)} model(s) not accessible: fix them before training")
    if args.check_only:
        return
    for n in names:
        path = snapshot_download(n, allow_patterns=WEIGHTS, token=token)
        print(f"  cached {n} -> {path}")
    print("=== all models cached: training needs no downloads")


if __name__ == "__main__":
    main()
