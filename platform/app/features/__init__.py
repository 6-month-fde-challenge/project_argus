"""Feature registry: every feature module contributes routers here. Adding a feature = one module + one line."""
from .aws.router import router as aws
from .jobs.router import router as jobs
from .knowledge.router import router as knowledge
from .ops.router import router as ops
from .profiles.router import router as profiles
from .registry.router import router as registry
from .runs.router import compare_router as compare
from .runs.router import router as runs
from .scoring.router import router as scoring
from .settings.router import router as settings
from .stream.router import router as stream
from .system.router import router as system

ROUTERS = [system, profiles, runs, compare, jobs, registry, scoring, stream, knowledge, settings, ops, aws]
