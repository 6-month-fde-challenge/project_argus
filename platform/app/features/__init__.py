"""Feature registry: every feature module contributes routers here. Adding a feature = one module + one line."""
from .jobs.router import router as jobs
from .knowledge.router import router as knowledge
from .profiles.router import router as profiles
from .registry.router import router as registry
from .runs.router import compare_router as compare
from .runs.router import router as runs
from .scoring.router import router as scoring
from .stream.router import router as stream
from .system.router import router as system

ROUTERS = [system, profiles, runs, compare, jobs, registry, scoring, stream, knowledge]
