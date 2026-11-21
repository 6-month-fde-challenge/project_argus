FROM python:3.12.16-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv

COPY requirements.txt requirements-dev.txt ./
ARG WITH_TESTS=false
# optional PyPI mirror for slow networks (same packages, same pinned versions)
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --retries 10 --timeout 120 -r requirements.txt \
 && if [ "$WITH_TESTS" = "true" ]; then pip install -r requirements-dev.txt; fi

COPY app ./app
COPY tests ./tests
COPY pytest.ini ./

EXPOSE 8090
HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090/api/v1/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
