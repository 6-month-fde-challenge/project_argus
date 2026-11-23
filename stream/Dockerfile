# ARGUS stream service: Kinesis producer + real-time scorer (calls the Fraud Model API) + analytics API, :8003.
#   docker build -t argus-stream stream          (tests: --build-arg WITH_TESTS=true, then python -m pytest -q)
FROM python:3.12.16-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 STORE_PATH=/data/argus_stream.db
WORKDIR /srv

COPY requirements.txt requirements-dev.txt ./
ARG WITH_TESTS=false
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --retries 10 --timeout 120 -r requirements.txt \
 && if [ "$WITH_TESTS" = "true" ]; then pip install -r requirements-dev.txt; fi

COPY argus_stream ./argus_stream
COPY tests ./tests
COPY pytest.ini ./
RUN mkdir -p /data

EXPOSE 8003
HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/live/health')"
CMD ["uvicorn", "argus_stream.api:app", "--host", "0.0.0.0", "--port", "8003"]
