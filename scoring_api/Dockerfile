# Fraud Model API (CRD 10 / LLD 4-6). Built on the training image: same torch / transformers / compressed-tensors
# stack, so the FP8 scorer, calibrators and feature code load exactly as they were trained.
ARG BASE_IMAGE=argus-training:latest
FROM ${BASE_IMAGE}

WORKDIR /workspace
COPY argus_api ./argus_api

ENV ARGUS_CONFIG=configs/local_demo.yaml
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/v1/health')"
CMD ["uvicorn", "argus_api.app:app", "--host", "0.0.0.0", "--port", "8080"]
