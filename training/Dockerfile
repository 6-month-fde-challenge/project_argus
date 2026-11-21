FROM pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime

# PIP_BREAK_SYSTEM_PACKAGES: the base image's Python is a PEP 668 system install; this container runs one app only
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/workspace/.hf_cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --retries 10 --timeout 120 -r requirements.txt

COPY argus_ft ./argus_ft
COPY configs ./configs

# data/, outputs/ and the HF cache are bind-mounted by docker-compose so they survive the container
CMD ["python", "-m", "argus_ft.pipeline", "--config", "configs/local_demo.yaml"]
