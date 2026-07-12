# Small linux/arm64 sandbox image for local Kaggle-in-Kaggle rehearsal runs.
# Replaces the amd64-only gcr.io/kaggle-images/python (which is enormous and
# emulated on Apple Silicon). The harness sandbox runs with network disabled,
# so every package the agent may import must be preinstalled here.
FROM --platform=linux/arm64 python:3.11-slim

# libgomp1 is required at runtime by lightgbm/xgboost OpenMP builds on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    lightgbm \
    xgboost \
    catboost

CMD ["python"]
