# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PM4PY_SHOW_PROGRESS_BAR=False \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app

# graphviz supports Petri-net rendering; libgomp is used by numerical wheels;
# curl and CA roots support the optional remote-checkpoint build mode.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        curl \
        graphviz \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY lara_align ./lara_align
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.0" \
    && python -m pip install .

RUN groupadd --system --gid 10001 lara \
    && useradd --system --uid 10001 --gid lara --home-dir /app --shell /usr/sbin/nologin lara

COPY --chown=lara:lara lara_ui ./lara_ui
COPY --chown=lara:lara pages ./pages
COPY --chown=lara:lara files ./files
COPY --chown=lara:lara streamlit_app.py ./streamlit_app.py

# CHECKPOINT_SOURCE accepts:
#   local    - copy the repository's existing runs/ directory (default)
#   download - fetch and extract CHECKPOINT_URL into /app/runs
# CHECKPOINT_SHA256 is optional because the "latest" archive may change. Set it
# in reproducible deployments to make a changed archive fail the image build.
ARG CHECKPOINT_SOURCE=local
ARG CHECKPOINT_URL=https://www.alessandroberti.it/checkpoint_lara_latest.tar.gz
ARG CHECKPOINT_SHA256=""
RUN --mount=type=bind,source=.,target=/tmp/build-context,ro \
    set -eux; \
    mkdir -p /app/runs; \
    case "${CHECKPOINT_SOURCE}" in \
        local) \
            test -d /tmp/build-context/runs; \
            cp -a /tmp/build-context/runs/. /app/runs/; \
            ;; \
        download) \
            curl --fail --location --silent --show-error \
                --output /tmp/checkpoint_lara_latest.tar.gz \
                "${CHECKPOINT_URL}"; \
            if [ -n "${CHECKPOINT_SHA256}" ]; then \
                echo "${CHECKPOINT_SHA256}  /tmp/checkpoint_lara_latest.tar.gz" \
                    | sha256sum --check --strict -; \
            fi; \
            tar --extract --gzip \
                --file /tmp/checkpoint_lara_latest.tar.gz \
                --directory /app \
                --no-same-owner \
                --no-same-permissions; \
            ;; \
        *) \
            echo "Unsupported CHECKPOINT_SOURCE=${CHECKPOINT_SOURCE}; expected local or download" >&2; \
            exit 2; \
            ;; \
    esac; \
    test -f /app/runs/lara/best.pt; \
    test -f /app/runs/lara/last.pt; \
    chown -R lara:lara /app/runs; \
    rm -f /tmp/checkpoint_lara_latest.tar.gz

USER lara

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).read()"]

CMD ["streamlit", "run", "streamlit_app.py", \
    "--server.address=0.0.0.0", \
    "--server.port=8501", \
    "--server.headless=true", \
    "--server.fileWatcherType=none", \
    "--browser.gatherUsageStats=false"]
