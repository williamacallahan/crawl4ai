# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

# C4ai version
ARG C4AI_VER=0.8.9
ARG SOURCE_COMMIT
ARG C4AI_GIT_SHA=${SOURCE_COMMIT}
RUN test -n "$C4AI_GIT_SHA"
ENV C4AI_VERSION=$C4AI_VER \
    C4AI_GIT_SHA=$C4AI_GIT_SHA
LABEL c4ai.version=$C4AI_VER \
      org.opencontainers.image.version=$C4AI_VER \
      org.opencontainers.image.revision=$C4AI_GIT_SHA

# Set build arguments
ARG APP_HOME=/app
ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    DEBIAN_FRONTEND=noninteractive \
    REDIS_HOST=localhost \
    REDIS_PORT=6379 \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    UV_LINK_MODE=copy

ARG PYTHON_VERSION=3.12
ARG INSTALL_TYPE=default
ARG ENABLE_GPU=false
ARG TARGETARCH

# Redis version — pinned to a CVE-patched release by default.
# Override with --build-arg REDIS_VERSION="" for latest, or
# --build-arg REDIS_VERSION="6:7.2.7-1rl1~bookworm1" for a specific version.
ARG REDIS_VERSION="6:7.2.7-1rl1~bookworm1"

LABEL maintainer="unclecode"
LABEL description="🔥🕷️ Crawl4AI: Open-source LLM Friendly Web Crawler & scraper"
LABEL version="1.0"

# Install curl and gnupg first (needed to add Redis repo)
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade the base image before installing version-pinned third-party packages below. Running a
# distribution upgrade after the Redis install silently replaces the requested version.
RUN apt-get update && apt-get dist-upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Add official Redis repository for security-patched versions
RUN curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb bookworm main" \
    > /etc/apt/sources.list.d/redis.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    gnupg \
    git \
    cmake \
    pkg-config \
    python3-dev \
    libjpeg-dev \
    redis-tools${REDIS_VERSION:+=$REDIS_VERSION} \
    redis-server${REDIS_VERSION:+=$REDIS_VERSION} \
    supervisor \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

RUN if [ -n "$REDIS_VERSION" ]; then \
        test "$(dpkg-query -W -f='${Version}' redis-server)" = "$REDIS_VERSION"; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxcb1 \
    libxkbcommon0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$ENABLE_GPU" = "true" ] && [ "$TARGETARCH" = "amd64" ] ; then \
    apt-get update && apt-get install -y --no-install-recommends \
    nvidia-cuda-toolkit \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/* ; \
else \
    echo "Skipping NVIDIA CUDA Toolkit installation (unsupported platform or GPU disabled)"; \
fi

RUN if [ "$TARGETARCH" = "arm64" ]; then \
    echo "🦾 Installing ARM-specific optimizations"; \
    apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*; \
elif [ "$TARGETARCH" = "amd64" ]; then \
    echo "🖥️ Installing AMD64-specific optimizations"; \
    apt-get update && apt-get install -y --no-install-recommends \
    libomp-dev \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*; \
else \
    echo "Skipping platform-specific optimizations (unsupported platform)"; \
fi

# Create a non-root user and group
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

# Create and set permissions for appuser home directory
RUN mkdir -p /home/appuser && chown -R appuser:appuser /home/appuser

WORKDIR /tmp/project
COPY . /tmp/project/

RUN --mount=type=cache,target=/root/.cache/uv \
    case "$INSTALL_TYPE" in \
      default) uv sync --locked --no-dev --no-editable --group server ;; \
      torch) uv sync --locked --no-dev --no-editable --group server --extra torch ;; \
      transformer) uv sync --locked --no-dev --no-editable --group server --extra transformer \
        && python -m crawl4ai.model_loader ;; \
      all) uv sync --locked --no-dev --no-editable --group server --extra all \
        && python -m nltk.downloader punkt stopwords \
        && python -m crawl4ai.model_loader ;; \
      *) echo "Unsupported INSTALL_TYPE: $INSTALL_TYPE"; exit 64 ;; \
    esac

RUN python -c "import crawl4ai, fastapi, gunicorn, mcp, redis, websockets"

RUN crawl4ai-setup

RUN python -c "from patchright.sync_api import sync_playwright; p = sync_playwright().start(); b = p.chromium.launch(headless=True, args=['--no-sandbox']); b.close(); p.stop()"

WORKDIR ${APP_HOME}

# Copy supervisor config first (might need root later, but okay for now)
COPY deploy/docker/supervisord.conf .

RUN mkdir -p /home/appuser/.cache/ms-playwright \
    && cp -r /root/.cache/ms-playwright/chromium-* /home/appuser/.cache/ms-playwright/ \
    && chown -R appuser:appuser /home/appuser/.cache/ms-playwright

RUN crawl4ai-doctor

# Ensure all cache directories belong to appuser
# This fixes permission issues with .cache/url_seeder and other runtime cache dirs
RUN mkdir -p /home/appuser/.cache \
    && chown -R appuser:appuser /home/appuser/.cache

# Copy application code
COPY deploy/docker/* ${APP_HOME}/

# copy the playground + any future static assets
COPY deploy/docker/static ${APP_HOME}/static

# Change ownership of the application directory to the non-root user
RUN chown -R appuser:appuser ${APP_HOME}

# give permissions to redis persistence dirs if used
RUN mkdir -p /var/lib/redis /var/log/redis && chown -R appuser:appuser /var/lib/redis /var/log/redis

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD bash -c '\
    MEM=$(free -m | awk "/^Mem:/{print \$2}"); \
    if [ $MEM -lt 2048 ]; then \
        echo "⚠️ Warning: Less than 2GB RAM available! Your container might need a memory boost! 🚀"; \
        exit 1; \
    fi && \
    redis-cli ping > /dev/null && \
    curl -f http://localhost:11235/health || exit 1'

EXPOSE 6379
# Switch to the non-root user before starting the application
USER appuser

# Set environment variables to ptoduction
ENV PYTHON_ENV=production 

# Start the application using supervisord
CMD ["supervisord", "-c", "supervisord.conf"]
