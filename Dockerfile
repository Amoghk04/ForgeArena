# ── Stage 1: build the Vite/React UI ─────────────────────────────────────────
FROM node:20-slim AS ui-builder

WORKDIR /ui
COPY ui/package.json ui/package-lock.json* ./
RUN npm ci --prefer-offline
COPY ui/ ./
RUN npm run build          # outputs to /ui/dist


# ── Stage 2: Python backend + nginx + supervisord ────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime deps: nginx (reverse proxy) + supervisord (process manager)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Python package
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e "."

# Project data files
COPY openenv.yaml ./
COPY tasks/ ./tasks/

# nginx config
COPY nginx.conf ./nginx.conf

# supervisord config
COPY supervisord.conf /etc/supervisor/conf.d/forge-arena.conf

# Vite build output (served as static files by nginx)
COPY --from=ui-builder /ui/dist ./ui/dist

# HF Spaces runs containers as uid 1000 (non-root).
# nginx needs to write tmp files; /var/log/nginx must be writable too.
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app \
    && mkdir -p /var/log/nginx /var/lib/nginx/body /tmp/nginx_client_body \
                /tmp/nginx_proxy /tmp/nginx_fastcgi /tmp/nginx_uwsgi /tmp/nginx_scgi \
    && chown -R appuser:appuser /var/log/nginx /var/lib/nginx \
    && chown -R appuser:appuser /tmp/nginx_client_body /tmp/nginx_proxy \
                                /tmp/nginx_fastcgi /tmp/nginx_uwsgi /tmp/nginx_scgi \
    && ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log

USER appuser

# HuggingFace Spaces expects port 7860 (nginx listens here; uvicorn on :8000 internally)
EXPOSE 7860

# supervisord starts both uvicorn and nginx
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/forge-arena.conf"]
