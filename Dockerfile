# 40kPlayer — service web (parties à deux sauvegardées, annulation, journaux JSON pour l'entraînement)
#
#   docker build -t 40kplayer .
#   docker run -p 8040:8040 -v fortyk-data:/data 40kplayer
#
# Les parties (un JSON par partie) et les listes importées vont dans le volume /data.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FORTYK_HOST=0.0.0.0 \
    PORT=8040 \
    FORTYK_DATA_DIR=/data/games \
    FORTYK_LISTS_DIR=/data/lists

WORKDIR /app
RUN pip install --no-cache-dir "numpy>=1.24"

# code et données du jeu (fiches Wahapedia versionnées, carte, empreintes, listes livrées)
COPY pyproject.toml README.md ./
COPY fortyk ./fortyk
COPY scripts ./scripts
COPY data/wahapedia/raw ./data/wahapedia/raw
COPY data/layouts ./data/layouts
COPY data/lists ./data/lists
COPY data/hulls.json ./data/hulls.json

RUN python -c "from fortyk.data import load_catalog; load_catalog()" \
    && useradd --system --uid 10001 --home /app fortyk \
    && mkdir -p /data/games /data/lists \
    && chown -R fortyk /data

USER fortyk
VOLUME ["/data"]
EXPOSE 8040
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT', '8040'), timeout=4)"

CMD ["python", "scripts/serve.py", "--no-browser"]
