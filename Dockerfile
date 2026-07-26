FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/IHGGM-Aachen/crstlmeth"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    CRSTLMETH_LOGFILE=/tmp/crstlmeth.log

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tabix \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip \
    && pip install ".[web]"

EXPOSE 8501

CMD ["crstlmeth", "web"]