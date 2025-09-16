# Use a slim Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    CRSTLMETH_LOGFILE=/tmp/crstlmeth.log

# Install OS-level deps (for pip, wheels, htslib, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tabix \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working dir and copy code
WORKDIR /app
COPY . /app

# Install Python dependencies (assume PEP 517 with pyproject.toml)
RUN pip install --upgrade pip \
    && pip install .

# Expose Streamlit port
EXPOSE 8501

# Default to Streamlit app
CMD ["crstlmeth", "web"]
