FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY research_mcp ./research_mcp
# For neural embeddings use: pip install ".[st]" (larger image; downloads a model on first run).
RUN pip install --no-cache-dir .

ENV RESEARCH_DATA_DIR=/data \
    RESEARCH_TRANSPORT=http \
    RESEARCH_HOST=0.0.0.0 \
    RESEARCH_PORT=8000
VOLUME /data
EXPOSE 8000
CMD ["research-mcp"]
