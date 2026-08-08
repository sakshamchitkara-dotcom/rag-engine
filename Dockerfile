FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY rag_engine ./rag_engine
COPY examples ./examples
RUN pip install --no-cache-dir ".[all]" && useradd --create-home rag && mkdir /data && chown rag /data

USER rag
WORKDIR /data
ENV RAG_INDEX=/data/index.sqlite PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000
ENTRYPOINT ["rag"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
