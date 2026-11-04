# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS build
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache .

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 meridian
COPY --from=build /usr/local /usr/local
WORKDIR /app
COPY examples ./examples
COPY meridian.example.toml ./meridian.toml
ENV MERIDIAN_CONFIG=/app/meridian.toml PYTHONUNBUFFERED=1
USER meridian
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz').status==200 else 1)"
CMD ["uvicorn", "meridian.api.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
