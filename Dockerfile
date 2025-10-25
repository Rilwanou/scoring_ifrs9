FROM python:3.10-slim as builder

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry pip install --no-root --no-dev

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

COPY src/ /app/src/

CMD ["python", "src/main.py"]