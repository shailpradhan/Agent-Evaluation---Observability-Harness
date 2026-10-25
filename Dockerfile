FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser pyproject.toml README.md ./
COPY --chown=appuser:appuser agent_eval ./agent_eval
COPY --chown=appuser:appuser datasets ./datasets

RUN python -m pip install --no-cache-dir .

USER appuser
CMD ["python", "-m", "agent_eval.persisted_demo"]
