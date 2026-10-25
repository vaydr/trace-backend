FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md uv.lock ./
COPY licenses ./licenses
COPY src ./src
RUN pip install --no-cache-dir uv && uv sync --frozen --extra ml --no-dev
COPY configs ./configs
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["trace", "serve", "--host", "0.0.0.0", "--port", "8000"]
