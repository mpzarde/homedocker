FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies including PyYAML
RUN pip install --no-cache-dir fastapi uvicorn httpx orjson openai pyyaml

# Copy router code and configuration
COPY ollama-router.py .
COPY config.yaml .

EXPOSE 8000

CMD ["uvicorn", "ollama-router:app", "--host", "0.0.0.0", "--port", "8000"]
