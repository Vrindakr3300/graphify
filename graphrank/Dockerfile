FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY graphrank/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the graphify package and graphrank application
COPY graphify /app/graphify
COPY graphrank /app/graphrank

# Install graphify in editable mode
RUN pip install --no-cache-dir -e /app

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "graphrank.backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
