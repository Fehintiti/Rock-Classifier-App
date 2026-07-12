FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files. The model itself (~116MB) is NOT copied here:
# it's too large for Git, so it's downloaded from Cloud Storage at container
# startup instead (see main.py / DEPLOYMENT.md).
COPY main.py .
COPY deployment_info.json .

# Cloud Run injects PORT (defaults to 8080) and expects the container to listen on it
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
