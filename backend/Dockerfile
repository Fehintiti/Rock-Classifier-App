FROM python:3.10-slim

WORKDIR /app

# Install system dependencies and AWS CLI
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    unzip \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY deployment_info.json .

# Create startup script that downloads model then starts server
RUN echo '#!/bin/bash\n\
echo "Downloading model from S3..."\n\
aws s3 cp s3://rock-classifier-model/model_cleaned_best.pth /app/model_cleaned_best.pth --region us-east-1\n\
echo "Model downloaded! Starting server..."\n\
exec uvicorn main:app --host 0.0.0.0 --port 8000\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Run startup script
CMD ["/app/start.sh"]