FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /data/models /data/output

# Set environment variables
ENV MODELS_DIR=/data/models
ENV OUTPUT_DIR=/data/output
ENV FLASK_APP=civitai_manager/web_app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Start the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
