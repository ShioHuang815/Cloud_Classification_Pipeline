# Base image with Python 3.9
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create directories for data and artifacts if they don't exist
RUN mkdir -p /app/data /app/artifacts

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Volume mount points for data input and artifact output
VOLUME ["/app/data", "/app/artifacts"]

# Default command to run the pipeline
CMD ["python", "pipeline.py"]

# Add an entrypoint script for flexibility
COPY dockerfiles/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]