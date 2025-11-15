FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application modules
COPY main.py .
COPY agents.py .
COPY tasks.py .
COPY vision_extractor.py .
COPY valuation_rag.py .

# Create directories for data
RUN mkdir -p /data/financials /data/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FILE_SHARE_PATH=/data/financials
ENV VALUATION_PDF_PATH=/data/valuation_parameters.pdf
ENV OUTPUT_PATH=/data/output
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV VISION_MODEL=qwen2.5vl:7b
ENV ANALYSIS_MODEL=gpt-oss:20b
ENV EMBEDDING_MODEL=nomic-embed-text

# Run the application
CMD ["python", "main.py"]