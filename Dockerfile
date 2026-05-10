# Use a minimal Python runtime
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy repository contents
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the batch process
CMD ["python", "-m", "main"]