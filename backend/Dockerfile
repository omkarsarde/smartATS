# Backend Dockerfile
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies (if any needed for psycopg2 or others)
RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure langchain-huggingface is installed (for updated API compatibility)
RUN pip install --no-cache-dir langchain-huggingface>=0.0.2

# Pre-download models to speed up first request
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Create directory for resume storage
RUN mkdir -p resumes

# Copy backend code
COPY ./ ./

# Expose port for FastAPI
EXPOSE 8000

# Set env var to accept larger file uploads for resume processing
ENV UPLOAD_LIMIT="10MB"

# Start FastAPI server with hot reload for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]