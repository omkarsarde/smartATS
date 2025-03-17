#!/bin/bash
# This script starts the resume matching system

echo "Starting the Resume Matching System..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please run setup.sh first."
    exit 1
fi

# Check if OpenAI API key is set
OPENAI_KEY=$(grep OPENAI_API_KEY .env | cut -d '=' -f2)
if [ "$OPENAI_KEY" = "your_openai_key_here" ]; then
    echo "Warning: Please update your OpenAI API key in the .env file for full functionality."
fi

# Create backend directory for add_sample_resumes.py if it doesn't exist
if [ ! -f backend/add_sample_resumes.py ]; then
    echo "Sample resume script not found. Please run setup.sh first."
    exit 1
fi

# Start the containers
echo "Starting containers with docker-compose..."
docker compose up --build -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 15

# Check if the backend is running
if ! docker ps | grep resume_backend > /dev/null; then
    echo "Error: Backend container is not running. Check the logs with 'docker logs resume_backend'."
    exit 1
fi

# Check if the frontend is running
if ! docker ps | grep resume_frontend > /dev/null; then
    echo "Error: Frontend container is not running. Check the logs with 'docker logs resume_frontend'."
    exit 1
fi

# Add sample resumes if needed
echo "Checking if sample resumes need to be added..."
docker exec resume_backend python /app/add_sample_resumes.py

echo ""
echo "System is now running!"
echo "Frontend UI: http://localhost:8501"
echo "Backend API: http://localhost:8000/docs"
echo ""
echo "To stop the system, run: docker compose down"
echo "To view logs, run: docker compose logs -f"