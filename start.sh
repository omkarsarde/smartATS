#!/bin/bash
# This script starts the resume matching system

echo "Starting the Resume Matching System..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Creating with default values..."
    cat > .env << EOL
DB_USER=resuser
DB_PASS=respass
DB_NAME=resumedb
# Replace with your actual API key
OPENAI_API_KEY=your_openai_key_here
EOL
    echo "Created .env file with default values. Please edit it to add your OpenAI API key."
fi

# Check if OpenAI API key is set
OPENAI_KEY=$(grep OPENAI_API_KEY .env | cut -d '=' -f2)
if [ "$OPENAI_KEY" = "your_openai_key_here" ]; then
    echo "Warning: Please update your OpenAI API key in the .env file for full functionality."
fi

# Create backend directory for add_sample_resumes.py if it doesn't exist
if [ ! -f backend/add_sample_resumes.py ]; then
    echo "Sample resume script not found. Please ensure the codebase is properly set up."
    exit 1
fi

# First check if the system is already running
if docker ps | grep resume_ > /dev/null; then
    echo "Some resume system containers are already running."
    echo "Do you want to restart the system? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Stopping existing containers..."
        ./down.sh
    else
        echo "Keeping existing containers. Exiting."
        exit 0
    fi
fi

# Start the containers
echo "Starting containers with docker-compose..."
docker compose up --build -d

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to start containers. Check docker compose logs for details."
    exit 1
fi

echo "Waiting for services to start..."

# Wait for backend to be ready using a more reliable method
echo "Waiting for backend to be ready..."
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8000/docs > /dev/null; then
        echo "Backend is ready!"
        break
    fi
    attempt=$((attempt+1))
    echo "Waiting for backend... ($attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "Warning: Backend may not be ready. Continuing anyway..."
fi

# Wait for frontend to be ready
echo "Waiting for frontend to be ready..."
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8501 > /dev/null; then
        echo "Frontend is ready!"
        break
    fi
    attempt=$((attempt+1))
    echo "Waiting for frontend... ($attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "Warning: Frontend may not be ready. Continuing anyway..."
fi

# Only add sample resumes if specifically requested
echo "Do you want to add sample resumes to the database? (y/n)"
read -r add_samples
if [[ "$add_samples" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Adding sample resumes..."
    docker exec resume_backend python /app/add_sample_resumes.py
fi

echo ""
echo "System is now running!"
echo "Frontend UI: http://localhost:8501"
echo "Backend API: http://localhost:8000/docs"
echo ""
echo "To stop the system and clean up data, run: ./down.sh"
echo "To view logs, run: docker compose logs -f"