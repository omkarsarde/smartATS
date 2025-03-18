#!/bin/bash
# down.sh - Script to stop the resume matching system and clean up resources

echo "Stopping the Resume Matching System..."

# Check if docker-compose is installed and running
if ! command -v docker compose &> /dev/null; then
    echo "Error: docker compose is not installed or not in PATH"
    exit 1
fi

# Check if containers are running before attempting to stop
if ! docker ps | grep -q "resume_"; then
    echo "No resume system containers appear to be running."
fi

# Stop containers and remove volumes
echo "Stopping containers and removing volumes..."
docker compose down -v

# Clean up any orphaned volumes that might be related to our app
echo "Checking for orphaned volumes..."
ORPHANED_VOLUMES=$(docker volume ls -qf dangling=true | grep -E 'db_data|resumes_data' 2>/dev/null)
if [ -n "$ORPHANED_VOLUMES" ]; then
    echo "Removing orphaned volumes related to the application..."
    echo "$ORPHANED_VOLUMES" | xargs docker volume rm
fi

# Remove any created resume files in the local directory
if [ -d "backend/resumes" ]; then
    echo "Removing sample resume files..."
    rm -rf backend/resumes
fi

echo ""
echo "System has been shut down and data has been cleared."
echo "Run ./start.sh to restart the system with a clean state." 