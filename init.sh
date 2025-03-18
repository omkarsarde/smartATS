#!/bin/bash
# init.sh - Script to initialize the system with required dependencies

echo "Initializing the Resume Matching System..."

# Ensure we're in the project directory
cd "$(dirname "$0")" || exit 1

# Check if Dockerfile exists, to confirm correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: docker-compose.yml not found. Please run this script from the project root directory."
    exit 1
fi

# Update backend dependencies to include langchain-huggingface
echo "Updating backend dependencies..."
if ! grep -q "langchain-huggingface" backend/requirements.txt; then
    # Add the langchain-huggingface package to requirements.txt if not present
    sed -i '/langchain-community/i langchain-huggingface>=0.0.2  # Updated package for HuggingFacePipeline' backend/requirements.txt
    echo "Added langchain-huggingface to requirements.txt"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file with default values..."
    cat > .env << EOL
DB_USER=resuser
DB_PASS=respass
DB_NAME=resumedb
# Replace with your actual API key
OPENAI_API_KEY=your_openai_key_here
EOL
    echo "Created .env file. Please edit it to add your OpenAI API key."
else
    echo ".env file already exists."
fi

# Make scripts executable
echo "Making scripts executable..."
chmod +x start.sh down.sh

echo ""
echo "Initialization complete!"
echo "To start the system, run: ./start.sh"
echo "Note: The first start may take longer as it needs to download models and build Docker images." 