# smartATS - Resume Matching and Ranking System

This project is a local implementation of a resume matching and ranking system, consisting of a FastAPI backend with a PostgreSQL database and a Streamlit frontend. It uses AI techniques (OpenAI embeddings and LLMs via CrewAI) to rank resumes against job descriptions and provide detailed explanations for the matches.

## Features

- **FastAPI Backend**: Provides API endpoints for uploading resumes and matching them to job descriptions.
- **PostgreSQL Database**: Stores structured information about resumes (name, contact info, skills, experience, education) and references to the resume files.
- **FAISS Vector Store**: Enables semantic search through resume embeddings for efficient candidate retrieval based on job description similarity.
- **Sentence Transformers**: Uses the 'all-MiniLM-L6-v2' model to transform resumes and job descriptions into high-dimensional vectors for semantic comparison, all running locally without API costs.
- **CrewAI Agents**: Autonomous AI agents generate explanations for why a resume is a good (or bad) match, including matched skills, related skills, and improvement suggestions.
- **Streamlit Frontend**: User-friendly web interface to input job descriptions, upload resumes, and view ranked matches with explanations.
- **Dockerized Deployment**: Easily start all components with a single `docker-compose up` command. The setup is modular and can be extended or deployed to cloud services (e.g., Azure ML) in the future.

## Getting Started

### Prerequisites

- **Docker** and **Docker Compose** installed on your system.
- (Optional) OpenAI or Anthropic API key only if you want to use external LLMs with CrewAI for generating explanations. The system can function with local models.

### Setup and Running the App

1. **Clone the repository**
2. **Create a `.env` file** in the project root (if not already present) and fill in the necessary values:
   - `DB_USER`, `DB_PASS`, `DB_NAME`: PostgreSQL database credentials (you can use the defaults provided or change them, but ensure consistency with the compose file).
   - (Optional) `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`: Only needed if you want to use these services with CrewAI.
3. **Build and start the containers**:
   ```bash
   docker-compose up --build
   ```
   This will start three services:
   - `db` (PostgreSQL database)
   - `backend` (FastAPI app, accessible at http://localhost:8000)
   - `frontend` (Streamlit app, accessible at http://localhost:8501)

4. **Access the application**:
   - Open your web browser to **http://localhost:8501** to see the Streamlit frontend.

### Using the Application

- In the Streamlit UI, enter a **Job Description** in the text area. This can be several paragraphs detailing the role, responsibilities, and required skills.
- You have two options to provide resumes for matching:
  1. **Search in Database**: If you want to match against resumes already in the system, ensure no files are uploaded in the "Upload resumes" section. The system will use all resumes stored in the database and return the top matches.
  2. **Upload Resumes for One-off Match**: You can upload one or multiple resume files (PDF or DOCX) using the file uploader. These will be used for matching **without** saving them to the database.
- Click the **"Match Resumes"** button. The backend will process the job description and resumes:
  - It generates an embedding for the job description.