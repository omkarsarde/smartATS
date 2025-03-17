#!/bin/bash
# This script prepares the system for deployment

echo "Preparing the resume matching system for deployment..."

# Create the sample resumes script in the backend directory
echo "Creating sample resumes script..."
cat > backend/add_sample_resumes.py << 'EOL'
#!/usr/bin/env python
"""
This script adds sample resumes to the database for testing workflow 1.
Run this after the database has been initialized.
"""
import os
import sys
from sqlalchemy.orm import Session

# Import from the backend modules
from database import SessionLocal, Resume
from vectorstore import add_resume_vector
from sentence_transformers import SentenceTransformer

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def add_sample_resumes():
    """Add sample resumes to the database"""
    print("Adding sample resumes to the database...")
    db = SessionLocal()
    
    # Sample resume data
    sample_resumes = [
        {
            "name": "John Smith",
            "contact": "john.smith@example.com",
            "skills": "Python, JavaScript, React, Node.js, SQL, Docker, AWS",
            "education": "Bachelor's in Computer Science, Stanford University",
            "years_of_experience": 5,
            "resume_path": "resumes/john_smith.pdf"
        },
        {
            "name": "Sarah Johnson",
            "contact": "sarah.johnson@example.com",
            "skills": "Python, Data Analysis, Machine Learning, TensorFlow, PyTorch, SQL, Tableau",
            "education": "Master's in Data Science, MIT",
            "years_of_experience": 3,
            "resume_path": "resumes/sarah_johnson.pdf"
        },
        {
            "name": "Michael Chen",
            "contact": "michael.chen@example.com",
            "skills": "Java, Spring, Hibernate, MySQL, AWS, Microservices, CI/CD",
            "education": "Bachelor's in Software Engineering, UC Berkeley",
            "years_of_experience": 7,
            "resume_path": "resumes/michael_chen.pdf"
        },
        {
            "name": "Emily Davis",
            "contact": "emily.davis@example.com",
            "skills": "UX/UI Design, Figma, Sketch, HTML, CSS, JavaScript, User Research",
            "education": "Bachelor's in Graphic Design, RISD",
            "years_of_experience": 4,
            "resume_path": "resumes/emily_davis.pdf"
        },
        {
            "name": "David Wilson",
            "contact": "david.wilson@example.com",
            "skills": "DevOps, Kubernetes, Docker, Terraform, AWS, Azure, CI/CD, Python, Go",
            "education": "Bachelor's in Computer Engineering, Georgia Tech",
            "years_of_experience": 6,
            "resume_path": "resumes/david_wilson.pdf"
        }
    ]
    
    # Create directory for resume files if it doesn't exist
    os.makedirs("resumes", exist_ok=True)
    
    # Add resumes to the database
    for resume_data in sample_resumes:
        # Create dummy file if it doesn't exist
        if not os.path.exists(resume_data["resume_path"]):
            with open(resume_data["resume_path"], "wb") as f:
                f.write(f"Dummy resume file for {resume_data['name']}".encode('utf-8'))
        
        # Check if resume already exists in database
        existing = db.query(Resume).filter(Resume.name == resume_data["name"]).first()
        if existing:
            print(f"Resume for {resume_data['name']} already exists, skipping...")
            continue
        
        # Create new resume record
        new_resume = Resume(**resume_data)
        db.add(new_resume)
        db.commit()
        db.refresh(new_resume)
        
        # Generate embedding for vector search
        content_to_embed = f"Skills: {new_resume.skills}. Education: {new_resume.education}. {new_resume.years_of_experience} years of experience."
        vector = model.encode(content_to_embed).tolist()
        add_resume_vector(new_resume.id, vector)
        
        print(f"Added resume for {new_resume.name} (ID: {new_resume.id})")
    
    print("Sample resumes added successfully!")
    db.close()

if __name__ == "__main__":
    add_sample_resumes()
EOL

# Update agents.py with our improved version
echo "Updating agents.py..."
cat > backend/agents.py << 'EOL'
import os
from crewai import Agent, Task, Crew, Process

# Try to import OpenAI LLM adapter
try:
    from crewai.llms import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("WARNING: Could not import OpenAI adapter from crewai.llms")

# Try to import local LLM support
try:
    from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HAS_LOCAL_LLM = True
except ImportError:
    HAS_LOCAL_LLM = False

# Stub for PDF resume parsing (to be implemented in the future with PDF/DOCX parsing libraries)
def parse_resume_pdf(file_bytes: bytes) -> dict:
    """
    Placeholder function to parse resume PDF content into structured data.
    Returns a dictionary with keys: name, contact, skills, education, years_of_experience, text.
    Currently, this is a stub that returns minimal dummy data.
    """
    # In a real implementation, we would use libraries like PyMuPDF, pdfplumber, or pymupdf
    # to extract text from PDF and then potentially use NLP to extract structured data
    
    # For testing, return some sample data
    return {
        "name": "Test Candidate",
        "contact": "test@example.com",
        "skills": "Python, FastAPI, SQL, Data Analysis, Machine Learning",
        "education": "Bachelor's in Computer Science",
        "years_of_experience": 3,
        "text": "Experienced software engineer with skills in Python, FastAPI, SQL, and data analysis. 3 years of experience in software development and machine learning projects."
    }

# Select the appropriate LLM to use
llm = None

# Try using OpenAI if API key is available
if os.getenv("OPENAI_API_KEY") and HAS_OPENAI:
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
    print("Using OpenAI GPT-3.5 for explanations")
# If no OpenAI, try Anthropic if API key is available
elif os.getenv("ANTHROPIC_API_KEY"):
    try:
        from crewai.llms import Anthropic
        llm = Anthropic(model="claude-3-haiku-20240307")
        print("Using Anthropic Claude for explanations")
    except ImportError:
        print("Could not import Anthropic adapter from crewai.llms")

# Fall back to local LLM as last resort
if llm is None and HAS_LOCAL_LLM:
    try:
        # Try to load a smaller model that can run on CPU
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model that can run on CPU
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                  device_map="auto", 
                                                  load_in_8bit=True,  # Reduce memory usage
                                                  trust_remote_code=True)
        
        # Create a pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.7,
            repetition_penalty=1.1,
        )
        
        # Create LangChain wrapper
        local_llm = HuggingFacePipeline(pipeline=pipe)
        llm = local_llm
        print("Successfully loaded local LLM model for explanations")
    except Exception as e:
        print(f"Could not load local LLM model: {e}")
        llm = None  # CrewAI will use its default

print(f"LLM configured: {llm.__class__.__name__ if llm else 'Using CrewAI default'}")

# Initialize a CrewAI agent for matching and explanation
matching_agent = Agent(
    role="HR Matchmaker",
    goal="Assess candidate resume against job description and provide match score with detailed reasoning.",
    backstory=("You are an expert HR recruiter with deep knowledge of job requirements and resume analysis. "
               "You compare job descriptions with candidate resumes, identify skill matches, related (adjacent) skills, "
               "missing skills, and suggest improvements for the candidate."),
    llm=llm,  # Use the selected LLM
    verbose=True,  # Enable verbose mode for debugging
)
 
def generate_match_explanation(job_description: str, resume_structured: dict) -> str:
    """
    Use the CrewAI agent to generate a detailed explanation for how well the resume matches the job description.
    Returns a text explanation including matched skills, adjacent skills, missing skills, and improvement recommendations.
    """
    # Prepare the content for the agent task using structured data (to minimize token usage)
    name = resume_structured.get("name") or "Candidate"
    skills = resume_structured.get("skills") or ""
    education = resume_structured.get("education") or ""
    years = resume_structured.get("years_of_experience")
    years_text = f"{years} years" if years is not None else "N/A"
    resume_info = (
        f"Resume of {name}:\n"
        f"- Skills: {skills}\n"
        f"- Education: {education}\n"
        f"- Experience: {years_text}\n"
    )
    # Define the task prompt for the agent
    task_description = (
        f"Job Description:\n{job_description}\n\n"
        f"{resume_info}\n"
        "Analyze how well this resume matches the job. Provide:\n"
        "1. A match score out of 10 (higher is better).\n"
        "2. Matched skills (skills in resume that meet job requirements).\n"
        "3. Adjacent skills (resume skills that are relevant or similar to required skills, even if not exact).\n"
        "4. Missing important skills from the resume (that the job description asks for).\n"
        "5. A short recommendation on how the candidate could improve their resume for this job."
    )
    
    # Create a CrewAI task for this analysis
    task = Task(
        description=task_description,
        agent=matching_agent,
        expected_output="A detailed analysis of how well the resume matches the job description with specific sections for matched skills, adjacent skills, missing skills, and improvement recommendations."
    )
    
    # Create a crew with just our matching agent
    crew = Crew(
        agents=[matching_agent],
        tasks=[task],
        verbose=True,
        process=Process.sequential  # Use sequential processing for a single agent
    )
    
    try:
        result = crew.kickoff()  # Execute the agent task
        explanation = str(result)
    except Exception as e:
        # Fallback if there's an error with the agent or LLM
        print(f"Agent explanation generation failed: {e}")
        explanation = (
            f"Score: 5/10\n\n"
            f"Analysis could not be generated due to an error. The system was able to determine "
            f"that this resume is somewhat relevant to the job description, but a detailed analysis "
            f"could not be produced.\n\n"
            f"Technical Error: {str(e)}\n\n"
            f"Please ensure your OpenAI API key is properly configured if you'd like to receive "
            f"detailed match explanations."
        )
    
    return explanation
EOL

# Make sure the backend Dockerfile is updated to include the sample resumes script
echo "Updating the backend Dockerfile..."
cat > backend/Dockerfile << 'EOL'
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

# Copy backend code
COPY ./ ./

# Create directory for resumes
RUN mkdir -p resumes

# Expose port for FastAPI
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOL

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOL
DB_USER=resuser
DB_PASS=respass
DB_NAME=resumedb
# Replace with your actual API key
OPENAI_API_KEY=your_openai_key_here
EOL
    echo "Created .env file. Please edit it to add your OpenAI API key."
fi

echo "Setup complete! Now you can run the system with:"
echo "./start.sh"