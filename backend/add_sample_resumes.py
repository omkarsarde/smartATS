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
from database import Base, engine
Base.metadata.create_all(bind=engine)

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
