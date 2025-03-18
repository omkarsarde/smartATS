#!/usr/bin/env python
"""
This script adds sample resumes to the database for testing workflow 1.
Run this after the database has been initialized.
"""
import os
import sys
import json
from sqlalchemy.orm import Session

# Import from the backend modules
from database import SessionLocal, Resume, Base, engine
from vectorstore import add_resume_vector
from sentence_transformers import SentenceTransformer

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create tables if they don't exist
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
            "resume_path": "resumes/john_smith.pdf",
            "work_experience": json.dumps([
                {
                    "company": "Tech Solutions Inc.",
                    "title": "Senior Developer",
                    "duration": "2020-2023",
                    "description": "Led development of cloud-native applications using AWS, Docker, and Python."
                },
                {
                    "company": "Startup Innovations",
                    "title": "Full Stack Developer",
                    "duration": "2018-2020",
                    "description": "Built React/Node.js applications and implemented REST APIs."
                }
            ])
        },
        {
            "name": "Sarah Johnson",
            "contact": "sarah.johnson@example.com",
            "skills": "Python, Data Analysis, Machine Learning, TensorFlow, PyTorch, SQL, Tableau",
            "education": "Master's in Data Science, MIT",
            "years_of_experience": 3,
            "resume_path": "resumes/sarah_johnson.pdf",
            "work_experience": json.dumps([
                {
                    "company": "Data Insights Corp",
                    "title": "Data Scientist",
                    "duration": "2021-2023",
                    "description": "Developed machine learning models for customer segmentation and prediction."
                },
                {
                    "company": "Research Analytics",
                    "title": "Data Analyst",
                    "duration": "2019-2021",
                    "description": "Created data visualizations and reports using Tableau and Python."
                }
            ])
        },
        {
            "name": "Michael Chen",
            "contact": "michael.chen@example.com",
            "skills": "Java, Spring, Hibernate, MySQL, AWS, Microservices, CI/CD",
            "education": "Bachelor's in Software Engineering, UC Berkeley",
            "years_of_experience": 7,
            "resume_path": "resumes/michael_chen.pdf",
            "work_experience": json.dumps([
                {
                    "company": "Enterprise Solutions",
                    "title": "Technical Lead",
                    "duration": "2020-2023",
                    "description": "Architected microservices applications with Spring Boot and AWS."
                },
                {
                    "company": "Financial Systems Inc.",
                    "title": "Java Developer",
                    "duration": "2016-2020",
                    "description": "Built high-performance Java applications with Spring Framework."
                },
                {
                    "company": "Tech Consultants",
                    "title": "Junior Developer",
                    "duration": "2014-2016",
                    "description": "Developed and maintained Java applications."
                }
            ])
        },
        {
            "name": "Emily Davis",
            "contact": "emily.davis@example.com",
            "skills": "UX/UI Design, Figma, Sketch, HTML, CSS, JavaScript, User Research",
            "education": "Bachelor's in Graphic Design, RISD",
            "years_of_experience": 4,
            "resume_path": "resumes/emily_davis.pdf",
            "work_experience": json.dumps([
                {
                    "company": "Design Agency",
                    "title": "Senior UX Designer",
                    "duration": "2021-2023",
                    "description": "Led user research and designed interfaces for mobile and web applications."
                },
                {
                    "company": "Creative Solutions",
                    "title": "UI/UX Designer",
                    "duration": "2019-2021",
                    "description": "Created wireframes, prototypes, and final designs using Figma and Sketch."
                }
            ])
        },
        {
            "name": "David Wilson",
            "contact": "david.wilson@example.com",
            "skills": "DevOps, Kubernetes, Docker, Terraform, AWS, Azure, CI/CD, Python, Go",
            "education": "Bachelor's in Computer Engineering, Georgia Tech",
            "years_of_experience": 6,
            "resume_path": "resumes/david_wilson.pdf",
            "work_experience": json.dumps([
                {
                    "company": "Cloud Enterprises",
                    "title": "DevOps Engineer",
                    "duration": "2021-2023",
                    "description": "Managed Kubernetes clusters and implemented CI/CD pipelines."
                },
                {
                    "company": "Infrastructure Solutions",
                    "title": "Site Reliability Engineer",
                    "duration": "2019-2021",
                    "description": "Automated infrastructure deployment with Terraform and CloudFormation."
                },
                {
                    "company": "Tech Systems",
                    "title": "Systems Administrator",
                    "duration": "2017-2019",
                    "description": "Managed Linux-based infrastructure and deployed applications."
                }
            ])
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
        
        # Add work experience details to the embedding
        try:
            work_exp = json.loads(new_resume.work_experience) if new_resume.work_experience else []
            for job in work_exp:
                content_to_embed += f" {job.get('title')} at {job.get('company')}. {job.get('description')}"
        except Exception as e:
            print(f"Warning: Could not parse work experience for {new_resume.name}: {e}")
        
        # Generate and store the vector embedding
        vector = model.encode(content_to_embed).tolist()
        new_resume.embedding = json.dumps(vector)
        db.commit()
        
        # Add to vector index
        add_resume_vector(new_resume.id, vector)
        
        print(f"Added resume for {new_resume.name} (ID: {new_resume.id})")
    
    print("Sample resumes added successfully!")
    db.close()

if __name__ == "__main__":
    add_sample_resumes()
