import os
import numpy as np
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
import hashlib
import re

from database import Base, engine, SessionLocal, Resume
from vectorstore import add_resume_vector, search_similar, save_index, load_index, clear_index, get_index_stats
from agents import parse_resume_pdf, generate_match_explanation, llm

# Load the sentence transformer model
# Using a lightweight but effective model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FastAPI app
app = FastAPI(title="Resume Matching API")

# CORS middleware (if needed for local frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# No need for OpenAI API key anymore as we're using sentence-transformers

# On startup, create tables and load existing resumes into FAISS index
@app.on_event("startup")
def startup_event():
    # Create database tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Try to load existing FAISS index from disk
    if load_index():
        print("Successfully loaded existing FAISS index")
    else:
        print("No existing FAISS index found, will build from database")
    
    # Always rebuild/verify index from database to ensure it's up to date
    print("Verifying FAISS index with database records...")
    db = SessionLocal()
    resumes = db.query(Resume).all()
    
    # Track if we need to update any embeddings
    updated = False
    valid_vectors = 0
    
    # Count resumes and provide debug info
    print(f"Found {len(resumes)} resumes in database")
    
    for res in resumes:
        try:
            if res.embedding:
                # If embedding exists in the database, use it
                vector = json.loads(res.embedding)
                add_resume_vector(res.id, vector)
                valid_vectors += 1
                continue
                
            # If no embedding in DB, generate it
            content = ""
            if res.skills:
                content += f"Skills: {res.skills}. "
            if res.education:
                content += f"Education: {res.education}. "
            if res.years_of_experience is not None:
                content += f"{res.years_of_experience} years of experience. "
            if res.work_experience:
                content += f"Work experience: {res.work_experience}. "
            if not content:
                print(f"Warning: Resume {res.id} ({res.name}) has no content for embedding")
                # Use name if nothing else is available
                content = res.name
                
            # Generate embedding for the resume content
            embed_vector = model.encode(content).tolist()
            
            # Save embedding to database
            res.embedding = json.dumps(embed_vector)
            db.commit()
            
            # Add to FAISS index
            add_resume_vector(res.id, embed_vector)
            valid_vectors += 1
            updated = True
            
            print(f"Generated new embedding for resume {res.id} ({res.name})")
            
        except Exception as e:
            print(f"Embedding load failed for resume {res.id}: {e}")
    
    db.close()
    
    # Log the final count of vectors
    print(f"FAISS index contains {valid_vectors} resume vectors")
    
    # Save the index to disk for future use
    if updated or valid_vectors > 0:
        save_index()
        print("Saved updated FAISS index to disk")

# On shutdown, save the index
@app.on_event("shutdown")
def shutdown_event():
    save_index()

# Dependency function to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Endpoint: Upload a single resume
@app.post("/resumes")
def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db), background_tasks: BackgroundTasks = None):
    """
    Upload a resume PDF/DOCX file to the system.
    Parses the resume, stores structured info in DB, and indexes the resume for matching.
    Also validates the parsed data and checks for duplicates.
    """
    # Basic validation
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    # Read file content
    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")
    # Parse resume to structured data (with validation if possible)
    parsed = parse_resume_pdf(file_bytes, validate=True)
    
    # Extract fields from parsed data
    name = parsed.get("name") or filename.rsplit(".", 1)[0]  # fallback: use filename (without extension) as name
    contact = parsed.get("contact")
    skills = parsed.get("skills")
    education = parsed.get("education")
    years = parsed.get("years_of_experience")
    work_experience = parsed.get("work_experience")
    resume_hash = parsed.get("resume_hash")
    validation_status = parsed.get("validation_status", "not_validated")
    ai_parsed_data = parsed.get("ai_parsed_data")
    validated_data = parsed.get("validated_data")
    
    # Check for duplicate based on hash
    if resume_hash:
        existing = db.query(Resume).filter(Resume.resume_hash == resume_hash).first()
        if existing:
            return {
                "id": existing.id, 
                "name": existing.name, 
                "detail": "Resume already exists in the database",
                "is_duplicate": True
            }
    
    # Save file to disk (in a designated resumes directory)
    os.makedirs("resumes", exist_ok=True)
    file_path = os.path.join("resumes", filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    
    # Generate embedding for the resume
    content_to_embed = parsed.get("text")
    if not content_to_embed:
        # If full text is not available, use structured fields as a proxy for embedding content
        content_to_embed = ""
        if skills:
            content_to_embed += f"Skills: {skills}. "
        if education:
            content_to_embed += f"Education: {education}. "
        if years:
            content_to_embed += f"{years} years of experience. "
        if work_experience:
            # Add work experience to embedding content
            try:
                if isinstance(work_experience, str):
                    work_exp_json = json.loads(work_experience)
                else:
                    work_exp_json = work_experience
                    
                for job in work_exp_json:
                    content_to_embed += f"{job.get('title')} at {job.get('company')}. {job.get('description')} "
            except:
                # If we can't parse work experience as JSON, use it as a string
                content_to_embed += f"Work experience: {work_experience}. "
                
        if not content_to_embed:
            content_to_embed = name  # at least use name if nothing else
    
    try:
        # Generate embedding using sentence-transformers
        vector = model.encode(content_to_embed).tolist()
        
        # Create Resume record in DB with embedding and validation fields
        new_resume = Resume(
            name=name,
            contact=contact,
            skills=(",".join(skills) if isinstance(skills, list) else skills) if skills else None,
            education=education,
            years_of_experience=years,
            work_experience=json.dumps(work_experience) if isinstance(work_experience, list) else work_experience,
            resume_path=file_path,
            embedding=json.dumps(vector),  # Store embedding in database
            resume_hash=resume_hash,
            ai_parsed_data=ai_parsed_data,
            validated_data=validated_data,
            validation_status=validation_status,
            is_duplicate=False
        )
        db.add(new_resume)
        db.commit()
        db.refresh(new_resume)  # get the generated ID
        
        # Add to vector index
        add_resume_vector(new_resume.id, vector)
        
        # Persist the updated index in the background
        if background_tasks:
            background_tasks.add_task(save_index)
        else:
            # Fallback to immediate save if background_tasks not available
            save_index()
        
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process resume: {e}")
        
    return {
        "id": new_resume.id, 
        "name": new_resume.name, 
        "detail": "Resume uploaded and indexed successfully.",
        "validation_status": validation_status
    }

# Endpoint: Match job description against all stored resumes
@app.post("/match")
def match_resumes(job_description: str = Form(...), top_k: int = 5, db: Session = Depends(get_db)):
    """
    Given a job description, return top matching resumes from the database, ranked by relevance.
    Also provides detailed reasoning for each match and improvement recommendations.
    
    This endpoint implements workflow 1: User provides a job description, system returns best
    candidates from our database with explanations and improvement suggestions.
    """
    if not job_description or job_description.strip() == "":
        raise HTTPException(status_code=400, detail="Job description is required.")
    
    # Check if there are any resumes in the database first
    resume_count = db.query(Resume).count()
    if resume_count == 0:
        return {
            "matches": [],
            "status": "empty_database",
            "message": "No resumes found in the database. Please add resumes first using /resumes or /add_test_samples endpoints."
        }
    
    # Get FAISS index stats before searching
    index_stats = get_index_stats()
    if index_stats["vector_count"] == 0:
        # If database has resumes but index is empty, try to rebuild index
        print("FAISS index is empty but database has resumes. Rebuilding index...")
        # Import necessary function
        from vectorstore import clear_index
        
        # Clear and rebuild index
        clear_index()
        
        # Add vectors from database
        updated = False
        valid_vectors = 0
        
        for res in db.query(Resume).all():
            try:
                if res.embedding:
                    vector = json.loads(res.embedding)
                    if add_resume_vector(res.id, vector):
                        valid_vectors += 1
                        updated = True
            except Exception as e:
                print(f"Error adding vector for resume {res.id}: {e}")
        
        # Save the updated index
        if updated:
            save_index()
            print(f"Rebuilt index with {valid_vectors} vectors")
        
        # If still no vectors, return helpful message
        if valid_vectors == 0:
            return {
                "matches": [],
                "status": "empty_index",
                "message": "No vector embeddings found. Please add resumes with proper embeddings."
            }
    
    # Embed the job description using sentence-transformers
    try:
        job_vector = model.encode(job_description).tolist()
    except Exception as e:
        print(f"Error generating job description embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation error: {e}")
    
    # Search in FAISS index for similar resumes
    results = search_similar(job_vector, top_k=top_k)
    if not results:
        # Check if it's because the index is empty or no good matches
        index_stats = get_index_stats()
        if index_stats["vector_count"] == 0:
            return {
                "matches": [],
                "status": "empty_index",
                "message": "No resume vectors found in the index. Please add resumes first."
            }
        else:
            return {
                "matches": [],
                "status": "no_matches",
                "message": f"No matching resumes found for the given job description among {index_stats['vector_count']} indexed resumes."
            }
    
    # Retrieve resume details from DB and generate explanations
    matches = []
    for res_id, score in results:
        resume_rec = db.query(Resume).get(res_id)
        if not resume_rec:
            continue
        
        # Prepare structured data for explanation (from DB fields)
        structured = {
            "name": resume_rec.name,
            "contact": resume_rec.contact,
            "skills": resume_rec.skills,
            "education": resume_rec.education,
            "years_of_experience": resume_rec.years_of_experience,
            "work_experience": resume_rec.work_experience  # Include work experience for better matching
        }
        
        # Generate comprehensive match explanation and improvement plan
        explanation = generate_match_explanation(job_description, structured)
        
        # Extract the match score from the explanation rather than using FAISS similarity
        # Look for patterns like "Match Score: 75/100" in the explanation
        score_match = re.search(r'[Mm]atch [Ss]core:?\s*(\d+)(?:/100)?', explanation)
        if score_match:
            similarity_pct = int(score_match.group(1))
            print(f"Extracted match score {similarity_pct} from explanation for {resume_rec.name}")
        else:
            # Fallback to FAISS similarity if we can't extract a score
            similarity_pct = int((score * 100)) if score <= 1 else int(score)
            print(f"Using FAISS similarity score {similarity_pct} for {resume_rec.name}")
        
        matches.append({
            "resume_id": res_id,
            "name": resume_rec.name,
            "score": similarity_pct,
            "explanation": explanation
        })
    
    # Sort matches by score in descending order
    matches = sorted(matches, key=lambda x: x["score"], reverse=True)
    
    return {
        "matches": matches,
        "status": "success",
        "count": len(matches)
    }

# Endpoint: Match job description against a list of uploaded resumes (without storing them in DB)
@app.post("/match_files")
def match_multiple_resumes(
    job_description: str = Form(...), 
    resumes: list[UploadFile] = File(...),
    store_in_db: bool = Form(False),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Given a job description and multiple resume files, return a ranked list of these resumes with match explanations.
    
    This endpoint implements workflow 2: User uploads multiple resumes and a job description, 
    system ranks the resumes and provides explanations.
    
    If store_in_db is True, the resumes will also be stored in the database for future use.
    """
    if not job_description:
        raise HTTPException(status_code=400, detail="Job description is required.")
    if not resumes or len(resumes) == 0:
        raise HTTPException(status_code=400, detail="No resumes provided.")
    
    print(f"Processing match for job description ({len(job_description)} chars) against {len(resumes)} resumes")
    
    # Embed the job description once using sentence-transformers
    try:
        job_vector = model.encode(job_description).tolist()
        # Normalize job vector for cosine similarity
        jvec = np.array(job_vector, dtype='float32')
        norm = np.linalg.norm(jvec)
        if norm != 0:
            jvec = jvec / norm
        print(f"Successfully embedded job description, vector length: {len(job_vector)}")
    except Exception as e:
        print(f"Error embedding job description: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation error: {e}")
    
    temporary_results = []
    stored_resume_ids = []
    duplicates = []
    
    # Process each resume file: parse, embed, and score against job vector
    for file_idx, file in enumerate(resumes):
        try:
            print(f"Processing resume {file_idx+1}/{len(resumes)}: {file.filename}")
            content = file.file.read()
            if not content:
                print(f"Skipping empty file: {file.filename}")
                continue
            
            # Parse resume to get structured data with validation
            parsed = parse_resume_pdf(content, validate=True)
            validation_status = parsed.get("validation_status", "unknown")
            resume_hash = parsed.get("resume_hash")
            
            print(f"Parsed resume {file.filename}, validation status: {validation_status}")
            
            # Check for duplicates if storing in DB
            if store_in_db and resume_hash:
                existing = db.query(Resume).filter(Resume.resume_hash == resume_hash).first()
                if existing:
                    print(f"Resume {file.filename} is a duplicate of resume ID {existing.id}")
                    duplicates.append({
                        "filename": file.filename,
                        "duplicate_of_id": existing.id,
                        "duplicate_of_name": existing.name
                    })
                    # Skip further processing for this file
                    continue
            
            # Use structured fields to form an embedding input
            embed_text = parsed.get("text")
            if not embed_text:
                embed_text = ""
                if parsed.get("skills"):
                    embed_text += f"Skills: {parsed['skills']}. "
                if parsed.get("education"):
                    embed_text += f"Education: {parsed['education']}. "
                if parsed.get("years_of_experience"):
                    embed_text += f"{parsed['years_of_experience']} years of experience. "
                
                # Add work experience to embedding if available
                work_exp = parsed.get("work_experience")
                if work_exp:
                    try:
                        work_experience = json.loads(work_exp) if isinstance(work_exp, str) else work_exp
                        for job in work_experience:
                            embed_text += f" {job.get('title')} at {job.get('company')}. {job.get('description')}"
                    except Exception as e:
                        print(f"Error processing work experience: {e}")
                
                if embed_text == "":
                    print(f"Warning: No content extracted for {file.filename}, using filename")
                    embed_text = file.filename  # fallback to using filename as content
            
            print(f"Generated embedding text ({len(embed_text)} chars) for {file.filename}")
            
            # Compute embedding for the resume
            resume_vector = model.encode(embed_text).tolist()
            
            # Normalize resume vector for cosine similarity
            rvec = np.array(resume_vector, dtype='float32')
            norm_r = np.linalg.norm(rvec)
            if norm_r != 0:
                rvec = rvec / norm_r
            
            # Compute similarity score (cosine similarity)
            score = float(np.dot(jvec, rvec))
            print(f"Similarity score for {file.filename}: {score:.4f}")
            
            # Store resume in DB if requested
            resume_id = None
            if store_in_db:
                # Need to reset the file position for storage
                file.file.seek(0)
                filename = file.filename
                os.makedirs("resumes", exist_ok=True)
                file_path = os.path.join("resumes", filename)
                
                with open(file_path, "wb") as f:
                    f.write(content)
                
                # Create Resume record in DB with validation data
                new_resume = Resume(
                    name=parsed.get("name") or filename.split(".")[0],
                    contact=parsed.get("contact"),
                    skills=(",".join(parsed.get("skills")) if isinstance(parsed.get("skills"), list) else parsed.get("skills")) if parsed.get("skills") else None,
                    education=parsed.get("education"),
                    years_of_experience=parsed.get("years_of_experience"),
                    work_experience=json.dumps(parsed.get("work_experience")) if isinstance(parsed.get("work_experience"), list) else parsed.get("work_experience"),
                    resume_path=file_path,
                    embedding=json.dumps(resume_vector),
                    resume_hash=resume_hash,
                    ai_parsed_data=parsed.get("ai_parsed_data"),
                    validated_data=parsed.get("validated_data"),
                    validation_status=validation_status,
                    is_duplicate=False
                )
                
                db.add(new_resume)
                db.commit()
                db.refresh(new_resume)
                resume_id = new_resume.id
                
                # Add to vector index
                add_resume_vector(new_resume.id, resume_vector)
                stored_resume_ids.append(resume_id)
                print(f"Stored resume in database with ID {resume_id}")
            
            # Generate explanation via agent
            structured = {
                "name": parsed.get("name") or file.filename.split(".")[0],
                "contact": parsed.get("contact"),
                "skills": parsed.get("skills"),
                "education": parsed.get("education"),
                "years_of_experience": parsed.get("years_of_experience"),
                "work_experience": parsed.get("work_experience")
            }
            
            print(f"Generating match explanation for {file.filename}")
            explanation = generate_match_explanation(job_description, structured)
            
            # Extract the match score from the explanation rather than using similarity
            score_match = re.search(r'[Mm]atch [Ss]core:?\s*(\d+)(?:/100)?', explanation)
            if score_match:
                similarity_pct = int(score_match.group(1))
                print(f"Extracted match score {similarity_pct} from explanation for {file.filename}")
            else:
                # Fallback to FAISS similarity if we can't extract a score
                similarity_pct = int(score * 100)
                print(f"Using FAISS similarity score {similarity_pct} for {file.filename}")
            
            temporary_results.append({
                "name": structured["name"],
                "score": similarity_pct,
                "explanation": explanation,
                "resume_id": resume_id,  # Will be None if not stored in DB
                "validation_status": validation_status
            })
            
            print(f"Completed processing for {file.filename}")
            
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            # Continue with other files if one fails
    
    # If we stored any resumes, save the index in the background
    if store_in_db and stored_resume_ids:
        if background_tasks:
            background_tasks.add_task(save_index)
            print(f"Added save_index task to background tasks")
        else:
            save_index()
            print(f"Saved FAISS index immediately")
    
    # Sort the results by score (highest first)
    ranked = sorted(temporary_results, key=lambda x: x["score"], reverse=True)
    
    print(f"Returning {len(ranked)} matches, {len(stored_resume_ids)} stored resumes, {len(duplicates)} duplicates")
    
    return {
        "matches": ranked,
        "stored_resume_ids": stored_resume_ids if store_in_db else [],
        "duplicates": duplicates
    }

# Endpoint: Get diagnostic information about the system
@app.get("/diagnostics")
def get_diagnostics(db: Session = Depends(get_db)):
    """
    Return diagnostic information about the system, including:
    - Database status and counts
    - FAISS index statistics
    - Environment information
    
    This is useful for debugging system issues.
    """
    try:
        # Get database stats
        resume_count = db.query(Resume).count()
        resume_samples = []
        
        # Get a few sample resumes for inspection
        for resume in db.query(Resume).limit(3).all():
            resume_samples.append({
                "id": resume.id,
                "name": resume.name,
                "skills": resume.skills,
                "has_embedding": resume.embedding is not None,
                "embedding_length": len(json.loads(resume.embedding)) if resume.embedding else 0,
                "work_experience": resume.work_experience is not None,
                "validation_status": resume.validation_status
            })
            
        # Get index stats
        index_stats = get_index_stats()
        
        # Get environment info
        env_info = {
            "embedding_model": "all-MiniLM-L6-v2",
            "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "upload_limit": os.getenv("UPLOAD_LIMIT", "default"),
            "faiss_index_file_exists": os.path.exists("faiss_index.bin"),
            "resumes_directory_exists": os.path.exists("resumes"),
            "resumes_directory_count": len(os.listdir("resumes")) if os.path.exists("resumes") else 0
        }
        
        return {
            "status": "ok",
            "timestamp": str(np.datetime64('now')),
            "database": {
                "resume_count": resume_count,
                "resume_samples": resume_samples
            },
            "faiss_index": index_stats,
            "environment": env_info
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": str(np.datetime64('now'))
        }

# Endpoint: Force rebuild of the FAISS index
@app.post("/rebuild_index")
def rebuild_index(background_tasks: BackgroundTasks = None, db: Session = Depends(get_db)):
    """
    Force a complete rebuild of the FAISS index from the database.
    This is useful if the index becomes corrupted or out of sync.
    """
    try:
        # Clear the existing index
        clear_index()
        
        # Get all resumes from the database
        resumes = db.query(Resume).all()
        print(f"Rebuilding index with {len(resumes)} resumes from database")
        
        # Track progress
        valid_vectors = 0
        errors = 0
        
        for res in resumes:
            try:
                if res.embedding:
                    # If embedding exists in the database, use it
                    vector = json.loads(res.embedding)
                    if add_resume_vector(res.id, vector):
                        valid_vectors += 1
                    else:
                        errors += 1
                else:
                    # Log missing embedding
                    print(f"Resume {res.id} has no embedding")
                    errors += 1
            except Exception as e:
                print(f"Error processing resume {res.id}: {e}")
                errors += 1
        
        # Save the index
        if valid_vectors > 0:
            if background_tasks:
                background_tasks.add_task(save_index)
                print(f"Added save_index task to background tasks")
            else:
                save_index()
                print(f"Saved FAISS index immediately")
            
        return {
            "status": "success",
            "vectors_added": valid_vectors,
            "errors": errors,
            "total_resumes": len(resumes)
        }
    except Exception as e:
        print(f"Error rebuilding index: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

# Endpoint: Add test samples to the database for quick testing
@app.post("/add_test_samples")
def add_test_samples(background_tasks: BackgroundTasks = None, db: Session = Depends(get_db)):
    """
    Add some test resume samples to the database for quick testing.
    This is useful during development and debugging.
    
    Uses a CrewAI agent to generate 10 diverse resumes with different skills and backgrounds.
    """
    try:
        # Check if we already have samples
        existing_count = db.query(Resume).count()
        if existing_count > 0:
            return {
                "status": "skipped",
                "message": f"Database already contains {existing_count} resumes. No new samples added."
            }
            
        print("Generating test resume samples for the database")
        
        # Create a function to generate resumes using CrewAI if available
        def generate_diverse_resumes(count=10):
            # If we have LLM access, use it to generate diverse resumes
            if llm:
                from crewai import Agent, Task, Crew
                
                resume_generator = Agent(
                    role="Resume Profile Generator",
                    goal="Create diverse and realistic resume profiles for testing",
                    backstory="""You are an expert resume creator who can generate realistic 
                    candidate profiles across different industries, experience levels, and skill sets.
                    You create detailed work histories, education backgrounds, and skill lists that
                    would be found in real resumes.""",
                    llm=llm,
                    verbose=True
                )
                
                generation_task = Task(
                    description=f"""
                    Create {count} diverse and realistic resume profiles for testing an AI resume matching system.
                    
                    Each resume should include:
                    1. Full name (diverse backgrounds)
                    2. Contact information (email)
                    3. Skills (specific to the role/industry)
                    4. Education details
                    5. Years of experience (varying from 0-15 years)
                    6. Work experience (2-4 positions with company, title, duration, and description)
                    
                    Create resumes across these industries and roles:
                    - Software development (frontend, backend, fullstack)
                    - Data science and analytics
                    - Design (UX/UI, graphic design)
                    - Project/product management
                    - Marketing and content creation
                    - DevOps and cloud engineering
                    - Cybersecurity
                    - Finance and accounting
                    - Sales and business development
                    - Healthcare and biotech
                    
                    Make the resumes realistic with skills that match the positions.
                    
                    Return your answer as a JSON array where each resume is an object with the keys:
                    "name", "contact", "skills", "education", "years_of_experience", "work_experience"
                    
                    The work_experience field should be an array of objects with "company", "title", "duration" and "description" keys.
                    
                    IMPORTANT: Return ONLY the JSON array with no additional text.
                    """,
                    agent=resume_generator,
                    expected_output="A JSON array of resume objects"
                )
                
                try:
                    # Create and run the crew
                    crew = Crew(
                        agents=[resume_generator],
                        tasks=[generation_task],
                        verbose=True
                    )
                    
                    print("Starting resume generation crew...")
                    result = crew.kickoff()
                    print(f"Resume generation complete, got result of length {len(result) if result else 0}")
                    
                    # Extract the JSON array from the result
                    json_match = re.search(r'(\[.*\])', result, re.DOTALL)
                    if json_match:
                        resumes_json = json.loads(json_match.group(1))
                        print(f"Successfully parsed JSON from result, found {len(resumes_json)} resumes")
                        return resumes_json
                except Exception as e:
                    print(f"Error generating resumes with agent: {e}")
            
            # Fallback to pre-defined samples if agent generation fails
            print("Using pre-defined samples since agent generation failed or is unavailable")
            return [
                {
                    "name": "John Smith",
                    "contact": "john.smith@example.com",
                    "skills": "Python, JavaScript, React, Node.js, MongoDB",
                    "education": "Bachelor of Science, Computer Science",
                    "years_of_experience": 5,
                    "work_experience": [
                        {
                            "company": "Tech Solutions Inc.",
                            "title": "Full Stack Developer",
                            "duration": "2018-2022",
                            "description": "Developed web applications using React frontend and Node.js backend with MongoDB database."
                        },
                        {
                            "company": "StartupXYZ",
                            "title": "Junior Developer",
                            "duration": "2016-2018",
                            "description": "Built customer-facing features and internal tools using JavaScript and Python."
                        }
                    ]
                },
                {
                    "name": "Emma Davis",
                    "contact": "emma.davis@example.com",
                    "skills": "Python, R, SQL, Data Analysis, Machine Learning, Statistical Modeling, Tableau",
                    "education": "Master of Science, Data Science",
                    "years_of_experience": 3,
                    "work_experience": [
                        {
                            "company": "Data Insights Co.",
                            "title": "Data Scientist",
                            "duration": "2020-2022",
                            "description": "Developed predictive models for customer churn. Created data visualizations with Tableau."
                        },
                        {
                            "company": "Analytics Firm",
                            "title": "Data Analyst",
                            "duration": "2018-2020",
                            "description": "Performed SQL queries for business intelligence. Created reports using R and Python."
                        }
                    ]
                },
                # Include the rest of the predefined samples here
                {
                    "name": "Michael Chen",
                    "contact": "michael.chen@example.com",
                    "skills": "DevOps, AWS, Docker, Kubernetes, CI/CD, Terraform, Python",
                    "education": "Bachelor of Engineering, Computer Engineering",
                    "years_of_experience": 7,
                    "work_experience": [
                        {
                            "company": "Cloud Solutions Ltd.",
                            "title": "DevOps Engineer",
                            "duration": "2019-2022",
                            "description": "Managed AWS infrastructure using Terraform. Implemented CI/CD pipelines with Jenkins."
                        },
                        {
                            "company": "Tech Infrastructure Inc.",
                            "title": "System Administrator",
                            "duration": "2015-2019",
                            "description": "Maintained Linux servers and deployed applications using Docker containers."
                        }
                    ]
                },
                {
                    "name": "Sophia Rodriguez",
                    "contact": "sophia.rodriguez@example.com",
                    "skills": "UI/UX Design, Figma, Adobe XD, HTML, CSS, Wireframing, Prototyping",
                    "education": "Bachelor of Fine Arts, Graphic Design",
                    "years_of_experience": 4,
                    "work_experience": [
                        {
                            "company": "Design Studio",
                            "title": "UI/UX Designer",
                            "duration": "2019-2022",
                            "description": "Created wireframes and prototypes for mobile applications. Conducted user testing."
                        },
                        {
                            "company": "Creative Agency",
                            "title": "Graphic Designer",
                            "duration": "2017-2019",
                            "description": "Designed visual assets for web and print. Collaborated with marketing team on brand identity."
                        }
                    ]
                },
                {
                    "name": "Robert Johnson",
                    "contact": "robert.johnson@example.com",
                    "skills": "Java, Spring Boot, Microservices, REST APIs, PostgreSQL, Hibernate, JUnit",
                    "education": "Master of Computer Applications",
                    "years_of_experience": 8,
                    "work_experience": [
                        {
                            "company": "Enterprise Solutions",
                            "title": "Senior Backend Developer",
                            "duration": "2018-2022",
                            "description": "Designed and implemented microservices architecture. Developed RESTful APIs using Spring Boot."
                        },
                        {
                            "company": "Financial Tech",
                            "title": "Java Developer",
                            "duration": "2014-2018",
                            "description": "Built backend systems for financial applications. Optimized database queries and performance."
                        }
                    ]
                }
            ]
            
        # Generate diverse resumes
        print("Generating diverse resume samples...")
        resume_samples = generate_diverse_resumes(10)
        
        # Add samples to the database
        added_count = 0
        for sample in resume_samples:
            try:
                # Create a dummy resume path
                filename = f"{sample['name'].replace(' ', '_')}_resume.pdf"
                filepath = os.path.join("resumes", filename)
                os.makedirs("resumes", exist_ok=True)
                
                # Create an empty file if it doesn't exist
                if not os.path.exists(filepath):
                    with open(filepath, "w") as f:
                        f.write(f"Dummy resume for {sample['name']}")
                
                # Ensure work_experience is properly formatted as JSON string
                if isinstance(sample['work_experience'], list):
                    work_experience_json = json.dumps(sample['work_experience'])
                else:
                    work_experience_json = sample['work_experience']
                
                # Create content for embedding
                content_to_embed = f"Skills: {sample['skills']}. "
                content_to_embed += f"Education: {sample['education']}. "
                content_to_embed += f"{sample['years_of_experience']} years of experience. "
                
                # Add work experience details
                if isinstance(sample['work_experience'], list):
                    work_exp = sample['work_experience']
                else:
                    work_exp = json.loads(sample['work_experience'])
                    
                for job in work_exp:
                    content_to_embed += f"{job['title']} at {job['company']}. {job['description']} "
                
                # Generate embedding
                vector = model.encode(content_to_embed).tolist()
                
                # Create a unique hash for this sample
                resume_hash = hashlib.md5(content_to_embed.encode('utf-8')).hexdigest()
                
                # Create the resume record
                new_resume = Resume(
                    name=sample['name'],
                    contact=sample['contact'],
                    skills=sample['skills'],
                    education=sample['education'],
                    years_of_experience=sample['years_of_experience'],
                    work_experience=work_experience_json,
                    resume_path=filepath,
                    embedding=json.dumps(vector),
                    resume_hash=resume_hash,
                    validation_status="validated",
                    is_duplicate=False
                )
                
                db.add(new_resume)
                db.commit()
                db.refresh(new_resume)
                
                # Add to vector index
                add_resume_vector(new_resume.id, vector)
                added_count += 1
                print(f"Added resume for {sample['name']}")
                
            except Exception as e:
                print(f"Error adding sample {sample.get('name', 'Unknown')}: {e}")
        
        # Save the index if we added samples
        if added_count > 0:
            if background_tasks:
                background_tasks.add_task(save_index)
                print(f"Scheduled save_index task to run in background")
            else:
                save_index()
                print(f"Saved FAISS index with {added_count} vectors")
        
        return {
            "status": "success",
            "samples_added": added_count,
            "total_samples": len(resume_samples)
        }
        
    except Exception as e:
        print(f"Error adding test samples: {e}")
        return {
            "status": "error",
            "error": str(e)
        }