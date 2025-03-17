import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer

from database import Base, engine, SessionLocal, Resume
from vectorstore import add_resume_vector, search_similar
from agents import parse_resume_pdf, generate_match_explanation

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
    # Load existing resumes from DB and add to FAISS index
    db = SessionLocal()
    resumes = db.query(Resume).all()
    for res in resumes:
        try:
            # For existing resumes, we assume embedding was generated and stored or can be re-generated.
            # Here, we attempt to re-generate embeddings using available data (skills + education as proxy for content).
            content = ""
            if res.skills:
                content += f"Skills: {res.skills}. "
            if res.education:
                content += f"Education: {res.education}. "
            if res.years_of_experience is not None:
                content += f"{res.years_of_experience} years of experience. "
            if not content:
                # If no structured content available, skip embedding for now
                continue
            # Generate embedding for the resume content using sentence-transformers
            embed_vector = model.encode(content).tolist()
            add_resume_vector(res.id, embed_vector)
        except Exception as e:
            print(f"Embedding load failed for resume {res.id}: {e}")
    db.close()

# Dependency function to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Endpoint: Upload a single resume
@app.post("/resumes")
def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a resume PDF/DOCX file to the system.
    Parses the resume (stub), stores structured info in DB, and indexes the resume for matching.
    """
    # Basic validation
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    # Read file content
    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")
    # Parse resume to structured data (using stub parser for now)
    parsed = parse_resume_pdf(file_bytes)
    name = parsed.get("name") or filename.rsplit(".", 1)[0]  # fallback: use filename (without extension) as name
    contact = parsed.get("contact")
    skills = parsed.get("skills")
    education = parsed.get("education")
    years = parsed.get("years_of_experience")
    # Save file to disk (in a designated resumes directory)
    os.makedirs("resumes", exist_ok=True)
    file_path = os.path.join("resumes", filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    # Create Resume record in DB
    new_resume = Resume(
        name=name,
        contact=contact,
        skills=(",".join(skills) if isinstance(skills, list) else skills) if skills else None,
        education=education,
        years_of_experience=years,
        resume_path=file_path
    )
    db.add(new_resume)
    db.commit()
    db.refresh(new_resume)  # get the generated ID
    # Generate embedding for the resume (we use whatever content we have; if text not parsed, use structured fields)
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
        if not content_to_embed:
            content_to_embed = name  # at least use name if nothing else
    try:
        # Generate embedding using sentence-transformers
        vector = model.encode(content_to_embed).tolist()
        add_resume_vector(new_resume.id, vector)
    except Exception as e:
        print(f"Embedding generation failed: {e}")
    return {"id": new_resume.id, "name": new_resume.name, "detail": "Resume uploaded and indexed successfully."}

# Endpoint: Match job description against all stored resumes
@app.post("/match")
def match_resumes(job_description: str = Form(...), top_k: int = 5, db: Session = Depends(get_db)):
    """
    Given a job description, return top matching resumes from the database, ranked by relevance.
    Also provides detailed reasoning for each match.
    """
    if not job_description or job_description.strip() == "":
        raise HTTPException(status_code=400, detail="Job description is required.")
    # Embed the job description using sentence-transformers
    try:
        job_vector = model.encode(job_description).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation error: {e}")
    # Search in FAISS index for similar resumes
    results = search_similar(job_vector, top_k=top_k)
    if not results:
        return {"matches": []}
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
            "years_of_experience": resume_rec.years_of_experience
        }
        explanation = generate_match_explanation(job_description, structured)
        # Compute a percentage or scaled score for display (e.g., 0-100)
        similarity_pct = int((score * 100)) if score <= 1 else int(score)
        matches.append({
            "resume_id": res_id,
            "name": resume_rec.name,
            "score": similarity_pct,
            "explanation": explanation
        })
    return {"matches": matches}

# Endpoint: Match job description against a list of uploaded resumes (without storing them in DB)
@app.post("/match_files")
def match_multiple_resumes(job_description: str = Form(...), resumes: list[UploadFile] = File(...)):
    """
    Given a job description and multiple resume files, return a ranked list of these resumes with match explanations.
    This endpoint does not permanently store resumes in the database; it is useful for one-off comparisons.
    """
    if not job_description:
        raise HTTPException(status_code=400, detail="Job description is required.")
    if not resumes or len(resumes) == 0:
        raise HTTPException(status_code=400, detail="No resumes provided.")
    # Embed the job description once using sentence-transformers
    try:
        job_vector = model.encode(job_description).tolist()
        # Normalize job vector
        jvec = np.array(job_vector, dtype='float32')
        norm = np.linalg.norm(jvec)
        if norm != 0:
            jvec = jvec / norm
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation error: {e}")
    temporary_results = []
    # Process each resume file: parse, embed, and score against job vector
    for file in resumes:
        content = file.file.read()
        if not content:
            continue
        # Parse resume (stub) to get structured data and text
        parsed = parse_resume_pdf(content)
        # Use structured fields to form an embedding input (fall back to raw text if available)
        embed_text = parsed.get("text")
        if not embed_text:
            embed_text = ""
            if parsed.get("skills"):
                embed_text += f"Skills: {parsed['skills']}. "
            if parsed.get("education"):
                embed_text += f"Education: {parsed['education']}. "
            if parsed.get("years_of_experience"):
                embed_text += f"{parsed['years_of_experience']} years of experience. "
            if embed_text == "":
                embed_text = file.filename  # fallback to using filename as content
        # Compute embedding for the resume using sentence-transformers
        try:
            resume_vector = model.encode(embed_text).tolist()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding generation error: {e}")
        # Normalize resume vector
        rvec = np.array(resume_vector, dtype='float32')
        norm_r = np.linalg.norm(rvec)
        if norm_r != 0:
            rvec = rvec / norm_r
        # Compute similarity score with job vector (cosine similarity via dot product)
        score = float(np.dot(jvec, rvec))
        # Generate explanation via agent
        structured = {
            "name": parsed.get("name") or file.filename.rsplit(".", 1)[0],
            "contact": parsed.get("contact"),
            "skills": parsed.get("skills"),
            "education": parsed.get("education"),
            "years_of_experience": parsed.get("years_of_experience")
        }
        explanation = generate_match_explanation(job_description, structured)
        similarity_pct = int(score * 100)
        temporary_results.append({
            "name": structured["name"],
            "score": similarity_pct,
            "explanation": explanation
        })
    # Sort the results by score (highest first)
    ranked = sorted(temporary_results, key=lambda x: x["score"], reverse=True)
    return {"matches": ranked}