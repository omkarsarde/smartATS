import os
import json
import hashlib
import io
from crewai import Agent, Task, Crew, LLM
from typing import Dict, List, Optional


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using pypdf"""
    try:
        # Import here to avoid loading the module if not necessary
        from pypdf import PdfReader
        
        # Create a file-like object from bytes
        pdf_file = io.BytesIO(file_bytes)
        
        # Create a PDF reader object
        pdf = PdfReader(pdf_file)
        
        # Extract text from all pages
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        if not text.strip():
            # If no text was extracted (possibly an image-based PDF)
            print("Warning: No text extracted from PDF. It may be image-based or corrupted.")
            # Return a portion of raw bytes as fallback
            return file_bytes[:5000].decode('utf-8', errors='ignore')
        
        print(f"Successfully extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        print(f"PDF extraction error: {e}")
        # Fallback to naive extraction
        return file_bytes[:5000].decode('utf-8', errors='ignore')

def setup_openai_llm():
    """Configure and return an OpenAI LLM instance for CrewAI"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        return None
        
    try:
        # Use GPT-4o-mini for analysis
        llm = LLM(model="gpt-4o-mini",
                  temperature=0.7,
                  seed=42)
        print("Successfully configured OpenAI GPT-4o-mini for resume analysis")
        return llm
    except Exception as e:
        print(f"Failed to initialize OpenAI: {e}")
        try:
            # Try with gpt-3.5-turbo as fallback
            llm = LLM(model="gpt-3.5-turbo",
                     temperature=0.2,
                     seed=42)
            print("Fallback to gpt-3.5-turbo successful")
            return llm
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return None

# Initialize the OpenAI LLM
llm = setup_openai_llm()
if not llm:
    print("WARNING: Could not initialize OpenAI LLM. Resume matching will be limited.")

# Create the specialized agents for our resume matching system
def create_agents():
    """Create the specialized agents for our resume matching system"""
    
    # PDF Parser for extracting structured data from resumes
    parser = Agent(
        role="Resume Parser",
        goal="Extract structured information from resume PDFs",
        backstory="""You are an expert resume parser with exceptional ability to extract 
        structured information from resume text. You can identify contact details, 
        skills, education, work history, and other relevant information from raw resume content.
        You're meticulous and accurate, ensuring all key information is captured.""",
        llm=llm,
        verbose=True
    )
    
    # Resume Validator for confirming the parsed data
    validator = Agent(
        role="Resume Validation Specialist",
        goal="Verify the accuracy of parsed resume data",
        backstory="""You are a detail-oriented validation specialist with years of experience
        in resume screening and verification. Your job is to double-check parsed resume data
        for accuracy, completeness, and consistency. You have a keen eye for inconsistencies
        and can detect when information might be incorrect or missing.""",
        llm=llm,
        verbose=True
    )
    
    # HR Specialist for evaluating resume-job matches
    matcher = Agent(
        role="Senior HR Specialist",
        goal="Evaluate resumes against job requirements with extreme precision",
        backstory="""You are a seasoned HR professional with 15+ years of experience in technical recruiting.
        You have an exceptional ability to identify matching skills, recognize adjacent or transferable skills,
        and understand the nuances of job requirements. You've helped hundreds of companies find the perfect
        candidates for their positions.""",
        llm=llm,
        verbose=True
    )
    
    # Career Coach for providing improvement recommendations
    coach = Agent(
        role="Technical Career Development Coach",
        goal="Create personalized skill development plans to help candidates better match job requirements",
        backstory="""You are an expert career coach specializing in technical skill development.
        You have deep knowledge of learning paths for different technical skills, realistic timeframes
        for skill acquisition, and practical advice for career advancement. You've helped thousands
        of professionals enhance their skills and advance their careers.""",
        llm=llm,
        verbose=True
    )
    
    return parser, validator, matcher, coach

# Initialize our specialized agents
resume_parser, resume_validator, hr_matcher, career_coach = create_agents() if llm else (None, None, None, None)

def parse_resume_pdf(file_bytes: bytes, validate: bool = True) -> dict:
    """
    Use CrewAI to parse resume PDF content into structured data.
    Returns a dictionary with keys: name, contact, skills, education, years_of_experience, work_experience, text.
    
    Args:
        file_bytes: The binary content of the resume file
        validate: Whether to use a second agent to validate the parsed data
    
    Returns:
        A dictionary with the parsed resume data and validation information
    """
    # Generate file hash for deduplication
    resume_hash = hashlib.sha256(file_bytes).hexdigest()
    
    # If no LLM is available, return basic placeholder data
    if not llm or not resume_parser:
        placeholder = _generate_placeholder_resume_data(file_bytes)
        placeholder["resume_hash"] = resume_hash
        return placeholder
    
    # Try to extract text from the PDF for the agent to analyze
    try:
        # Extract text using proper PDF parsing
        text_sample = extract_text_from_pdf(file_bytes)
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        text_sample = "Unable to extract text from PDF"
    
    print(f"Extracted {len(text_sample)} characters from the PDF")
    
    # Define the task for the resume parser agent
    parsing_task = Task(
        description=f"""
        You are analyzing a resume extract. The content is as follows:
        
        {text_sample}
        
        Based on this content, extract the following structured information:
        1. Full name of the candidate
        2. Contact information (email, phone)
        3. Skills (as a comma-separated list)
        4. Education details (degree, institution)
        5. Total years of professional experience (as a number)
        6. Work experience as a list of positions, each containing:
           - Company name
           - Job title
           - Duration (e.g., "2018-2020")
           - Brief description of responsibilities
        
        Return your answer as a valid JSON object with the following keys:
        "name", "contact", "skills", "education", "years_of_experience", "work_experience"
        
        The work_experience should be a list of objects, each with keys: "company", "title", "duration", "description"
        
        IMPORTANT: Make sure your response is a valid, parseable JSON object. Don't include any non-JSON text in your response.
        """,
        agent=resume_parser,
        expected_output="A structured JSON object containing the parsed resume information."
    )
    
    # Execute the parsing task
    try:
        crew = Crew(
            agents=[resume_parser],
            tasks=[parsing_task],
            verbose=True
        )
        
        print("Starting resume parsing crew...")
        result = crew.kickoff()
        print(f"Resume parsing complete, got result of length {len(result) if result else 0}")
        
        # Try to parse the result as JSON
        try:
            # Look for JSON in the response (the agent might include extra text)
            import re
            json_match = re.search(r'({[\s\S]*})', result)
            
            if json_match:
                parsed_data = json.loads(json_match.group(1))
                print(f"Successfully parsed JSON from result, found {len(parsed_data)} fields")
            else:
                # Fallback if no JSON object found
                print("No JSON found in parser result, using placeholder data")
                parsed_data = _generate_placeholder_resume_data(file_bytes)
                parsed_data["text"] = str(result)  # Include the raw result for debugging
        except json.JSONDecodeError as e:
            # If JSON parsing fails, use the placeholder data
            print(f"Failed to parse JSON from result: {e}")
            parsed_data = _generate_placeholder_resume_data(file_bytes)
            parsed_data["text"] = str(result)  # Include the raw result for debugging
            
        # Store the AI parsed data
        parsed_data["resume_hash"] = resume_hash
        parsed_data["ai_parsed_data"] = json.dumps(parsed_data)
        
        # Don't validate if not requested
        if not validate or not resume_validator:
            parsed_data["validation_status"] = "not_validated"
            return parsed_data
            
        # Validation process with second agent
        validated_data = validate_resume_data(text_sample, parsed_data)
        parsed_data["validated_data"] = json.dumps(validated_data)
        
        # Check if validation succeeded and if the data is consistent
        if validated_data:
            # Determine validation status by comparing key fields
            validation_issues = []
            
            # Check name consistency
            if parsed_data.get("name") != validated_data.get("name"):
                validation_issues.append("name")
                
            # Check skills (allow for minor variations)
            parser_skills = set([s.strip().lower() for s in str(parsed_data.get("skills", "")).split(",") if s.strip()])
            validator_skills = set([s.strip().lower() for s in str(validated_data.get("skills", "")).split(",") if s.strip()])
            if len(parser_skills.symmetric_difference(validator_skills)) > len(parser_skills) * 0.2:  # More than 20% different
                validation_issues.append("skills")
                
            # Check education (more lenient)
            if not (str(parsed_data.get("education", "")).lower() in str(validated_data.get("education", "")).lower() or 
                   str(validated_data.get("education", "")).lower() in str(parsed_data.get("education", "")).lower()):
                validation_issues.append("education")
                
            # Check work experience count
            parser_jobs = parsed_data.get("work_experience", [])
            validator_jobs = validated_data.get("work_experience", [])
            if abs(len(parser_jobs) - len(validator_jobs)) > 1:  # Allow one job difference
                validation_issues.append("work_experience_count")
            
            # Set validation status
            if validation_issues:
                parsed_data["validation_status"] = "conflict"
                parsed_data["validation_issues"] = ",".join(validation_issues)
            else:
                parsed_data["validation_status"] = "validated"
                
            # Use validated data for some fields where appropriate
            if parsed_data["validation_status"] == "validated":
                # Merge the validated data into our result
                parsed_data["name"] = validated_data.get("name", parsed_data.get("name"))
                parsed_data["contact"] = validated_data.get("contact", parsed_data.get("contact"))
                parsed_data["years_of_experience"] = validated_data.get("years_of_experience", parsed_data.get("years_of_experience"))
        else:
            parsed_data["validation_status"] = "validation_failed"
        
        return parsed_data
        
    except Exception as e:
        print(f"Resume parsing failed: {e}")
        placeholder = _generate_placeholder_resume_data(file_bytes)
        placeholder["resume_hash"] = resume_hash
        placeholder["validation_status"] = "parsing_failed"
        placeholder["error"] = str(e)
        return placeholder

def validate_resume_data(text_sample: str, parsed_data: dict) -> dict:
    """
    Use a second agent to validate the parsed resume data
    
    Args:
        text_sample: The text extracted from the resume
        parsed_data: The initially parsed resume data
        
    Returns:
        A dictionary with the validated resume data, or None if validation failed
    """
    if not llm or not resume_validator:
        return None
        
    # Convert the parsed data to a string representation for the validator
    parsed_str = json.dumps(parsed_data, indent=2)
    
    # Define the validation task
    validation_task = Task(
        description=f"""
        You are validating parsed resume data for accuracy. 
        
        Original resume text:
        {text_sample}
        
        Parsed data to validate:
        {parsed_str}
        
        Your task is to independently extract the same information from the resume text and validate the parsed data.
        If you find any discrepancies or errors, correct them in your response.
        
        Return a valid JSON object with the following keys:
        "name", "contact", "skills", "education", "years_of_experience", "work_experience"
        
        The work_experience should be a list of objects, each with keys: "company", "title", "duration", "description"
        
        IMPORTANT: Make sure your response is a valid, parseable JSON object. Don't include any non-JSON text in your response.
        """,
        agent=resume_validator,
        expected_output="A structured JSON object containing the validated resume information."
    )
    
    # Execute the validation task
    try:
        crew = Crew(
            agents=[resume_validator],
            tasks=[validation_task],
            verbose=True
        )
        
        print("Starting resume validation crew...")
        result = crew.kickoff()
        print(f"Resume validation complete, got result of length {len(result) if result else 0}")
        
        # Try to parse the result as JSON
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'({[\s\S]*})', result)
            
            if json_match:
                validated_data = json.loads(json_match.group(1))
                print(f"Successfully parsed validation JSON, found {len(validated_data)} fields")
                return validated_data
            else:
                print("No JSON found in validator result")
                return None
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from validator result: {e}")
            return None
            
    except Exception as e:
        print(f"Resume validation failed: {e}")
        return None

def _generate_placeholder_resume_data(file_bytes: bytes) -> dict:
    """Generate placeholder resume data when parsing fails"""
    # Try to extract a filename if available
    filename = "unknown_resume"
    
    # Extract some basic text from the PDF for context
    try:
        text_sample = file_bytes[:100].decode('utf-8', errors='ignore')
    except:
        text_sample = ""
        
    # Sample work experience for testing
    work_experience = [
        {
            "company": "Sample Company Inc.",
            "title": "Software Engineer",
            "duration": "2018-2020",
            "description": "Developed backend systems using Python and FastAPI."
        },
        {
            "company": "Tech Startup",
            "title": "Junior Developer",
            "duration": "2016-2018",
            "description": "Worked on frontend applications using React."
        }
    ]
        
    return {
        "name": filename,
        "contact": "example@email.com",
        "skills": "Python, Data Analysis",
        "education": "Bachelor's Degree in Computer Science",
        "years_of_experience": 4,
        "work_experience": work_experience,
        "text": text_sample,
        "validation_status": "placeholder"
    }

def generate_match_explanation(job_description: str, resume_structured: dict) -> str:
    """
    Use CrewAI agents to generate a detailed explanation for how well the resume matches the job description.
    
    Returns a text explanation including:
    - Overall match score
    - Matched skills breakdown
    - Adjacent/transferable skills
    - Missing skills with importance
    - Detailed improvement plan with timeline
    """
    # Add safety checks
    if not job_description or not resume_structured:
        return "Error: Missing job description or resume data"
    
    # If we don't have OpenAI LLM configured, provide basic response
    if not llm or not hr_matcher or not career_coach:
        return _generate_basic_match_explanation(job_description, resume_structured)
    
    # Prepare the resume data in a detailed format
    name = resume_structured.get("name") or "Candidate"
    skills = resume_structured.get("skills") or ""
    education = resume_structured.get("education") or ""
    years = resume_structured.get("years_of_experience")
    years_text = f"{years} years" if years is not None else "N/A"
    
    # Parse the work experience if available
    work_exp_text = ""
    try:
        work_exp = resume_structured.get("work_experience")
        if work_exp:
            # Handle both string and list formats
            work_experience = json.loads(work_exp) if isinstance(work_exp, str) else work_exp
            for job in work_experience:
                work_exp_text += f"- {job.get('title')} at {job.get('company')} ({job.get('duration')}): {job.get('description')}\n"
    except Exception as e:
        print(f"Failed to parse work experience: {e}")
        work_exp_text = "No structured work experience available"
    
    # Compose the resume summary
    resume_info = (
        f"Resume of {name}:\n"
        f"- Skills: {skills}\n"
        f"- Education: {education}\n"
        f"- Years of Experience: {years_text}\n"
        f"- Work History:\n{work_exp_text}\n"
    )
    
    print(f"Starting match explanation for resume: {name}")
    
    # Define the evaluation task for the HR matcher
    evaluation_task = Task(
        description=f"""
        Job Description:
        {job_description}
        
        {resume_info}
        
        Your task is to perform a comprehensive evaluation of this resume against the job description.
        
        Provide your response in the following format:
        
        Start with: "Match Score: [NUMBER]/100" where [NUMBER] is your overall rating from 0-100.
        
        Then provide:
        1. A detailed breakdown of matched skills (skills in the resume that directly meet job requirements).
        2. Adjacent/transferable skills (skills in the resume that are relevant or similar to required skills).
        3. Critical missing skills (important skills from the job description that are absent in the resume).
        4. Any red flags or misalignments in the candidate's experience or qualifications.
        
        Be extremely specific about why you assigned the match score. Focus on technical skills, experience,
        and education. Don't be overly generous - maintain high standards.
        """,
        agent=hr_matcher,
        expected_output="A detailed evaluation with match score (clearly stated at the beginning), matched skills, adjacent skills, missing skills, and red flags."
    )
    
    # Define the improvement plan task for the career coach
    improvement_task = Task(
        description=f"""
        Job Description:
        {job_description}
        
        {resume_info}
        
        Using the resume and job requirements above, create a personalized improvement plan for this candidate.
        
        Your response must follow this format:
        
        Start with confirming the same "Match Score: [NUMBER]/100" that was assigned by the HR evaluation.
        
        Then provide:
        1. The top 3-5 skills the candidate should develop to better match this job.
        2. For each skill, estimate the time required to develop it (in months) for someone with their background.
        3. Specific resources or courses they could use to acquire these skills.
        4. A step-by-step roadmap for improving their candidacy for this role.
        5. Realistic estimate of what score they could achieve with these improvements.
        
        Be very specific and practical. Assume the candidate has the aptitude and motivation to learn.
        """, 
        agent=career_coach,
        expected_output="A detailed improvement plan that begins with the same match score, followed by specific skills to develop, timeline, resources, and roadmap."
    )
    
    # Create a crew with our specialized agents
    crew = Crew(
        agents=[hr_matcher, career_coach],
        tasks=[evaluation_task, improvement_task],
        verbose=True
    )
    
    try:
        # Execute both tasks and get results
        print("Starting match explanation crew...")
        results = crew.kickoff()
        # Convert the CrewOutput to string first since it doesn't support len()
        results_str = str(results)
        print(f"Match explanation complete, got results of length {len(results_str) if results_str else 0}")
        return results_str
    except Exception as e:
        print(f"Agent explanation generation failed: {e}")
        # Fallback to basic explanation
        return _generate_basic_match_explanation(job_description, resume_structured)

def _generate_basic_match_explanation(job_description: str, resume_structured: dict) -> str:
    """Generate a basic match explanation when the agent system is unavailable"""
    name = resume_structured.get("name") or "Candidate"
    skills = resume_structured.get("skills") or ""
    education = resume_structured.get("education") or ""
    years = resume_structured.get("years_of_experience")
    years_text = f"{years} years" if years is not None else "N/A"
    
    # Do a simple keyword matching for skills
    job_description_lower = job_description.lower()
    skills_list = [s.strip().lower() for s in skills.split(",") if s.strip()]
    matched_skills = [skill for skill in skills_list if skill in job_description_lower]
    missing_words = ["python", "javascript", "react", "aws", "cloud", "data", "analysis", 
                     "ml", "ai", "database", "sql", "nosql", "devops", "agile"]
    missing_skills = [word for word in missing_words 
                      if word in job_description_lower and word not in " ".join(skills_list).lower()]
    
    # Generate a simple score based on matched skills
    score = min(95, max(50, len(matched_skills) * 15))
    
    return (
        f"Match Score: {score}/100\n\n"
        f"Analysis for {name}:\n"
        f"- Education: {education}\n"
        f"- Experience: {years_text}\n\n"
        f"Matched Skills: {', '.join(matched_skills) if matched_skills else 'No direct skill matches found'}\n\n"
        f"Missing Skills: {', '.join(missing_skills) if missing_skills else 'No obvious missing skills identified'}\n\n"
        f"Improvement Recommendations:\n"
        f"This candidate could improve their match for this position by developing skills in "
        f"{', '.join(missing_skills[:3]) if missing_skills else 'areas mentioned in the job description'}. "
        f"With focused study and practice, these skills could likely be developed within 3-6 months."
    )