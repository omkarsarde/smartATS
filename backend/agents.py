import os
from crewai import Agent, Task, Crew
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
    Currently, this is a stub and returns minimal dummy data.
    """
    # TODO: Implement actual PDF parsing logic (e.g., using PyMuPDF or pdfminer.six)
    # For now, we return empty or placeholder values.
    return {
        "name": None,
        "contact": None,
        "skills": None,
        "education": None,
        "years_of_experience": None,
        "text": None
    }

# Initialize a local LLM if dependencies are available and no API keys are set
# This is optional and falls back to CrewAI's default if not available
local_llm = None
if HAS_LOCAL_LLM and not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
    try:
        # Try to load a smaller model that can run on CPU
        # This is just an example; adjust based on hardware capabilities
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
        print("Successfully loaded local LLM model for explanations")
    except Exception as e:
        print(f"Could not load local LLM model: {e}")
        local_llm = None

# Initialize a CrewAI agent for matching and explanation
matching_agent = Agent(
    role="HR Matchmaker",
    goal="Assess candidate resume against job description and provide match score with detailed reasoning.",
    backstory=("You are an expert HR recruiter with deep knowledge of job requirements and resume analysis. "
               "You compare job descriptions with candidate resumes, identify skill matches, related (adjacent) skills, "
               "missing skills, and suggest improvements for the candidate."),
    llm=local_llm  # This will be None if local LLM couldn't be loaded, and CrewAI will use its default
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
    task = Task(description=task_description, agent=matching_agent)
    crew = Crew(agents=[matching_agent], tasks=[task])
    
    try:
        result = crew.kickoff()  # Execute the agent task
        explanation = str(result)
    except Exception as e:
        # Fallback if there's an error with the agent or LLM
        print(f"Agent explanation generation failed: {e}")
        explanation = f"Score: 5/10\n\nAnalysis could not be generated. The system was able to match this resume, but detailed analysis requires configured LLM access.\n\nError: {str(e)}"
    
    return explanation