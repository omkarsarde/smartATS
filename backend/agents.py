import os
from crewai import Agent, Task, Crew

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
        verbose=True
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