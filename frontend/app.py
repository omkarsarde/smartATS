import os
import requests
import streamlit as st

# Configuration
st.set_page_config(page_title="Resume Matcher", layout="wide")
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.title("🔍 Resume Matching and Ranking System")
st.markdown("""
This system helps you find the best candidates for a job by analyzing resumes against job descriptions.
Use it to discover top matches from existing candidates or evaluate new resumes.
""")

# Create tabs for different workflows
tab1, tab2 = st.tabs(["Find Matches in Database", "Upload & Match New Resumes"])

# Tab 1: Find matches in existing database (Workflow 1)
with tab1:
    st.header("Find Candidates in Database")
    st.write("Enter a job description to find the best matching candidates from our existing database.")
    
    # Input field for job description
    job_description_db = st.text_area(
        "Job Description", 
        height=200, 
        placeholder="Enter the detailed job description here...",
        key="jd_db"
    )
    
    # Number of results to return
    num_results = st.slider("Number of top candidates to show", min_value=1, max_value=10, value=5)
    
    # When the user clicks the search button
    if st.button("Search Database"):
        if not job_description_db or job_description_db.strip() == "":
            st.error("Please enter a job description.")
        else:
            with st.spinner("Searching for matching candidates..."):
                try:
                    # Call the /match endpoint
                    response = requests.post(
                        f"{BACKEND_URL}/match",
                        data={"job_description": job_description_db, "top_k": num_results}
                    )
                    
                    if response.status_code != 200:
                        st.error(f"Error from server: {response.status_code} - {response.text}")
                    else:
                        data = response.json()
                        matches = data.get("matches", [])
                        
                        if not matches:
                            st.warning("No matching candidates found in the database.")
                        else:
                            st.success(f"Found {len(matches)} matching candidate(s).")
                            
                            # Display a summary table
                            summary_data = {
                                "Name": [m.get("name", "Unknown") for m in matches],
                                "Match Score": [f"{m.get('score', 0)}/100" for m in matches]
                            }
                            st.dataframe(summary_data, use_container_width=True)
                            
                            # Display detailed explanations
                            st.subheader("Detailed Candidate Analysis")
                            for i, m in enumerate(matches):
                                name = m.get("name", "Unknown")
                                score = m.get("score", 0)
                                explanation = m.get("explanation", "")
                                
                                with st.expander(f"#{i+1}: {name} — Match Score: {score}/100", expanded=(i==0)):
                                    # Print the explanation text, preserving formatting
                                    st.markdown(explanation.replace("\n", "  \n"))
                
                except Exception as e:
                    st.error(f"Failed to get response from backend: {e}")

# Tab 2: Upload and match new resumes (Workflow 2)
with tab2:
    st.header("Evaluate New Resumes")
    st.write("Upload resumes and a job description to evaluate and rank new candidates.")
    
    # Input field for job description
    job_description_upload = st.text_area(
        "Job Description", 
        height=200, 
        placeholder="Enter the detailed job description here...",
        key="jd_upload"
    )
    
    # File uploader for resumes
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="resume_upload"
    )
    
    # Option to store resumes in database
    store_in_db = st.checkbox("Store these resumes in database for future matching", value=False)
    
    # When the user clicks the analyze button
    if st.button("Analyze Resumes"):
        if not job_description_upload or job_description_upload.strip() == "":
            st.error("Please enter a job description.")
        elif not uploaded_files:
            st.error("Please upload at least one resume file.")
        else:
            with st.spinner("Analyzing resumes against job description..."):
                try:
                    # Prepare files for upload
                    files = []
                    for file in uploaded_files:
                        files.append(("resumes", (file.name, file.read(), file.type)))
                    
                    # Call the /match_files endpoint
                    response = requests.post(
                        f"{BACKEND_URL}/match_files",
                        files=files,
                        data={
                            "job_description": job_description_upload,
                            "store_in_db": "true" if store_in_db else "false"
                        }
                    )
                    
                    if response.status_code != 200:
                        st.error(f"Error from server: {response.status_code} - {response.text}")
                    else:
                        data = response.json()
                        matches = data.get("matches", [])
                        stored_ids = data.get("stored_resume_ids", [])
                        
                        if not matches:
                            st.warning("No results returned from analysis.")
                        else:
                            st.success(f"Successfully analyzed {len(matches)} resume(s).")
                            
                            if stored_ids:
                                st.info(f"Stored {len(stored_ids)} resume(s) in the database for future matching.")
                            
                            # Display a summary table
                            summary_data = {
                                "Rank": list(range(1, len(matches) + 1)),
                                "Name": [m.get("name", "Unknown") for m in matches],
                                "Match Score": [f"{m.get('score', 0)}/100" for m in matches]
                            }
                            st.dataframe(summary_data, use_container_width=True)
                            
                            # Display detailed explanations
                            st.subheader("Detailed Analysis & Recommendations")
                            for i, m in enumerate(matches):
                                name = m.get("name", "Unknown")
                                score = m.get("score", 0)
                                explanation = m.get("explanation", "")
                                
                                with st.expander(f"#{i+1}: {name} — Match Score: {score}/100", expanded=(i==0)):
                                    # Print the explanation text, preserving formatting
                                    st.markdown(explanation.replace("\n", "  \n"))
                
                except Exception as e:
                    st.error(f"Failed to get response from backend: {e}")

# Add a small footer with information
st.markdown("---")
st.markdown(
    "This system uses advanced AI to compare resumes against job descriptions. "
    "Recommendations include skill matches, missing skills, and personalized improvement plans."
)