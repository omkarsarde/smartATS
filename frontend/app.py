import os
import requests
import streamlit as st

# Configuration
st.set_page_config(page_title="Resume Matcher", layout="wide")
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.title("🔍 Resume Matching and Ranking System")
st.write("Upload resumes and input a job description to find the best candidate matches. "
         "The system will rank resumes and provide detailed explanations for the matching.")

# Input fields
job_description = st.text_area("Job Description", height=200, placeholder="Enter the job description here...")

uploaded_files = st.file_uploader(
    "Upload one or multiple resumes (PDF or DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# When the user clicks the match button
if st.button("Match Resumes"):
    if not job_description or job_description.strip() == "":
        st.error("Please enter a job description.")
    else:
        # Determine which endpoint to call based on whether files are uploaded
        try:
            if uploaded_files:
                # Use the /match_files endpoint for on-the-fly matching
                files = []
                for file in uploaded_files:
                    files.append(("resumes", (file.name, file.read(), file.type)))
                response = requests.post(
                    f"{BACKEND_URL}/match_files",
                    files=files,
                    data={"job_description": job_description}
                )
            else:
                # No files uploaded, search in stored resumes (/match endpoint)
                response = requests.post(
                    f"{BACKEND_URL}/match",
                    data={"job_description": job_description, "top_k": 5}
                )
            if response.status_code != 200:
                st.error(f"Error from server: {response.status_code} - {response.text}")
            else:
                data = response.json()
                matches = data.get("matches", [])
                if not matches:
                    st.warning("No matching resumes found.")
                else:
                    st.success(f"Found {len(matches)} matching resume(s).")
                    # Display each match
                    for m in matches:
                        name = m.get("name", "Unknown")
                        score = m.get("score", 0)
                        explanation = m.get("explanation", "")
                        with st.expander(f"🎓 {name} — Match Score: {score}/100", expanded=True):
                            # Print the explanation text, preserving formatting
                            st.markdown(explanation.replace("\n", "  \n"))
        except Exception as e:
            st.error(f"Failed to get response from backend: {e}")