import streamlit as st
import torch
from resume_parser import extract_resume_text
from matcher import tfidf_match, bert_match, find_skill_gaps

torch.classes.__path__ = []

st.set_page_config(page_title="AI Resume Matcher", layout="centered")
st.title("AI Resume to Job Matcher")

# Resume upload and processing
resume_file = st.file_uploader("Upload your resume (.pdf or .docx)", type=["pdf", "docx"])
resume_text = ""

if resume_file:
    resume_text = extract_resume_text(resume_file)
    st.success("Resume uploaded and processed!")

# Job descriptions input
st.subheader("Enter Job Descriptions (up to 5)")
num_jobs = st.number_input("Number of jobs", min_value=1, max_value=5, value=1, step=1)

job_entries = []
for i in range(num_jobs):
    with st.expander(f"Job #{i + 1}"):
        title = st.text_input(f"Job Title #{i + 1}", key=f"title_{i}")
        description = st.text_area(f"Job Description #{i + 1}", height=200, key=f"desc_{i}")
        if title and description:
            job_entries.append((title, description))


# matching method selector
method = st.radio("Choose Matching Method", ["TF-IDF", "BERT"], horizontal=True)

# Match resume to jobs
if st.button("Match Resume to Jobs"):
    if not resume_text:
        st.error("Please upload a resume first!")
    elif len(job_entries) == 0:
        st.error("Please enter at least one job description.")
    else:
        jd_texts = [desc for _, desc in job_entries]

        if method == "TF-IDF":
            results = tfidf_match(resume_text, jd_texts)
        else:
            results = bert_match(resume_text, jd_texts)

        st.subheader("Match Results")
        sorted_results = sorted(zip(job_entries, results), key=lambda x: x[1][1], reverse=True)

        for i, ((title, desc), (_, score)) in enumerate (sorted_results):
            st.write(f"**{i + 1}. {title}** - Match Score: {round(score*100, 2)}%")
            st.text_area("Preview", desc[:500] + "...", height=150)

            # Find skill gaps
            missing_skills = find_skill_gaps(resume_text, desc)
            if missing_skills: 
                st.warning(f"Missing skills for this role: {', '.join(missing_skills)}")
            else:
                st.success("All key skills matched for this job!")
