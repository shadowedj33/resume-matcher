import streamlit as st
import torch
import plotly.graph_objects as go
from resume_parser import extract_resume_text
from matcher import (
    tfidf_match,
    bert_match,
    find_skill_gaps,
    get_top_matched_keywords,
    extract_skills,
    get_tfidf_vectors,
    generate_resume_suggestions,
)
from llm_utils import get_resume_improvement_suggestions, generate_tailored_resume
from export_utils import export_as_docx, export_as_pdf

torch.classes.__path__ = []

st.set_page_config(page_title="AI Resume Matcher", layout="centered")
st.title("AI Resume to Job Matcher")

# Resume Upload 
resume_file = st.file_uploader("Upload your resume (.pdf or .docx)", type=["pdf", "docx"])
resume_text = ""

if resume_file:
    resume_text = extract_resume_text(resume_file)
    st.success("Resume uploaded and processed!")

# Job Descriptions Input 
st.subheader("Enter Job Descriptions (up to 5)")
num_jobs = st.number_input("Number of jobs", min_value=1, max_value=5, value=1, step=1)

job_entries = []
for i in range(num_jobs):
    with st.expander(f"Job #{i + 1}"):
        title = st.text_input(f"Job Title #{i + 1}", key=f"title_{i}")
        description = st.text_area(f"Job Description #{i + 1}", height=200, key=f"desc_{i}")
        if title and description:
            job_entries.append((title, description))

# Matching Method Selector 
method = st.radio("Choose Matching Method", ["TF-IDF", "BERT"], horizontal=True)

# Toggle for AI Suggestions 
use_ai_suggestions = st.checkbox("Enable AI Resume Suggestions", value=True)

# Match Button Logic 
if st.button("Match Resume to Jobs"):
    if not resume_text:
        st.error("Please upload a resume first!")
    elif len(job_entries) == 0:
        st.error("Please enter at least one job description.")
    else:
        jd_texts = [desc for _, desc in job_entries]

        if method == "TF-IDF":
            results = tfidf_match(resume_text, jd_texts)
            tfidf_matrix, feature_names = get_tfidf_vectors([resume_text] + jd_texts)
        else:
            results = bert_match(resume_text, jd_texts)

        st.subheader("Match Results")
        sorted_results = sorted(zip(job_entries, results), key=lambda x: x[1][1], reverse=True)

        for i, ((title, desc), (_, score)) in enumerate(sorted_results):
            st.write(f"**{i + 1}. {title}** - Match Score: {round(score * 100, 2)}%")
            st.text_area("Job Preview", desc[:500] + "...", height=150)

            # Top Matching Keywords (TF-IDF Only)
            if method == "TF-IDF":
                job_index = jd_texts.index(desc) + 1
                keywords = get_top_matched_keywords(tfidf_matrix, feature_names, resume_index=0, job_index=job_index, top_n=10)
                st.markdown("**Top Matching Keywords:**")
                for word, kw_score in keywords:
                    st.write(f"• {word} — `{round(kw_score, 4)}`")

            # Skill Gap Analysis 
            missing_skills = find_skill_gaps(resume_text, desc)
            if missing_skills:
                st.warning(f"Missing skills for this role: {', '.join(missing_skills)}")
                
                if use_ai_suggestions:
                    with st.spinner("Generating resume improvement suggestions..."):
                        suggestions = get_resume_improvement_suggestions(resume_text, desc, missing_skills)
                        st.markdown("**Resume Improvement Suggestions:**")
                        st.info(suggestions)
            else:
                st.success("All key skills matched for this job!")

            # Radar Chart Explanation 
            total_skills = len(set(extract_skills(desc)))
            matched_skills = total_skills - len(missing_skills) if total_skills > 0 else 0
            skill_match_pct = (matched_skills / total_skills) * 100 if total_skills > 0 else 0

            if method == "TF-IDF":
                avg_keyword_score = sum(score for _, score in keywords) / len(keywords) if keywords else 0
            else:
                avg_keyword_score = score  # fallback for BERT

            labels = ["Match Score", "Skill Match %", "Keyword Relevance"]
            values = [score * 100, skill_match_pct, avg_keyword_score * 100]
            fig = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill='toself',
                name='Match Breakdown'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)

            st.plotly_chart(fig, use_container_width=True)

            # Tailored Resume Generation and Export
            st.subheader("Generate Tailored Resume")

            selected_job_title = st.selectbox(
                "Select a job to tailor your resume for", [title for title, _ in job_entries]
            )
            selected_job_desc = next(
                desc for title, desc in job_entries if title == selected_job_title
            )

            if st.button("Generate Tailored Resume"):
                with st.spinner("Generating tailored resume..."):
                    tailored_resume, changes_summary = generate_tailored_resume(
                        resume_text, selected_job_desc
                    )

                    st.markdown("**Changes Made:**")
                    st.code(changes_summary, language="markdown")

                    # Save version history in session state
                    if "version_history" not in st.session_state:
                        st.session_state.version_history = []
                    st.session_state.version_history.append(
                        {
                            "job_title": selected_job_title,
                            "changes": changes_summary,
                            "resume": tailored_resume,
                        }
                    )

                    st.markdown("**Tailored Resume Preview:**")
                    st.text_area("Tailored Resume", tailored_resume, height=300)

                    format_choice = st.radio("Export format", ["DOCX", "PDF"], horizontal=True)

                    if st.button("Download Tailored Resume"):
                        filename = f"Tailored_Resume_for_{selected_job_title.replace(' ', '_')}"
                        if format_choice == "DOCX":
                            export_as_docx(tailored_resume, filename + ".docx")
                        else:
                            export_as_pdf(tailored_resume, filename + ".pdf")

# Tailored Resume Version History
if "version_history" in st.session_state and st.session_state.version_history:
    st.subheader("Resume Tailoring History")
    for idx, version in enumerate(reversed(st.session_state.version_history)):
        with st.expander(f"Version {len(st.session_state.version_history) - idx}: {version['job_title']}"):
            st.markdown("**Changes Made:**")
            st.code(version["changes"], language="markdown")
            st.markdown("**Tailored Resume:**")
            st.text_area("Tailored Resume Snapshot", version["resume"], height=300, disabled=True)
