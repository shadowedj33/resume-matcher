import os
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import difflib

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_data(show_spinner=False)
def get_resume_improvement_suggestions(resume_text, job_description, missing_skills):
    """
    Generate targeted resume improvement suggestions based on missing skills and job description.
    """
    prompt = f"""
    You are a career coach AI assistant. You are given a candidate's resume text, a job description, and a list of skills that are missing from the resume.
    Your task is to generate tailored resume improvement suggestions that highlight the candidate's skills and experience in relation to the job description.

    Given the following resume text:
    {resume_text}

    And the following job description:
    {job_description}

    And the following skills that are missing from the resume:
    {', '.join(missing_skills) if missing_skills else 'None'}

    1. Suggest specific keywords or phrases that the candidate should add or emphasize in their resume to better align with the job description.
    2. Rephrase these bullet points from the resume to better fit the job description language.
    3. Provide 2-3 realistic action steps the candidate can take to close the skill gaps or strengthen their application.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
            n=1,
            stop=None,
        )
        suggestion = response.choices[0].message.content.strip()
        return suggestion
    
    except Exception as e:
        return f"Error generating suggestions: {str(e)}"


@st.cache_data(show_spinner=False)
def generate_tailored_resume(resume_text, job_description, missing_skills=None):
    skills_notes = (
        f"\nInclude the following missing skills only if the resume logically supports them: {', '.join(missing_skills)}"
        if missing_skills else ""
    )

    prompt = f"""
    You are a professional resume editor. Improve the following resume to better match the job description. 
    Ensure the resume remains truthful and professional. {skills_notes}

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Return the improved resume only.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


# Tailored Resume Generator V2
@st.cache_data(show_spinner=False)
def generate_tailored_resume_v2(resume_text, job_title, job_description, intensity="medium"):
    prompt = f"""
    You are a professional AI resume coach. A user is applyong for the job title "{job_title}" with the following job description:
    ---
    {job_description}
    ---

    The user's current resume is as follows:
    ---
    {resume_text}
    ---
    Tailor the resume by updating only relevant sections (Summary, Experience, Skills), keeping other sections unchanged.
    Add missing skills and improve keyword relevance without sounding robotic.
    Tailoring intensity: {intensity.upper()}.
    Return the full tailored resume content. 
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


# Highlight changes in resume
def highlight_resume_changes(original: str, modified: str) -> str:
    """
    Highlight differences between original and modified text.
    Additions will be wrapped in **bold** for visibility.
    """
    diff = difflib.ndiff(original.split(), modified.split())
    highlighted = []

    for token in diff:
        if token.startswith("+ "): # Added word
            highlighted.append(f"**{token[2]}**")
        elif token.startswith("  "): # Same word
            highlighted.append(token[2])

    return " ".join(highlighted)
