import os
import openai
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

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
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
            n=1,
            stop=None,
        )
        suggestion = response['choices'][0]['message']['content'].strip()
        return suggestion
    
    except Exception as e:
        return f"Error generating suggestions: {str(e)}"
