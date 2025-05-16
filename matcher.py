from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

# Common skill extractor using regex and filtering, replace with spaCy or a model later
COMMON_SKILLS = [
    "python", "java", "javascript", "sql", "aws", "azure", "linux", "git", "docker",
    "kubernetes", "react", "nodejs", "express", "mongodb", "mysql", "postgresql",
    "c#", "c++", "c", "ruby", "php", "swift", "kotlin", "go", "rust", "scala",
    "html", "css", "sass", "less", "typescript", "jquery", "json", "xml", "yaml",
    "bash", "powershell", "shell", "npm", "yarn", "gradle", "maven", "sbt", "ivy",
    "jenkins", "circleci", "gitlab", "github", "travis", "circle", "appveyor",
    "selenium", "appium", "cypress", "jest", "mocha", "chai", "jasmine", "pytest",
    "pytest", "unittest", "nose", "nose2", "tox", "coverage", "mockito", "junit",
    "mockito", "junit", "testng", "spock", "cucumber", "behave", "robot", "wiremock",
    "machine learning", "deep learning", "ai", "ml", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "scipy",
    "data analysis", "data science", "data engineering", "data visualization",
    "data cleaning", "data wrangling", "data processing", "data modeling",
    "communication", "teamwork", "leadership", "problem solving", "critical thinking",
    "project management", "agile", "scrum", "kanban", "waterfall", "devops", "cicd",
]

# Extract skills from text
def extract_skills(text):
    text_lower = text.lower()
    found = [skill for skill in COMMON_SKILLS if skill in text_lower]
    return list(set(found))

# Find skill gaps
def find_skill_gaps(resume_text, job_text):
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))
    missing = job_skills - resume_skills
    return list(missing)


# TF-IDF-based matcher
def tfidf_match(resume_text, job_texts):
    vectorizer = TfidfVectorizer()
    docs = [resume_text] + job_texts
    tfidf_matrix = vectorizer.fit_transform(docs)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return list(zip(job_texts, similarity))

# BERT-based semantic matcher
def bert_match(resume_text, job_texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([resume_text] + job_texts)
    similarity = cosine_similarity([embeddings[0]], embeddings[1:]).flatten()
    return list(zip(job_texts, similarity))
