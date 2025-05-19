from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy

nlp = spacy.load("en_core_web_sm")

# Common skill list — replace or extend with spaCy-based extraction later
COMMON_SKILLS = [
    "python", "java", "javascript", "sql", "aws", "azure", "linux", "git", "docker",
    "kubernetes", "react", "nodejs", "express", "mongodb", "mysql", "postgresql",
    "c#", "c++", "c", "ruby", "php", "swift", "kotlin", "go", "rust", "scala",
    "html", "css", "sass", "less", "typescript", "jquery", "json", "xml", "yaml",
    "bash", "powershell", "shell", "npm", "yarn", "gradle", "maven", "sbt", "ivy",
    "jenkins", "circleci", "gitlab", "github", "travis", "appveyor",
    "selenium", "appium", "cypress", "jest", "mocha", "chai", "jasmine", "pytest",
    "unittest", "nose", "tox", "coverage", "mockito", "junit", "testng", "spock",
    "cucumber", "behave", "robot", "wiremock",
    "machine learning", "deep learning", "ai", "ml", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "scipy",
    "data analysis", "data science", "data engineering", "data visualization",
    "data cleaning", "data wrangling", "data processing", "data modeling",
    "communication", "teamwork", "leadership", "problem solving", "critical thinking",
    "project management", "agile", "scrum", "kanban", "waterfall", "devops", "cicd",
]


def tfidf_match(resume_text, job_descriptions):
    documents = [resume_text] + job_descriptions
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    resume_vec = tfidf_matrix[0]
    job_vecs = tfidf_matrix[1:]
    similarities = cosine_similarity(resume_vec, job_vecs).flatten()
    return list(zip(job_descriptions, similarities))

def get_tfidf_vectors(documents):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

def get_top_matched_keywords(tfidf_matrix, feature_names, resume_index, job_index, top_n=10):
    resume_vec = tfidf_matrix[resume_index].toarray()[0]
    job_vec = tfidf_matrix[job_index].toarray()[0]
    
    keyword_scores = []
    for i, feature in enumerate(feature_names):
        if resume_vec[i] > 0 and job_vec[i] > 0:
            keyword_scores.append((feature, resume_vec[i] * job_vec[i]))

    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    return keyword_scores[:top_n]

# Extract skills from text (simple keyword matching)
def extract_skills(text):
    doc = nlp(text)
    return [ent.text.lower() for ent in doc.ents if ent.label_ in ["SKILL", "ORG", "PRODUCT"] or ent.text.lower() in COMMON_SKILLS]

# Find skill gaps (skills in job description but missing in resume)
def find_skill_gaps(resume_text, job_text):
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))
    return list(job_skills - resume_skills)

# BERT matcher
def bert_match(resume_text, job_texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([resume_text] + job_texts)
    similarity = cosine_similarity([embeddings[0]], embeddings[1:]).flatten()
    return list(zip(job_texts, similarity))

# Resume suggestion generator
"""
def generate_resume_suggestions(missing_skills, job_description):
    if not missing_skills:
        return "Your resume covers all key skills for this job."
    
    skill_list = ", ".join(missing_skills)
    prompt = (
        f"You are a career coach. A job description requires the following key skills missing from the candidate's resume: {skill_list}.\n"
        f"Write 3-5 concise, impactful resume bullet points that demonstrate proficiency in these skills."
        f"Keep it professional and relevant to the following job context:\n{job_description}\n"
    )

    # WIP: Implement LLM call here later

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for resume improvement."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
            n=1,
        )
        suggestions = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        suggestions = f"Error generating suggestions: {str(e)}"

    return suggestions
"""
