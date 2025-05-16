# 🧠 AI Resume Matcher & Job Relevance Screener

An intelligent resume screening application built with Python, Streamlit, and Machine Learning. This app allows users to upload a resume (PDF or DOCX) and compare it against up to 5 job descriptions to determine the best-fit roles based on semantic similarity and skill matching.

---

## 🚀 Features

- ✅ Upload and parse resumes (PDF/DOCX)
- ✅ Paste multiple job descriptions (up to 5)
- ✅ Match using TF-IDF and BERT-based sentence embeddings
- ✅ Visual similarity scores per job
- ✅ Extract and compare skills (gap detection)
- ✅ Clean and interactive Streamlit interface
- ✅ NLP-based preprocessing and smart parsing

---

## 🖼️ Demo

> *Coming Soon: Hosted version on Streamlit Cloud or Hugging Face Spaces*

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **NLP**: spaCy, Sentence Transformers, scikit-learn
- **File Parsing**: pdfplumber, docx2txt
- **ML Matching**: BERT embeddings, cosine similarity

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/resume-matcher.git
cd resume-matcher
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv/Scripts/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run the App
```bash
streamlit run app.py
```

---

## Project Structure
```bash
resume-matcher/
├── app.py                      # Streamlit application
├── matcher.py                 # TF-IDF & BERT-based matching logic
├── resume_parser.py          # Resume parsing for PDF and DOCX
├── skill_gap.py              # Skill extraction and comparison
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

---

## Future Improvements
- Resume tailoring with GPT (generate job-specific edits)
- OCR support for scanned PDFs
- Save and compare match history
- User accounts and resume library
- Export match reports as PDF

---

## Author
James DeFoggia

---

## License
MIT License - free to use and modify for personal or commercial use

---

## Contributing
Pull requests welcome! For major changes, please open an issue first to discuss what you'd like to change or improve.
