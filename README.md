# 🧠 ATS Resume Score Checker (AI‑assisted) — Streamlit

A one‑file web app that scores a resume against a job description (ATS‑style) and optionally uses OpenAI to generate actionable improvement tips.

## ✨ Features
- Upload **PDF/DOCX/TXT** resume or paste text
- Paste job description or upload TXT
- **Deterministic ATS score (0–100)** with a **clear breakdown**
- Missing/covered keyword analysis (unigrams + key phrases)
- Section checks (Education, Experience, Projects, Skills), contact info validation
- Bullet quality estimate (metrics/action verbs)
- Length/readability heuristics
- **Optional AI** suggestions using OpenAI Responses API (`gpt-4o-mini` by default)
- Downloadable analysis report

## 🧩 Tech
- Python 3.9+
- Streamlit
- OpenAI Python SDK (optional)
- PyPDF2, docx2txt

## 🚀 Quickstart

1) **Create & activate a virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2) **Install dependencies**
```bash
pip install -r requirements.txt
```

3) **(Optional) Set OpenAI API key**  
Create a `.env` file (copy `.env.example`) or pass in the UI.
```bash
# .env
OPENAI_API_KEY=your_key_here
```

4) **Run the app**
```bash
streamlit run app.py
```

5) Open your browser at the URL printed in the terminal.

---

## 🔑 About AI integration
This app supports OpenAI's **Responses API**. It will only call the API if an API key is detected (in `.env` or provided in the sidebar). Otherwise, it runs fully offline using heuristics.

> Docs: OpenAI API Quickstart, API Reference, and Responses API guides (see OpenAI docs).

## 📁 Project layout
```
ats-resume-score-checker/
├─ app.py
├─ requirements.txt
├─ .env.example
├─ sample_data/
│  ├─ sample_resume.txt
│  └─ sample_job_description.txt
└─ README.md
```

## 🧪 Try with sample data
- `sample_data/sample_resume.txt`
- `sample_data/sample_job_description.txt`

Paste them into the UI to see a demo score and suggestions.

## ⚠️ Disclaimer
ATS systems vary. This tool provides **heuristics** and **AI‑generated suggestions** and should be used as guidance—not a guarantee of how a particular ATS will score your resume.