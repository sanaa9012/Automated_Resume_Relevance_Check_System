import os
import re
import io
import time
import textwrap
from typing import Dict

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx2txt
from sentence_transformers import SentenceTransformer
import numpy as np

# ---------------------- Config ----------------------
APP_TITLE = "🧠 ATS Resume Score Checker (AI-assisted)"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------- Utils ----------------------
def extract_text_from_pdf(file) -> str:
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(file) -> str:
    return docx2txt.process(file)

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_embedding(text: str):
    embedding = embedder.encode([text])[0]  # numpy array
    return embedding

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ---------------------- Scoring ----------------------
def calculate_relevance(resume_text: str, jd_text: str) -> Dict:
    resume_text = preprocess_text(resume_text)
    jd_text = preprocess_text(jd_text)

    # --- Hard Match: keyword overlap ---
    jd_keywords = set(jd_text.split())
    resume_words = set(resume_text.split())
    matched_keywords = jd_keywords.intersection(resume_words)
    keyword_score = len(matched_keywords) / max(1, len(jd_keywords))

    # --- Semantic Match ---
    jd_embedding = get_embedding(jd_text)
    resume_embedding = get_embedding(resume_text)
    semantic_score = cosine_similarity(jd_embedding, resume_embedding)

    # --- Weighted Final Score ---
    final_score = (0.5 * keyword_score + 0.5 * semantic_score) * 100

    # Verdict
    if final_score >= 75:
        verdict = "High"
    elif final_score >= 50:
        verdict = "Medium"
    else:
        verdict = "Low"

    return {
        "final_score": round(final_score, 2),
        "keyword_score": round(keyword_score * 100, 2),
        "semantic_score": round(semantic_score * 100, 2),
        "verdict": verdict,
        "matched_keywords": list(matched_keywords),
        "missing_keywords": list(jd_keywords - matched_keywords),
    }

# ---------------------- Streamlit UI ----------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # --- JD Upload ---
    st.header("📄 Upload Job Description")
    jd_file = st.file_uploader("Upload JD (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key="jd")

    jd_text = ""
    if jd_file is not None:
        if jd_file.type == "application/pdf":
            jd_text = extract_text_from_pdf(jd_file)
        elif jd_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            jd_text = extract_text_from_docx(jd_file)
        else:
            jd_text = jd_file.read().decode("utf-8")

        st.success("✅ Job Description uploaded!")

    # --- Resume Upload ---
    st.header("👤 Upload Resume")
    resume_file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key="resume")

    resume_text = ""
    if resume_file is not None:
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(resume_file)
        else:
            resume_text = resume_file.read().decode("utf-8")

        st.success("✅ Resume uploaded!")

    # --- Evaluate ---
    if st.button("🔍 Evaluate Resume Relevance") and jd_text and resume_text:
        with st.spinner("Evaluating..."):
            result = calculate_relevance(resume_text, jd_text)

        st.subheader("📊 Results")
        st.metric("Final Relevance Score", f"{result['final_score']}%")
        st.write(f"**Verdict:** {result['verdict']}")
        st.write(f"**Keyword Match Score:** {result['keyword_score']}%")
        st.write(f"**Semantic Match Score:** {result['semantic_score']}%")

        st.subheader("✅ Matched Keywords")
        st.write(", ".join(result["matched_keywords"]) if result["matched_keywords"] else "None")

        st.subheader("❌ Missing Keywords")
        st.write(", ".join(result["missing_keywords"]) if result["missing_keywords"] else "None")

if __name__ == "__main__":
    main()
