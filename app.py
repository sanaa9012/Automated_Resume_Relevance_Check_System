# app.py
import os
import re
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx2txt
from sentence_transformers import SentenceTransformer
import hashlib
import pickle
import google.generativeai as genai  # Gemini AI placeholder

# ---------------- Config ----------------
load_dotenv()
APP_TITLE = "Automated Resume Relevance Check System"
DB_PATH = "evaluations.db"
EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)

# ---------------- Database Helpers ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Evaluations table
    c.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_name TEXT,
            jd_name TEXT,
            keyword_score REAL,
            semantic_score REAL,
            final_score REAL,
            verdict TEXT,
            matched_keywords TEXT,
            missing_keywords TEXT,
            ai_suggestions TEXT,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Embeddings cache table
    c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_type TEXT,
            name TEXT,
            text_hash TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

def save_evaluation(resume_name, jd_name, kw_score, sem_score, final_score, verdict, matched, missing, ai_text):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO evaluations
        (resume_name, jd_name, keyword_score, semantic_score, final_score, verdict,
         matched_keywords, missing_keywords, ai_suggestions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (resume_name, jd_name, kw_score, sem_score, final_score, verdict,
          ",".join(matched), ",".join(missing), ai_text))
    conn.commit()
    conn.close()

def delete_evaluations_by_name(jd_names=[], resume_names=[]):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if jd_names:
        c.execute(f"DELETE FROM evaluations WHERE jd_name IN ({','.join(['?']*len(jd_names))})", jd_names)
    if resume_names:
        c.execute(f"DELETE FROM evaluations WHERE resume_name IN ({','.join(['?']*len(resume_names))})", resume_names)
    conn.commit()
    conn.close()

def load_all_evaluations():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM evaluations ORDER BY upload_time DESC", conn)
    conn.close()
    return df

# ---------------- Embedding Helpers ----------------
def text_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def get_cached_embedding(text_type, name, text):
    h = text_hash(text)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT embedding FROM embeddings WHERE text_type=? AND name=? AND text_hash=?", (text_type, name, h))
    row = c.fetchone()
    conn.close()
    if row:
        return pickle.loads(row[0])
    return None

def cache_embedding(text_type, name, text, emb):
    h = text_hash(text)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO embeddings (text_type,name,text_hash,embedding) VALUES (?,?,?,?)",
              (text_type, name, h, pickle.dumps(emb)))
    conn.commit()
    conn.close()

def get_embeddings_batch_cached(text_type, names_texts):
    embeddings = []
    for name, text in names_texts.items():
        emb = get_cached_embedding(text_type, name, text)
        if emb is None:
            emb = embedder.encode(text, show_progress_bar=False)
            cache_embedding(text_type, name, text, emb)
        embeddings.append(emb)
    return np.array(embeddings)

# ---------------- Text Extraction ----------------
def extract_text(uploaded_file):
    text = ""
    name = getattr(uploaded_file, "name", "")
    lower = name.lower()
    try:
        if lower.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            for p in reader.pages:
                page_text = p.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif lower.endswith(".docx"):
            try:
                text = docx2txt.process(uploaded_file)
            except Exception:
                b = uploaded_file.read()
                tmp = f"/tmp/{name}"
                with open(tmp, "wb") as f:
                    f.write(b)
                text = docx2txt.process(tmp)
                os.remove(tmp)
        elif lower.endswith(".txt"):
            raw = uploaded_file.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            text = raw
    except Exception as e:
        st.warning(f"Text extraction failed for {name}: {e}")
    return (text or "").strip()

def preprocess_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip() if t else ""

def cosine_sim_matrix(emb_a, emb_b):
    if emb_a.size == 0 or emb_b.size == 0:
        return np.zeros((emb_a.shape[0], emb_b.shape[0]))
    a_norm = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
    b_norm = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

# ---------------- Scoring ----------------
def hard_match_score(resume_text, jd_text):
    r = set(preprocess_text(resume_text).split())
    j = set(preprocess_text(jd_text).split())
    if not j:
        return 0.0, [], []
    matched = list(j.intersection(r))
    missing = list(j - r)
    score = len(matched) / max(1, len(j)) * 100
    return round(score, 2), matched, missing

def compute_verdict(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"

# ---------------- AI Suggestions ----------------
def ai_suggestions(resume_text, jd_text):
    return "(Gemini AI suggestions placeholder)"

# ---------------- Streamlit UI ----------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("AI suggestions for improving your resume against the Job Description.")
    init_db()

    # ----------- Session State -----------
    if "jd_texts" not in st.session_state: st.session_state.jd_texts = {}
    if "resume_texts" not in st.session_state: st.session_state.resume_texts = {}

    # ----------- Sidebar Uploads & Delete -----------
    st.sidebar.header("Upload Files")

    # Upload JDs
    jd_files = st.sidebar.file_uploader("Upload JDs (PDF/DOCX/TXT)", type=["pdf","docx","txt"], accept_multiple_files=True)
    if jd_files:
        for f in jd_files: st.session_state.jd_texts[f.name] = extract_text(f)
        st.sidebar.success(f"{len(jd_files)} JD(s) uploaded.")

    if st.session_state.jd_texts:
        st.sidebar.subheader("Uploaded JDs")
        for jd in st.session_state.jd_texts: st.sidebar.text(jd)

    # Upload Resumes
    resume_files = st.sidebar.file_uploader("Upload Resumes (PDF/DOCX/TXT)", type=["pdf","docx","txt"], accept_multiple_files=True)
    if resume_files:
        for f in resume_files: st.session_state.resume_texts[f.name] = extract_text(f)
        st.sidebar.success(f"{len(resume_files)} resume(s) uploaded.")

    if st.session_state.resume_texts:
        st.sidebar.subheader("Uploaded Resumes")
        for r in st.session_state.resume_texts: st.sidebar.text(r)

    # Delete JDs
    if st.session_state.jd_texts:
        jd_to_delete = st.sidebar.multiselect("Select JD(s) to delete", list(st.session_state.jd_texts.keys()))
        if jd_to_delete:
            for jd_name in jd_to_delete: st.session_state.jd_texts.pop(jd_name, None)
            delete_evaluations_by_name(jd_names=jd_to_delete)
            st.sidebar.info(f"Deleted JD(s) and related evaluations: {', '.join(jd_to_delete)}")

    # Delete Resumes
    if st.session_state.resume_texts:
        resume_to_delete = st.sidebar.multiselect("Select Resume(s) to delete", list(st.session_state.resume_texts.keys()))
        if resume_to_delete:
            for resume_name in resume_to_delete: st.session_state.resume_texts.pop(resume_name, None)
            delete_evaluations_by_name(resume_names=resume_to_delete)
            st.sidebar.info(f"Deleted Resume(s) and related evaluations: {', '.join(resume_to_delete)}")

    jd_texts = st.session_state.jd_texts
    resumes_texts = st.session_state.resume_texts

    # ----------- Main Page -----------
    st.subheader("Resume Evaluations Overview")
    df_all = load_all_evaluations()

    if not df_all.empty:
        verdict_means = df_all.groupby("verdict")["final_score"].mean()
        st.bar_chart(verdict_means, height=250)

        verdict_counts = df_all["verdict"].value_counts()
        col1, col2, col3 = st.columns(3)
        col1.metric("High", verdict_counts.get('High',0))
        col2.metric("Medium", verdict_counts.get('Medium',0))
        col3.metric("Low", verdict_counts.get('Low',0))

    # Evaluate All Button
    if st.button("🔍 Evaluate All"):
        if not jd_texts and not resumes_texts:
            st.warning("Upload at least one JD or one Resume.")
            st.stop()

        jd_names = list(jd_texts.keys()) or ["No JD"]
        resume_names = list(resumes_texts.keys()) or ["No Resume"]

        st.info("Computing embeddings (cached)...")
        jd_embeddings = get_embeddings_batch_cached("JD", jd_texts) if jd_texts else np.zeros((len(jd_names), embedder.get_sentence_embedding_dimension()))
        resume_embeddings = get_embeddings_batch_cached("Resume", resumes_texts) if resumes_texts else np.zeros((len(resume_names), embedder.get_sentence_embedding_dimension()))
        sim_matrix = cosine_sim_matrix(jd_embeddings, resume_embeddings)
        sem_matrix = np.round(np.clip(sim_matrix, -1,1)*100,2)

        total_pairs = len(jd_names)*len(resume_names)
        progress = st.progress(0)
        status = st.empty()
        pair_idx = 0

        for ji, jd_name in enumerate(jd_names):
            jd_text = jd_texts.get(jd_name,"")
            for ri, resume_name in enumerate(resume_names):
                pair_idx +=1
                status.text(f"Scoring {pair_idx}/{total_pairs}: {resume_name} ⇢ {jd_name}")
                kw_score, matched, missing = hard_match_score(resumes_texts.get(resume_name,""), jd_text)
                sem_score = float(sem_matrix[ji, ri])
                final = round(kw_score*0.5 + sem_score*0.5,2)
                verdict = compute_verdict(final)
                save_evaluation(resume_name, jd_name, kw_score, sem_score, final, verdict, matched, missing, "")
                progress.progress(int(pair_idx/total_pairs*100))

        df_all = load_all_evaluations()

    # Show evaluations
    if not df_all.empty:
        st.subheader("Evaluations by JD")
        for jd in df_all["jd_name"].unique():
            st.markdown(f"### {jd}")
            df_jd = df_all[df_all["jd_name"]==jd]
            display_cols = ["resume_name","final_score","verdict","upload_time"]
            st.dataframe(df_jd[display_cols])
            csv_bytes = df_jd[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(f"⬇️ Download CSV for {jd}", data=csv_bytes,
                               file_name=f"evaluations_{jd}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv")

    # AI Suggestions
    if not df_all.empty:
        st.subheader("AI Suggestions (expanders)")
        for jd in df_all["jd_name"].unique():
            df_jd = df_all[df_all["jd_name"]==jd]
            top_resumes = df_jd.sort_values("final_score",ascending=False).head(3)
            for _, row in top_resumes.iterrows():
                resume_name = row["resume_name"]
                eval_id = row["id"]
                with st.expander(f"{resume_name} ⇢ {jd}"):
                    ai_txt = ai_suggestions("", "")
                    st.text_area("Gemini AI Suggestions", value=ai_txt or "(empty)", height=300,
                                 key=f"{eval_id}_ai_suggestions")

if __name__=="__main__":
    main()
