# app.py
import os
import re
import sqlite3
import numpy as np
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx2txt
from sentence_transformers import SentenceTransformer
import hashlib
import pickle

# ---------------- Config ----------------
load_dotenv()
APP_TITLE = "Automated Resume Relevance Check System"
DB_PATH = "evaluations.db"

EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)

# ---------------- DB helpers ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
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

def save_evaluation(resume_name, jd_name, keyword_score, semantic_score, final_score, verdict, matched, missing, ai_text):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO evaluations
        (resume_name, jd_name, keyword_score, semantic_score, final_score, verdict,
         matched_keywords, missing_keywords, ai_suggestions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        resume_name, jd_name, keyword_score, semantic_score, final_score, verdict,
        ",".join(matched), ",".join(missing), ai_text
    ))
    conn.commit()
    conn.close()

def load_all_evaluations():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM evaluations ORDER BY upload_time DESC", conn)
    conn.close()
    return df

# ---------------- Embedding caching ----------------
def text_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def get_cached_embedding(text_type, name, text):
    h = text_hash(text)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT embedding FROM embeddings WHERE text_type=? AND name=? AND text_hash=?",(text_type,name,h))
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
              (text_type,name,h,pickle.dumps(emb)))
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

# ---------------- Text extraction ----------------
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
    if not t:
        return ""
    s = t.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def cosine_sim_matrix(emb_a, emb_b):
    if emb_a.size == 0 or emb_b.size == 0:
        return np.zeros((emb_a.shape[0], emb_b.shape[0]))
    a_norm = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
    b_norm = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

# ---------------- scoring ----------------
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
    return "(AI suggestions not available in this version)"

# ---------------- Streamlit UI ----------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("AI suggestions for doing better for your next Job.")

    init_db()

    # ---------- Sidebar ----------
    st.sidebar.header("Upload Files")
    jd_files = st.sidebar.file_uploader("Upload JDs (PDF/DOCX/TXT)", type=["pdf","docx","txt"], accept_multiple_files=True)
    resume_files = st.sidebar.file_uploader("Upload Resumes (PDF/DOCX/TXT)", type=["pdf","docx","txt"], accept_multiple_files=True)

    jd_texts = {}
    if jd_files:
        for f in jd_files:
            jd_texts[f.name] = extract_text(f)
        st.sidebar.success(f"{len(jd_files)} JD(s) uploaded.")

    resumes_texts = {}
    if resume_files:
        for f in resume_files:
            resumes_texts[f.name] = extract_text(f)
        st.sidebar.success(f"{len(resume_files)} resume(s) uploaded.")

    # ---------------- Main Page ----------------
    st.subheader("Resume Evaluations Overview")
    df_all = load_all_evaluations()

    if not df_all.empty:
        # Graph on top
        st.bar_chart(df_all.groupby("verdict")["final_score"].mean())

    # Button to evaluate all
    if st.button("🔍 Evaluate All"):
        if not jd_texts or not resumes_texts:
            st.warning("Upload at least one JD and one Resume.")
            st.stop()

        jd_names = list(jd_texts.keys())
        resume_names = list(resumes_texts.keys())

        st.info("Computing embeddings (cached)...")
        jd_embeddings = get_embeddings_batch_cached("JD", jd_texts)
        resume_embeddings = get_embeddings_batch_cached("Resume", resumes_texts)
        sim_matrix = cosine_sim_matrix(jd_embeddings, resume_embeddings)
        sem_matrix = np.round(np.clip(sim_matrix, -1,1)*100,2)

        total_pairs = len(jd_names)*len(resume_names)
        progress = st.progress(0)
        status = st.empty()
        pair_idx = 0

        # numeric scoring
        for ji, jd_name in enumerate(jd_names):
            jd_text = jd_texts[jd_name]
            for ri, resume_name in enumerate(resume_names):
                pair_idx +=1
                status.text(f"Scoring {pair_idx}/{total_pairs}: {resume_name} ⇢ {jd_name}")
                kw_score, matched, missing = hard_match_score(resumes_texts[resume_name], jd_text)
                sem_score = float(sem_matrix[ji, ri])
                final = round(kw_score*0.5 + sem_score*0.5,2)
                verdict = compute_verdict(final)
                save_evaluation(resume_name, jd_name, kw_score, sem_score, final, verdict, matched, missing, "")
                progress.progress(int(pair_idx/total_pairs*100))

        # Refresh data
        df_all = load_all_evaluations()

    # ---------------- Show evaluations per JD ----------------
    if not df_all.empty:
        st.subheader("Evaluations by JD")
        jd_names = df_all["jd_name"].unique()
        for jd in jd_names:
            st.markdown(f"### {jd}")
            df_jd = df_all[df_all["jd_name"]==jd]
            display_cols = ["resume_name","final_score","verdict","upload_time"]
            st.dataframe(df_jd[display_cols])

            # CSV download per JD
            csv_bytes = df_jd[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(f"⬇️ Download CSV for {jd}", data=csv_bytes,
                               file_name=f"evaluations_{jd}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv")

    # ---------------- AI Suggestions (bottom expanders) ----------------
    if not df_all.empty:
        st.subheader("AI Suggestions (expanders)")
        for jd in df_all["jd_name"].unique():
            df_jd = df_all[df_all["jd_name"]==jd]
            top_resumes = df_jd.sort_values("final_score",ascending=False).head(3)
            for _, row in top_resumes.iterrows():
                resume_name = row["resume_name"]
                eval_id = row["id"]  # unique DB ID
                with st.expander(f"{resume_name} ⇢ {jd}"):
                    ai_txt = ai_suggestions("", "")
                    st.text_area(
                        "Gemini AI Suggestions",
                        value=ai_txt or "(empty)",
                        height=300,
                        key=f"{eval_id}_ai_suggestions"  
                    )

if __name__=="__main__":
    main()
