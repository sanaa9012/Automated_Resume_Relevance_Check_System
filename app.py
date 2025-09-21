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
import google.generativeai as genai
import hashlib
import pickle
import altair as alt  # For dynamic graphs

# ---------------- Config ----------------
load_dotenv()
APP_TITLE = "🧠 ATS Resume Score Checker — Persistent SQLite + Embeddings"
DB_PATH = "evaluations.db"
DEFAULT_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()

TOP_K_AI_PER_JD = 3
EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)

if DEFAULT_GEMINI_KEY:
    try:
        genai.configure(api_key=DEFAULT_GEMINI_KEY)
    except Exception:
        pass

# ---------------- DB helpers ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_name TEXT,
            jd_name TEXT,
            resume_text TEXT,
            jd_text TEXT,
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

def save_evaluation(resume_name, jd_name, resume_text, jd_text,
                    keyword_score, semantic_score, final_score, verdict,
                    matched, missing, ai_text):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO evaluations
        (resume_name, jd_name, resume_text, jd_text,
         keyword_score, semantic_score, final_score, verdict,
         matched_keywords, missing_keywords, ai_suggestions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        resume_name, jd_name, resume_text, jd_text,
        keyword_score, semantic_score, final_score, verdict,
        ",".join(matched), ",".join(missing), ai_text
    ))
    conn.commit()
    conn.close()

def load_all_evaluations():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM evaluations ORDER BY upload_time DESC", conn)
    conn.close()

    # Clean up final_score column
    def safe_convert(val):
        if isinstance(val, (bytes, bytearray)):
            try:
                import struct
                return float(struct.unpack("f", val)[0])
            except Exception:
                return None
        return val

    if "final_score" in df.columns:
        df["final_score"] = df["final_score"].apply(safe_convert)
        df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")

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
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

# ---------------- AI Suggestions ----------------
def ai_suggestions(gemini_key, resume_text, jd_text):
    if not gemini_key:
        return ""
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""You are an expert ATS and resume coach.
JOB DESCRIPTION:
{jd_text[:8000]}

RESUME:
{resume_text[:8000]}

Provide:
1) Top 10 missing/weak skills
2) 5–8 bullet rewrites (metrics-driven, action verbs)
3) Quick checklist to improve ATS compatibility
Output concise actionable bullet points.
"""
        resp = model.generate_content(prompt)
        return resp.text if hasattr(resp, "text") else str(resp)
    except Exception as e:
        return f"(AI error) {e}"

# ---------------- Streamlit UI ----------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Persistent SQLite DB + Embedding caching. Expanders show AI suggestions.")

    init_db()

    st.markdown("**Step 1 — Upload JDs (multiple)**")
    jd_files = st.file_uploader("Upload JDs (PDF/DOCX/TXT)", type=["pdf","docx","txt"], accept_multiple_files=True)
    jd_texts = {}
    if jd_files:
        for f in jd_files:
            jd_texts[f.name] = extract_text(f)
        st.success(f"{len(jd_files)} JD(s) uploaded.")

    st.markdown("**Step 2 — Upload Resumes (multiple)**")
    resume_files = st.file_uploader("Upload Resumes (PDF/DOCX/TXT)", type=["pdf","docx","txt"], accept_multiple_files=True)
    resumes_texts = {}
    if resume_files:
        for f in resume_files:
            resumes_texts[f.name] = extract_text(f)
        st.success(f"{len(resume_files)} resume(s) uploaded.")

    st.markdown("---")
    st.markdown("**Options**")
    hard_weight = st.slider("Hard match (keyword) weight", 0.0, 1.0, 0.5, 0.05)
    top_k = st.number_input("Top K resumes per JD for Gemini AI", min_value=0, max_value=20, value=TOP_K_AI_PER_JD)
    gemini_key_input = st.text_input("Gemini API Key (optional)", value=DEFAULT_GEMINI_KEY, type="password")

    # Load persistent data
    df_all = load_all_evaluations()
    st.subheader("Previous Evaluations (persistent)")
    if not df_all.empty:
        st.dataframe(df_all[["resume_name","jd_name","final_score","verdict","upload_time"]])
        csv_bytes = df_all.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Full Evaluations CSV", data=csv_bytes,
                           file_name=f"evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")
    else:
        st.info("No evaluations yet.")

    # Evaluate all
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
        results = []
        ai_texts = {}
        pair_idx = 0

        # numeric scoring
        for ji, jd_name in enumerate(jd_names):
            jd_text = jd_texts[jd_name]
            for ri, resume_name in enumerate(resume_names):
                pair_idx +=1
                status.text(f"Scoring {pair_idx}/{total_pairs}: {resume_name} ⇢ {jd_name}")
                kw_score, matched, missing = hard_match_score(resumes_texts[resume_name], jd_text)
                sem_score = float(sem_matrix[ji, ri])
                final = round(kw_score*hard_weight + sem_score*(1-hard_weight),2)
                verdict = compute_verdict(final)
                save_evaluation(resume_name, jd_name, resumes_texts[resume_name], jd_text,
                                kw_score, sem_score, final, verdict, matched, missing, "")
                results.append({"Resume": resume_name,"JD":jd_name,"Keyword Score":kw_score,"Semantic Score":sem_score,"Final Score":final,"Verdict":verdict})
                progress.progress(int(pair_idx/total_pairs*100))

        # AI suggestions
        if top_k>0 and gemini_key_input:
            st.info(f"Running Gemini AI for top {top_k} resumes per JD...")
            df_results = pd.DataFrame(results)
            for jd_name in jd_names:
                df_j = df_results[df_results["JD"]==jd_name].sort_values("Final Score",ascending=False).head(top_k)
                for _, row in df_j.iterrows():
                    resume_name = row["Resume"]
                    ai_txt = ai_suggestions(gemini_key_input, resumes_texts[resume_name], jd_texts[jd_name])
                    ai_texts[(jd_name,resume_name)] = ai_txt
                    # update DB
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute("SELECT id FROM evaluations WHERE resume_name=? AND jd_name=? ORDER BY upload_time DESC LIMIT 1",(resume_name,jd_name))
                        res = c.fetchone()
                        if res:
                            row_id = res[0]
                            c.execute("UPDATE evaluations SET ai_suggestions=? WHERE id=?",(ai_txt,row_id))
                            conn.commit()
                        conn.close()
                    except Exception as e:
                        st.warning(f"Failed to update AI text: {e}")

        # show table
        df_all = load_all_evaluations()
        st.subheader("Updated Evaluations")
        st.dataframe(df_all[["resume_name","jd_name","final_score","verdict","upload_time"]])

        # CSV download
        csv_bytes = df_all.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Full Evaluations CSV", data=csv_bytes,
                           file_name=f"evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")

        # Expanders for AI
        if ai_texts:
            st.markdown("---")
            st.subheader("AI Suggestions (expanders)")
            for (jd_name,resume_name), ai_txt in ai_texts.items():
                with st.expander(f"{resume_name} ⇢ {jd_name}"):
                    st.text_area("Gemini AI Suggestions", value=ai_txt or "(empty)", height=300)

        # ---------------- Dashboard Graphs ----------------
        st.markdown("---")
        st.subheader("📊 Dynamic Dashboard")

        if not df_all.empty:
            # 1. Final Score Distribution
            st.markdown("**Final Score Distribution**")
            st.bar_chart(df_all["final_score"])

            # 2. Verdict Counts per JD
            st.markdown("**Verdict Counts per Job Description**")
            verdict_count = df_all.groupby(["jd_name","verdict"]).size().unstack(fill_value=0)
            st.bar_chart(verdict_count)

            # 3. Top Resumes per JD
            st.markdown("**Top Resumes per JD**")
            for jd_name in df_all["jd_name"].unique():
                st.markdown(f"**{jd_name}**")
                top_res = df_all[df_all["jd_name"]==jd_name].sort_values("final_score", ascending=False).head(5)
                st.bar_chart(top_res.set_index("resume_name")["final_score"])

            # 4. Keyword vs Semantic Score Scatter
            st.markdown("**Keyword Score vs Semantic Score**")
            scatter = alt.Chart(df_all).mark_circle(size=60).encode(
                x="keyword_score",
                y="semantic_score",
                color="verdict",
                tooltip=["resume_name","jd_name","final_score"]
            ).interactive()
            st.altair_chart(scatter, use_container_width=True)

if __name__=="__main__":
    main()
