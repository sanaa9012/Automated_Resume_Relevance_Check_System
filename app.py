# app.py
import os, re, sqlite3, pickle, hashlib, numpy as np, pandas as pd, streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx2txt
from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
load_dotenv()
APP_TITLE = "Automated Resume Relevance Check System"
DB_PATH = "evaluations.db"
EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)

# ---------------- Database ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        resume_name TEXT, jd_name TEXT, keyword_score REAL,
        semantic_score REAL, final_score REAL, verdict TEXT,
        matched_keywords TEXT, missing_keywords TEXT,
        ai_suggestions TEXT, upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text_type TEXT, name TEXT, text_hash TEXT, embedding BLOB
    )""")
    conn.commit(); conn.close()

def save_eval(resume_name, jd_name, kw_score, sem_score, final_score, verdict, matched, missing, ai_text):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO evaluations
        (resume_name, jd_name, keyword_score, semantic_score, final_score, verdict,
        matched_keywords, missing_keywords, ai_suggestions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (resume_name, jd_name, kw_score, sem_score, final_score, verdict,
         ",".join(matched), ",".join(missing), ai_text))
    conn.commit(); conn.close()

def delete_eval(jd_names=[], resume_names=[]):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    if jd_names:
        c.execute(f"DELETE FROM evaluations WHERE jd_name IN ({','.join(['?']*len(jd_names))})", jd_names)
    if resume_names:
        c.execute(f"DELETE FROM evaluations WHERE resume_name IN ({','.join(['?']*len(resume_names))})", resume_names)
    conn.commit(); conn.close()

def load_all_evals():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM evaluations ORDER BY upload_time DESC", conn)
    conn.close(); return df

# ---------------- Embeddings ----------------
def text_hash(text): return hashlib.md5(text.encode("utf-8")).hexdigest()

def get_cached_emb(text_type, name, text):
    h = text_hash(text)
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT embedding FROM embeddings WHERE text_type=? AND name=? AND text_hash=?", (text_type,name,h))
    row = c.fetchone(); conn.close()
    return pickle.loads(row[0]) if row else None

def cache_emb(text_type, name, text, emb):
    h = text_hash(text)
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("INSERT INTO embeddings (text_type,name,text_hash,embedding) VALUES (?,?,?,?)", (text_type,name,h,pickle.dumps(emb)))
    conn.commit(); conn.close()

def get_embeddings_cached(text_type, names_texts):
    embeddings = []
    for name, text in names_texts.items():
        emb = get_cached_emb(text_type, name, text)
        if emb is None:
            emb = embedder.encode(text, show_progress_bar=False)
            cache_emb(text_type, name, text, emb)
        embeddings.append(emb)
    return np.array(embeddings)

# ---------------- Text ----------------
def extract_text(file):
    text = ""; name = getattr(file, "name", "").lower()
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(file)
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
        elif name.endswith(".docx"):
            try: text = docx2txt.process(file)
            except Exception:
                tmp = f"/tmp/{name}"
                with open(tmp,"wb") as f: f.write(file.read())
                text = docx2txt.process(tmp); os.remove(tmp)
        elif name.endswith(".txt"):
            raw = file.read(); text = raw.decode("utf-8",errors="ignore") if isinstance(raw,bytes) else raw
    except Exception as e: st.warning(f"Text extraction failed for {name}: {e}")
    return text.strip()

def preprocess_text(t): return re.sub(r"\s+"," ", t.lower()).strip() if t else ""

def cosine_sim_matrix(a,b):
    if a.size==0 or b.size==0: return np.zeros((a.shape[0],b.shape[0]))
    a_norm = a/np.linalg.norm(a,axis=1,keepdims=True)
    b_norm = b/np.linalg.norm(b,axis=1,keepdims=True)
    return np.dot(a_norm,b_norm.T)

# ---------------- Scoring ----------------
def hard_match_score(resume_text,jd_text):
    r,j = set(preprocess_text(resume_text).split()), set(preprocess_text(jd_text).split())
    matched = list(j.intersection(r)); missing = list(j-r)
    score = round(len(matched)/max(1,len(j))*100,2) if j else 0.0
    return score, matched, missing

def compute_verdict(score):
    if score>=70: return "High"
    elif score>=40: return "Medium"
    else: return "Low"

# ---------------- AI Suggestions ----------------
def ai_suggestions(resume_text,jd_text): return "(Gemini AI suggestions placeholder)"

# ---------------- Streamlit UI ----------------
def sidebar_upload_delete():
    st.sidebar.header("Upload Files")
    # Upload JDs
    jd_files = st.sidebar.file_uploader("Upload JDs", type=["pdf","docx","txt"], accept_multiple_files=True)
    for f in jd_files or []: st.session_state.jd_texts[f.name]=extract_text(f)
    # Upload Resumes
    resume_files = st.sidebar.file_uploader("Upload Resumes", type=["pdf","docx","txt"], accept_multiple_files=True)
    for f in resume_files or []: st.session_state.resume_texts[f.name]=extract_text(f)
    # Delete
    if st.session_state.jd_texts:
        to_del = st.sidebar.multiselect("Delete JD(s)", list(st.session_state.jd_texts.keys()))
        for jd in to_del: st.session_state.jd_texts.pop(jd); delete_eval(jd_names=[jd])
    if st.session_state.resume_texts:
        to_del = st.sidebar.multiselect("Delete Resume(s)", list(st.session_state.resume_texts.keys()))
        for r in to_del: st.session_state.resume_texts.pop(r); delete_eval(resume_names=[r])

def evaluate_all():
    jd_texts, resumes_texts = st.session_state.jd_texts, st.session_state.resume_texts
    if not jd_texts and not resumes_texts: st.warning("Upload at least one JD or Resume."); return
    jd_names, resume_names = list(jd_texts.keys()) or ["No JD"], list(resumes_texts.keys()) or ["No Resume"]
    st.info("Computing embeddings (cached)...")
    jd_emb = get_embeddings_cached("JD", jd_texts) if jd_texts else np.zeros((len(jd_names), embedder.get_sentence_embedding_dimension()))
    res_emb = get_embeddings_cached("Resume", resumes_texts) if resumes_texts else np.zeros((len(resume_names), embedder.get_sentence_embedding_dimension()))
    sim_matrix = np.round(np.clip(cosine_sim_matrix(jd_emb,res_emb),-1,1)*100,2)
    total = len(jd_names)*len(resume_names); progress=st.progress(0); status=st.empty(); idx=0
    for ji,jd in enumerate(jd_names):
        jd_text = jd_texts.get(jd,"")
        for ri,resume in enumerate(resume_names):
            idx+=1; status.text(f"Scoring {idx}/{total}: {resume} ⇢ {jd}")
            kw, matched, missing = hard_match_score(resumes_texts.get(resume,""), jd_text)
            sem = float(sim_matrix[ji,ri]); final = round(kw*0.5+sem*0.5,2)
            save_eval(resume,jd,kw,sem,final,compute_verdict(final),matched,missing,"")
            progress.progress(int(idx/total*100))
    st.success("Evaluation Complete!")

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE); st.caption("AI suggestions for improving your resume against the Job Description.")
    init_db()
    if "jd_texts" not in st.session_state: st.session_state.jd_texts={}
    if "resume_texts" not in st.session_state: st.session_state.resume_texts={}
    sidebar_upload_delete()

    # ----------- Main Dashboard -----------
    st.subheader("Resume Evaluations Overview")
    df = load_all_evals()
    if not df.empty:
        verdict_means = df.groupby("verdict")["final_score"].mean()
        st.bar_chart(verdict_means, height=250)
        counts = df["verdict"].value_counts()
        c1,c2,c3 = st.columns(3)
        c1.metric("High", counts.get('High',0)); c2.metric("Medium", counts.get('Medium',0)); c3.metric("Low", counts.get('Low',0))

    if st.button("🔍 Evaluate All"): evaluate_all(); df = load_all_evals()

    if not df.empty:
        st.subheader("Evaluations by JD")
        for jd in df["jd_name"].unique():
            st.markdown(f"### {jd}")
            df_jd = df[df["jd_name"]==jd]; cols=["resume_name","final_score","verdict","upload_time"]
            st.dataframe(df_jd[cols])
            csv_bytes = df_jd[cols].to_csv(index=False).encode("utf-8")
            st.download_button(f"⬇️ Download CSV for {jd}", data=csv_bytes,
                               file_name=f"evaluations_{jd}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

        st.subheader("AI Suggestions (Top 3 resumes)")
        for jd in df["jd_name"].unique():
            top = df[df["jd_name"]==jd].sort_values("final_score",ascending=False).head(3)
            for _, row in top.iterrows():
                with st.expander(f"{row['resume_name']} ⇢ {jd}"):
                    st.text_area("Gemini AI Suggestions", value=ai_suggestions("",""), height=300, key=f"{row['id']}_ai")

if __name__=="__main__": main()
