import os
import re
import io
import math
import time
import textwrap
from dataclasses import dataclass
from typing import List, Dict, Tuple

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx2txt

# ---------------------- Config ----------------------
APP_TITLE = "🧠 ATS Resume Score Checker (AI‑assisted)"
AI_MODEL = "gpt-4o-mini"  # Change if you prefer another model name

# Load environment variables
load_dotenv()
DEFAULT_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Try to import the new OpenAI SDK (Responses API)
Client = None
try:
    from openai import OpenAI
    Client = OpenAI
except Exception:
    OpenAI = None
    Client = None

# ---------------------- Helpers ----------------------
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before being below between both
but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd
i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours  ourselves out over own same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this
those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

COMMON_SECTIONS = [
    "education", "experience", "work experience", "projects", "skills",
    "certifications", "achievements", "publications", "summary", "objective"
]

ACTION_VERBS = [
    "led","built","developed","designed","implemented","deployed","optimized","created","launched","migrated",
    "improved","reduced","increased","automated","integrated","analyzed","trained","evaluated","deployed","scaled",
    "refactored","orchestrated","owned","delivered","shipped","spearheaded","achieved"
]

KNOWN_SKILLS = [
    # Programming / DS
    "python","java","c++","sql","pandas","numpy","scikit-learn","tensorflow","pytorch","keras",
    "nlp","computer vision","opencv","transformers","hugging face",
    # Web / backend
    "fastapi","flask","django","streamlit","react","node","express",
    # DevOps / cloud
    "docker","kubernetes","linux","git","github","gitlab","aws","azure","gcp","ec2","s3","lambda","cloudwatch",
    # DB
    "postgresql","mysql","mongodb","redis","elasticsearch",
    # Tools
    "power bi","tableau","airflow","mlflow"
]

@dataclass
class ScoreBreakdown:
    keyword_coverage: float
    phrase_coverage: float
    sections: float
    contact: float
    bullets: float
    length: float
    formatting: float

    def total(self) -> float:
        w = dict(keyword_coverage=0.35, phrase_coverage=0.15, sections=0.1,
                 contact=0.1, bullets=0.1, length=0.1, formatting=0.1)
        return sum([self.keyword_coverage*w['keyword_coverage'],
                    self.phrase_coverage*w['phrase_coverage'],
                    self.sections*w['sections'],
                    self.contact*w['contact'],
                    self.bullets*w['bullets'],
                    self.length*w['length'],
                    self.formatting*w['formatting']])

# ----------- Text extraction -----------
def extract_text_from_pdf(file) -> str:
    try:
        reader = PdfReader(file)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\\n".join(text)
    except Exception as e:
        return ""

def extract_text_from_docx(file) -> str:
    try:
        # docx2txt expects a path or file‑like object; we can write to temp if needed
        # Save to in‑memory bytes, then to tmp file
        data = file.read()
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as t:
            t.write(data)
            t.flush()
            path = t.name
        text = docx2txt.process(path) or ""
        try:
            os.unlink(path)
        except Exception:
            pass
        return text
    except Exception:
        return ""

def read_text_upload(upload) -> str:
    if upload is None:
        return ""
    name = upload.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(upload)
    elif name.endswith(".docx"):
        return extract_text_from_docx(upload)
    else:
        try:
            return upload.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""

# ----------- NLP utils -----------
def normalize_text(t: str) -> str:
    t = t.replace("\\x00", " ")
    t = re.sub(r"[\\r\\t]", " ", t)
    t = re.sub(r"[ ]{2,}", " ", t)
    return t.strip()

def tokenize(text: str) -> List[str]:
    text = re.sub(r"[^a-zA-Z0-9\\+\\#\\- ]+", " ", text.lower())
    tokens = [tok for tok in text.split() if tok not in STOPWORDS and len(tok) > 1]
    return tokens

def extract_phrases(text: str, phrases: List[str]) -> List[str]:
    text_norm = " " + re.sub(r"\\s+", " ", text.lower()) + " "
    found = []
    for p in phrases:
        p_norm = " " + p.lower().strip() + " "
        if p_norm in text_norm:
            found.append(p)
    return found

def top_keywords(tokens: List[str], whitelist: List[str]=None, k:int=40) -> List[str]:
    from collections import Counter
    c = Counter(tokens)
    ranking = [w for w,_ in c.most_common(200)]
    if whitelist:
        ranking = [w for w in ranking if (w in whitelist or any(w in ks for ks in whitelist))]
    # De‑duplicate keep order
    dedup = []
    for w in ranking:
        if w not in dedup:
            dedup.append(w)
    return dedup[:k]

# ----------- Scoring -----------
def section_presence(resume_text: str) -> float:
    r = resume_text.lower()
    hits = 0
    required = ["education","experience","projects","skills"]
    for s in required:
        if s in r:
            hits += 1
    return (hits / len(required)) * 100.0

def contact_presence(resume_text: str) -> float:
    email_ok = bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", resume_text))
    phone_ok = bool(re.search(r"(\\+\\d{1,3}[- ]?)?\\d{10}", resume_text))
    pct = 0.0
    if email_ok: pct += 50.0
    if phone_ok: pct += 50.0
    return pct

def bullet_quality(resume_text: str) -> float:
    lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
    bullets = [l for l in lines if l.startswith(("-", "•", "*"))]
    if not bullets:
        # also consider lines that start with an action verb
        bullets = [l for l in lines if any(l.lower().startswith(v) for v in ACTION_VERBS)]
    if not bullets:
        return 30.0  # small credit even if formatted differently
    # score by proportion having a number or % and an action verb
    score = 0
    for b in bullets:
        has_num = bool(re.search(r"\\d", b))
        has_verb = any((" " + v + " ") in (" " + b.lower() + " ") for v in ACTION_VERBS)
        if has_num and has_verb:
            score += 1.0
        elif has_num or has_verb:
            score += 0.5
    frac = score / max(len(bullets), 1)
    return min(100.0, 50 + 50*frac)  # baseline 50 if bullets exist

def length_readability(resume_text: str) -> float:
    words = len(resume_text.split())
    # Ideal between 350 and 1200 words (roughly 1–2 pages plain text)
    if 350 <= words <= 1200:
        return 100.0
    if words < 200:
        return 40.0
    if words > 2000:
        return 50.0
    # interpolate
    if words < 350:
        return 40.0 + (words - 200) * (60.0/150.0)
    else:
        # 1200 -> 100; 2000 -> 50
        over = min(words, 2000) - 1200
        return 100.0 - (over * (50.0/800.0))

def formatting_health(resume_text: str) -> float:
    # penalize emojis/unicode, images refs, excessive symbols
    weird = len(re.findall(r"[\\u2600-\\u27BF]", resume_text))  # emojis block-ish
    odd = len(re.findall(r"[{}|<>~^`]", resume_text))
    img = len(re.findall(r".(png|jpg|jpeg|gif)", resume_text.lower()))
    penalties = weird*5 + odd*1 + img*10
    return max(40.0, 100.0 - min(60.0, penalties))

def keyword_scores(resume_text: str, jd_text: str) -> Tuple[float, float, List[str], List[str]]:
    rtoks = tokenize(resume_text)
    jtoks = tokenize(jd_text)

    # expand JD keywords with known skills so we bias towards skills
    jd_keys = list(dict.fromkeys(top_keywords(jtoks + [w for w in KNOWN_SKILLS if w in jd_text.lower()], k=50)))
    # unigram coverage
    covered = [k for k in jd_keys if k in rtoks or k in resume_text.lower()]
    missing = [k for k in jd_keys if k not in covered]

    # phrase coverage (multi‑word from KNOWN_SKILLS)
    phrases = [p for p in KNOWN_SKILLS if " " in p and p in jd_text.lower()]
    found_phrases = extract_phrases(resume_text, phrases)
    missing_phrases = [p for p in phrases if p not in found_phrases]

    uni_cov = (len(covered) / max(len(jd_keys), 1)) * 100.0
    phr_cov = (len(found_phrases) / max(len(phrases), 1)) * 100.0 if phrases else (100.0 if found_phrases else 0.0)
    return uni_cov, phr_cov, missing, missing_phrases

def compute_score(resume_text: str, jd_text: str) -> Tuple[ScoreBreakdown, Dict]:
    resume_text = normalize_text(resume_text)
    jd_text = normalize_text(jd_text)

    uni_cov, phr_cov, missing_unigrams, missing_phrases = keyword_scores(resume_text, jd_text)
    sections = section_presence(resume_text)
    contact = contact_presence(resume_text)
    bullets = bullet_quality(resume_text)
    length = length_readability(resume_text)
    formatting = formatting_health(resume_text)

    breakdown = ScoreBreakdown(
        keyword_coverage=uni_cov,
        phrase_coverage=phr_cov,
        sections=sections,
        contact=contact,
        bullets=bullets,
        length=length,
        formatting=formatting
    )
    detail = dict(
        missing_keywords=missing_unigrams,
        missing_phrases=missing_phrases
    )
    return breakdown, detail

# ----------- AI suggestions -----------
def ai_suggestions(openai_key: str, resume_text: str, jd_text: str) -> str:
    if not openai_key:
        return ""
    if Client is None:
        st.warning("OpenAI SDK not available. Install `openai>=1.40.0`.")
        return ""

    client = Client(api_key=openai_key)
    prompt = f"""You are an expert ATS and resume coach.
Given the JOB DESCRIPTION and the RESUME, analyze fit and provide:
1) The top 10 missing or weak skills/keywords to add (prioritize hard skills).
2) 5–8 bullet rewrites that are metrics‑driven, using strong action verbs from the JD context.
3) A quick checklist to improve ATS compatibility (sections, formatting, wording).

JOB DESCRIPTION:
{jd_text[:8000]}

RESUME:
{resume_text[:8000]}

Output using concise bullet points. Keep it actionable and tailored to the JD.
"""

    try:
        resp = client.responses.create(
            model=AI_MODEL,
            input=[
                {"role": "system", "content": "You are a precise, practical resume and ATS optimization expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        # Compatible with official SDKs that expose output_text
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text
        # Fallback: assemble text from outputs
        try:
            parts = []
            for item in getattr(resp, "output", []):
                if hasattr(item, "content"):
                    for c in item.content:
                        if c.type == "output_text":
                            parts.append(c.text)
            return "\\n".join(parts)
        except Exception:
            return "AI suggestions generated, but could not parse response text. Please update the OpenAI SDK."
    except Exception as e:
        return f"(AI error) {e}"

# ----------- UI -----------
def ui():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
    st.title(APP_TITLE)
    st.caption("Score your resume against a job description using ATS‑style heuristics and optional AI coaching.")

    with st.sidebar:
        st.subheader("Job Description")
        jd_upload = st.file_uploader("Upload JD (TXT optional)", type=["txt"], key="jd_upload")
        jd_text_area = st.text_area("Or paste JD text", height=180, key="jd_text")
        st.markdown("---")
        st.subheader("Resume")
        res_upload = st.file_uploader("Upload resume (PDF, DOCX, TXT)", type=["pdf","docx","txt"], key="res_upload")
        res_text_area = st.text_area("Or paste resume text", height=180, key="res_text")

        st.markdown("---")
        st.subheader("AI (optional)")
        api_key = st.text_input("OpenAI API Key", value=DEFAULT_OPENAI_KEY, type="password")
        use_ai = st.checkbox("Use AI suggestions", value=bool(api_key))

        st.markdown("---")
        st.write("Try samples:")
        if st.button("Load sample data"):
            try:
                base = os.path.join(os.path.dirname(__file__), "sample_data")
                with open(os.path.join(base, "sample_job_description.txt"), "r", encoding="utf-8") as f:
                    st.session_state["jd_text"] = f.read()
                with open(os.path.join(base, "sample_resume.txt"), "r", encoding="utf-8") as f:
                    st.session_state["res_text"] = f.read()
                st.success("Loaded sample JD and resume.")
            except Exception:
                st.warning("Sample files not found.")

    # Compose inputs
    jd_text = ""
    if jd_upload:
        jd_text = read_text_upload(jd_upload)
    if not jd_text:
        jd_text = jd_text_area

    resume_text = ""
    if res_upload:
        resume_text = read_text_upload(res_upload)
    if not resume_text:
        resume_text = res_text_area

    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Job Description (preview)")
        st.text_area(" ", value=jd_text[:4000], height=220, label_visibility="collapsed")
    with col2:
        st.subheader("Resume (preview)")
        st.text_area("  ", value=resume_text[:4000], height=220, label_visibility="collapsed")

    st.markdown("### Analyze")
    run = st.button("⚡ Score my resume", type="primary", use_container_width=True)

    if run:
        if not jd_text.strip() or not resume_text.strip():
            st.error("Please provide both Job Description and Resume (upload or paste).")
            st.stop()

        with st.spinner("Scoring..."):
            breakdown, detail = compute_score(resume_text, jd_text)
            total = round(breakdown.total(), 1)
            time.sleep(0.2)

        st.success(f"ATS Score: **{total}/100**")
        st.progress(min(100, int(total)))

        # Breakdown
        bcols = st.columns(3)
        bcols[0].metric("Keyword coverage (unigrams)", f"{breakdown.keyword_coverage:.1f}")
        bcols[1].metric("Keyword coverage (phrases)", f"{breakdown.phrase_coverage:.1f}")
        bcols[2].metric("Sections", f"{breakdown.sections:.1f}")
        bcols = st.columns(4)
        bcols[0].metric("Contact info", f"{breakdown.contact:.1f}")
        bcols[1].metric("Bullet quality", f"{breakdown.bullets:.1f}")
        bcols[2].metric("Length/readability", f"{breakdown.length:.1f}")
        bcols[3].metric("Formatting health", f"{breakdown.formatting:.1f}")

        st.markdown("---")
        # Missing keywords
        st.subheader("Keyword analysis")
        missing_all = detail["missing_keywords"][:20]
        missing_phr = detail["missing_phrases"][:10]
        if missing_all:
            st.write("**Top missing keywords** (add where relevant):")
            st.code(", ".join(missing_all))
        else:
            st.write("Great! No critical unigrams missing.")

        if missing_phr:
            st.write("**Missing key phrases**:")
            st.code(", ".join(missing_phr))

        # Covered keywords (quick view)
        rtoks = set(tokenize(resume_text))
        jtoks = set(tokenize(jd_text))
        covered = [k for k in jtoks if k in rtoks]
        if covered:
            st.write("**Already covered keywords** (good to keep):")
            st.code(", ".join(sorted(list(covered))[:40]))

        # AI suggestions
        if use_ai and (DEFAULT_OPENAI_KEY or st.session_state.get("OPENAI_API_KEY") or True):
            with st.spinner("Generating AI suggestions..."):
                tips = ai_suggestions(api_key, resume_text, jd_text)
            if tips:
                st.subheader("AI suggestions")
                st.markdown(tips)
            else:
                st.info("No AI suggestions generated (missing API key or SDK).")

        # Downloadable report
        st.markdown("---")
        report = f"""ATS Score Report

Total Score: {total}/100

Breakdown:
- Keyword coverage (unigrams): {breakdown.keyword_coverage:.1f}
- Keyword coverage (phrases): {breakdown.phrase_coverage:.1f}
- Sections: {breakdown.sections:.1f}
- Contact info: {breakdown.contact:.1f}
- Bullet quality: {breakdown.bullets:.1f}
- Length/readability: {breakdown.length:.1f}
- Formatting health: {breakdown.formatting:.1f}

Top missing keywords: {", ".join(missing_all) if missing_all else "None"}
Missing phrases: {", ".join(missing_phr) if missing_phr else "None"}

Notes:
- Use strong action verbs and quantify impact.
- Mirror critical skills/keywords from the JD (honestly and accurately).
- Keep layout simple (no tables/columns/images), consistent fonts, and clear sections.
"""
        st.download_button("Download report (.txt)", data=report.encode("utf-8"),
                           file_name="ats_score_report.txt", mime="text/plain")

        st.caption("Tip: iterate your resume, then re‑run the score to see improvements.")

if __name__ == "__main__":
    ui()