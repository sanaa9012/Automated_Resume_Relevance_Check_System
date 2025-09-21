# Automated Resume Relevance Check System

A **Streamlit-based web application** to evaluate resumes against job descriptions (JDs) using **keyword matching** and **semantic similarity embeddings**.
The system also provides **AI suggestions** (placeholder for Gemini AI) to improve resumes for better alignment with JDs.

---

## Features

* Upload multiple **Resumes** and **Job Descriptions** in PDF, DOCX, or TXT format.
* **Evaluate all resumes** against all JDs:

  * **Keyword match score**
  * **Semantic similarity score** (using `sentence-transformers`)
  * **Final score** & **verdict** (High / Medium / Low)
* **View summary charts** of evaluation results.
* **Download CSV reports** per JD.
* **AI suggestions** for top resumes (Gemini AI placeholder).
* **Embedded caching** of text embeddings for faster processing.
* **Delete uploaded files** and associated evaluations.

---

## Requirements

* Python 3.9+
* [Streamlit](https://streamlit.io/)
* [Sentence Transformers](https://www.sbert.net/)
* PyPDF2
* docx2txt
* pandas
* numpy
* python-dotenv
* google-generativeai (placeholder for AI suggestions)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

1. Clone this repository:

```bash
git clone <repo_url>
cd <repo_folder>
```

2. Create a `.env` file (optional) to store API keys for Gemini AI:

```
GENAI_API_KEY=<your_key_here>
```

3. Run the application:

```bash
streamlit run app.py
```

---

## Usage

1. Open the app in your browser (Streamlit will provide a local URL).
2. Upload **Job Descriptions** and **Resumes** via the sidebar.
3. Click **Evaluate All** to compute scores.
4. View **evaluation summary**, **charts**, and **top candidates**.
5. Expand AI suggestions for resume improvement.
6. Download CSV reports per JD.

---

## File Structure

```
.
├── app.py               # Main Streamlit application
├── evaluations.db       # SQLite database for evaluations & embeddings
├── requirements.txt     # Python dependencies
├── README.md
├── .env                 # Environment variables (optional)
```

---

## How it Works

1. **Text Extraction**: Supports PDF, DOCX, TXT.
2. **Keyword Matching**: Computes percentage of JD keywords present in resume.
3. **Semantic Similarity**: Uses `sentence-transformers` embeddings and cosine similarity.
4. **Final Score**: Weighted average of keyword score & semantic similarity.
5. **Verdict**:

   * `High` (>=70)
   * `Medium` (40–69)
   * `Low` (<40)
6. **AI Suggestions**: Placeholder for Gemini AI suggestions.

---

## Notes

* Embeddings are cached in the SQLite database to improve performance.
* Supports multiple uploads and incremental evaluations.
* AI suggestions currently use a placeholder. Integrate Gemini AI for real-time suggestions.

---
