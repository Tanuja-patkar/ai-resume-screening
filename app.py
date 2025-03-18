import streamlit as st
import spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Preprocessing function
def preprocess_text(text):
    doc = nlp(text.lower())  # Convert to lowercase and tokenize
    words = [token.lemma_ for token in doc if token.is_alpha and token.text not in string.punctuation]
    return " ".join(words)

# Function to compute similarity
def compute_similarity(resume_text, job_description):
    corpus = [preprocess_text(resume_text), preprocess_text(job_description)]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_score = cosine_similarity(tfidf_matrix)[0][1]
    return similarity_score

# Streamlit UI
st.title("📝 AI Resume Screening System")

st.header("📌 Enter Job Description")
job_description = st.text_area("Paste Job Description here")

st.header("📤 Upload Resumes")
uploaded_files = st.file_uploader("Upload resume files (TXT format)", accept_multiple_files=True, type=["txt"])

if st.button("🔍 Analyze Resumes"):
    if uploaded_files and job_description:
        results = []
        for file in uploaded_files:
            resume_text = file.read().decode("utf-8")
            score = compute_similarity(resume_text, job_description)
            results.append((file.name, score))

        # Sort by highest match
        results.sort(key=lambda x: x[1], reverse=True)

        st.subheader("✅ Top Matching Resumes")
        for name, score in results:
            st.write(f"**📄 {name}:** {score:.2f} similarity")

    else:
        st.warning("⚠️ Please upload resumes and provide a job description.")
