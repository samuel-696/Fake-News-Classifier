import streamlit as st
import joblib
import re
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b1220; color: #e6e6e6; }
    .verdict-box {
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .fake { background-color: #3a1414; border: 1px solid #b33; color: #ff8080; }
    .real { background-color: #133a1a; border: 1px solid #2d7a3a; color: #7fdb8f; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    model = joblib.load("Models/fakenews_model.pkl")
    vectorizer = joblib.load("Models/fakenews_vectorizer.pkl")
    return make_pipeline(vectorizer, model)

pipeline = load_pipeline()
explainer = LimeTextExplainer(class_names=["Fake", "Real"])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\(reuters\)', '', text)
    text = re.sub(r'reuters', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("Fake News Detector")
st.caption(
    "Logistic Regression on TF-IDF features (unigrams + bigrams), trained on the "
    "Fake and Real News dataset. Paste an article's title and body to see the model's "
    "verdict and which words drove the decision (via LIME)."
)

article = st.text_area("Paste article text here", height=200)
run = st.button("Check article")

if run and article.strip():
    cleaned = clean_text(article)
    proba = pipeline.predict_proba([cleaned])[0]
    pred = proba.argmax()

    verdict = "REAL" if pred == 1 else "FAKE"
    css_class = "real" if pred == 1 else "fake"

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            f'<div class="verdict-box {css_class}">Verdict: {verdict}<br>'
            f'P(Fake) = {proba[0]:.4f} | P(Real) = {proba[1]:.4f}</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.write("Model confidence")
        st.progress(float(proba[pred]))

    st.subheader("Why the model thinks this")
    exp = explainer.explain_instance(cleaned, pipeline.predict_proba, num_features=10)
    lime_html = exp.as_html()
    wrapped_html = f'<div style="background-color:#ffffff;padding:12px;border-radius:8px;">{lime_html}</div>'
    st.iframe(wrapped_html, height=420)

elif run:
    st.warning("Paste some article text first.")

st.divider()
st.caption(
    "Note: the 'Real' class in training data is dominated by Reuters-style wire "
    "reporting. Articles written in other real-news styles (local news, opinion, etc.) "
    "may be misclassified — this is a known limitation of the underlying dataset, not "
    "a universal fake-news detector."
)