from __future__ import annotations

import html
import io
import re
from pathlib import Path
from typing import Any

import joblib
import kagglehub
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "fake_review_nlp_model.pkl"
DATASET_SLUG = "mexwell/fake-reviews-dataset"
REQUIRED_COLUMNS = {"label", "text_"}

st.set_page_config(
    page_title="ReviewLens | Fake Review Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #17253d;
        --muted: #718096;
        --line: #e8edf4;
        --blue: #425df5;
        --teal: #13a98b;
        --surface: #ffffff;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f6f8fc; color: var(--ink); }
    [data-testid="stSidebar"] {
        background: #101b32;
        border-right: 1px solid #26324a;
    }
    [data-testid="stSidebar"] * { color: #e9eefb; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { color: #dce4f5; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: #192741;
        border: 1px dashed #52617d;
        border-radius: 12px;
    }
    .brand { padding: 12px 4px 24px; }
    .brand-mark {
        display: inline-flex; width: 38px; height: 38px; align-items: center;
        justify-content: center; border-radius: 12px; background: #425df5;
        font-size: 19px; margin-right: 10px; vertical-align: middle;
    }
    .brand-name { font: 800 18px 'Manrope', sans-serif; letter-spacing: -.4px; }
    .brand-caption {
        color: #92a1bd !important; font-size: 11px; letter-spacing: 1.6px;
        text-transform: uppercase; margin: 9px 0 0 49px;
    }
    .eyebrow {
        color: #5269e9; font-size: 11px; font-weight: 700;
        letter-spacing: 1.5px; text-transform: uppercase;
    }
    .page-title {
        color: #17253d; font: 800 34px 'Manrope', sans-serif;
        letter-spacing: -1.2px; margin: 5px 0 2px;
    }
    .page-subtitle { color: #718096; font-size: 15px; margin-bottom: 24px; }
    .hero {
        background: linear-gradient(120deg, #172643 0%, #263c69 62%, #4058c6 100%);
        border-radius: 22px; padding: 28px 32px; color: white; margin: 8px 0 22px;
        box-shadow: 0 14px 35px rgba(35, 54, 101, .15);
    }
    .hero h2 { color: white; font: 800 25px 'Manrope', sans-serif; margin: 5px 0 8px; }
    .hero p { color: #d4def3; margin: 0; font-size: 14px; }
    .hero-tag {
        display: inline-block; color: #d6ddff; border: 1px solid #7385d2;
        border-radius: 99px; padding: 5px 11px; font-size: 11px; letter-spacing: .5px;
    }
    .metric-card {
        background: white; border: 1px solid var(--line); border-radius: 16px;
        padding: 18px 20px; min-height: 108px;
    }
    .metric-label { color: #718096; font-size: 12px; font-weight: 600; }
    .metric-value { color: #17253d; font: 800 25px 'Manrope', sans-serif; margin-top: 7px; }
    .section-title { color: #1b2942; font: 700 18px 'Manrope', sans-serif; margin: 16px 0 10px; }
    .result-card {
        border: 1px solid var(--line); background: #fff; border-radius: 18px;
        padding: 23px; margin: 12px 0;
    }
    .result-label { font: 800 24px 'Manrope', sans-serif; margin: 6px 0; }
    .result-note { color: #718096; font-size: 13px; }
    .pill {
        display: inline-block; border-radius: 99px; padding: 5px 11px;
        font-size: 11px; font-weight: 700; letter-spacing: .5px;
    }
    .pill-fake { color: #ac4a14; background: #fff1e7; }
    .pill-real { color: #08775e; background: #e2f7f1; }
    .pill-neutral { color: #536178; background: #edf1f7; }
    .small-note { color: #8793a6; font-size: 12px; }
    div[data-testid="stTextArea"] textarea {
        border-radius: 12px; border-color: #dfe5ef; background: #fff;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        border-radius: 10px; font-weight: 700; border: 0;
        background: linear-gradient(90deg, #425df5, #566df3);
    }
    div[data-testid="stButton"] button { border-radius: 10px; }
    div[data-testid="stExpander"] { border-color: var(--line); border-radius: 12px; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model() -> Pipeline:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Place fake_review_nlp_model.pkl next to app.py."
        )
    model = joblib.load(MODEL_PATH)
    if not isinstance(model, Pipeline) or not {"tfidf", "classifier"}.issubset(model.named_steps):
        raise ValueError("The saved model must be a TF-IDF and classifier scikit-learn pipeline.")
    if not hasattr(model, "decision_function"):
        raise ValueError("The saved classifier must provide a decision function.")
    return model


@st.cache_resource
def load_text_processing_resources() -> tuple[set[str], WordNetLemmatizer]:
    for resource, package in (("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet")):
        try:
            nltk.data.find(resource)
        except LookupError:
            if not nltk.download(package, quiet=True):
                raise RuntimeError(f"Could not download the NLTK {package} data required for predictions.")

    return set(stopwords.words("english")), WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    stop_words, lemmatizer = load_text_processing_resources()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]
    return " ".join(words)


@st.cache_data(show_spinner=False)
def download_dataset() -> str:
    return str(kagglehub.dataset_download(DATASET_SLUG))


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lstrip("\ufeff").lower() for column in normalized.columns]
    missing = REQUIRED_COLUMNS - set(normalized.columns)
    if missing:
        raise ValueError(
            "Dataset must include the columns label and text_. "
            f"Missing: {', '.join(sorted(missing))}. Found: {', '.join(map(str, normalized.columns))}."
        )
    return normalized


@st.cache_data(show_spinner=False)
def load_dataset_file(file_path: str) -> pd.DataFrame:
    return normalize_columns(pd.read_csv(file_path))


@st.cache_data(show_spinner=False)
def load_dataset_upload(contents: bytes) -> pd.DataFrame:
    return normalize_columns(pd.read_csv(io.BytesIO(contents)))


def find_dataset_csv(dataset_dir: str) -> Path:
    csv_files = sorted(Path(dataset_dir).rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found in the Kaggle dataset directory: {dataset_dir}")

    unreadable: list[str] = []
    for csv_path in csv_files:
        try:
            headers = pd.read_csv(csv_path, nrows=0).columns
        except (OSError, UnicodeError, pd.errors.ParserError) as error:
            unreadable.append(f"{csv_path.name}: {error}")
            continue
        columns = {str(column).strip().lstrip("\ufeff").lower() for column in headers}
        if REQUIRED_COLUMNS.issubset(columns):
            return csv_path

    details = f" CSV read errors: {'; '.join(unreadable)}" if unreadable else ""
    raise ValueError(
        f"Could not find a CSV containing both label and text_ under {dataset_dir}.{details}"
    )


def label_details(label: object) -> tuple[str, str]:
    value = str(label).strip()
    if value.upper() == "CG":
        return "Likely computer-generated", "fake"
    if value.upper() == "OR":
        return "Likely original", "real"
    return value, "neutral"


def predict_review(model: Pipeline, review: str) -> tuple[str, float]:
    cleaned_review = preprocess_text(review)
    label = str(model.predict([cleaned_review])[0])
    decision_score = float(model.decision_function([cleaned_review])[0])
    return label, decision_score


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div><div class="page-title">{title}</div>'
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="small-note">{html.escape(detail)}</div></div>',
        unsafe_allow_html=True,
    )


def render_prediction(model: Pipeline, review: str) -> None:
    label, decision_score = predict_review(model, review.strip())
    description, style = label_details(label)
    st.markdown(
        f'<div class="result-card"><span class="pill pill-{style}">{html.escape(description.upper())}</span>'
        f'<div class="result-label">{html.escape(description)}</div>'
        f'<div class="result-note">Dataset label: {html.escape(label)} · '
        f'Raw decision score: {decision_score:.3f}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The decision score is an uncalibrated LinearSVC margin, not a probability. "
        "Predictions are estimates, not proof that a review is genuine or fake."
    )


def render_classifier(model: Pipeline) -> None:
    page_header("REVIEW ANALYSIS", "Is this review real?", "Check a review with the trained text-classification model.")
    st.markdown(
        '<div class="hero"><span class="hero-tag">TF-IDF · LINEAR SVC</span>'
        '<h2>Spot patterns, not just star ratings.</h2>'
        '<p>Paste a product review to estimate whether it resembles computer-generated '
        'or original reviews in the training dataset.</p></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.45, 0.8], gap="large")
    with left:
        review = st.text_area(
            "Review text",
            height=210,
            placeholder="Paste a product review here…",
            help="The model analyzes text only; it does not use product or reviewer metadata.",
            key="review_input",
        )
        analyze = st.button("Analyze review", type="primary", use_container_width=True)
    with right:
        st.markdown('<div class="section-title">How to use</div>', unsafe_allow_html=True)
        st.markdown(
            "1. Paste a complete review.\n"
            "2. Select **Analyze review**.\n"
            "3. Review the predicted class and raw decision score.\n\n"
            "**CG** means computer-generated; **OR** means original in the dataset."
        )
        st.info("Longer, informative review text generally gives the classifier more to work with.")
    if analyze:
        if not review.strip():
            st.warning("Enter a review before running the analysis.")
        else:
            try:
                render_prediction(model, review)
            except (LookupError, OSError, RuntimeError, ValueError, TypeError, AttributeError) as error:
                st.error(f"Could not analyze this review: {error}")


def render_dataset(source_path: str | None, uploaded_file: Any) -> None:
    page_header("DATA EXPLORER", "Explore the reviews", "Understand the label distribution and inspect review text.")

    try:
        if uploaded_file is not None:
            frame = load_dataset_upload(uploaded_file.getvalue())
            source_name = uploaded_file.name
        elif source_path:
            csv_path = find_dataset_csv(source_path)
            frame = load_dataset_file(str(csv_path))
            source_name = csv_path.name
        else:
            st.markdown(
                '<div class="hero"><span class="hero-tag">40K+ REVIEW ROWS</span>'
                '<h2>Bring the dataset into view.</h2>'
                '<p>Use the sidebar to download the Kaggle dataset or upload your own CSV '
                'containing the label and text_ columns.</p></div>',
                unsafe_allow_html=True,
            )
            st.caption("Kaggle dataset: mexwell/fake-reviews-dataset")
            return
    except (OSError, ValueError, pd.errors.ParserError, UnicodeError) as error:
        st.error(f"Unable to load the dataset: {error}")
        return

    valid = frame.dropna(subset=["label", "text_"]).copy()
    valid["label"] = valid["label"].astype(str)
    valid["text_"] = valid["text_"].astype(str)
    valid = valid[valid["text_"].str.strip().ne("")]
    total_rows = len(frame)
    label_count = valid["label"].nunique()
    first, second, third = st.columns(3)
    with first:
        metric_card("TOTAL ROWS", f"{total_rows:,}", source_name)
    with second:
        metric_card("USABLE REVIEWS", f"{len(valid):,}", "Rows with label and review text")
    with third:
        metric_card("DISTINCT LABELS", str(label_count), "Categories in the label column")

    if valid.empty:
        st.warning("There are no rows with both a label and non-empty review text.")
        return

    st.markdown('<div class="section-title">Label distribution</div>', unsafe_allow_html=True)
    label_counts = valid["label"].value_counts().rename_axis("label").to_frame("reviews")
    chart_col, summary_col = st.columns([1.25, 0.75], gap="large")
    with chart_col:
        st.bar_chart(label_counts, color="#5269e9", height=290)
    with summary_col:
        st.dataframe(
            label_counts.reset_index().rename(columns={"label": "Label", "reviews": "Reviews"}),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Class meanings follow the dataset convention: CG = computer-generated, OR = original.")

    st.markdown('<div class="section-title">Find a review</div>', unsafe_allow_html=True)
    filter_col, search_col = st.columns([0.7, 1.3])
    with filter_col:
        labels = ["All labels", *sorted(valid["label"].unique().tolist())]
        selected_label = st.selectbox("Filter by label", labels)
    with search_col:
        search = st.text_input("Search review text", placeholder="Try a word or phrase…")

    shown = valid
    if selected_label != "All labels":
        shown = shown[shown["label"] == selected_label]
    if search.strip():
        shown = shown[shown["text_"].str.contains(search.strip(), case=False, na=False, regex=False)]
    st.caption(f"Showing {min(len(shown), 100):,} of {len(shown):,} matching reviews.")
    display = shown[["label", "text_"]].head(100).rename(
        columns={"label": "Label", "text_": "Review"}
    )
    st.dataframe(display, hide_index=True, use_container_width=True, height=390)


def main() -> None:
    try:
        model = load_model()
    except (OSError, ValueError, TypeError) as error:
        st.error(f"Could not load the classifier: {error}")
        st.stop()

    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="brand-mark">⌕</span>'
            '<span class="brand-name">ReviewLens</span>'
            '<div class="brand-caption">FAKE REVIEW INTELLIGENCE</div></div>',
            unsafe_allow_html=True,
        )
        page = st.radio("WORKSPACE", ["Classify a review", "Explore dataset", "About"], label_visibility="visible")
        st.markdown("---")
        st.markdown("**DATA SOURCE**")
        st.caption("Download the public Kaggle dataset or bring in a CSV.")
        if st.button("Download Kaggle dataset", use_container_width=True):
            try:
                with st.spinner("Downloading dataset from Kaggle…"):
                    st.session_state["dataset_path"] = download_dataset()
                st.success("Dataset downloaded. Open Explore dataset to view it.")
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f"Kaggle download failed: {error}")
        uploaded_file = st.file_uploader(
            "Or upload a CSV",
            type=["csv"],
            help="CSV must have columns named label and text_.",
            key="dataset_upload",
        )
        if uploaded_file is not None:
            st.caption(f"Using {uploaded_file.name}")
        st.markdown("---")
        st.caption("Local model · Text-only predictions")

    if page == "Classify a review":
        render_classifier(model)
    elif page == "Explore dataset":
        render_dataset(st.session_state.get("dataset_path"), uploaded_file)
    else:
        page_header("ABOUT REVIEWLENS", "A small model, made explorable.", "An interactive front end for your review classifier.")
        st.markdown(
            '<div class="hero"><span class="hero-tag">MODEL CARD</span>'
            '<h2>Text classification for review research.</h2>'
            '<p>The saved pipeline combines TF-IDF features with a LinearSVC classifier. '
            'It was trained to distinguish the dataset labels CG and OR.</p></div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-title">What it does</div>', unsafe_allow_html=True)
            st.write(
                "• Predicts one of the labels learned from the training dataset.\n"
                "• Shows an uncalibrated decision score, not a probability.\n"
                "• Explores the label and text_ fields in a CSV."
            )
        with right:
            st.markdown('<div class="section-title">Important limitations</div>', unsafe_allow_html=True)
            st.write(
                "This is a statistical estimate, not a fact check. Writing style alone cannot establish "
                "who authored a review. Results may be inaccurate, especially for short, translated, "
                "or out-of-domain text."
            )
        st.caption(f"Model file: {MODEL_PATH.name} · Dataset source: {DATASET_SLUG}")


if __name__ == "__main__":
    main()