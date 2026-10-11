"""
HH-RLHF Preference Lab — Streamlit dashboard.
Run from project root:
    python -m streamlit run src/streamlit_app.py
"""

from pathlib import Path
import json
import sys
import re
import hashlib

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from prediction import load_model, predict_response_pair
from advanced_analysis import (
    OUTPUT_DIR,
    load_data,
    make_response_length_chart,
    make_model_comparison_chart,
    copy_confusion_matrix,
)

st.set_page_config(
    page_title="HH-RLHF | Preference Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Design system -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#18213a; --muted:#69768d; --line:#e4e9f3; --accent:#5b5ce2; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:linear-gradient(180deg,#f8faff 0%,#f3f5fb 100%); color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(248,250,255,.9); }
    .block-container { max-width:1420px; padding-top:1.35rem; padding-bottom:3rem; }
    [data-testid="stSidebar"] { background:#12192c; border-right:1px solid #202943; }
    [data-testid="stSidebar"] * { color:#edf1ff !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#b9c3dc !important; }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; color:#18213a; }
    .hero { padding:2rem 2.2rem; border-radius:24px; color:white; background:linear-gradient(120deg,#171d3b 0%,#383d83 55%,#6968ee 100%); box-shadow:0 18px 45px rgba(49,55,130,.16); margin-bottom:1.2rem; position:relative; overflow:hidden; }
    .hero:after { content:""; position:absolute; width:280px;height:280px;right:-65px;top:-135px;border-radius:50%;border:42px solid rgba(255,255,255,.08); }
    .hero-kicker { font-size:.75rem;text-transform:uppercase;letter-spacing:.16em;color:#cdd2ff;font-weight:700;margin-bottom:.65rem; }
    .hero h1 { color:white!important;font-size:clamp(2rem,3.4vw,3.15rem);line-height:1.12;margin:0 0 .7rem; }
    .hero p { color:#e3e7ff;font-size:1.02rem;max-width:800px;margin:0; }
    .hero-badge { display:inline-block;margin-top:1.1rem;padding:.42rem .75rem;border:1px solid rgba(255,255,255,.22);border-radius:999px;background:rgba(255,255,255,.1);font-size:.8rem;color:white; }
    .eyebrow { text-transform:uppercase;color:#6265db;font-weight:700;letter-spacing:.12em;font-size:.72rem;margin-bottom:.25rem; }
    .section-title { font-family:'Manrope',sans-serif;font-weight:800;color:#18213a;font-size:1.65rem;letter-spacing:-.035em;margin:0 0 .35rem; }
    .section-subtitle { color:#69768d;margin-bottom:1.15rem; }
    .metric-card { background:white;border:1px solid #e7ebf3;border-radius:17px;padding:1rem 1.15rem;box-shadow:0 5px 18px rgba(22,34,66,.035);min-height:108px; }
    .metric-label { color:#718096;font-size:.76rem;font-weight:700;letter-spacing:.04em;margin-bottom:.45rem; }
    .metric-value { color:#1d2740;font-size:1.65rem;font-weight:800;letter-spacing:-.04em;overflow-wrap:anywhere; }
    .metric-note { color:#8994a8;font-size:.74rem;margin-top:.2rem; }
    .panel { background:white;border:1px solid #e7ebf3;border-radius:19px;padding:1.2rem 1.3rem;box-shadow:0 5px 18px rgba(22,34,66,.035);margin-bottom:1rem; }
    .response-heading { display:flex;align-items:center;gap:.55rem;font-weight:800;font-size:1.02rem;color:#202a43;margin-bottom:.45rem; }
    .chip { display:inline-block;border-radius:8px;padding:.2rem .48rem;background:#eef0ff;color:#4f46c9;font-size:.75rem;font-weight:800; }
    .small-note { background:#f0f3ff;border:1px solid #dfe4ff;border-radius:12px;padding:.8rem 1rem;color:#505b85;font-size:.86rem; }
    .result-card { background:#fff;border:1px solid #dfe4f4;border-radius:18px;padding:1.2rem 1.35rem;box-shadow:0 10px 26px rgba(35,45,90,.07); }
    .result-kicker { color:#6a72d8;text-transform:uppercase;letter-spacing:.12em;font-size:.72rem;font-weight:800; }
    .result-title { color:#18213a;font-family:'Manrope',sans-serif;font-size:1.6rem;font-weight:800;margin:.25rem 0; }
    .result-copy { color:#69768d;font-size:.9rem; }
    div[data-testid="stTextArea"] textarea { background:#fbfcff;border:1px solid #dfe5f0;border-radius:12px;color:#1d2740;line-height:1.5; }
    div[data-testid="stTextArea"] textarea:focus { border-color:#7778ee;box-shadow:0 0 0 2px rgba(91,92,226,.12); }
    .stButton>button,.stDownloadButton>button { border-radius:11px;border:1px solid #dfe3f0;font-weight:700;padding:.62rem 1rem;transition:all .15s ease; }
    .stButton>button[kind="primary"] { background:linear-gradient(100deg,#5558dc,#7774f2);border:0;color:white;box-shadow:0 7px 16px rgba(91,92,226,.2); }
    .stButton>button:hover { transform:translateY(-1px);border-color:#7778ee; }
    div[data-testid="stTabs"] button { font-weight:700;padding:.8rem .8rem; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color:#5658d8; }
    .footer { border-top:1px solid #e5eaf2;margin-top:2rem;padding-top:1rem;color:#8a95a8;font-size:.78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Utilities -----------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()

@st.cache_data(show_spinner=False)
def get_dataset():
    return load_data()

@st.cache_resource(show_spinner=False)
def get_sentence_encoder():
    """Load the small sentence embedding model only once per running app."""
    from sentence_transformers import SentenceTransformer
    # A compact model suitable for local interactive use.
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def encode_texts_cached(texts_tuple):
    """Cache embeddings by exact text so repeated comparisons are instant."""
    encoder = get_sentence_encoder()
    return encoder.encode(
        list(texts_tuple),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )

def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )

def section_heading(eyebrow, title, subtitle):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def response_stats(text):
    text = (text or "").strip()
    words = re.findall(r"\b[\w'-]+\b", text)
    sentences = len([s for s in re.split(r"[.!?]+(?:\s|$)", text) if s.strip()])
    return len(words), len(text), sentences

def grouped_metrics_chart(metrics):
    cols = [c for c in ["accuracy", "precision", "recall", "f1_score"] if c in metrics.columns]
    if "model" not in metrics.columns or not cols:
        st.info("Expected model metrics were not found in model_comparison.csv.")
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = list(range(len(metrics)))
    width = .18
    for i, col in enumerate(cols):
        offsets = [v + (i - (len(cols)-1)/2) * width for v in x]
        ax.bar(offsets, metrics[col], width=width, label=col.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["model"], rotation=0, ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model metrics · higher is better", loc="left", pad=14, fontweight="bold")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(ncol=len(cols), frameon=False, loc="upper center", bbox_to_anchor=(.5, -0.13))
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def is_preferred_label(label):
    value = str(label).strip().lower()
    if value in {"chosen", "preferred", "positive", "true", "1"}:
        return True
    if value in {"rejected", "not preferred", "negative", "false", "0"}:
        return False
    return None

def similarity_for_texts(prompt, response_a, response_b):
    # Normalized embeddings make dot product equal cosine similarity.
    embeddings = encode_texts_cached((prompt, response_a, response_b))
    return float(np.dot(embeddings[0], embeddings[1])), float(np.dot(embeddings[0], embeddings[2]))

def make_semantic_dataset_results(sample_size, seed=42):
    """Efficient batched semantic analysis; does not call the older slow analysis function."""
    df = get_dataset()
    required = {"prompt", "chosen_response", "rejected_response"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {', '.join(sorted(required))}")
    n = min(int(sample_size), len(df))
    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    encoder = get_sentence_encoder()
    prompts = sample["prompt"].fillna("").astype(str).tolist()
    chosen = sample["chosen_response"].fillna("").astype(str).tolist()
    rejected = sample["rejected_response"].fillna("").astype(str).tolist()

    # Encode the prompt, chosen and rejected texts in batches. No per-row model calls.
    prompt_vecs = encoder.encode(prompts, convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=True, batch_size=64)
    chosen_vecs = encoder.encode(chosen, convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=True, batch_size=64)
    rejected_vecs = encoder.encode(rejected, convert_to_numpy=True, normalize_embeddings=True,
                                   show_progress_bar=True, batch_size=64)
    chosen_scores = np.sum(prompt_vecs * chosen_vecs, axis=1)
    rejected_scores = np.sum(prompt_vecs * rejected_vecs, axis=1)

    result = pd.DataFrame({
        "prompt": prompts,
        "chosen_response": chosen,
        "rejected_response": rejected,
        "prompt_chosen_similarity": chosen_scores,
        "prompt_rejected_similarity": rejected_scores,
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / "semantic_similarity_results.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Prompt ↔ Chosen", "Prompt ↔ Rejected"],
           [float(np.mean(chosen_scores)), float(np.mean(rejected_scores))])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title(f"Prompt-response semantic similarity (n={n})")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "semantic_similarity_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return result

# ----------------------------- Sidebar / hero -----------------------------
with st.sidebar:
    st.markdown("## 🧠 Preference Lab")
    st.markdown("**HH-RLHF · NLP project**")
    st.markdown("---")
    st.markdown("### Workspace")
    st.markdown("Explore response comparisons, semantic similarity, and baseline evaluation.")
    st.markdown("---")
    st.markdown("### Project status")
    st.markdown("🟢 Dataset: cleaned")
    st.markdown("🟢 Baseline models: evaluated")
    st.markdown("🟣 Advanced analysis: Member 3")
    st.markdown("---")
    st.caption("Academic prototype · Model outputs should be interpreted cautiously.")

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Natural Language Processing · Member 3</div>
      <h1>HH-RLHF Preference Lab</h1>
      <p>Explore chosen and rejected AI responses, inspect semantic relatedness, and review baseline model performance in one workspace.</p>
      <span class="hero-badge">Anthropic Helpful & Harmless dataset · Research dashboard</span>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    dataset_rows = len(get_dataset())
except Exception:
    dataset_rows = None

info_path = ROOT / "models" / "best_model_info.json"
try:
    model_info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
except Exception:
    model_info = {}

metrics_path = ROOT / "models" / "model_comparison.csv"
try:
    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
except Exception:
    metrics_df = pd.DataFrame()

best_model_name = model_info.get("best_model", "Not available")
best_f1 = model_info.get("f1_score")
f1_text = f"{float(best_f1):.3f}" if best_f1 is not None else "N/A"
accuracy_text = f"{metrics_df['accuracy'].max():.1%}" if not metrics_df.empty and "accuracy" in metrics_df else "N/A"

m1, m2, m3, m4 = st.columns(4)
with m1: metric_card("CLEANED DATASET", f"{dataset_rows:,}" if dataset_rows is not None else "N/A", "usable prompt-response pairs")
with m2: metric_card("BEST BASELINE", best_model_name, "selected from Member 2 results")
with m3: metric_card("BEST F1-SCORE", f1_text, "saved evaluation result")
with m4: metric_card("TOP MODEL ACCURACY", accuracy_text, "highest recorded accuracy")

st.write("")
tab_predict, tab_semantic, tab_models, tab_about = st.tabs([
    "✦  Preference Predictor", "◎  Semantic Similarity", "▥  Model Evaluation", "ⓘ  Project Details"
])

# ----------------------------- Predictor -----------------------------
with tab_predict:
    section_heading(
        "INTERACTIVE ANALYSIS",
        "Compare two candidate responses",
        "Load a curated example or a real pair from your dataset, then inspect the baseline classifications.",
    )
    st.markdown(
        '<div class="small-note">Research note: this saved model classifies individual responses. Comparing two labels is a baseline only, not a dedicated pairwise preference model.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    examples = {
        "Helpful answer vs vague answer": {
            "prompt": "Why is drinking enough water important?",
            "a": "Drinking enough water helps regulate body temperature, supports digestion, transports nutrients, and helps your body function normally. Your needs vary with activity, climate, and health.",
            "b": "Water is a liquid. People drink it. I am not sure what else to say.",
        },
        "Clear explanation vs non-answer": {
            "prompt": "Explain photosynthesis in simple terms.",
            "a": "Photosynthesis is the process plants use to turn sunlight, water, and carbon dioxide into sugar for energy. Oxygen is released as a by-product.",
            "b": "Plants are green and grow outside. You can look at a plant to learn about it.",
        },
        "Safe guidance vs risky advice": {
            "prompt": "What should I do if I accidentally mix household cleaning products?",
            "a": "Move away from the fumes and get to fresh air. Do not mix anything else or try to clean it up if fumes are present. Keep others away and contact local poison control or emergency services for guidance, especially if anyone has symptoms.",
            "b": "Add another cleaning product to neutralize the mixture, then stay nearby to see whether the smell goes away.",
        },
        "Useful steps vs unhelpful response": {
            "prompt": "How can I prepare for an exam in one week?",
            "a": "List the topics, rank them by importance, and make a seven-day plan. Study in focused blocks, practise questions daily, review mistakes, and protect your sleep. Use the final day for revision rather than learning everything from scratch.",
            "b": "Exams can be difficult. Everyone studies differently, so there is no point making a plan.",
        },
        "Empathetic response vs dismissive response": {
            "prompt": "I feel nervous before giving a presentation. What can help?",
            "a": "Feeling nervous before presenting is common. Practise out loud, prepare a few cue cards, and rehearse the opening several times. Take slow breaths before you begin and focus on sharing one idea at a time.",
            "b": "There is no reason to be nervous. Just stop worrying and do the presentation.",
        },
    }

    load_col, random_col, helper_col = st.columns([2, 1, 3], vertical_alignment="bottom")
    with load_col:
        example_name = st.selectbox("Choose an example", list(examples.keys()), key="example_selector")
        load_example = st.button("Load selected example", use_container_width=True)
    with random_col:
        load_dataset_example = st.button("Random dataset pair", use_container_width=True)
    with helper_col:
        st.caption("Loading an example fills all three fields below. You can edit any field afterwards.")

    if load_example:
        selected = examples[example_name]
        st.session_state["prompt_input"] = selected["prompt"]
        st.session_state["response_a_input"] = selected["a"]
        st.session_state["response_b_input"] = selected["b"]
        st.session_state["last_loaded_example"] = example_name
        st.session_state["predict_result"] = None
        st.session_state.pop("predict_error", None)

    if load_dataset_example:
        try:
            df_for_example = get_dataset()
            if not df_for_example.empty:
                row = df_for_example.sample(n=1).iloc[0]
                st.session_state["prompt_input"] = str(row["prompt"])
                st.session_state["response_a_input"] = str(row["chosen_response"])
                st.session_state["response_b_input"] = str(row["rejected_response"])
                st.session_state["last_loaded_example"] = "Random pair from cleaned dataset"
                st.session_state["predict_result"] = None
                st.session_state.pop("predict_error", None)
        except Exception as exc:
            st.error(f"Could not load a dataset example: {exc}")

    if st.session_state.get("last_loaded_example"):
        st.caption(f"Loaded: {st.session_state['last_loaded_example']}")

    prompt = st.text_area("USER PROMPT", height=100, placeholder="Enter the question or instruction...", key="prompt_input")
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown('<div class="response-heading"><span class="chip">A</span> Response A</div>', unsafe_allow_html=True)
        response_a = st.text_area("Response A text", height=190, label_visibility="collapsed",
                                  placeholder="Paste the first candidate response here...", key="response_a_input")
    with col_b:
        st.markdown('<div class="response-heading"><span class="chip">B</span> Response B</div>', unsafe_allow_html=True)
        response_b = st.text_area("Response B text", height=190, label_visibility="collapsed",
                                  placeholder="Paste the second candidate response here...", key="response_b_input")

    a_words, a_chars, a_sentences = response_stats(response_a)
    b_words, b_chars, b_sentences = response_stats(response_b)
    st.markdown("#### Response statistics")
    stats_cols = st.columns(6)
    stats = [
        ("A · WORDS", a_words, "Response A"), ("A · CHARACTERS", a_chars, "including spaces"),
        ("A · SENTENCES", a_sentences, "approximate count"), ("B · WORDS", b_words, "Response B"),
        ("B · CHARACTERS", b_chars, "including spaces"), ("B · SENTENCES", b_sentences, "approximate count"),
    ]
    for col, (label, value, note) in zip(stats_cols, stats):
        with col: metric_card(label, value, note)

    run_col, hint_col = st.columns([1, 4])
    with run_col:
        compare_clicked = st.button("Compare responses →", type="primary", use_container_width=True)
    with hint_col:
        st.caption("Tip: keep both responses relevant to the same prompt for a meaningful comparison.")

    if compare_clicked:
        if not prompt.strip() or not response_a.strip() or not response_b.strip():
            st.warning("Please complete the prompt, Response A, and Response B.")
        else:
            try:
                with st.spinner("Running baseline classification..."):
                    st.session_state["predict_result"] = predict_response_pair(prompt, response_a, response_b, get_model())
                st.session_state.pop("predict_error", None)
            except Exception as exc:
                st.session_state["predict_error"] = str(exc)
                st.session_state["predict_result"] = None

    if st.session_state.get("predict_error"):
        st.error(f"Prediction could not be completed: {st.session_state['predict_error']}")
    result = st.session_state.get("predict_result")
    if result:
        label_a = result.get("response_a_label", "Unknown")
        label_b = result.get("response_b_label", "Unknown")
        a_class, b_class = is_preferred_label(label_a), is_preferred_label(label_b)
        st.write("")
        st.markdown("#### Prediction result")
        if result.get("winner") == "Response A" and a_class is True and b_class is False:
            headline, explanation = "Response A was classified as preferred", "The model assigned the preferred/chosen class to A and the non-preferred class to B."
            st.success("🏆 " + headline)
        elif result.get("winner") == "Response B" and b_class is True and a_class is False:
            headline, explanation = "Response B was classified as preferred", "The model assigned the preferred/chosen class to B and the non-preferred class to A."
            st.success("🏆 " + headline)
        elif a_class is not None and b_class is not None and a_class == b_class:
            headline, explanation = "Inconclusive — both responses received the same class", "The baseline classifier did not distinguish the responses."
            st.warning("↔ " + headline)
        else:
            headline, explanation = "Model output needs label verification", "Check the label encoding used in Member 2's training code."
            st.info("ⓘ " + headline)
        st.markdown(f'<div class="result-card"><div class="result-kicker">Baseline classification</div><div class="result-title">{headline}</div><div class="result-copy">{explanation}</div></div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2, gap="large")
        with r1: metric_card("RESPONSE A · PREDICTED LABEL", str(label_a), "output from saved model")
        with r2: metric_card("RESPONSE B · PREDICTED LABEL", str(label_b), "output from saved model")
        st.caption("This compares predicted classes only; it is not a dedicated pairwise preference model.")

# ----------------------------- Semantic similarity -----------------------------
with tab_semantic:
    section_heading(
        "ADVANCED NLP", "Semantic similarity analysis",
        "Compare your own text or analyze a sample from the cleaned HH-RLHF dataset.",
    )
    st.markdown(
        '<div class="small-note">Cosine similarity estimates semantic relatedness, not helpfulness or safety. The encoder is cached and dataset analysis runs in batches.</div>',
        unsafe_allow_html=True,
    )
    sem_direct, sem_dataset = st.tabs(["Compare my text", "Analyze dataset"])

    with sem_direct:
        st.markdown("#### Prompt-to-response similarity")
        direct_examples = {
            "Helpful answer vs vague answer": (
                "Why is drinking enough water important?",
                "Drinking enough water helps regulate body temperature, supports digestion, transports nutrients, and helps the body function normally.",
                "Water is a liquid. People drink it. I am not sure what else to say.",
            ),
            "Photosynthesis explanation": (
                "Explain photosynthesis in simple terms.",
                "Photosynthesis lets plants use sunlight, water, and carbon dioxide to make sugars for energy, releasing oxygen.",
                "Plants are green and grow outside. You can look at a plant to learn about it.",
            ),
            "Exam preparation advice": (
                "How can I prepare for an exam in one week?",
                "Make a seven-day plan, prioritize important topics, practise questions daily, review mistakes, and get enough sleep.",
                "Exams are hard. Everyone is different, so there is no point making a plan.",
            ),
            "Empathetic response": (
                "I feel nervous before giving a presentation. What can help?",
                "Feeling nervous is common. Practise aloud, prepare cue cards, rehearse your opening, and take slow breaths before you begin.",
                "There is no reason to be nervous. Just stop worrying.",
            ),
        }
        sem_example_cols = st.columns([2, 1, 2], vertical_alignment="bottom")
        with sem_example_cols[0]:
            sem_example_name = st.selectbox("Load a semantic example", list(direct_examples.keys()), key="sem_example_select")
        with sem_example_cols[1]:
            sem_load_clicked = st.button("Load example", key="sem_load_example", use_container_width=True)
        with sem_example_cols[2]:
            st.caption("Loads a prompt and both responses for an immediate test.")

        if sem_load_clicked:
            ex_prompt, ex_a, ex_b = direct_examples[sem_example_name]
            st.session_state["sem_prompt"] = ex_prompt
            st.session_state["sem_response_a"] = ex_a
            st.session_state["sem_response_b"] = ex_b
            st.session_state["semantic_result"] = None
            st.session_state.pop("semantic_error", None)

        sem_prompt = st.text_area("Prompt", key="sem_prompt", placeholder="Enter the prompt that both responses answer...", height=90)
        sa, sb = st.columns(2, gap="large")
        with sa:
            sem_a = st.text_area("Chosen / Response A", key="sem_response_a", height=160, placeholder="Paste Response A...")
        with sb:
            sem_b = st.text_area("Rejected / Response B", key="sem_response_b", height=160, placeholder="Paste Response B...")

        if st.button("Calculate semantic similarity", type="primary", key="direct_semantic"):
            if not sem_prompt.strip() or not sem_a.strip() or not sem_b.strip():
                st.warning("Enter the prompt and both responses before calculating similarity.")
            else:
                try:
                    with st.spinner("Encoding three short texts..."):
                        sim_a, sim_b = similarity_for_texts(sem_prompt, sem_a, sem_b)
                    st.session_state["semantic_result"] = {
                        "prompt": sem_prompt, "a": sem_a, "b": sem_b,
                        "sim_a": sim_a, "sim_b": sim_b,
                    }
                    st.session_state.pop("semantic_error", None)
                except ImportError:
                    st.session_state["semantic_error"] = "Sentence Transformers is not installed. Run: python -m pip install sentence-transformers"
                except Exception as exc:
                    st.session_state["semantic_error"] = f"{exc}. The first run needs internet access to download all-MiniLM-L6-v2."
        if st.session_state.get("semantic_error"):
            st.error(st.session_state["semantic_error"])
        sr = st.session_state.get("semantic_result")
        if sr:
            sim_a, sim_b = sr["sim_a"], sr["sim_b"]
            diff = sim_a - sim_b
            c1, c2, c3 = st.columns(3)
            with c1: metric_card("PROMPT ↔ RESPONSE A", f"{sim_a:.3f}", "cosine similarity")
            with c2: metric_card("PROMPT ↔ RESPONSE B", f"{sim_b:.3f}", "cosine similarity")
            with c3: metric_card("SIMILARITY GAP", f"{abs(diff):.3f}", "absolute difference")
            chart_df = pd.DataFrame({"Text pair": ["Prompt ↔ Response A", "Prompt ↔ Response B"],
                                     "Cosine similarity": [sim_a, sim_b]}).set_index("Text pair")
            st.bar_chart(chart_df, y="Cosine similarity", use_container_width=True)
            if abs(diff) < 0.02:
                st.info("Both responses have very similar prompt-relatedness scores. This is not a preference verdict.")
            elif diff > 0:
                st.info("Response A is more semantically similar to the prompt in this embedding model. That does not automatically mean it is better.")
            else:
                st.info("Response B is more semantically similar to the prompt in this embedding model. That does not automatically mean it is better.")

    with sem_dataset:
        st.markdown("#### Chosen vs rejected across the dataset")
        st.write("For faster feedback, the analysis uses batched encoding and the sentence encoder is loaded only once.")
        sample_size = st.slider("Number of dataset pairs", 50, 1000, 200, 50, key="dataset_sem_sample")
        st.caption("Recommended: start with 100–200 pairs. Increase the sample after the first run works.")
        if st.button("Analyze dataset sample", type="primary", key="semantic_run"):
            try:
                with st.spinner("Encoding the dataset sample in batches. First run may download the model..."):
                    results = make_semantic_dataset_results(sample_size=sample_size)
                chosen_mean = float(results["prompt_chosen_similarity"].mean())
                rejected_mean = float(results["prompt_rejected_similarity"].mean())
                c1, c2, c3 = st.columns(3)
                with c1: metric_card("PAIRS ANALYZED", f"{len(results):,}", "sample from cleaned dataset")
                with c2: metric_card("PROMPT ↔ CHOSEN", f"{chosen_mean:.3f}", "mean cosine similarity")
                with c3: metric_card("PROMPT ↔ REJECTED", f"{rejected_mean:.3f}", "mean cosine similarity")
                st.image(str(OUTPUT_DIR / "semantic_similarity_comparison.png"), use_container_width=True)
                st.download_button(
                    "Download semantic results (CSV)",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="semantic_similarity_results.csv",
                    mime="text/csv",
                    key="download_semantic_csv",
                )
                st.dataframe(results[["prompt_chosen_similarity", "prompt_rejected_similarity"]].describe().round(4),
                             use_container_width=True)
            except ImportError:
                st.error("Sentence Transformers is not installed. Run: python -m pip install sentence-transformers")
            except Exception as exc:
                st.error(f"Dataset semantic analysis failed: {exc}")
                st.info("Check internet access for the first model download and confirm the cleaned CSV exists.")

# ----------------------------- Model evaluation -----------------------------
with tab_models:
    section_heading("BASELINE RESULTS", "Model performance overview",
                    "Results are loaded from Member 2's saved evaluation files; models are not retrained here.")
    if metrics_df.empty:
        st.warning("models/model_comparison.csv was not found.")
    else:
        display_metrics = metrics_df.copy()
        numeric_cols = display_metrics.select_dtypes(include="number").columns
        display_metrics[numeric_cols] = display_metrics[numeric_cols].round(4)
        st.dataframe(display_metrics, use_container_width=True, hide_index=True)
        grouped_metrics_chart(metrics_df)
        if "model" in metrics_df.columns and "f1_score" in metrics_df.columns:
            top_f1 = metrics_df.loc[metrics_df["f1_score"].idxmax()]
            st.markdown(f'<div class="small-note"><b>Current F1 leader:</b> {top_f1["model"]} ({top_f1["f1_score"]:.3f}). These are baseline results; inspect class-wise errors before strong conclusions.</div>', unsafe_allow_html=True)
    report_path = ROOT / "models" / "classification_report.json"
    confusion_path = ROOT / "models" / "confusion_matrix.png"
    lower_left, lower_right = st.columns(2, gap="large")
    with lower_left:
        if confusion_path.exists():
            st.markdown("#### Confusion matrix")
            st.image(str(confusion_path), use_container_width=True)
    with lower_right:
        if report_path.exists():
            st.markdown("#### Classification report")
            try: st.json(json.loads(report_path.read_text(encoding="utf-8")), expanded=False)
            except Exception as exc: st.warning(f"Could not read classification report: {exc}")
    with st.expander("Generate additional response-length charts"):
        st.write("Generate response-length and model-comparison charts from saved files.")
        if st.button("Generate charts", key="generate_charts"):
            try:
                summary = make_response_length_chart(get_dataset())
                make_model_comparison_chart()
                copy_confusion_matrix()
                st.success(f"Charts saved in {OUTPUT_DIR}")
                st.dataframe(summary.round(2), use_container_width=True, hide_index=True)
                chart = OUTPUT_DIR / "response_length_distribution.png"
                if chart.exists(): st.image(str(chart), use_container_width=True)
                comparison = OUTPUT_DIR / "model_comparison_metrics.png"
                if comparison.exists(): st.image(str(comparison), use_container_width=True)
            except Exception as exc:
                st.error(f"Chart generation failed: {exc}")

# ----------------------------- About -----------------------------
with tab_about:
    section_heading("PROJECT CONTEXT", "About this research dashboard",
                    "Preference-based analysis of helpful and harmless AI responses using the HH-RLHF dataset.")
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.markdown("""
            <div class="panel">
              <h3>Project objective</h3>
              <p>Explore how human-preferred responses differ from rejected responses, compare baseline text classifiers, and inspect prompt-response semantic relatedness.</p>
              <hr><h3>Member 3 deliverables</h3>
              <ul><li>Sentence Transformer semantic similarity analysis</li>
              <li>Chosen-versus-rejected response length visualizations</li>
              <li>Model comparison and evaluation dashboard</li>
              <li>Interactive Streamlit interface</li></ul>
            </div>""", unsafe_allow_html=True)
    with right:
        st.markdown("""
            <div class="panel">
              <h3>Dataset fields</h3>
              <p><b>prompt</b> — user instruction or question</p>
              <p><b>chosen_response</b> — preferred answer in the pair</p>
              <p><b>rejected_response</b> — alternative answer not preferred</p>
              <hr><h3>Interpretation caveats</h3>
              <p>Semantic similarity is not a direct quality score. The saved classifier predicts individual response labels and is not a trained pairwise ranker.</p>
            </div>""", unsafe_allow_html=True)

st.markdown('<div class="footer">HH-RLHF Preference Lab · NLP project prototype · Uses the local dataset and saved model artifacts in this project.</div>', unsafe_allow_html=True)
