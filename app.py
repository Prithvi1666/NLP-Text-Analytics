"""
NLP & Text Analytics Dashboard
Rotten Tomatoes movie reviews — sentiment analysis, topic modeling,
semantic similarity search.
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import matplotlib
matplotlib.use("Agg")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP & Text Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TEMPLATE       = "plotly_dark"
SENTIMENT_COLORS = {"Positive": "#4CAF50", "Negative": "#F44336"}


# ── Load pre-trained artifacts ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts() -> dict:
    return joblib.load("models/artifacts.pkl")


@st.cache_resource
def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── Section: Dataset Overview ──────────────────────────────────────────────────
def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Dataset Overview")
    st.caption("Rotten Tomatoes movie reviews — critic quotes labeled positive or negative")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Reviews", f"{len(df):,}")
    m2.metric("Positive",      f"{(df['sentiment'] == 'Positive').sum():,}")
    m3.metric("Negative",      f"{(df['sentiment'] == 'Negative').sum():,}")
    m4.metric("Avg Length",    f"{df['word_count_raw'].mean():.0f} words")

    col1, col2 = st.columns([1, 2])

    with col1:
        counts = df["sentiment"].value_counts()
        fig = px.pie(
            names=counts.index, values=counts.values,
            color=counts.index,
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.45, title="Sentiment Distribution", template=TEMPLATE,
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df, x="word_count_raw", color="sentiment",
            barmode="overlay", nbins=50, opacity=0.75,
            color_discrete_map=SENTIMENT_COLORS,
            title="Review Length Distribution by Sentiment",
            labels={"word_count_raw": "Word Count"},
            template=TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Top 30 Most Frequent Words (after cleaning)**")
    all_words = " ".join(df["cleaned"]).split()
    freq = Counter(all_words).most_common(30)
    freq_df = pd.DataFrame(freq, columns=["word", "count"])
    fig = px.bar(
        freq_df, x="word", y="count",
        color="count", color_continuous_scale="Blues",
        title="Word Frequency (cleaned corpus)", template=TEMPLATE,
    )
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Sample Reviews**")
    sample = df[["review", "sentiment", "word_count_raw", "word_count_cleaned"]].sample(8, random_state=1)
    st.dataframe(sample, use_container_width=True, hide_index=True)


# ── Section: Text Preprocessing ───────────────────────────────────────────────
def render_preprocessing(df: pd.DataFrame) -> None:
    st.subheader("Text Preprocessing Pipeline")

    steps = [
        ("1. Lowercase",      "Convert all characters to lowercase to ensure consistent token matching."),
        ("2. Remove HTML tags","Strip `<br>`, `<i>`, and other HTML markup using a regex tag pattern."),
        ("3. Remove URLs",     "Drop any `http://` or `https://` tokens that carry no sentiment signal."),
        ("4. Keep letters only","Replace non-alphabetic characters (punctuation, digits) with spaces via `[^a-z\\s]`."),
        ("5. Collapse whitespace","Normalise multiple spaces to single spaces and strip leading/trailing whitespace."),
        ("6. Remove stopwords","Drop NLTK English stopwords such as 'the', 'is', 'at', etc."),
        ("7. Remove short tokens","Discard tokens of length ≤ 2 which rarely carry semantic meaning."),
    ]

    for title, desc in steps:
        with st.expander(title, expanded=False):
            st.markdown(desc)

    st.code(
        """def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>",  " ", text)   # HTML tags
    text = re.sub(r"http\\S+",  " ", text)   # URLs
    text = re.sub(r"[^a-z\\s]", " ", text)   # non-alpha
    text = re.sub(r"\\s+",      " ", text).strip()
    tokens = [t for t in text.split()
              if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)""",
        language="python",
    )

    st.markdown("**Before / After Comparison**")
    idx = st.slider("Select review index:", 0, len(df) - 1, 0)
    row = df.iloc[idx]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Raw text**")
        st.info(row["review"])
    with c2:
        st.markdown("**Cleaned text**")
        st.success(row["cleaned"] if row["cleaned"] else "_All tokens removed by cleaning_")

    c3, c4 = st.columns(2)
    c3.metric("Raw tokens",     row["word_count_raw"])
    c4.metric("Cleaned tokens", row["word_count_cleaned"])

    reduction = df["word_count_raw"] - df["word_count_cleaned"]
    fig = px.histogram(
        reduction, nbins=40,
        title="Token Reduction per Review (raw − cleaned)",
        labels={"value": "Tokens removed"},
        template=TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Section: Topic Explorer ────────────────────────────────────────────────────
def render_topics(df, lda_top_words, nmf_top_words, lda_doc_topics, nmf_doc_topics) -> None:
    st.subheader("Topic Explorer")
    st.caption(
        "Unsupervised topic discovery running independently of the sentiment labels. "
        "NMF (TF-IDF) tends to produce cleaner, more interpretable topics; "
        "LDA (count matrix) captures probabilistic word-to-topic memberships."
    )

    model_choice = st.radio("Topic model:", ["NMF", "LDA"], horizontal=True)
    top_words_map = nmf_top_words if model_choice == "NMF" else lda_top_words
    doc_topics    = nmf_doc_topics if model_choice == "NMF" else lda_doc_topics

    topic_names   = list(top_words_map.keys())
    selected_name = st.selectbox("Select topic:", topic_names)
    topic_idx     = topic_names.index(selected_name)

    words        = top_words_map[selected_name][:15]
    proxy_scores = list(range(len(words), 0, -1))

    fig = px.bar(
        x=proxy_scores, y=words, orientation="h",
        color=proxy_scores, color_continuous_scale="Viridis",
        title=f"{model_choice} — {selected_name}: Top 15 Words",
        template=TEMPLATE,
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Topic Distribution Across Dataset**")
    topic_col = "nmf_topic" if model_choice == "NMF" else "lda_topic"
    topic_counts = df[topic_col].value_counts().sort_index().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    topic_counts["Topic"] = topic_counts["Topic"].apply(lambda x: f"Topic {x + 1}")
    fig2 = px.bar(
        topic_counts, x="Topic", y="Count",
        color="Count", color_continuous_scale="Blues",
        title=f"{model_choice} — Review Count per Dominant Topic",
        template=TEMPLATE,
    )
    fig2.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"**Top 5 Reviews — {selected_name}**")
    scores   = doc_topics[:, topic_idx]
    top_idxs = np.argsort(scores)[::-1][:5]
    for rank, i in enumerate(top_idxs, 1):
        row = df.iloc[i]
        color = "#4CAF50" if row["sentiment"] == "Positive" else "#F44336"
        st.markdown(
            f"**{rank}.** {row['review'][:300]}{'...' if len(row['review']) > 300 else ''}"
        )
        st.caption(
            f"Sentiment: :{'green' if row['sentiment'] == 'Positive' else 'red'}[**{row['sentiment']}**]"
            f" | Topic Score: {scores[i]:.4f}"
        )
        if rank < 5:
            st.divider()


# ── Section: Sentiment Analysis ────────────────────────────────────────────────
def render_sentiment(X_te, y_te, y_pred, cv_results, top_words, pipe) -> None:
    st.subheader("Sentiment Classifier — TF-IDF + Logistic Regression")
    st.caption("5-fold stratified CV on the training split; evaluation on 20% hold-out test set")

    c1, c2, c3 = st.columns(3)
    c1.metric("CV Accuracy", f"{cv_results['Accuracy']:.3f}")
    c2.metric("CV F1",       f"{cv_results['F1']:.3f}")
    c3.metric("CV ROC-AUC",  f"{cv_results['ROC-AUC']:.3f}")

    col1, col2 = st.columns(2)

    with col1:
        cm  = confusion_matrix(y_te, y_pred)
        fig = px.imshow(
            cm, text_auto=True,
            x=["Predicted: Neg", "Predicted: Pos"],
            y=["Actual: Neg",    "Actual: Pos"],
            color_continuous_scale="Blues",
            title="Confusion Matrix — Test Set",
            template=TEMPLATE,
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        report = classification_report(
            y_te, y_pred,
            target_names=["Negative", "Positive"],
            output_dict=True,
        )
        rep_df = pd.DataFrame(report).T.drop("accuracy").round(3)
        st.markdown("**Classification Report — Test Set**")
        st.dataframe(rep_df, use_container_width=True)

    st.markdown("**Top Discriminative Words by Sentiment**")
    col3, col4 = st.columns(2)
    with col3:
        pos_words = top_words[top_words["direction"] == "Positive"].head(20)
        fig = px.bar(
            pos_words, x="coef", y="word", orientation="h",
            color="coef", color_continuous_scale="Greens",
            title="Top 20 Positive Words", template=TEMPLATE,
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        neg_words = top_words[top_words["direction"] == "Negative"].head(20).copy()
        neg_words["coef_abs"] = neg_words["coef"].abs()
        fig = px.bar(
            neg_words, x="coef_abs", y="word", orientation="h",
            color="coef_abs", color_continuous_scale="Reds",
            title="Top 20 Negative Words", template=TEMPLATE,
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**ROC Curve — Test Set**")
    y_prob = pipe.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="gray"))
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"TF-IDF + LR  (AUC = {roc_auc:.3f})",
        line=dict(width=3, color="#4CAF50"),
    ))
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="ROC Curve",
        template=TEMPLATE,
        legend=dict(x=0.55, y=0.05),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Live Sentiment Prediction**")
    user_text = st.text_area("Enter a movie review snippet:", height=120,
                             placeholder="e.g. 'A breathtaking performance with stunning visuals.'")
    if user_text.strip():
        import re
        from nltk.corpus import stopwords as sw
        stop = set(sw.words("english"))
        cleaned = re.sub(r"[^a-z\s]", " ", user_text.lower())
        cleaned = " ".join(t for t in cleaned.split() if t not in stop and len(t) > 2)
        prob   = pipe.predict_proba([cleaned])[0, 1]
        label  = "Positive" if prob >= 0.5 else "Negative"
        color  = "#4CAF50" if label == "Positive" else "#F44336"
        st.markdown(
            f"""<div style="padding:20px; border-radius:10px; background:{color}22;
                border:2px solid {color}; text-align:center; margin-top:12px;">
  <h3 style="color:{color}; margin:0;">{label}</h3>
  <p style="margin:6px 0 0 0; opacity:.8;">Confidence: <strong>{max(prob, 1-prob)*100:.1f}%</strong></p>
</div>""",
            unsafe_allow_html=True,
        )


# ── Section: Semantic Search ───────────────────────────────────────────────────
def render_search(df: pd.DataFrame, embeddings: np.ndarray) -> None:
    st.subheader("Semantic Similarity Search")
    st.caption(
        "Type any phrase and retrieve the most semantically similar reviews using "
        "sentence-level embeddings (all-MiniLM-L6-v2) and cosine similarity. "
        "This goes beyond keyword matching — the model understands meaning."
    )

    query = st.text_input(
        "Search query:",
        placeholder="e.g. 'slow pacing but beautiful cinematography'  or  'laugh-out-loud comedy'",
    )
    top_k = st.slider("Number of results:", min_value=3, max_value=20, value=8)

    col_sent = st.multiselect(
        "Filter by sentiment:",
        options=["Positive", "Negative"],
        default=["Positive", "Negative"],
    )

    if query.strip():
        with st.spinner("Finding similar reviews..."):
            encoder    = load_encoder()
            query_emb  = encoder.encode([query], normalize_embeddings=True)
            scores     = (embeddings @ query_emb.T).flatten()

        mask     = df["sentiment"].isin(col_sent)
        filtered = np.where(mask)[0]
        ranked   = filtered[np.argsort(scores[filtered])[::-1]][:top_k]

        st.markdown(f"**Top {top_k} results for: _{query}_**")
        for rank, i in enumerate(ranked, 1):
            row   = df.iloc[i]
            sim   = scores[i]
            color = "#4CAF50" if row["sentiment"] == "Positive" else "#F44336"
            st.markdown(
                f"""<div style="padding:14px; margin-bottom:10px; border-radius:8px;
                    border-left:4px solid {color}; background:{color}11;">
  <strong style="color:{color};">#{rank}  {row['sentiment']}</strong>
  <span style="float:right; opacity:.6;">Similarity: {sim:.3f}</span><br/>
  <span style="opacity:.9;">{row['review']}</span>
</div>""",
                unsafe_allow_html=True,
            )


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("🎬 NLP & Text Analytics — Movie Review Intelligence")
    st.markdown(
        "**Dataset:** Rotten Tomatoes critic quotes — 8,530 labeled movie reviews. "
        "**Pipeline:** text cleaning → TF-IDF sentiment classifier → NMF + LDA topic modeling → "
        "sentence-embedding semantic search."
    )

    arts = load_artifacts()
    df             = arts["df"]
    X_te           = arts["X_te"]
    y_te           = arts["y_te"]
    y_pred         = arts["y_pred"]
    cv_results     = arts["cv_results"]
    top_words      = arts["top_words"]
    pipe           = arts["sentiment_pipe"]
    lda_top_words  = arts["lda_top_words"]
    nmf_top_words  = arts["nmf_top_words"]
    lda_doc_topics = arts["lda_doc_topics"]
    nmf_doc_topics = arts["nmf_doc_topics"]
    embeddings     = arts["embeddings"]

    tabs = st.tabs([
        "📊 Dataset Overview",
        "⚙️ Preprocessing",
        "🗂️ Topic Explorer",
        "🎯 Sentiment Analysis",
        "🔎 Semantic Search",
    ])

    with tabs[0]: render_overview(df)
    with tabs[1]: render_preprocessing(df)
    with tabs[2]: render_topics(df, lda_top_words, nmf_top_words, lda_doc_topics, nmf_doc_topics)
    with tabs[3]: render_sentiment(X_te, y_te, y_pred, cv_results, top_words, pipe)
    with tabs[4]: render_search(df, embeddings)


if __name__ == "__main__":
    main()
