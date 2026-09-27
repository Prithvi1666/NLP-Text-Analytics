"""
Pre-training script: NLP & Text Analytics Pipeline
Dataset: Rotten Tomatoes movie reviews (HuggingFace datasets)

Trains a sentiment classifier, fits NMF + LDA topic models, and
computes sentence embeddings — then serializes everything to
models/artifacts.pkl so the Streamlit app starts instantly.
"""

import os
import re
import warnings

import joblib
import nltk
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ── NLTK resources ─────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))
N_TOPICS   = 10


# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)   # HTML tags
    text = re.sub(r"http\S+",  " ", text)   # URLs
    text = re.sub(r"[^a-z\s]", " ", text)   # non-alpha
    text = re.sub(r"\s+",      " ", text).strip()
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("── Loading Rotten Tomatoes dataset ──────────────────")
    ds = load_dataset("rotten_tomatoes")
    parts = [ds["train"].to_pandas(), ds["validation"].to_pandas(), ds["test"].to_pandas()]
    df = pd.concat(parts, ignore_index=True)
    df.columns = ["review", "label"]
    df["sentiment"] = df["label"].map({1: "Positive", 0: "Negative"})
    print(f"   Total reviews: {len(df)}")
    print(f"   Positive: {(df['sentiment'] == 'Positive').sum()}  "
          f"Negative: {(df['sentiment'] == 'Negative').sum()}")

    # ── Preprocessing ──────────────────────────────────────────────────────────
    print("\n── Cleaning text ─────────────────────────────────────")
    df["cleaned"]            = df["review"].apply(clean_text)
    df["word_count_raw"]     = df["review"].apply(lambda x: len(x.split()))
    df["word_count_cleaned"] = df["cleaned"].apply(lambda x: len(x.split()))
    print(f"   Avg raw length:    {df['word_count_raw'].mean():.0f} words")
    print(f"   Avg cleaned length:{df['word_count_cleaned'].mean():.0f} words")

    # ── Sentiment classifier ───────────────────────────────────────────────────
    print("\n── Training sentiment classifier (5-fold CV) ─────────")
    X, y = df["cleaned"], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10_000, ngram_range=(1, 2),
            min_df=2, max_df=0.95,
        )),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42,
        )),
    ])

    cv = cross_validate(pipe, X_tr, y_tr, cv=5,
                        scoring=["accuracy", "f1", "roc_auc"])
    cv_results = {
        "Accuracy": round(float(cv["test_accuracy"].mean()), 4),
        "F1":       round(float(cv["test_f1"].mean()),       4),
        "ROC-AUC":  round(float(cv["test_roc_auc"].mean()),  4),
    }
    print(f"   Accuracy: {cv_results['Accuracy']:.4f}  "
          f"F1: {cv_results['F1']:.4f}  "
          f"AUC: {cv_results['ROC-AUC']:.4f}")

    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    print(classification_report(y_te, y_pred, target_names=["Negative", "Positive"]))

    # Top discriminative words (positive vs negative)
    vocab  = pipe.named_steps["tfidf"].get_feature_names_out()
    coefs  = pipe.named_steps["clf"].coef_[0]
    top_pos = pd.DataFrame({
        "word": vocab[np.argsort(coefs)[::-1][:30]],
        "coef": np.sort(coefs)[::-1][:30],
        "direction": "Positive",
    })
    top_neg = pd.DataFrame({
        "word": vocab[np.argsort(coefs)[:30]],
        "coef": np.sort(coefs)[:30],
        "direction": "Negative",
    })
    top_words = pd.concat([top_pos, top_neg], ignore_index=True)

    # ── Topic modeling ─────────────────────────────────────────────────────────
    print(f"\n── Topic modeling (NMF + LDA, {N_TOPICS} topics each) ─────")
    count_vec = CountVectorizer(
        max_features=5000, min_df=3, max_df=0.9, stop_words="english"
    )
    tfidf_vec = TfidfVectorizer(
        max_features=5000, min_df=3, max_df=0.9, stop_words="english"
    )
    dtm_count = count_vec.fit_transform(df["cleaned"])
    dtm_tfidf = tfidf_vec.fit_transform(df["cleaned"])

    lda = LatentDirichletAllocation(
        n_components=N_TOPICS, random_state=42, n_jobs=-1, max_iter=20
    )
    nmf = NMF(n_components=N_TOPICS, random_state=42, max_iter=500)

    lda_doc_topics = lda.fit_transform(dtm_count)
    nmf_doc_topics = nmf.fit_transform(dtm_tfidf)

    count_vocab = count_vec.get_feature_names_out()
    tfidf_vocab = tfidf_vec.get_feature_names_out()

    lda_top_words = {
        f"Topic {i + 1}": [count_vocab[j] for j in comp.argsort()[:-16:-1]]
        for i, comp in enumerate(lda.components_)
    }
    nmf_top_words = {
        f"Topic {i + 1}": [tfidf_vocab[j] for j in comp.argsort()[:-16:-1]]
        for i, comp in enumerate(nmf.components_)
    }

    df["lda_topic"] = np.argmax(lda_doc_topics, axis=1)
    df["nmf_topic"] = np.argmax(nmf_doc_topics, axis=1)
    print("   LDA + NMF trained ✓")

    # ── Sentence embeddings ────────────────────────────────────────────────────
    print("\n── Computing sentence embeddings (all-MiniLM-L6-v2) ─")
    encoder    = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = encoder.encode(
        df["cleaned"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # dot product == cosine similarity
    )
    print(f"   Embeddings shape: {embeddings.shape}")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    print("\n── Saving artifacts ─────────────────────────────────")
    os.makedirs("models", exist_ok=True)
    artifacts = {
        "df":              df,
        "X_te":            X_te,
        "y_te":            y_te,
        "y_pred":          y_pred,
        "sentiment_pipe":  pipe,
        "cv_results":      cv_results,
        "top_words":       top_words,
        "lda_top_words":   lda_top_words,
        "nmf_top_words":   nmf_top_words,
        "lda_doc_topics":  lda_doc_topics,
        "nmf_doc_topics":  nmf_doc_topics,
        "embeddings":      embeddings,
    }
    joblib.dump(artifacts, "models/artifacts.pkl", compress=3)
    print("   Saved → models/artifacts.pkl")
    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
