# NLP & Text Analytics

An end-to-end Natural Language Processing project focused on **sentiment classification, topic modeling, and semantic similarity search** using movie reviews.

## Project Overview

This project analyzes **8,530 Rotten Tomatoes reviews** using various NLP and machine learning techniques. The pipeline covers text preprocessing, feature extraction, sentiment classification, topic modeling, and semantic search.

## Features

* Text preprocessing and cleaning
* TF-IDF based feature extraction
* Sentiment classification using Logistic Regression
* Topic modeling using NMF and LDA
* Semantic similarity search using Sentence Transformers
* Interactive Streamlit dashboard
* Docker-based deployment

## Tech Stack

* **Language:** Python
* **Data Processing:** Pandas
* **NLP:** NLTK
* **Machine Learning:** Scikit-learn
* **Embeddings:** Sentence Transformers
* **Frontend/Dashboard:** Streamlit
* **Deployment:** Docker

## NLP Pipeline

```text
Raw Reviews
     ↓
Text Preprocessing
     ↓
Feature Extraction (TF-IDF)
     ↓
Sentiment Classification
     ↓
Topic Modeling
     ↓
Semantic Similarity Search
     ↓
Streamlit Dashboard
```

## Machine Learning Techniques

### Sentiment Classification

Used **TF-IDF** for text feature extraction and **Logistic Regression** for classifying review sentiment.

### Topic Modeling

Applied:

* **NMF (Non-negative Matrix Factorization)**
* **LDA (Latent Dirichlet Allocation)**

to identify major topics and patterns within the reviews.

### Semantic Search

Used **Sentence Transformers** to generate text embeddings and perform semantic similarity search across reviews.

## Dataset

The project uses **8,530 Rotten Tomatoes movie reviews** for NLP analysis and machine learning tasks.

## Installation

```bash
git clone <your-github-repository-url>
cd nlp-text-analytics

pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Docker

Build the Docker image:

```bash
docker build -t nlp-text-analytics .
```

Run the container:

```bash
docker run -p 8501:8501 nlp-text-analytics
```

Then open:

```text
http://localhost:8501
```

## Key Learnings

* Practical implementation of NLP preprocessing techniques
* Text feature extraction using TF-IDF
* Supervised sentiment classification
* Unsupervised topic modeling
* Text embeddings and semantic search
* Building interactive ML applications with Streamlit
* Containerizing applications using Docker


