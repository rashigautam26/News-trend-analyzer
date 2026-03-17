import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

st.set_page_config(layout="wide")

st.title("📰 NLP News Analytics Dashboard")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("final_nlp_output.csv")

# =========================
# TOTAL NEWS
# =========================
st.metric("Total Headlines", len(df))

# =========================
# VIBE CHECK
# =========================
st.subheader("Vibe Check")

sentiment_count = df["sentiment"].value_counts()

fig = px.pie(
    values=sentiment_count.values,
    names=sentiment_count.index,
    title="Sentiment Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# TF-IDF KEYWORDS
# =========================
st.subheader("Top Keywords (TF-IDF)")

vectorizer = TfidfVectorizer(max_features=10)
X = vectorizer.fit_transform(df["processed_text"])

keywords = vectorizer.get_feature_names_out()

for word in keywords:
    st.button(word)

# =========================
# TOPIC MODELING
# =========================
st.subheader("Generated Topics (LDA)")

lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X)

words = vectorizer.get_feature_names_out()

for i, topic in enumerate(lda.components_):
    topic_words = [words[j] for j in topic.argsort()[-5:]]
    st.write(f"Topic {i+1}: {', '.join(topic_words)}")

# =========================
# DATA TABLE
# =========================
st.subheader("Headlines Analyzed")

st.dataframe(df[["title", "sentiment"]])