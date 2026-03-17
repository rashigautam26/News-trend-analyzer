# import pandas as pd
# import re
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from textblob import TextBlob

# print("Reading cleaned data...")

# df = pd.read_csv("news_cleaned.csv")

# # TEXT CLEANING
# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     return text

# df["cleaned"] = df["title"].apply(clean_text)

# print("Text cleaning done ✅")

# # TOKENIZATION
# stop_words = set(stopwords.words("english"))

# def preprocess(text):
#     tokens = word_tokenize(text)
#     words = [w for w in tokens if w not in stop_words and len(w) > 2]
#     return " ".join(words)

# df["processed"] = df["cleaned"].apply(preprocess)

# print("Tokenization done ✅")

# # TF-IDF
# vectorizer = TfidfVectorizer(max_features=10)
# tfidf = vectorizer.fit_transform(df["processed"])

# print("\nTop Keywords:")
# print(vectorizer.get_feature_names_out())

# # SENTIMENT
# def get_sentiment(text):
#     polarity = TextBlob(text).sentiment.polarity
#     if polarity > 0:
#         return "Positive"
#     elif polarity < 0:
#         return "Negative"
#     else:
#         return "Neutral"

# df["sentiment"] = df["processed"].apply(get_sentiment)

# df.to_csv("final_nlp_output.csv", index=False)

# print("\nSentiment Count:")
# print(df["sentiment"].value_counts())

# print("\nMilestone-2 Completed 🎉")
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation


nltk.download('punkt')
nltk.download('stopwords')

print("Reading cleaned data...")


df = pd.read_csv("news_cleaned.csv")

# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df["cleaned_text"] = df["title"].apply(clean_text)

print("Text cleaning done ✅")

# =========================
# TOKENIZATION + STOPWORD REMOVAL
# =========================
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(text)
    words = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(words)

df["processed_text"] = df["cleaned_text"].apply(preprocess)

print("Tokenization & Stopword removal done ✅")

# =========================
# TF-IDF → TOP KEYWORDS
# =========================
vectorizer = TfidfVectorizer(max_features=10)
tfidf = vectorizer.fit_transform(df["processed_text"])

print("\nTop Keywords:")
print(vectorizer.get_feature_names_out())

# =========================
# SENTIMENT ANALYSIS
# =========================
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["processed_text"].apply(get_sentiment)

print("\nSentiment Count:")
print(df["sentiment"].value_counts())

# =========================
# TOPIC MODELING (LDA)
# =========================
print("\nApplying Topic Modeling...")

lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(tfidf)

words = vectorizer.get_feature_names_out()

for i, topic in enumerate(lda.components_):
    print(f"\nTopic {i+1}:")
    topic_words = [words[j] for j in topic.argsort()[-5:]]
    print(topic_words)

# =========================
# FINAL FILE SAVE
# =========================
df.to_csv("final_nlp_output.csv", index=False)

print("\nMilestone-2 Completed Successfully 🎉")