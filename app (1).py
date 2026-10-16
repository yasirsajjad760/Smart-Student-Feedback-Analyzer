import re
from collections import Counter

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Smart Student Feedback Analyzer",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_sentiment_analyzer():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()

sia = load_sentiment_analyzer()

TOPIC_KEYWORDS = {
    "Teaching": [
        "teacher", "teaching", "lecturer", "lecture",
        "instructor", "professor", "explain", "teach"
    ],
    "Exams": [
        "exam", "exams", "test", "quiz", "marks",
        "grade", "grading", "assessment"
    ],
    "Assignments": [
        "assignment", "assignments", "homework",
        "deadline", "workload", "project"
    ],
    "Facilities": [
        "classroom", "library", "internet", "lab",
        "computer", "wifi", "facility", "facilities",
        "equipment", "building"
    ],
    "Course Content": [
        "course", "content", "syllabus", "curriculum",
        "subject", "material", "textbook"
    ]
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def analyze_sentiment(text):
    score = sia.polarity_scores(str(text))
    compound = score["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return pd.Series({
        "positive_score": score["pos"],
        "negative_score": score["neg"],
        "neutral_score": score["neu"],
        "compound_score": compound,
        "predicted_sentiment": label
    })

def classify_topic(text):
    text = clean_text(text)
    padded_text = f" {text} "

    for topic, keywords in TOPIC_KEYWORDS.items():
        for word in keywords:
            if f" {word} " in padded_text:
                return topic

    return "Other"

st.title("📊 Smart Student Feedback Analyzer")
st.write(
    "Upload student feedback, analyze sentiment, explore topics, "
    "and download the analyzed results."
)

uploaded_file = st.file_uploader(
    "Upload student feedback CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Start by uploading your student feedback CSV file.")
    st.stop()

try:
    try:
        data = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        data = pd.read_csv(uploaded_file, encoding="cp1252")
except Exception as e:
    st.error(f"CSV file read nahi ho saki: {e}")
    st.stop()

if data.empty:
    st.error("CSV file empty hai.")
    st.stop()

st.subheader("Select Feedback Column")

candidates = [
    col for col in data.columns
    if any(word in col.lower()
           for word in ["feedback", "comment", "review", "text"])
]

default_index = (
    data.columns.get_loc(candidates[0])
    if candidates else 0
)

text_column = st.selectbox(
    "Kaun se column mein student feedback hai?",
    options=data.columns.tolist(),
    index=default_index
)

data = data.dropna(subset=[text_column]).copy()
data[text_column] = data[text_column].astype(str)

if data.empty:
    st.warning("Selected column mein usable feedback nahi mila.")
    st.stop()


with st.spinner("Feedback analyze ho raha hai..."):

    # Pehle se mojood analysis columns remove karein
    analysis_columns = [
        "positive_score",
        "negative_score",
        "neutral_score",
        "compound_score",
        "predicted_sentiment",
        "cleaned_feedback",
        "topic"
    ]

    data = data.drop(
        columns=analysis_columns,
        errors="ignore"
    )

    # Sentiment analysis dobara karein
    sentiment_results = data[text_column].apply(analyze_sentiment)

    data = pd.concat(
        [
            data.reset_index(drop=True),
            sentiment_results.reset_index(drop=True)
        ],
        axis=1
    )

    # Topic aur cleaned text create karein
    data["cleaned_feedback"] = data[text_column].apply(clean_text)
    data["topic"] = data[text_column].apply(classify_topic)


st.success(f"Analysis complete! Total feedback: {len(data)}")

sentiment_counts = data["predicted_sentiment"].value_counts()
positive = int(sentiment_counts.get("Positive", 0))
negative = int(sentiment_counts.get("Negative", 0))
neutral = int(sentiment_counts.get("Neutral", 0))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Feedback", len(data))
m2.metric("Positive", positive)
m3.metric("Negative", negative)
m4.metric("Neutral", neutral)

st.subheader("Sentiment Distribution")
fig, ax = plt.subplots()
sentiment_counts.reindex(
    ["Positive", "Neutral", "Negative"], fill_value=0
).plot(kind="bar", ax=ax)
ax.set_xlabel("Sentiment")
ax.set_ylabel("Number of feedback records")
ax.tick_params(axis="x", rotation=0)
st.pyplot(fig)
plt.close(fig)

st.subheader("Feedback by Topic")
topic_counts = data["topic"].value_counts()
fig, ax = plt.subplots()
topic_counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Topic")
ax.set_ylabel("Number of feedback records")
ax.tick_params(axis="x", rotation=35)
st.pyplot(fig)
plt.close(fig)

st.subheader("Improvement Priorities")
negative_data = data[data["predicted_sentiment"] == "Negative"]
summary = pd.DataFrame({
    "Total Feedback": data["topic"].value_counts(),
    "Negative Feedback": negative_data["topic"].value_counts()
}).fillna(0).astype(int)

summary["Negative Percentage"] = (
    summary["Negative Feedback"]
    / summary["Total Feedback"].replace(0, float("nan"))
    * 100
).round(2)

st.dataframe(
    summary.sort_values("Negative Feedback", ascending=False),
    use_container_width=True
)

st.subheader("Explore Feedback")
sentiment_filter = st.selectbox(
    "Filter by sentiment",
    ["All", "Positive", "Negative", "Neutral"]
)
topic_filter = st.selectbox(
    "Filter by topic",
    ["All"] + sorted(data["topic"].unique().tolist())
)

filtered = data.copy()

if sentiment_filter != "All":
    filtered = filtered[
        filtered["predicted_sentiment"] == sentiment_filter
    ]

if topic_filter != "All":
    filtered = filtered[filtered["topic"] == topic_filter]

st.write(f"Matching records: {len(filtered)}")
st.dataframe(filtered, use_container_width=True)

st.subheader("Common Words in Feedback")
all_words = " ".join(data["cleaned_feedback"]).split()
stop_words = {
    "the", "and", "for", "was", "are", "with", "this", "that",
    "you", "your", "they", "their", "have", "has", "had", "our",
    "from", "but", "not", "too", "very", "all", "can", "should",
    "would", "could", "been", "being", "into", "its", "about",
    "when", "what", "who", "how", "than", "then", "there"
}
word_counts = Counter(
    word for word in all_words
    if len(word) > 2 and word not in stop_words
)

if word_counts:
    st.write(pd.DataFrame(
        word_counts.most_common(15),
        columns=["Word", "Count"]
    ))

st.subheader("Download Results")
csv_data = data.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Download analyzed CSV",
    data=csv_data,
    file_name="student_feedback_analyzed.csv",
    mime="text/csv"
)

st.caption(
    "Note: VADER is a rule-based sentiment analyzer. Its predictions "
    "may not match the original dataset labels exactly."
)
