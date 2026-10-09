"""Train the TF-IDF + Logistic Regression model and save it for the Streamlit app."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

BASE_DIR = "data/"


def combine_text(df):
    return (df["title"].fillna("") + " " + df["content"].fillna("")).astype(str)


train_df = pd.read_parquet(BASE_DIR + "train-00000-of-00001.parquet")
test_df = pd.read_parquet(BASE_DIR + "test-00000-of-00001.parquet")

model = make_pipeline(
    TfidfVectorizer(max_features=100_000, ngram_range=(1, 2), min_df=2, stop_words="english"),
    LogisticRegression(max_iter=200, class_weight="balanced"),
)
model.fit(combine_text(train_df), train_df["bias_text"])
print("Test accuracy:", accuracy_score(test_df["bias_text"], model.predict(combine_text(test_df))))

joblib.dump(model, "bias_model.joblib", compress=3)
print("Saved bias_model.joblib")
