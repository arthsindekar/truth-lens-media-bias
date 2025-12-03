import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os


# -----------------------------------------------------------
# 1. Load dataset (LOCAL FILES)
# -----------------------------------------------------------

BASE_DIR = "/Users/arthsindekar/Desktop/AI/ProjectMediaBias/data/"

def load_split(name: str):
    """Load the local parquet split."""
    files = {
        "train": "train-00000-of-00001.parquet",
        "valid": "valid-00000-of-00001.parquet",
        "test":  "test-00000-of-00001.parquet",
    }
    path = os.path.join(BASE_DIR, files[name])
    print(f"Loading: {path}")
    return pd.read_parquet(path)


train_df = load_split("train")
valid_df = load_split("valid")
test_df = load_split("test")


# -----------------------------------------------------------
# 2. Prepare text + labels
# -----------------------------------------------------------

def combine_text(df):
    return (
        df["title"].fillna("") + " " + df["content"].fillna("")
    ).astype(str)


X_train = combine_text(train_df)
y_train = train_df["bias"]

X_valid = combine_text(valid_df)
y_valid = valid_df["bias"]

X_test = combine_text(test_df)
y_test = test_df["bias"]


# -----------------------------------------------------------
# 3. Logistic Regression pipeline
# -----------------------------------------------------------

logreg_clf = make_pipeline(
    TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english"
    ),
    LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        n_jobs=-1
    )
)

print("\nTraining Logistic Regression...")
logreg_clf.fit(X_train, y_train)


# -----------------------------------------------------------
# 4. Evaluation
# -----------------------------------------------------------

print("\n=== VALIDATION RESULTS ===")
valid_pred = logreg_clf.predict(X_valid)
print("Accuracy:", accuracy_score(y_valid, valid_pred))
print(classification_report(y_valid, valid_pred))
print(confusion_matrix(y_valid, valid_pred))

print("\n=== TEST RESULTS ===")
test_pred = logreg_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))
print(confusion_matrix(y_test, test_pred))

print("\nDone.")
print("Train bias distribution:\n", train_df["bias"].value_counts(normalize=True))
print("Valid bias distribution:\n", valid_df["bias"].value_counts(normalize=True))
print("Test bias distribution:\n", test_df["bias"].value_counts(normalize=True))
