import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os


# -----------------------------------------------------------
# 1. Load dataset (LOCAL FILES)
# -----------------------------------------------------------

BASE_DIR = "data/"

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
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(
    train_df,
    test_size=0.1,         # 10% of train
    random_state=42,
    stratify=train_df["bias"]   # keep class distribution consistent
)

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
# 3. Decision Tree pipeline
# -----------------------------------------------------------

dt_clf = make_pipeline(
    TfidfVectorizer(
        max_features=10_000,    # Reduced for decision tree (less features = faster)
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english"
    ),
    DecisionTreeClassifier(
        max_depth=20,           # Limit depth to prevent overfitting
        min_samples_split=10,   # Minimum samples to split a node
        min_samples_leaf=5,     # Minimum samples in leaf node
        class_weight="balanced",
        random_state=42
    )
)

print("\nTraining Decision Tree...")
dt_clf.fit(X_train, y_train)


# -----------------------------------------------------------
# 4. Evaluation
# -----------------------------------------------------------

print("\n=== VALIDATION RESULTS ===")
valid_pred = dt_clf.predict(X_valid)
print("Accuracy:", accuracy_score(y_valid, valid_pred))
print(classification_report(y_valid, valid_pred))
print(confusion_matrix(y_valid, valid_pred))

print("\n=== TEST RESULTS ===")
test_pred = dt_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))
print(confusion_matrix(y_test, test_pred))

print("\nDone.")
print("Train bias distribution:\n", train_df["bias"].value_counts(normalize=True))
print("Valid bias distribution:\n", valid_df["bias"].value_counts(normalize=True))
print("Test bias distribution:\n", test_df["bias"].value_counts(normalize=True))


# -----------------------------------------------------------
# 5. Feature Importance (Bonus)
# -----------------------------------------------------------

print("\n=== FEATURE IMPORTANCE (Top 20) ===")
vectorizer = dt_clf.named_steps['tfidfvectorizer']
tree = dt_clf.named_steps['decisiontreeclassifier']

feature_names = vectorizer.get_feature_names_out()
importances = tree.feature_importances_

# Get top 20 most important features
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False).head(20)

print(importance_df.to_string(index=False))