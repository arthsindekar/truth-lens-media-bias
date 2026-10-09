import joblib
import streamlit as st

st.set_page_config(page_title="Media Bias Detector", page_icon="📰")

LABELS = {0: "Left", 1: "Right", 2: "Center"}  # bias_text mapping from data/README


@st.cache_resource
def load_model():
    return joblib.load("bias_model.joblib")


st.title("📰 Media Bias Detector")
st.write("Paste a news article and the model will estimate its political leaning (left / center / right).")

title = st.text_input("Title (optional)")
content = st.text_area("Article text", height=300)

if st.button("Analyze", type="primary"):
    text = f"{title} {content}".strip()
    if len(text.split()) < 20:
        st.warning("Please enter at least a few sentences for a meaningful prediction.")
    else:
        model = load_model()
        probs = model.predict_proba([text])[0]
        scores = dict(zip(model.classes_, probs))
        best = max(scores, key=scores.get)
        st.subheader(f"Prediction: {LABELS.get(best, best)}")
        st.bar_chart({LABELS.get(k, k): v for k, v in scores.items()})

st.caption("Model: TF-IDF + Logistic Regression. Predictions reflect patterns in the training data, not ground truth.")
