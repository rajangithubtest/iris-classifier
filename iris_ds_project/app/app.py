import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Iris Species Classifier", page_icon="🌸")

st.title("🌸 Iris Species Classifier")
st.caption("Simple end-to-end Data Science project demo")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "iris.csv"

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    else:
        st.error("Model file not found. Please run `python scripts/train.py` first.")
        st.stop()

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    else:
        st.error("Dataset not found.")
        st.stop()

model = load_model()
df = load_data()

st.subheader("Input features")
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal length", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
    sepal_width  = st.number_input("Sepal width",  min_value=0.0, max_value=10.0, value=3.5, step=0.1)
with col2:
    petal_length = st.number_input("Petal length", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
    petal_width  = st.number_input("Petal width",  min_value=0.0, max_value=10.0, value=0.2, step=0.1)

if st.button("Predict"):
    X = pd.DataFrame([{
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    labels = model.classes_
    st.success(f"Predicted species: **{pred}**")
    st.write("Class probabilities:")
    st.write(pd.DataFrame({"species": labels, "probability": proba}).sort_values("probability", ascending=False).reset_index(drop=True))

with st.expander("See sample of training data"):
    st.dataframe(df.head(10))
