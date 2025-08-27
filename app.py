import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import pandas as pd

# ----------------------------
# Load model, tokenizer, and label mappings
# ----------------------------
MODEL_PATH = "fine_tuned_disease_model"

@st.cache_resource  # cache model so it doesn’t reload every run
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    with open(f"{MODEL_PATH}/label_mappings.json", "r") as f:
        mappings = json.load(f)
    id_to_label = {int(k): v for k, v in mappings["id_to_label"].items()}
    return tokenizer, model, id_to_label

tokenizer, model, id_to_label = load_model_and_tokenizer()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Doctor", page_icon="🩺", layout="centered")

st.title("🩺 AI Doctor – Disease Prediction")
st.write("Enter your symptoms and get the top 10 predicted diseases with probabilities.")

# Text input
symptoms_input = st.text_area("Describe your symptoms:", height=150)

if st.button("Predict"):
    if symptoms_input.strip() == "":
        st.warning("⚠️ Please enter some symptoms first.")
    else:
        # Tokenize input
        inputs = tokenizer(symptoms_input, return_tensors="pt")

        # Run model
        with torch.no_grad():
            outputs = model(**inputs)

        # Convert logits → probabilities
        probs = F.softmax(outputs.logits, dim=-1)[0]

        # Get top 10 predictions
        topk = torch.topk(probs, 10)

        # Create a dataframe for display
        results = []
        for prob, idx in zip(topk.values, topk.indices):
            disease_name = id_to_label[int(idx)]
            results.append({"Disease": disease_name, "Probability": f"{prob.item():.4f}"})

        df = pd.DataFrame(results)
        st.subheader("Top 10 Predicted Diseases")
        st.table(df)
