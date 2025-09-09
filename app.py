# app.py (updated to fix mapping/probability issues)
import os
import json
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

from disease_info_agent import get_disease_info  # keep this if you use the LLM-guidance step

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = "fine_tuned_disease_model"
TOP_K_DEFAULT = 10
HIGH_CONF_THRESHOLD_DEFAULT = 0.3

st.set_page_config(page_title="AI Doctor", page_icon="🩺", layout="centered")
st.title("🩺 AI Doctor — Disease Prediction + Guidance")
st.info("Demo only — not medical advice. Always consult a healthcare professional.")

# ---------------------------
# Load model/tokenizer/mappings (cached)
# ---------------------------
@st.cache_resource
def load_model_artifacts(model_path: str):
    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Move to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Try to get mapping from model.config.id2label first (most reliable)
    id_to_label = {}
    cfg = getattr(model, "config", None)
    if cfg:
        # cfg.id2label may have int keys or string keys
        cfg_id2label = getattr(cfg, "id2label", None)
        if cfg_id2label:
            # Convert keys to int if needed
            id_to_label = {int(k): v for k, v in cfg_id2label.items()}
    # Fallback to label_mappings.json (convert JSON string keys to int)
    mappings_path = os.path.join(model_path, "label_mappings.json")
    if not id_to_label and os.path.exists(mappings_path):
        with open(mappings_path, "r") as f:
            mappings = json.load(f)
        id_map = mappings.get("id_to_label", {})
        # JSON keys are strings — convert to int
        id_to_label = {int(k): v for k, v in id_map.items()}

    # Final fallback: auto-generate generic labels if no mapping found
    num_labels = getattr(cfg, "num_labels", None)
    if num_labels is None:
        # try to infer from model params (not guaranteed)
        try:
            # create a dummy input to infer logits size? skip here
            num_labels = None
        except Exception:
            num_labels = None

    if not id_to_label:
        if num_labels:
            id_to_label = {i: f"Class_{i}" for i in range(num_labels)}
            st.warning("No label mapping found — using generic Class_i labels. This may mismatch your real labels.")
        else:
            raise FileNotFoundError("Could not find label mapping (model.config.id2label or label_mappings.json).")

    # Sanity check: warn if sizes mismatch
    if num_labels:
        if len(id_to_label) != num_labels:
            st.warning(
                f"Warning: number of labels in mapping ({len(id_to_label)}) "
                f"!= model.config.num_labels ({num_labels}). This can cause wrong name–probability pairing."
            )

    return tokenizer, model, id_to_label, device

# Load artifacts
tokenizer, model, id_to_label, device = load_model_artifacts(MODEL_PATH)

# ---------------------------
# UI controls
# ---------------------------
symptoms_input = st.text_area("Describe your symptoms:", height=180)
col1, col2 = st.columns([2, 1])
with col1:
    max_results = st.number_input("Show top K predictions", min_value=1, max_value=100, value=TOP_K_DEFAULT, step=1)
with col2:
    threshold = st.slider("High-confidence threshold", min_value=0.0, max_value=1.0, value=HIGH_CONF_THRESHOLD_DEFAULT, step=0.01)

debug_show_raw = st.checkbox("Show raw top-k (index, probability) for debugging", value=False)

predict_btn = st.button("Predict")

# ---------------------------
# Helper: get top-k predictions (robust)
# ---------------------------
def predict_topk(text: str, k: int = 10):
    # Tokenize (move inputs to same device as model)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.logits shape: (batch_size=1, num_labels)
    logits = outputs.logits.detach().cpu()[0]  # tensor shape [num_labels]
    probs = F.softmax(logits, dim=-1)

    # Get topk indexes and probs (descending)
    topk = torch.topk(probs, min(k, probs.shape[0]))
    topk_values = topk.values.tolist()
    topk_indices = topk.indices.tolist()

    results = []
    for prob, idx in zip(topk_values, topk_indices):
        cls_id = int(idx)
        name = id_to_label.get(cls_id, f"Class_{cls_id}")
        results.append({"id": cls_id, "disease": name, "prob": float(prob)})
    return results, list(zip(topk_indices, topk_values))  # return both mapped result and raw pairs

# ---------------------------
# Cache disease info calls (if using the LLM-guidance)
# ---------------------------
@st.cache_data(show_spinner=False)
def fetch_disease_info_cached(disease_prob_dict_json: str):
    disease_prob_dict = json.loads(disease_prob_dict_json)
    return get_disease_info(disease_prob_dict)

# ---------------------------
# Run
# ---------------------------
if predict_btn:
    if not symptoms_input.strip():
        st.warning("Please enter symptoms.")
    else:
        with st.spinner("Running inference..."):
            preds, raw_pairs = predict_topk(symptoms_input, k=int(max_results))

        # Show raw debug pairs if requested
        if debug_show_raw:
            st.write("Raw (index, probability) pairs (descending):")
            st.write(raw_pairs)

        # Display predictions table (sorted by prob desc already)
        df_preds = pd.DataFrame([{"Rank": i+1, "Disease": p["disease"], "Probability": f"{p['prob']*10:.4f}", "Class_ID": p["id"]} for i, p in enumerate(preds)])
        st.subheader("Top Predictions")
        st.table(df_preds)

        # High confidence filter
        high_conf = {p["disease"]: p["prob"] for p in preds if p["prob"]*10 > float(threshold)}
        if high_conf:
            st.subheader("High-confidence diseases (fetching guidance)")
            for d, p in high_conf.items():
                st.write(f"- **{d}** — probability: {p*10:.4f}")

            with st.spinner("Fetching web-backed guidance..."):
                try:
                    disease_info_json_str = json.dumps(high_conf, ensure_ascii=False)
                    guidance = fetch_disease_info_cached(disease_info_json_str)
                except Exception as e:
                    st.error(f"Failed to fetch disease guidance: {e}")
                    guidance = {}

            for disease, info in guidance.items():
                st.markdown(f"---\n### 🦠 {disease}")
                if isinstance(info, dict):
                    st.write(f"**Medication:** {info.get('medication', 'Not Found')}")
                    st.write(f"**When to take:** {info.get('when_to_take', 'Not Found')}")
                    st.write(f"**Prevention:** {info.get('prevention', 'Not Found')}")
                    st.write(f"**Other:** {info.get('other', 'Not Found')}")
                    sources = info.get("sources", []) or []
                    if sources:
                        st.write("**Sources:**")
                        for s in sources:
                            st.write(s)
                    if info.get("raw"):
                        with st.expander("Raw LLM output"):
                            st.write(info["raw"])
                else:
                    st.write(info)
        else:
            st.info(f"No disease exceeded the threshold of {threshold:.2f}. Lower the threshold to get more guidance.")
