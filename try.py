import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load mappings
with open("fine_tuned_disease_model/label_mappings.json", "r") as f:
    mappings = json.load(f)

id_to_label = {int(k): v for k, v in mappings["id_to_label"].items()}
label_to_id = {k: int(v) for k, v in mappings["label_to_id"].items()}

# Load model + tokenizer
model_path = "fine_tuned_disease_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Example inference
inputs = tokenizer("""Chest pain when you breathe or cough
Confusion or changes in mental awareness (in adults age 65 and older)
Cough, which may produce phlegm
Fatigue
Fever, sweating and shaking chills
Lower than normal body temperature (in adults older than age 65 and people with weak immune systems)
Nausea, vomiting or diarrhea
Shortness of breath""", return_tensors="pt")

outputs = model(**inputs)

# Get probabilities
logits = outputs.logits
probs = F.softmax(logits, dim=-1)[0]

# Get top 10 predictions
topk = torch.topk(probs, 10)

print("\nTop 10 predicted diseases:")
for i, (prob, idx) in enumerate(zip(topk.values, topk.indices)):
    disease_name = id_to_label[int(idx)]
    print(f"{i+1}. {disease_name}: {prob.item():.4f}")
