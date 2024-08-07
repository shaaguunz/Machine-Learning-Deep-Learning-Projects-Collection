import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the model and tokenizer
model_path = "./saved_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Function to classify text
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions.item()

# Streamlit app
st.title("Text Classification with DistilBERT")

text_input = st.text_area("Enter text to classify")

if st.button("Classify"):
    if text_input:
        prediction = classify_text(text_input)
        st.write(f"Prediction: {prediction}")
    else:
        st.write("Please enter some text to classify.")
