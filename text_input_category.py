import streamlit as st
from transformers import LongformerTokenizer, LongformerForSequenceClassification
import torch

# --- CACHED MODEL AND TOKENIZER LOADING ---
@st.cache_resource
def load_model():
    """Load and cache the Longformer model and tokenizer."""
    checkpoint = "./allenlongformer"  # Local folder containing Longformer model
    tokenizer = LongformerTokenizer.from_pretrained(checkpoint)
    model = LongformerForSequenceClassification.from_pretrained(
        checkpoint, num_labels=10, device_map='auto', torch_dtype=torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Load the model to the correct device
    return tokenizer, model, device

# Load the model and tokenizer
tokenizer, classification_model, device = load_model()

# Predefined categories
categories = [
    "First Party Collection/Use",
    "Third Party Sharing/Collection",
    "User Choice/Control",
    "User Access, Edit, and Deletion",
    "Data Retention",
    "Data Security",
    "Policy Change",
    "Do Not Track",
    "International and Specific Audiences",
    "Other"
]

# --- CLASSIFICATION FUNCTION ---
def classify_text(input_text):
    """Classify the input text using the Longformer model."""
    # Tokenize and move inputs to the same device as the model
    inputs = tokenizer(input_text, truncation=True, padding=True, max_length=4096, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = classification_model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()

    return categories[predicted_label]

# --- STREAMLIT APP ---
def main():
    """Main function to run the Streamlit app."""

    st.title("Text Classification Tool")
    st.markdown("**Classify your input into predefined categories:**")

    # User input
    user_input = st.text_area("Enter text to classify:")

    if st.button("Classify Text"):
        if user_input.strip():
            # Perform classification
            category = classify_text(user_input)

            # Display result
            st.markdown(f"### Predicted Category: **{category}**")
        else:
            st.error("Please enter some text to classify.")

if __name__ == '__main__':
    main()
