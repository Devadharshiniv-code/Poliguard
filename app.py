import streamlit as st
from transformers import LongformerTokenizer, LongformerForSequenceClassification, BartTokenizer, BartForConditionalGeneration
import torch

# --- CACHED MODEL AND TOKENIZER LOADING ---
@st.cache_resource
def load_models():
    """Load and cache the Longformer model, tokenizer, and summarization model."""
    # Load classification model and tokenizer
    classification_checkpoint = "./allenlongformer"
    classification_tokenizer = LongformerTokenizer.from_pretrained(classification_checkpoint)
    classification_model = LongformerForSequenceClassification.from_pretrained(
        classification_checkpoint, num_labels=10, device_map="auto", torch_dtype=torch.float32
    )
    
    # Load summarization model and tokenizer
    summarization_checkpoint = "./bart"
    summarization_tokenizer = BartTokenizer.from_pretrained(summarization_checkpoint)
    summarization_model = BartForConditionalGeneration.from_pretrained(summarization_checkpoint)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classification_model.to(device)
    summarization_model.to(device)

    return classification_tokenizer, classification_model, summarization_tokenizer, summarization_model, device

# Load the models and tokenizers
classification_tokenizer, classification_model, summarization_tokenizer, summarization_model, device = load_models()

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

# --- FILE PREPROCESSING FUNCTION ---
def preprocess_document(uploaded_file):
    """Read and preprocess the uploaded document."""
    file_type = uploaded_file.type

    if file_type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    elif file_type == "application/pdf":
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = "".join(page.extract_text() for page in pdf_reader.pages)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        import docx
        doc = docx.Document(uploaded_file)
        text = "\n".join(para.text for para in doc.paragraphs)
    else:
        text = ""

    return text.split("\n")  # Split into lines for classification

# --- CLASSIFICATION FUNCTION ---
def classify_line(line):
    """Classify a single line using the Longformer model."""
    inputs = classification_tokenizer(line, truncation=True, padding=True, max_length=4096, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = classification_model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()

    return predicted_label

# --- SUMMARIZATION FUNCTION ---
def summarize_text(text_lines):
    """Summarize extracted lines using the summarization model."""
    combined_text = " ".join(text_lines)  # Combine lines into a single input
    inputs = summarization_tokenizer(
        combined_text, max_length=1024, truncation=True, return_tensors="pt", padding="max_length"
    ).to(device)

    with torch.no_grad():
        summary_ids = summarization_model.generate(
            inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True
        )
    return summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- STREAMLIT APP ---
def main():
    """Main function to run the Streamlit app."""

    st.title("Poliguard")
    st.markdown("**Upload a document, classify its content, and summarize related lines:**")

    # File upload
    uploaded_file = st.file_uploader("Upload a document:", type=["txt", "pdf", "docx"])

    if uploaded_file:
        document_lines = preprocess_document(uploaded_file)

        if not document_lines:
            st.error("Unable to extract text from the uploaded document.")
            return

        # Category selection
        selected_category = st.selectbox("Select a category:", categories)

        if selected_category:
            category_index = categories.index(selected_category)

            # Extract lines matching the selected category
            related_lines = [
                line for line in document_lines if classify_line(line) == category_index
            ]

            # Display the results
            st.markdown(f"### Lines related to **{selected_category}:**")

            if related_lines:
                for line in related_lines:
                    st.write(f"- {line}")

                # Add summarization button
                if st.button("Summarize"):
                    summary = summarize_text(related_lines)
                    st.markdown("### Summary of Selected Category:")
                    st.write(summary)
            else:
                st.write("No related lines found for the selected category.")

if __name__ == "__main__":
    main()

