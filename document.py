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
    inputs = tokenizer(line, truncation=True, padding=True, max_length=4096, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = classification_model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()

    return predicted_label

# --- STREAMLIT APP ---
def main():
    """Main function to run the Streamlit app."""

    st.title("Document Classification by Category")
    st.markdown("**Upload a document and select a category to extract related lines:**")

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
            else:
                st.write("No related lines found for the selected category.")

if __name__ == '__main__':
    main()

