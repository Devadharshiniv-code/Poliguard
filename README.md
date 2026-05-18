# PoliGuard 🛡️
### Privacy Policy Summarization & Classification using Fine-Tuned Transformers

> Upload any privacy policy. PoliGuard classifies every clause into 10 legal categories and generates a plain-English summary — so users actually understand what they're agreeing to.

---

## What It Does

Most people click "I Agree" without reading privacy policies. They're long, dense, and written in legal language. PoliGuard fixes that.

Upload a privacy policy document (PDF, TXT, or DOCX) and PoliGuard will:
1. **Classify** every line into one of 10 predefined legal categories (Data Security, User Rights, Third-Party Sharing, etc.)
2. **Summarise** the relevant clauses in plain English using an abstractive NLP model

---

## Demo

| Upload a policy | Select a category | Get a plain-English summary |
|---|---|---|
| ![upload](Result%20images/layout.png) | ![category](Result%20images/category%20index.png) | ![summary](Result%20images/suma-1.png) |

---

## Architecture

Two fine-tuned transformer models work in sequence:

```
Input Document (PDF / TXT / DOCX)
        │
        ▼
┌───────────────────────┐
│  Longformer Classifier │  ← fine-tuned on OPP-115
│  (allenai/longformer-  │     classifies each line into
│   base-4096)           │     1 of 10 privacy categories
└───────────────────────┘
        │
        ▼  (lines matching selected category)
┌───────────────────────┐
│  BART Summariser       │  ← fine-tuned on OPP-115
│  (facebook/bart-base)  │     generates abstractive
│                        │     plain-English summary
└───────────────────────┘
        │
        ▼
Plain-English Summary
```

---

## Models & Results

### Classification — Longformer (`allenai/longformer-base-4096`)

| Metric | Score |
|--------|-------|
| Accuracy | **81.84%** |
| Precision | 0.99 |
| Recall | 0.89 |
| F1-Score | 0.89 |

Fine-tuned on the OPP-115 dataset with 10 privacy policy categories. Stratified 80/10/10 train/val/test split.

### Summarisation — BART (`facebook/bart-base`)

| Metric | Score |
|--------|-------|
| ROUGE-1 | **0.61** |
| ROUGE-2 | **0.51** |
| ROUGE-L | **0.58** |

Fine-tuned on 22,765 (text, summary) pairs derived from OPP-115 sanitised policies and annotated pretty-print summaries.

---

## Privacy Policy Categories

The classifier maps text into the 10 OPP-115 categories:

| # | Category |
|---|----------|
| 1 | First Party Collection/Use |
| 2 | Third Party Sharing/Collection |
| 3 | User Choice/Control |
| 4 | User Access, Edit, and Deletion |
| 5 | Data Retention |
| 6 | Data Security |
| 7 | Policy Change |
| 8 | Do Not Track |
| 9 | International and Specific Audiences |
| 10 | Other |

---

## Dataset

**OPP-115** (Online Privacy Policies, 115 documents)  
Developed by Carnegie Mellon University's Usable Privacy Policy Project.

- 115 real-world privacy policies manually annotated by legal experts
- 266,713 words across all documents
- 10 annotators (3 per document)
- Used to create: 3,792 classification samples + 22,765 summarisation pairs

> Wilson, S. et al. "The Creation and Analysis of a Website Privacy Policy Corpus." ACL 2016.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Classification model | `allenai/longformer-base-4096` (HuggingFace) |
| Summarisation model | `facebook/bart-base` (HuggingFace) |
| Deep learning framework | PyTorch |
| Web app | Streamlit |
| Training environment | Google Colab (GPU) |
| Evaluation | ROUGE Score, Flesch-Kincaid Readability |

---

## Installation & Usage

### Prerequisites
- Python 3.9+
- ~3GB disk space for model weights

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/poliguard.git
cd poliguard
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add model weights
Place your fine-tuned model folders in the project root:
```
poliguard/
├── allenlongformer/    ← Longformer classification model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files...
├── bart/               ← BART summarisation model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files...
└── app.py
```

> Model weights are not included in this repo due to size (~2.2GB total).  
> Contact the author or see the Releases section to download them.

### 5. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Scripts

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app — full classify + summarise pipeline |
| `document.py` | Document upload + category-based classification only |
| `text-input-category.py` | Paste raw text and classify into a category |

---

## Project Structure

```
poliguard/
├── app.py                          # Main application
├── document.py                     # Classification-only app
├── text-input-category.py          # Text input classifier
├── requirements.txt
├── README.md
├── .gitignore
├── Result images/                  # Screenshots of the working app
│   ├── layout.png
│   ├── category index.png
│   ├── suma-1.png
│   └── ...
├── allenlongformer/                # Fine-tuned Longformer (gitignored)
└── bart/                           # Fine-tuned BART (gitignored)
```

---

## Research Context

This project was developed as an MSc Computing dissertation at **Northumbria University** (2025), supervised by Dr. Abdulrahman Salih.

**Research questions addressed:**
1. What are the key challenges in fine-tuning models for domain-specific text classification and summarisation?
2. How does fine-tuning on privacy policy data affect classification and summarisation performance without distorting original context?
3. What strategies handle documents that exceed model token limits?
4. How can model performance be comprehensively evaluated for summarisation tasks?
5. What are the legal and ethical challenges of using AI to summarise privacy policy documents?

---

## Future Work

- 🎨 **Colour-coded output** — visual risk flagging per category
- 🔊 **Voice AI** — text-to-speech summary for accessibility
- 💬 **Chatbot** — ask questions about a specific policy
- 🌐 **Browser extension** — summarise any privacy policy in real time
- 📊 **Risk scoring** — rate how user-friendly a policy is

---

## Author

**Devadharshini Venkatesan**  
MSc Computing (Distinction) — Northumbria University  
CompTIA Security+  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [Email](mailto:devaviji29@gmail.com)

---

## Citation

If you use this work, please cite:

```
Venkatesan, D. (2025). PoliGuard: Privacy Policy Summarization through 
Transformer-Based Fine-Tuning. MSc Dissertation, Northumbria University.
```

---

## Acknowledgements

- OPP-115 dataset — Carnegie Mellon University Usable Privacy Policy Project
- HuggingFace Transformers library
- Supervisor: Dr. Abdulrahman Salih, Northumbria University
