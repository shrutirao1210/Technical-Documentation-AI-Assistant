# Technical Documentation AI Assistant

> An AI-powered tool that reads your Python codebase, understands it semantically, and generates professional technical documentation — built entirely on Kaggle using RAG + Llama 3 + FAISS + Gradio.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Llama3](https://img.shields.io/badge/LLM-Llama%203%208B-purple.svg)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-orange.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF.svg)

---

## What It Does

Upload your Python project files and the assistant will:
- **Parse** every function, class, and method using Python's AST
- **Embed** your code semantically using SentenceTransformers
- **Retrieve** the most relevant code chunks for any question (RAG)
- **Generate** professional Markdown documentation using Llama 3
- **Auto-write docstrings** for undocumented functions
- **Export** a complete `AUTO_README.md` for your project

---

## Architecture

```
Your .py Files
     │
     ▼
[1] AST Parser          ← Extracts functions, classes, docstrings
     │
     ▼
[2] Embedding Model     ← all-MiniLM-L6-v2 (SentenceTransformers)
     │
     ▼
[3] FAISS Vector Store  ← Stores & indexes semantic embeddings
     │
  [User Query]
     │
     ▼
[4] RAG Retriever       ← Finds top-k relevant code chunks
     │
     ▼
[5] Llama 3 8B Instruct ← Generates professional documentation
     │
     ▼
[6] Gradio UI           ← Interactive chat + file upload interface
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Meta Llama 3 8B Instruct (4-bit quantized) |
| **Embeddings** | `all-MiniLM-L6-v2` via SentenceTransformers |
| **Vector Store** | FAISS (IndexFlatIP with cosine similarity) |
| **Code Parsing** | Python `ast` module |
| **UI** | Gradio (3-tab interface) |
| **Platform** | Kaggle (T4 GPU) |
| **Quantization** | BitsAndBytes (NF4, 4-bit) |

---

## How to Run on Kaggle

### Step 1: Open the Notebook
- Go to [kaggle.com](https://kaggle.com) → **Code** → **New Notebook**
- Click **File → Import Notebook** and upload `final-ai-tech-doc.ipynb`

### Step 2: Enable GPU
- Right sidebar → **Settings** → **Accelerator** → **GPU T4 x2**
- Make sure **Internet** is ON

### Step 3: Set Up HuggingFace Token (for Llama 3)
- Accept Meta's license at [huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → create a **Read** token
- In Kaggle: **Add-ons → Secrets** → add secret named `HF_TOKEN`

### Step 4: Run All
- Click **Run All** and wait ~5 minutes for setup
- A **Gradio public URL** will appear — click it to open the UI

---

## UI Features

### Tab 1 — 📤 Upload Your Code
- Upload your own `.py` files (multiple at once)
- Click **Upload & Index My Code** to rebuild the vector index
- Or reset to the built-in sample project anytime

### Tab 2 — 💬 Ask About Code
- Ask natural language questions about your codebase
- Get structured Markdown documentation as answers
- Sources (file + line number) shown for every answer

### Tab 3 — 🔍 Search Code Chunks
- Explore raw FAISS semantic search results
- See similarity scores for each retrieved chunk

### Tab 4 — 📂 Indexed Codebase
- View all currently indexed functions, classes, and methods

---

## Auto Documentation Features

### Auto-generate docstrings
The notebook includes a cell that:
1. Reads each uploaded `.py` file
2. Finds all functions/classes missing docstrings
3. Asks Llama 3 to write a one-line docstring for each
4. Surgically inserts them without modifying anything else
5. Validates Python syntax before saving

### Auto-generate README
After docstrings are written, one cell generates a complete `AUTO_README.md` covering every file, class, function, and method in your project.

---

## Repository Structure

```text
Technical-Documentation-AI-Assistant/
├── autoreadme_demo_inputs/          # Sample uploaded Python files used for AutoREADME generation
│   ├── model.py                     # Example Python file
│   └── train.py                     # Example Python file
│
├── gradio frontend screenshots/            # Gradio frontend screenshots
│   ├── S1.png                       # Home page
│   ├── S2.png                       # Project upload interface
│   ├── S3.png                       # Documentation generation output
│   └── S4.png                       # README generation output
│
├── AUTO_README.md                   # Auto-generated README for the uploaded project
├── final-ai-tech-doc.ipynb          # Main Kaggle notebook containing the complete pipeline
└── README.md                        # Project documentation
```

---

## Requirements

All installed automatically inside the notebook:

```
transformers>=4.40
accelerate
bitsandbytes
sentence-transformers
faiss-cpu
gradio
langchain
huggingface_hub
```

---

## Example Questions You Can Ask

Once your code is uploaded, try asking:

- *"How does the login function work?"*
- *"Explain the DatabaseManager class"*
- *"What does the require_auth decorator do?"*
- *"How is the password hashed before storing?"*
- *"What are the training phases in this model?"*
- *"How does the emotion gating mechanism work?"*

---

## Future Improvements

- Support for JavaScript, Java, and Go files via `tree-sitter`
- Fine-tune on (code, documentation) pairs using LoRA
- Generate Mermaid.js architecture diagrams automatically
- Persistent FAISS index across sessions
- Deploy to HuggingFace Spaces for permanent hosting

---


