# GDG Document-to-Report Assistant

This project is a Streamlit app that:

- accepts one **structure document** (`.txt`, `.md`, `.pdf`, `.docx`)
- accepts multiple **supporting documents** (primarily PDFs)
- runs a local **RAG pipeline** to generate a structured report
- lets you edit the full report in one markdown textbox
- provides a chat workspace for report Q&A and edits
- includes a live **Developer Observability** view (tool flow, logs, run insights)

The app uses **local Ollama models** for both generation and embeddings.

---

## Tech Stack

- Python + Streamlit
- Ollama (`llama3.2:latest`, `nomic-embed-text`)
- LangChain / LangChain Ollama
- PDF/DOCX parsing (`pypdf`, `PyPDF2`, `python-docx`)

---

## Prerequisites

- Python 3.10+ (recommended)
- pip
- Ollama installed and running (**required**)

> This project requires local Ollama. The app will not run correctly without Ollama and the required models.

If you do **not** have Ollama, follow the installation steps below.

---

## Setup

From the project root:

```bash
pip install -r requirements.txt
pip install streamlit
```

> `streamlit` is installed explicitly above to avoid missing-command issues.

---

## If You Don’t Have Ollama

1. Install Ollama from [https://ollama.com/download](https://ollama.com/download)
2. Start Ollama (desktop app or service)
3. Pull required models:

```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text
```

4. Verify:

```bash
ollama list
```

You should see both models in the list.

---

## Run the App

From project root:

```bash
streamlit run app.py
```

If `streamlit` command is not found:

```bash
python -m streamlit run app.py
```

---

## How to Use

1. Upload exactly one **main structure file** in the **Main File** uploader.
   - The main file is the skeleton/template for report generation.
   - Initial structure is extracted from this main file.
2. In **Supporting Files**, upload files from:
   - `test_case_supporting_documents/`
   - Example files:
     - `test_case_supporting_documents/Sustainable-Agriculture.pdf`
     - `test_case_supporting_documents/sustainability-07-07833.pdf`
3. Click **Generate Report**
4. Work in **Report Workspace**:
   - edit full markdown report directly
   - use chat for questions/edits
5. Open/monitor **Developer Observability** for live tool flow and run insights

---

## Key Behavior

- If the uploaded structure is weak for retrieval, the app can auto-propose a new structure.
- Report generation is optimized for speed (turbo mode enabled).
- Logs are stored per run and also as global event streams:
  - `logs/runs/<run_id>.json`
  - `logs/streams/<run_id>.log`
  - `logs/tool_events.jsonl`

---

## Troubleshooting

### 1) Torch/Streamlit watcher error

If you see:

`RuntimeError: Tried to instantiate class '__path__._path'...`

This project already disables the Streamlit watcher via config. Restart Streamlit after pulling latest files.

### 2) Model not found

Run:

```bash
ollama list
```

If missing, pull again:

```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text
```

### 3) Slow first run

First run builds embeddings/cache. Later runs are faster due to cached chunk index data.

