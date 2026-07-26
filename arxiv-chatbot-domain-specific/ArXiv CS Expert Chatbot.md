# ArXiv CS Expert Chatbot: Local Setup Guide

This project is a domain-specific chatbot designed to serve as an expert in Computer Science research. It uses the arXiv dataset, Retrieval-Augmented Generation (RAG), and Streamlit to provide a user-friendly interface for scientific discussion, paper summarization, and concept visualization.

---

## 📋 Prerequisites

- **Python 3.11** installed on your system.
- **VS Code** (or any preferred code editor).
- **OpenAI API Key** (or a local LLM setup like Ollama).
- **arXiv Dataset**: The `kaggle.json` file you uploaded.

---

## 🚀 Step-by-Step Installation

### 1. Open the Project in VS Code
Open your terminal in VS Code (`Ctrl + ` ` or `Terminal > New Terminal`).

### 2. Set Up a Virtual Environment
It is highly recommended to use a virtual environment to avoid library conflicts.

**Windows (PowerShell):**
```powershell
python -m venv venv
# If you get an 'Execution Policy' error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Once the environment is activated (you should see `(venv)` in your terminal), run:
```bash
pip install -r requirements.txt
```

---

## 📊 Data Preparation

The chatbot needs a filtered version of the massive arXiv dataset to run efficiently.

1. Place your `kaggle.json` file in the root folder.
2. Run the filtering script:
```bash
python filter_arxiv.py
```
This will generate `arxiv_cs.jsonl`, which contains only Computer Science related papers.

---

## 🔑 Configuration

### Set OpenAI API Key
The chatbot uses OpenAI's GPT models by default. You must set your API key in the terminal.

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your_actual_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your_actual_api_key_here
```

**Mac / Linux:**
```bash
export OPENAI_API_KEY='your_actual_api_key_here'
```

---

## 🏃 Running the Chatbot

Start the Streamlit application with the following command:
```bash
streamlit run app.py
```
A new tab should automatically open in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

- `app.py`: The main Streamlit frontend and UI logic.
- `rag_backend.py`: The core RAG engine handling document indexing and retrieval.
- `filter_arxiv.py`: Utility script to process the raw arXiv dataset.
- `requirements.txt`: List of all Python dependencies.
- `faiss_index/`: (Generated automatically) Local vector database for fast searching.
- `arxiv_cs.jsonl`: (Generated automatically) The filtered dataset.

---

## 🛠 Troubleshooting

### "Scripts are disabled on this system"
If you cannot activate the virtual environment on Windows, run this command in PowerShell as Administrator (or just for your user):
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Missing Modules
If you see `ModuleNotFoundError`, ensure you have activated your virtual environment `(venv)` and re-run `pip install -r requirements.txt`.

### Using a Local LLM
To run without an API key, install **Ollama**, pull a model (e.g., `ollama pull llama3`), and update line 18 in `rag_backend.py`:
```python
# Replace:
self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
# With:
from langchain_ollama import OllamaLLM
self.llm = OllamaLLM(model="llama3")
```
