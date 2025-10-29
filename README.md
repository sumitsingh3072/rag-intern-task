# Chat with your PDF — RAG intern task

This repository contains a small Retrieval-Augmented Generation (RAG) demo for chatting with PDF documents using LangChain, a hybrid retriever (BM25 + vector search), Hugging Face BGE embeddings, and Groq's ChatGroq LLM. The app exposes a Streamlit UI (`app.py`) and an example notebook (`rag_intern_task.ipynb`).

## Contents

- `app.py` — Streamlit application that: uploads/downloads a PDF, splits it, builds embeddings, sets up an ensemble retriever (BM25 + Chroma vector store), and creates a conversational QA chain with memory.
- `rag_intern_task.ipynb` — Example notebook with similar end-to-end steps for Colab/local experimentation.
- `requirements.txt` — Python dependency list used by the project.
- `.gitignore` — Ignored files (e.g., `venv`, `.env`).

## Short summary

The app loads a PDF (uploaded or via URL), splits it into chunks, embeds chunks using the BAAI BGE model, stores them in Chroma, and combines a BM25 retriever with a vector retriever via an `EnsembleRetriever`. Queries entered in the Streamlit UI are answered by a `ConversationalRetrievalChain` backed by Groq's `ChatGroq` LLM and a session memory buffer.

## Quickstart (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file at the repo root (or set environment variables in your system). Put your Groq API key there:

```
GROQ_API_KEY=your_groq_api_key_here
```

Note: `.env` is included in `.gitignore` by default to avoid accidentally committing secrets.

4. Run the Streamlit app:

```powershell
streamlit run app.py
```

5. Open the URL shown by Streamlit (typically http://localhost:8501) and use the sidebar to upload a PDF or provide a PDF URL. Click "Process PDF" to build the retriever and embeddings. Once processed, ask questions in the chat input.

## Files of interest and behavior

- `app.py` — main Streamlit application. Key behaviors:
  - Loads and splits PDFs using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
  - Creates embeddings via `HuggingFaceEmbeddings` (BGE model `BAAI/bge-small-en-v1.5`).
  - Stores embeddings to a Chroma vectorstore and creates a vector retriever.
  - Initializes a `BM25Retriever` for keyword-style fuzzy retrieval.
  - Combines them with `EnsembleRetriever` (equal weights) and then a `ConversationalRetrievalChain` with `ConversationBufferMemory`.

- `rag_intern_task.ipynb` — notebook containing the same logic (useful for Colab or interactive experiments). It also demonstrates how to run the pipeline from the command line or interactive session.

## Environment notes & troubleshooting

- GROQ API Key: The Streamlit app expects a Groq API key in `GROQ_API_KEY`. You can set it in a `.env` file or your environment. The app uses `load_dotenv()` — ensure `.env` is readable.

- GPU / CPU: The notebook and app attempt to use a model device hint for embeddings (the notebook suggests `cuda` for GPUs). If you don't have a GPU, change the model device to `cpu` or ensure the embeddings call uses a CPU-compatible config. The environment must have compatible libraries (PyTorch, CUDA) if you intend to run on GPU.

- Common errors:
  - "Groq API key not found": set `GROQ_API_KEY` in `.env` or environment variables.
  - "Error loading PDF": ensure file exists or the provided URL is reachable. Large PDFs may require more memory.
  - Dependency/installation issues: some packages (embedding models, chromadb) may require system dependencies. Check package documentation and install wheels compatible with your platform.

## Development notes & suggestions

- The `app.py` code uses Streamlit caching (`@st.cache_resource`) to keep models in memory across runs.
- The current device check in `app.py` uses a simple string test which is not robust; a better approach is to use PyTorch's `torch.cuda.is_available()` or the embedding library's recommended device detection. Example fix:

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model_kwargs = {"device": device}
```

- If you want persistent vector storage across restarts, configure Chroma to use a persistent directory (instead of ephemeral in-memory). See `chromadb` docs.

## Notebook usage

- `rag_intern_task.ipynb` is configured to run on Colab (it includes a Colab badge). It demonstrates the same pipeline step-by-step: download or upload a PDF, split, embed, setup retriever, create the conversational chain, then chat.

## Reproducing / testing locally

1. Follow Quickstart steps above to create venv and install dependencies.
2. Run `streamlit run app.py`.
3. Upload a small PDF (1–5 pages) first to confirm the pipeline works.

If something breaks, capture Streamlit error messages and logs, check `requirements.txt` for package versions, and confirm your `GROQ_API_KEY` is valid.

## Dependencies

See `requirements.txt` for the project's Python dependencies. Key packages include:

- streamlit — web UI
- langchain_community, langchain-huggingface, langchain-groq — LangChain integrations and Groq LLM
- chromadb — vector store
- pypdf / PyPDF2 / PyPDF — PDF loading
- rank_bm25 — BM25 implementation

