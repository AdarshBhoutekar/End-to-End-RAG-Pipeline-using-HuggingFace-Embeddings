# End-to-End RAG Pipeline (HuggingFace & ChromaDB)

A comprehensive Retrieval-Augmented Generation (RAG) pipeline for document-based question answering. This project uses HuggingFace embeddings for semantic representation and ChromaDB for efficient vector storage and retrieval. It integrates with state-of-the-art LLMs via the Groq and Google Gemini APIs to generate accurate, context-aware answers.

---

## Overview
This project implements a complete RAG pipeline that:
- **Ingests Documents**: Loads PDF and text documents from a local directory.
- **Splits Documents**: Chunks documents into manageable pieces using LangChain's `RecursiveCharacterTextSplitter`.
- **Generates Embeddings**: Uses HuggingFace's `SentenceTransformer` (`all-MiniLM-L6-v2`) to convert text chunks into dense vector embeddings.
- **Stores Vectors**: Persists embeddings locally using `ChromaDB` for efficient semantic search.
- **Retrieves Context**: Fetches the most relevant document chunks based on user queries using cosine similarity.
- **Generates Answers**: Synthesizes the retrieved context to generate final answers using Language Models like **Qwen (via Groq API)** and **Gemini 2.5 Flash (via Google GenAI API)**.

---

## What We Built (Core Implementation)
- **Built an End-to-End RAG System**: We successfully transitioned raw documents into actionable knowledge by implementing proper document loaders and recursive text chunking.
- **Integrated Multiple External LLMs**: To generate the final answers, **we utilized both the Groq API and the Google Gemini API**. By plugging directly into their powerful LLMs (specifically Qwen running on Groq, and Gemini 2.5 Flash), our pipeline synthesizes highly accurate answers natively based on the retrieved context!
- **Switched to ChromaDB**: We integrated ChromaDB to securely and persistently store our HuggingFace dense vector embeddings locally.

---

## Tech Stack
- **Language:** Python
- **Orchestration:** LangChain (`langchain`, `langchain-core`, `langchain-community`)
- **Embeddings:** HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Store:** ChromaDB
- **LLMs:** Groq API (`qwen/qwen3-32b`) and Google Generative AI (`gemini-2.5-flash`)
- **Document Loaders:** PyPDFLoader, TextLoader

---
## Project Structure
```text
RAG-Project/
│
├── RAG_pipeline.ipynb       # Main Jupyter Notebook containing the full pipeline code
├── README.md                # Project documentation
│
├── Data/                    # Input files (PDFs / text) and persistent vector store
│   ├── vector_store/        # Stored ChromaDB collections
│   ├── .pdf files           # Research papers and other PDF documents
│   └── .txt files           # Text documents
│
└── .env                     # Environment variables storing API keys (Groq, Gemini)
```
---

## Setup & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/AdarshBhoutekar/End-to-End-RAG-Pipeline-using-HuggingFace-Embeddings.git
   cd End-to-End-RAG-Pipeline-using-HuggingFace-Embeddings
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your API keys exactly like this:
   ```env
   API_KEY_GROQ = "your_groq_api_key_here"
   API_KEY_GEMINI = "your_gemini_api_key_here"
   ```
3. Place your PDF and text documents inside the `Data/` folder.
4. Run the cells in `RAG_pipeline.ipynb` to execute the ingestion, retrieval, and generation processes.

---

## Future Work
- Add support for more document types (e.g., Word, HTML).
- Implement advanced retrieval techniques (e.g., Multi-Query Retrieval, Hybrid Search).
- Develop an interactive UI using Streamlit or Gradio.
- Add robust evaluation metrics for RAG generated responses.
