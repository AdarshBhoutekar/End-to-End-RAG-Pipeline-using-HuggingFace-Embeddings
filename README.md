# End-to-End-RAG-Pipeline-using-HuggingFace-Embeddings
A Retrieval-Augmented Generation (RAG) pipeline for document-based question answering using HuggingFace embeddings and FAISS for efficient semantic search.
---

## Overview
This project implements a RAG pipeline that:
- Splits documents into chunks  
- Generates embeddings using HuggingFace  
- Retrieves relevant context using similarity search  
- Generates answers using a language model  

---

## Tech Stack
- **Language:** Python  
- **Embeddings:** HuggingFace Sentence Transformers  
- **Data Processing:** NumPy, Pandas  
- **Similarity Search:** Scikit-learn (Cosine Similarity)  
---
```
RAG-Project/
│
├── RAG_pipeline.ipynb        # Main notebook
├── README.md                # Project description
│
├── data/                    # Input files (PDFs / text)
│   └── sample.txt

```
---

## 🔮 Future Work
- FAISS / Chroma integration  
- Better retrieval & prompt tuning  
- UI (Streamlit/Gradio)  
