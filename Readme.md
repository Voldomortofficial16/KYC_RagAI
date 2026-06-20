# Banking KYC Assistant using RAG

## Overview
Built a Retrieval-Augmented Generation (RAG) chatbot for banking KYC/AML policy documents.

## Features
- PDF Upload
- Semantic Search
- FAISS Vector Database
- Sentence Transformer Embeddings
- FLAN-T5 Answer Generation
- Streamlit Interface

## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- Sentence Transformers
- FLAN-T5
- PyMuPDF

## Architecture

PDF
↓
Chunking
↓
Embeddings
↓
FAISS
↓
Retriever
↓
FLAN-T5
↓
Answer

## Run Locally

pip install -r requirements.txt

streamlit run app.py
