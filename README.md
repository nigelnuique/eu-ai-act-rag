# 🚀 EU AI Act RAG System

A Retrieval-Augmented Generation (RAG) system for querying the European Union's Artificial Intelligence Act using advanced AI search technology.

## 🎯 Features

- **Hybrid Search**: Semantic + keyword matching for superior retrieval
- **Smart Chunking**: Structural awareness of legal document organization  
- **Query Expansion**: Automatic legal terminology expansion
- **Smart Reranking**: Content-aware result prioritization

## 🚀 Setup & Run

### Automated Setup (Recommended)
```bash
python setup.py
```

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create vector database
python retriever.py

# Run application
streamlit run app.py
```

## 📋 Requirements

- Python 3.8+
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- Internet connection for document download

## 🔧 Usage

1. Run the setup script or manual setup steps
2. Access the app at `http://localhost:8501`
3. Ask questions about the EU AI Act:
   - "What are high-risk AI systems?"
   - "How is the AI Act enforced?"
   - "What are the obligations for AI providers?"

## 📁 Core Files

- `app.py` - Main Streamlit application
- `setup.py` - Automated setup script
- `retriever.py` - RAG retrieval system
- `structural_chunker.py` - Document chunking logic
- `requirements.txt` - Python dependencies
- `faiss_vectordb/` - Vector database storage

The system automatically downloads the EU AI Act from the official EUR-Lex website and creates an optimized vector database for intelligent question answering. 