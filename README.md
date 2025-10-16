# RAG Chatbot - Retrieval-Augmented Generation System

A robust, production-ready RAG (Retrieval-Augmented Generation) chatbot built with LangChain, ChromaDB, and Ollama. This system can handle 300+ documents with hybrid retrieval, advanced re-ranking, and comprehensive safety guardrails.

## 🌟 Features

- **Hybrid Retrieval**: Combines dense semantic search (Nomic Embed-Text) with sparse BM25 retrieval
- **Reciprocal Rank Fusion (RRF)**: Advanced re-ranking algorithm for optimal document selection
- **Safety Guardrails**: Input moderation and output validation to prevent misuse
- **Conversational Memory**: Maintains context across multiple turns of conversation
- **Local LLM**: Runs entirely on your machine using Ollama (Llama 3 8B)
- **Persistent Storage**: ChromaDB vector database with persistent storage
- **Modern UI**: Clean, responsive Streamlit interface

## 📋 Prerequisites

### Hardware Requirements

For optimal performance running Llama 3 8B locally:

- **CPU**: Modern 6-core processor (Intel/AMD)
- **RAM**: 16GB DDR4 minimum
- **GPU**: NVIDIA GPU with 8GB+ VRAM (e.g., GTX 1070, RTX 4060) - **Highly Recommended**
- **Storage**: 20GB free space for models and data

### Software Requirements

- Python 3.9 or higher
- Ollama (for running local LLMs)
- Git (optional, for cloning)

## 🚀 Installation

### Step 1: Clone or Download the Project

```bash
cd Bytaid
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install and Setup Ollama

#### Windows
1. Download Ollama from: https://ollama.ai/download
2. Run the installer
3. Open a new terminal/PowerShell window

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Mac
```bash
brew install ollama
```

### Step 5: Pull Required Models

```bash
# Pull the embedding model
ollama pull nomic-embed-text

# Pull the LLM model
ollama pull llama3

# Start Ollama server (if not running)
ollama serve
```

**Note**: Keep the Ollama server running in the background while using the chatbot.

## 📚 Usage

### Phase 1: Index Your Documents

1. **Add Documents**: Place your documents (PDF, TXT, DOCX, MD) in the `documents/` folder

2. **Run Indexing Pipeline**:
   ```bash
   python src/ingestion.py
   ```

   This will:
   - Load all documents from the `documents/` folder
   - Split them into optimal chunks (1000 tokens, 200 overlap)
   - Generate embeddings using Nomic Embed-Text
   - Store them in ChromaDB (persistent storage in `data/chroma_db/`)

   **Expected Output**:
   ```
   ✅ SUCCESS: Documents indexed and ready for retrieval
   Total documents processed: X
   Total chunks created: Y
   ```

### Phase 2: Run the Chatbot

```bash
streamlit run app.py
```

The chatbot interface will open in your default browser (usually http://localhost:8501)

## 🏗️ Project Structure

```
Bytaid/
├── documents/              # Place your source documents here (empty by default)
├── data/                   # Generated data (ChromaDB storage)
│   └── chroma_db/         # Vector database persistence
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration and prompts
│   ├── guardrails.py      # Safety features (input/output validation)
│   ├── retrieval.py       # Hybrid retrieval + RRF re-ranking
│   ├── utils.py           # Helper functions
│   └── ingestion.py       # Indexing pipeline (offline)
├── app.py                  # Streamlit UI and inference pipeline
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .gitignore             # Git ignore rules
```

## ⚙️ Configuration

All system settings can be customized in `src/config.py`:

### Key Parameters

```python
# Chunking
CHUNK_SIZE = 1000           # Size of document chunks
CHUNK_OVERLAP = 200         # Overlap between chunks

# Retrieval
DENSE_TOP_K = 50           # Documents from semantic search
SPARSE_TOP_K = 50          # Documents from BM25 search
FINAL_TOP_K = 5            # Final documents after RRF re-ranking

# LLM
LLM_TEMPERATURE = 0.3      # Lower = more focused responses
LLM_MAX_TOKENS = 2048      # Maximum response length
```

## 🛡️ Safety Features

### Input Guardrails
- **Profanity filtering**: Blocks inappropriate language
- **Prompt injection detection**: Prevents attempts to override system instructions
- **Input validation**: Checks for empty/malformed queries

### Output Guardrails
- **PII detection**: Prevents leakage of sensitive information
- **XML tag parsing**: Extracts clean answers from structured LLM responses
- **Contextual constraints**: Forces LLM to answer only from retrieved documents

### System Prompt Strategy
The chatbot uses a strict contextual guardrail that:
- Forces the LLM to answer ONLY from retrieved context
- Uses `<thinking>` and `<answer>` tags for structured reasoning
- Returns "I don't have that information" when context is insufficient

## 🔧 Troubleshooting

### Issue: "Vector database not found"
**Solution**: Run the indexing pipeline first:
```bash
python src/ingestion.py
```

### Issue: "Ollama connection failed"
**Solution**: Ensure Ollama is running:
```bash
ollama serve
```

### Issue: Slow response generation
**Possible causes**:
- Insufficient VRAM (model running on CPU instead of GPU)
- Too many documents in context
- System resources constrained

**Solutions**:
- Reduce `FINAL_TOP_K` in `config.py`
- Ensure GPU is being used (check Ollama logs)
- Close other applications

### Issue: "No documents loaded"
**Solution**: 
- Ensure documents are in the `documents/` folder
- Check file formats (PDF, TXT, DOCX, MD supported)
- Check file permissions

## 📊 Architecture Overview

### Indexing Phase (Offline)
```
Documents → Document Loaders → Text Splitter → Embedding Model → ChromaDB
```

### Inference Phase (Runtime)
```
User Query → Input Guardrails → Hybrid Retrieval (Dense + BM25) 
→ RRF Re-ranking → Prompt Construction → LLM → Output Guardrails → Response
```

## 🎯 Best Practices

1. **Document Preparation**: Clean, well-formatted documents yield better results
2. **Regular Re-indexing**: Re-run indexing when adding new documents
3. **Monitor VRAM**: Watch GPU memory usage to prevent slowdowns
4. **Chunk Size Tuning**: Adjust `CHUNK_SIZE` based on your document types
5. **Context Window**: Keep `FINAL_TOP_K` between 3-7 for optimal balance

## 📖 Technical Specifications

Based on the comprehensive RAG Chatbot Architectural Specification Document, this implementation follows industry best practices:

- **Decoupled Architecture**: Separate indexing and inference pipelines
- **State-of-the-art Models**: Nomic Embed-Text (8192 context) and Llama 3 8B
- **Optimized Chunking**: 1000 tokens with 200 overlap (as per research recommendations)
- **Hybrid Search**: Combines semantic understanding with lexical precision
- **RRF Re-ranking**: Reciprocal Rank Fusion with k=60 constant

## 🤝 Contributing

This project is designed as a learning resource for building production-ready RAG systems. Feel free to:
- Experiment with different models
- Adjust chunking strategies
- Enhance guardrails
- Improve the UI

## 📝 License

This project is provided as-is for educational purposes.

## 🙏 Acknowledgments

Built following the architectural specification for robust, production-ready RAG systems with:
- LangChain for RAG orchestration
- ChromaDB for vector storage
- Ollama for local LLM deployment
- Streamlit for rapid UI development

---

**Note**: This is a local-first application. All processing happens on your machine, ensuring privacy and eliminating API costs.

For detailed technical information, refer to the "RAG Chatbot Architectural Specification Document.md" in the project root.

