# Troubleshooting Guide

## Issue Fixed: ONNXRuntime Import Error

### Problem
When running `streamlit run app.py`, you encountered:
```
ERROR:__main__:Failed to initialize RAG system: The onnxruntime python package is not installed
```

Even after manually installing `onnxruntime`, the error persisted.

### Root Cause
The actual issue was **NumPy version incompatibility**:
- ChromaDB 0.4.24 uses deprecated NumPy 1.x APIs (like `np.float_`)
- NumPy 2.0+ removed these deprecated APIs
- When ChromaDB was installed, it pulled in NumPy 2.2.6, causing import failures

### Solution Applied
1. **Downgraded NumPy** to version 1.26.4 (compatible with ChromaDB 0.4.24)
2. **Updated requirements.txt** to lock NumPy to version 1.x:
   ```
   numpy<2.0,>=1.22.5  # ChromaDB 0.4.24 requires NumPy 1.x
   ```
3. **Verified all dependencies** are properly installed and compatible

### Commands Run
```bash
# Install compatible NumPy version
pip install "numpy<2.0" --force-reinstall

# Reinstall ChromaDB with correct dependencies
pip install chromadb==0.4.24

# Install profanity check with dependencies
pip install alt-profanity-check==1.7.2
pip install onnxruntime==1.16.3
```

### Verification
All modules now import successfully:
- ✅ ChromaDB
- ✅ ONNXRuntime
- ✅ Profanity Check
- ✅ All custom modules (config, guardrails, retrieval, etc.)

## How to Run the App Now

### 1. Ensure Ollama is Running
```bash
ollama serve
```

### 2. Index Your Documents (First Time Only)
```bash
# Add documents to documents/ folder first
python src/ingestion.py
```

### 3. Run the Chatbot
```bash
streamlit run app.py
```

The app should start successfully at `http://localhost:8501`

## Common Issues

### Issue: "Vector database not found"
**Solution**: Run the indexing pipeline first
```bash
python src/ingestion.py
```

### Issue: "Ollama connection failed"
**Solution**: Make sure Ollama is running
```bash
ollama serve
```

### Issue: "Module not found" errors
**Solution**: Reinstall dependencies
```bash
pip install -r requirements.txt
```

### Issue: NumPy version conflicts in the future
**Solution**: Always use NumPy 1.x with ChromaDB 0.4.24
```bash
pip install "numpy<2.0" --force-reinstall
```

## Package Versions (Verified Working)

```
chromadb==0.4.24
numpy==1.26.4 (< 2.0)
onnxruntime==1.23.1 (or 1.16.3)
alt-profanity-check==1.7.2
langchain==0.1.20
langchain-community==0.0.38
streamlit==1.32.2
ollama==0.1.7
sentence-transformers==2.5.1
```

## System Architecture Note

The error message mentioning "onnxruntime" was misleading because:
1. ONNXRuntime WAS installed
2. The real issue was ChromaDB failing to import due to NumPy 2.x incompatibility
3. ChromaDB uses ONNXRuntime internally, so the error appeared to be about ONNXRuntime
4. Fixing NumPy resolved the entire chain of import errors

---

**Last Updated**: October 15, 2025
**Status**: ✅ All issues resolved


