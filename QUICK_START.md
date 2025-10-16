# 🚀 Quick Start Guide

## ✅ All Issues Fixed!

The ONNXRuntime import error has been resolved with the following fixes:
1. ✅ Added explicit `onnxruntime` and `numpy` imports at the top of `app.py`
2. ✅ Ensured NumPy version is locked to `<2.0` for ChromaDB compatibility
3. ✅ Updated Streamlit to latest version (1.50.0)
4. ✅ Created helper scripts for easy startup

---

## 📋 Prerequisites Checklist

Before running the app, ensure:

- [x] Python 3.9+ installed
- [x] Virtual environment created (`venv/` folder exists)
- [x] All packages installed (`pip install -r requirements.txt`)
- [x] Ollama installed and running
- [x] Models downloaded (`ollama pull llama3` and `ollama pull nomic-embed-text`)
- [x] Documents indexed (`python src/ingestion.py`)

---

## 🎯 Three Ways to Run the App

### **Method 1: Using the Batch Script (Recommended for Windows)**

Simply double-click `run_app.bat` or run in Command Prompt:
```cmd
run_app.bat
```

This script will:
- Check if the virtual environment exists
- Verify Ollama is running
- Start the Streamlit app with the correct Python

---

### **Method 2: Using PowerShell Script**

Run in PowerShell:
```powershell
.\run_app.ps1
```

If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\run_app.ps1
```

---

### **Method 3: Manual Command (If scripts don't work)**

```powershell
# Ensure you're in the project directory
cd D:\RAG\Bytaid

# Run with venv Python explicitly
.\venv\Scripts\python.exe -m streamlit run app.py
```

---

## 🔧 If You Still Get Errors

### Error: "onnxruntime not installed"

**Solution:**
```bash
.\venv\Scripts\python.exe -m pip install --force-reinstall onnxruntime numpy<2.0
```

### Error: "Vector database not found"

**Solution:**
```bash
# Add documents to documents/ folder first
.\venv\Scripts\python.exe src/ingestion.py
```

### Error: "Ollama connection failed"

**Solution:**
Open a new terminal and run:
```bash
ollama serve
```
Keep this terminal open.

### Error: "Module not found"

**Solution:**
```bash
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 📊 Expected Startup Sequence

When you run the app successfully, you should see:

```
============================================
Starting RAG Chatbot
============================================

Using Python from: venv\Scripts\python.exe

Checking Ollama connection...
✅ Ollama is running

Starting Streamlit app...
The app will open in your default browser at http://localhost:8501

Press Ctrl+C to stop the server
============================================

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## 🎉 Using the App

1. **Wait for initialization** - The first load takes ~10-30 seconds
2. **Check the sidebar** - Should show "✅ Vector store loaded"
3. **Ask a question** - Type in the chat input at the bottom
4. **Get responses** - The AI will answer based on your indexed documents

---

## 🛑 Stopping the App

Press `Ctrl + C` in the terminal where the app is running.

---

## 📝 Quick Reference Commands

```powershell
# Start Ollama (in a separate terminal)
ollama serve

# Index new documents
.\venv\Scripts\python.exe src/ingestion.py

# Run the chatbot
.\run_app.bat
# OR
.\venv\Scripts\python.exe -m streamlit run app.py

# Clear Streamlit cache (if needed)
.\venv\Scripts\python.exe -m streamlit cache clear
```

---

## 💡 Tips

1. **First time setup takes longer** - The app caches the vector store after first load
2. **Add documents anytime** - Just re-run `src/ingestion.py` after adding new files
3. **Keep Ollama running** - Start it once and leave it running in the background
4. **GPU recommended** - Responses will be faster with a dedicated GPU (8GB+ VRAM)

---

**Need more help?** Check `TROUBLESHOOTING.md` for detailed solutions.


