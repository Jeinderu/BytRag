"""
Configuration settings for the RAG Chatbot application.
Contains all system parameters, model settings, and prompt templates.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# Ollama Settings
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# ============================================================================
# CHUNKING PARAMETERS
# ============================================================================

# As per the specification: chunk size 1000, overlap 200
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Separators for Recursive Character Text Splitter
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ============================================================================
# RETRIEVAL PARAMETERS
# ============================================================================

# Number of documents to retrieve from each retriever (Dense + Sparse)
DENSE_TOP_K = 50
SPARSE_TOP_K = 50

# Final number of documents to pass to LLM after re-ranking
FINAL_TOP_K = 5

# Reciprocal Rank Fusion constant (typically 60)
RRF_K = 60

# ============================================================================
# LLM GENERATION PARAMETERS
# ============================================================================

LLM_TEMPERATURE = 0.3  # Lower temperature for more focused, factual responses
LLM_MAX_TOKENS = 2048
LLM_TOP_P = 0.9

# ============================================================================
# CHROMADB SETTINGS
# ============================================================================

COLLECTION_NAME = "rag_documents"
DISTANCE_METRIC = "cosine"  # Options: "cosine", "l2", "ip" (inner product)

# ============================================================================
# SAFETY AND GUARDRAILS
# ============================================================================

# Enable input moderation
ENABLE_INPUT_MODERATION = True

# Enable output guardrails
ENABLE_OUTPUT_VALIDATION = True

# Maximum conversation history to maintain
MAX_HISTORY_LENGTH = 10

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

SYSTEM_PROMPT = """You are a helpful, accurate AI assistant with access to a knowledge base of documents.

CRITICAL INSTRUCTIONS:
1. You MUST answer questions ONLY using information from the provided context below.
2. If the answer is not contained in the context, you MUST respond with: "I don't have that information in my knowledge base."
3. Do NOT use your general knowledge or training data to answer questions.
4. Always cite which part of the context you're using to form your answer.
5. Be concise, accurate, and helpful.

RESPONSE FORMAT:
First, in <thinking> tags, write down the exact quotes from the context that are relevant to the question.
Then, in <answer> tags, provide your final response to the user based only on those quotes.

Context:
{context}

Conversation History:
{history}

User Question: {question}

Remember: Answer ONLY from the provided context. If unsure, say so clearly."""

CONTEXTUAL_THINKING_PROMPT = """<thinking>
Let me identify the relevant information from the context:
{relevant_quotes}
</thinking>

<answer>
{final_answer}
</answer>"""

RE_PROMPT_MESSAGE = """I apologize, but I cannot process that request. Please:
- Ensure your question is appropriate and professional
- Ask questions related to the knowledge base
- Rephrase your query if it was unclear

How can I help you with information from the knowledge base?"""

NO_CONTEXT_MESSAGE = """I don't have that information in my knowledge base. The documents I have access to don't contain an answer to your question. 

Could you try:
- Rephrasing your question
- Asking about a different topic that might be covered in the knowledge base
- Being more specific about what you're looking for"""

# ============================================================================
# STREAMLIT UI SETTINGS
# ============================================================================

PAGE_TITLE = "Bytaid RAG Chatbot"
PAGE_ICON = "🤖"
LAYOUT = "wide"

WELCOME_MESSAGE = """👋 Welcome to the RAG Chatbot!

I'm an AI assistant with access to a specialized knowledge base. I can answer questions based on the documents that have been indexed.

**How to use:**
- Ask me questions about the topics covered in the knowledge base
- I'll provide answers based solely on the indexed documents
- I'll let you know if information isn't available in my knowledge base

**Note:** Please ensure the documents have been indexed by running `ingestion.py` before using this chatbot.

How can I help you today?"""

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

