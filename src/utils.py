"""
Utility functions for the RAG chatbot.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

# Configure logging
def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def ensure_directories_exist(directories: List[Path]) -> None:
    """
    Ensure all required directories exist.
    
    Args:
        directories: List of directory paths to create if they don't exist
    """
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory ensured: {directory}")


def get_supported_file_extensions() -> List[str]:
    """
    Get list of supported document file extensions.
    
    Returns:
        List of supported extensions
    """
    return ['.txt', '.pdf', '.docx', '.doc', '.md']


def validate_document_directory(doc_dir: Path) -> bool:
    """
    Validate that the document directory exists and contains files.
    
    Args:
        doc_dir: Path to the documents directory
        
    Returns:
        True if valid, False otherwise
    """
    if not doc_dir.exists():
        logging.error(f"Documents directory does not exist: {doc_dir}")
        return False
    
    # Check if there are any supported documents
    supported_extensions = get_supported_file_extensions()
    files = []
    
    for ext in supported_extensions:
        files.extend(list(doc_dir.glob(f"**/*{ext}")))
    
    if not files:
        logging.warning(f"No supported documents found in: {doc_dir}")
        logging.warning(f"Supported extensions: {supported_extensions}")
        return False
    
    logging.info(f"Found {len(files)} supported documents")
    return True


def format_conversation_history(history: List[Dict[str, str]], max_turns: int = 5) -> str:
    """
    Format conversation history for the LLM prompt.
    
    Args:
        history: List of message dictionaries with 'role' and 'content'
        max_turns: Maximum number of conversation turns to include
        
    Returns:
        Formatted history string
    """
    if not history:
        return "No previous conversation."
    
    # Take only the last N turns
    recent_history = history[-max_turns * 2:] if len(history) > max_turns * 2 else history
    
    formatted = []
    for msg in recent_history:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        if role == 'user':
            formatted.append(f"User: {content}")
        elif role == 'assistant':
            formatted.append(f"Assistant: {content}")
    
    return "\n".join(formatted)


def format_documents_for_context(documents: List[Any]) -> str:
    """
    Format retrieved documents into a context string for the LLM.
    
    Args:
        documents: List of Document objects with page_content and metadata
        
    Returns:
        Formatted context string
    """
    if not documents:
        return "No relevant documents found."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page', 'N/A')
        
        context_parts.append(
            f"[Document {i}] (Source: {source}, Page: {page})\n{content}\n"
        )
    
    return "\n---\n".join(context_parts)


def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count (simple heuristic: ~4 chars per token).
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Approximate token count
    """
    # Simple approximation: 1 token ≈ 4 characters
    return len(text) // 4


def truncate_context_if_needed(context: str, max_tokens: int = 6000) -> str:
    """
    Truncate context if it exceeds maximum token limit.
    
    Args:
        context: Context string
        max_tokens: Maximum allowed tokens
        
    Returns:
        Potentially truncated context
    """
    approx_tokens = count_tokens_approximate(context)
    
    if approx_tokens <= max_tokens:
        return context
    
    # Truncate to approximate character count
    max_chars = max_tokens * 4
    truncated = context[:max_chars]
    
    logging.warning(
        f"Context truncated from ~{approx_tokens} to ~{max_tokens} tokens"
    )
    
    return truncated + "\n\n[Context truncated due to length...]"


def display_startup_info():
    """Display startup information and system checks."""
    print("=" * 70)
    print("RAG CHATBOT SYSTEM")
    print("=" * 70)
    print("\nSystem Information:")
    print(f"- Python Version: {sys.version.split()[0]}")
    
    # Check if required directories exist
    from src.config import DOCUMENTS_DIR, DATA_DIR, CHROMA_PERSIST_DIR
    
    print(f"\nDirectories:")
    print(f"- Documents: {DOCUMENTS_DIR} {'✓' if DOCUMENTS_DIR.exists() else '✗'}")
    print(f"- Data: {DATA_DIR} {'✓' if DATA_DIR.exists() else '✗'}")
    print(f"- ChromaDB: {CHROMA_PERSIST_DIR} {'✓' if CHROMA_PERSIST_DIR.exists() else '✗'}")
    print("=" * 70)

