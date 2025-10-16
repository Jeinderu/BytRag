"""
Streamlit RAG Chatbot Application.
Main inference pipeline with hybrid retrieval, safety guardrails, and conversational UI.
"""

import os
# Disable ChromaDB telemetry to avoid warning messages
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import streamlit as st
import logging
from pathlib import Path
import sys

# Explicit import to prevent lazy-loading issues with Streamlit caching
import onnxruntime  # Required by ChromaDB
import numpy as np  # Ensure NumPy is loaded early

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.schema import HumanMessage, AIMessage

# Import project modules
from src.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    SYSTEM_PROMPT,
    RE_PROMPT_MESSAGE,
    NO_CONTEXT_MESSAGE,
    WELCOME_MESSAGE,
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT,
    DENSE_TOP_K,
    SPARSE_TOP_K,
    FINAL_TOP_K,
    RRF_K,
    ENABLE_INPUT_MODERATION,
    ENABLE_OUTPUT_VALIDATION,
    LLM_TEMPERATURE,
    MAX_HISTORY_LENGTH
)
from src.retrieval import create_hybrid_retriever
from src.guardrails import create_guardrails
from src.utils import (
    setup_logging,
    format_conversation_history,
    format_documents_for_context
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main answer section styling */
    .stMarkdown h3 {
        color: #1f77b4;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 5px;
        font-weight: 500;
    }
    
    /* Source document styling */
    .stTextArea textarea {
        font-size: 0.9rem;
        background-color: #fafafa;
    }
    
    /* Info box styling */
    .stAlert {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Divider styling */
    hr {
        margin: 0.5rem 0;
        border-color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system():
    """
    Initialize the RAG system components.
    Cached to avoid reloading on every interaction.
    
    Returns:
        Tuple of (vectorstore, retriever, llm, input_guard, output_guard)
    """
    logger.info("Initializing RAG system...")
    
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        # Load vector store
        vectorstore = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        
        # Get all documents for BM25 indexing
        all_docs = vectorstore.get()
        
        # Create document objects from the retrieved data
        from types import SimpleNamespace
        documents = []
        for i in range(len(all_docs['ids'])):
            doc = SimpleNamespace(
                page_content=all_docs['documents'][i],
                metadata=all_docs['metadatas'][i] if all_docs['metadatas'] else {}
            )
            documents.append(doc)
        
        # Initialize hybrid retriever
        retriever = create_hybrid_retriever(
            vectorstore=vectorstore,
            documents=documents,
            dense_top_k=DENSE_TOP_K,
            sparse_top_k=SPARSE_TOP_K,
            final_top_k=FINAL_TOP_K,
            rrf_k=RRF_K
        )
        
        # Initialize LLM
        llm = Ollama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE
        )
        
        # Initialize guardrails
        input_guard, output_guard = create_guardrails(
            enable_input=ENABLE_INPUT_MODERATION,
            enable_output=ENABLE_OUTPUT_VALIDATION
        )
        
        logger.info("RAG system initialized successfully")
        return vectorstore, retriever, llm, input_guard, output_guard
    
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise


def generate_response(query: str, retriever, llm, input_guard, output_guard, history: list):
    """
    Generate a response using the RAG pipeline.
    
    Args:
        query: User query
        retriever: Hybrid retriever instance
        llm: LLM instance
        input_guard: Input guardrail
        output_guard: Output guardrail
        history: Conversation history
        
    Returns:
        Tuple of (response_text, thinking_text, source_documents) or (error_message, None, None)
    """
    try:
        # Step 1: Input validation
        if input_guard:
            is_valid, reason = input_guard.validate(query)
            if not is_valid:
                logger.warning(f"Input validation failed: {reason}")
                return RE_PROMPT_MESSAGE, None, None
        
        # Step 2: Retrieve relevant documents
        logger.info(f"Processing query: {query[:100]}...")
        relevant_docs = retriever.retrieve(query)
        
        if not relevant_docs:
            logger.warning("No relevant documents found")
            return NO_CONTEXT_MESSAGE, None, None
        
        # Step 3: Format context and history
        context = format_documents_for_context(relevant_docs)
        history_text = format_conversation_history(history, max_turns=MAX_HISTORY_LENGTH)
        
        # Step 4: Construct prompt
        prompt = SYSTEM_PROMPT.format(
            context=context,
            history=history_text,
            question=query
        )
        
        # Step 5: Generate response
        logger.info("Generating response from LLM...")
        response = llm.invoke(prompt)
        
        # Step 6: Parse thinking and answer from response
        import re
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        
        thinking_text = thinking_match.group(1).strip() if thinking_match else None
        
        # If answer tags exist, use them; otherwise clean and use full response
        if answer_match:
            answer_text = answer_match.group(1).strip()
        else:
            # No tags found, remove any tags and use the response
            answer_text = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
            answer_text = re.sub(r'<answer>.*?</answer>', '', answer_text, flags=re.DOTALL | re.IGNORECASE)
            answer_text = answer_text.strip()
            
            # If still empty or very short, use original response
            if not answer_text or len(answer_text) < 10:
                answer_text = response.strip()
        
        # Step 7: Output validation (check for PII only, don't re-extract)
        if output_guard and answer_text:
            # Just check for PII, not re-extract since we already did that
            if output_guard.check_pii(answer_text):
                logger.warning("PII detected in output")
                return "I apologize, but I couldn't generate an appropriate response. Please try rephrasing your question.", None, None
        
        return answer_text, thinking_text, relevant_docs
    
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        return "I encountered an error while processing your request. Please try again.", None, None


def main():
    """Main Streamlit application."""
    
    # Initialize session state for conversation history FIRST
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": WELCOME_MESSAGE
        })
    
    # Initialize display settings
    if 'show_thinking' not in st.session_state:
        st.session_state.show_thinking = True  # Default ON
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = False  # Default OFF
    
    # Title
    st.title("🤖 Bytaid RAG Chatbot")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This is a **Retrieval-Augmented Generation (RAG)** chatbot that answers questions 
        based on indexed documents.
        
        **Features:**
        - 📚 Hybrid Retrieval (Dense + BM25)
        - 🎯 Reciprocal Rank Fusion Re-ranking
        - 🛡️ Safety Guardrails
        - 💬 Conversational Memory
        - 🔒 Local LLM (Llama 3)
        """)
        
        st.markdown("---")
        st.header("⚙️ System Status")
        
        # Check if vector store exists
        if CHROMA_PERSIST_DIR.exists():
            st.success("✅ Vector store loaded")
            
            # Try to get collection stats
            try:
                if 'vectorstore' in locals() or 'vectorstore' in globals():
                    count = vectorstore._collection.count()
                    st.info(f"📊 {count} chunks indexed")
            except:
                pass
        else:
            st.error("❌ Vector store not found")
            st.warning("Please run `python src/ingestion.py` first")
        
        st.markdown("---")
        st.header("🎨 Display Settings")
        
        # Settings for UI customization (already initialized at top of main())
        st.session_state.show_thinking = st.checkbox(
            "Expand reasoning by default",
            value=st.session_state.show_thinking,
            help="Show the AI's thinking process expanded by default"
        )
        
        st.session_state.show_sources = st.checkbox(
            "Expand sources by default",
            value=st.session_state.show_sources,
            help="Show source documents expanded by default"
        )
        
        st.markdown("---")
        st.header("📊 Session Stats")
        
        # Count messages (exclude welcome message)
        message_count = len([m for m in st.session_state.messages if m.get("content") != WELCOME_MESSAGE])
        user_messages = len([m for m in st.session_state.messages if m.get("role") == "user"])
        
        st.metric("Total Messages", message_count)
        st.metric("Questions Asked", user_messages)
        
        st.markdown("---")
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", type="primary"):
            st.session_state.messages = []
            st.rerun()
    
    # Check if ChromaDB exists
    if not CHROMA_PERSIST_DIR.exists():
        st.error("⚠️ **Vector database not found!**")
        st.info("""
        Please follow these steps:
        1. Add your documents to the `documents/` folder
        2. Run the indexing pipeline: `python src/ingestion.py`
        3. Restart this application
        """)
        st.stop()
    
    # Initialize RAG system
    try:
        with st.spinner("🔄 Initializing RAG system..."):
            vectorstore, retriever, llm, input_guard, output_guard = initialize_rag_system()
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {e}")
        st.error("Please ensure Ollama is running with the required models:")
        st.code(f"ollama pull {EMBEDDING_MODEL}\nollama pull {LLM_MODEL}")
        st.stop()
    
    # Display conversation history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # For assistant messages, check if we have stored metadata
            if message["role"] == "assistant" and i > 0:  # Skip welcome message
                # Display with enhanced formatting for stored messages
                if message["content"] != WELCOME_MESSAGE:
                    st.markdown("### 💡 Answer")
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                answer, thinking, sources = generate_response(
                    query=prompt,
                    retriever=retriever,
                    llm=llm,
                    input_guard=input_guard,
                    output_guard=output_guard,
                    history=st.session_state.messages[:-1]  # Exclude current message
                )
            
            # Display the final answer prominently
            st.markdown("### 💡 Answer")
            st.markdown(answer)
            
            # Display thinking process in an expander (if available)
            if thinking:
                with st.expander("🧠 Show Reasoning Process", expanded=st.session_state.get('show_thinking', False)):
                    st.markdown("**How I arrived at this answer:**")
                    st.info(thinking)
            
            # Display source documents
            if sources:
                with st.expander(f"📚 Source Documents ({len(sources)} retrieved)", expanded=st.session_state.get('show_sources', True)):
                    for i, doc in enumerate(sources, 1):
                        # Extract metadata
                        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                        source_file = metadata.get('source', 'Unknown')
                        page_num = metadata.get('page', 'N/A')
                        
                        # Get document content
                        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        
                        # Ensure content is a string and has content
                        if content and isinstance(content, str):
                            content = content.strip()
                            preview = content[:400] + "..." if len(content) > 400 else content
                        else:
                            preview = "[No content preview available]"
                        
                        # Display with nice formatting
                        st.markdown(f"""
                        **Source {i}:**  
                        📄 **File:** `{source_file}`  
                        📑 **Page:** {page_num}
                        """)
                        
                        # Display content in a code block for better readability
                        if preview != "[No content preview available]":
                            st.markdown("**Content Preview:**")
                            st.code(preview, language=None)
                        
                        if i < len(sources):
                            st.divider()
        
        # Add assistant response to history (only the answer, not the sources)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

