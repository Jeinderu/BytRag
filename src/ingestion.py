"""
Indexing Pipeline for RAG Chatbot.
Loads documents, chunks them, generates embeddings, and stores in ChromaDB.
"""

import logging
from pathlib import Path
from typing import List
import sys

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from tqdm import tqdm

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    DOCUMENTS_DIR,
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL
)
from src.utils import (
    setup_logging,
    ensure_directories_exist,
    validate_document_directory,
    get_supported_file_extensions
)

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Handles the offline indexing pipeline for RAG.
    Loads, chunks, embeds, and stores documents in ChromaDB.
    """
    
    def __init__(
        self,
        documents_dir: Path,
        persist_dir: Path,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        embedding_model: str = EMBEDDING_MODEL
    ):
        """
        Initialize the Document Indexer.
        
        Args:
            documents_dir: Directory containing source documents
            persist_dir: Directory for ChromaDB persistence
            chunk_size: Size of text chunks (in characters/tokens)
            chunk_overlap: Overlap between chunks
            embedding_model: Name of the Ollama embedding model
        """
        self.documents_dir = documents_dir
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        # Initialize embeddings
        logger.info(f"Initializing Ollama embeddings with model: {embedding_model}")
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=OLLAMA_BASE_URL
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=SEPARATORS,
            length_function=len,
        )
        
        logger.info("DocumentIndexer initialized")
    
    def load_documents(self) -> List:
        """
        Load all documents from the documents directory.
        
        Returns:
            List of loaded Document objects
        """
        logger.info(f"Loading documents from: {self.documents_dir}")
        
        all_documents = []
        
        # Load PDF files
        try:
            pdf_loader = DirectoryLoader(
                str(self.documents_dir),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
                use_multithreading=True
            )
            pdf_docs = pdf_loader.load()
            all_documents.extend(pdf_docs)
            logger.info(f"Loaded {len(pdf_docs)} PDF documents")
        except Exception as e:
            logger.warning(f"Error loading PDFs: {e}")
        
        # Load text files
        try:
            txt_loader = DirectoryLoader(
                str(self.documents_dir),
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            txt_docs = txt_loader.load()
            all_documents.extend(txt_docs)
            logger.info(f"Loaded {len(txt_docs)} text documents")
        except Exception as e:
            logger.warning(f"Error loading text files: {e}")
        
        # Load DOCX files
        try:
            docx_loader = DirectoryLoader(
                str(self.documents_dir),
                glob="**/*.docx",
                loader_cls=Docx2txtLoader,
                show_progress=True
            )
            docx_docs = docx_loader.load()
            all_documents.extend(docx_docs)
            logger.info(f"Loaded {len(docx_docs)} DOCX documents")
        except Exception as e:
            logger.warning(f"Error loading DOCX files: {e}")
        
        # Load markdown files
        try:
            md_loader = DirectoryLoader(
                str(self.documents_dir),
                glob="**/*.md",
                loader_cls=TextLoader,
                show_progress=True
            )
            md_docs = md_loader.load()
            all_documents.extend(md_docs)
            logger.info(f"Loaded {len(md_docs)} Markdown documents")
        except Exception as e:
            logger.warning(f"Error loading Markdown files: {e}")
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def chunk_documents(self, documents: List) -> List:
        """
        Split documents into chunks using RecursiveCharacterTextSplitter.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        logger.info(f"Chunking {len(documents)} documents...")
        logger.info(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        
        chunks = self.text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, chunks: List) -> Chroma:
        """
        Create ChromaDB vector store from document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            ChromaDB vector store instance
        """
        logger.info("Creating ChromaDB vector store...")
        logger.info(f"Persist directory: {self.persist_dir}")
        
        # Ensure persist directory exists
        ensure_directories_exist([self.persist_dir])
        
        # Create vector store with progress bar
        logger.info("Generating embeddings and storing in ChromaDB...")
        logger.info("This may take several minutes depending on the number of documents...")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir),
            collection_name=COLLECTION_NAME
        )
        
        logger.info(f"Vector store created with {len(chunks)} chunks")
        return vectorstore
    
    def index_documents(self) -> Chroma:
        """
        Run the complete indexing pipeline.
        
        Returns:
            ChromaDB vector store instance
        """
        logger.info("=" * 70)
        logger.info("STARTING INDEXING PIPELINE")
        logger.info("=" * 70)
        
        # Step 1: Load documents
        documents = self.load_documents()
        
        if not documents:
            logger.error("No documents loaded! Please add documents to the documents/ directory.")
            raise ValueError("No documents found to index")
        
        # Step 2: Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Step 3: Create vector store
        vectorstore = self.create_vectorstore(chunks)
        
        logger.info("=" * 70)
        logger.info("INDEXING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Total documents processed: {len(documents)}")
        logger.info(f"Total chunks created: {len(chunks)}")
        logger.info(f"Vector store location: {self.persist_dir}")
        logger.info("=" * 70)
        
        return vectorstore


def main():
    """Main function to run the indexing pipeline."""
    
    # Setup logging
    setup_logging("INFO")
    
    logger.info("\n" + "=" * 70)
    logger.info("RAG CHATBOT - INDEXING PIPELINE")
    logger.info("=" * 70)
    
    # Validate documents directory
    if not validate_document_directory(DOCUMENTS_DIR):
        logger.error("\n❌ INDEXING FAILED: No documents found")
        logger.error(f"Please add documents to: {DOCUMENTS_DIR}")
        logger.error(f"Supported formats: {get_supported_file_extensions()}")
        sys.exit(1)
    
    try:
        # Create indexer
        indexer = DocumentIndexer(
            documents_dir=DOCUMENTS_DIR,
            persist_dir=CHROMA_PERSIST_DIR
        )
        
        # Run indexing
        vectorstore = indexer.index_documents()
        
        logger.info("\n✅ SUCCESS: Documents indexed and ready for retrieval")
        logger.info(f"You can now run the chatbot application: streamlit run app.py")
        
    except Exception as e:
        logger.error(f"\n❌ INDEXING FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

