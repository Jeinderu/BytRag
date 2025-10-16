

# **Expert Blueprint for Building a Localized, Robust RAG Chatbot: Architecture, Optimization, and Safety for 300+ Documents**

## **I. Architectural Foundations: Understanding the RAG Pipeline**

A robust Retrieval-Augmented Generation (RAG) system, even when built by a beginner, must adhere to industry best practices, notably the separation of data preparation from live inference. A RAG application comprises two distinct, loosely coupled stages: the Indexing Phase and the Inference Phase.1

### **A. The RAG Lifecycle: Decoupling Indexing from Inference**

The **Indexing Phase** is performed offline and infrequently. Its primary function is to ingest source documents, process them into small, semantically meaningful chunks, transform those chunks into vector embeddings, and store them persistently in a vector database (VDB).1 This process is computationally intensive but not time-critical.

Conversely, the **Inference Phase** executes at runtime whenever a user submits a query. It involves retrieving the most relevant context from the index and generating a grounded response using the Large Language Model (LLM).1 This phase is highly time-critical, as latency directly impacts the user experience.

The architectural decision to decouple these two phases offers significant benefits, particularly when scaling or moving toward production. Separating the pipelines simplifies maintenance and debugging, as issues in the data ingestion pipeline (e.g., a corrupt PDF or a change in a chunking strategy) will not cause the live chatbot interface to crash.2 Furthermore, this modular design allows for independent scaling: the vector index generation can be versioned, tested, and promoted separately from the live model serving pipeline, enabling continuous improvement and robust operation in real-world scenarios.3

### **B. Project Technology Stack Selection**

The selected technology stack is based on open-source accessibility, high performance in local environments, and strong foundational support, offering the quickest learning curve for an IT student.

1. **Orchestration Framework:** **LangChain** is chosen for its established modularity, which simplifies the creation of the RAG chain and integration of various components like document loaders, vector stores, and LLMs.1  
2. **Vector Database (VDB):** **ChromaDB** is selected. It is a Python-first, open-source embedding database renowned for its simplicity and user-friendly API.5 Crucially, it offers persistent storage, which is necessary to reliably index and query the required 300+ documents without requiring complex server infrastructure setup.7  
3. **Embedding Model (Retrieval):** **Nomic Embed-Text** is preferred, run via Ollama. This open-source model boasts state-of-the-art performance, surpassing older models like OpenAI’s Ada-002 on retrieval benchmarks, and supports a generous 8192 context length.8  
4. **LLM (Generation):** **Llama 3 8B** is used, deployed locally via **Ollama**. Llama 3 represents the best-in-class open-source LLM, ensuring high-quality and contextually accurate response generation.10  
5. **Frontend Interface:** **Streamlit** provides the mechanism for rapid development of an interactive, functional chat interface entirely in Python.12

Table 1 details the core components chosen for this project blueprint:

Table 1: RAG Pipeline Component Mapping (Project Blueprint)

| Component | Phase | Selected Tool/Model | Rationale & Citation |
| :---- | :---- | :---- | :---- |
| Orchestration | Both | LangChain | Framework simplifies RAG chain creation.1 |
| Vector Store | Indexing | ChromaDB | Open-source, persistent, simple API for local scale.5 |
| Embedding Model | Both | Nomic-Embed-Text (Ollama) | High-performance open-source embedding model with large context.8 |
| LLM (Generator) | Inference | Llama 3 8B (Ollama) | State-of-the-art open-source LLM for high-quality generation.10 |
| Frontend | Inference | Streamlit | Rapid Python-based UI development for chat interface.12 |

## **II. The Offline Indexing Pipeline (300+ Documents)**

The Indexing Pipeline must efficiently transform the 300+ raw documents into a highly searchable format. This process requires meticulous attention to document preparation, chunking, embedding generation, and persistent storage.

### **A. Document Loading and Preprocessing for RAG**

The first step involves using LangChain's Document Loaders to read documents—whether they are plain text, PDFs, or other formats—and convert them into a unified Document object structure.1 Once loaded, basic preprocessing often occurs, such as removing boilerplate text, normalizing whitespace, and extracting relevant metadata (e.g., source file name or page number).

### **B. Strategic Chunking: Maximizing Contextual Integrity**

Chunking, the act of segmenting large documents into smaller pieces, is widely considered the single most important factor determining RAG retrieval performance.14 If chunks are too large, the context window of the LLM may be exceeded, or irrelevant details may dilute the vector embedding. If chunks are too small, they may lack the necessary contextual information for the LLM to synthesize a complete answer.14

The recommended approach for general RAG applications is the **Recursive Character Text Splitter**.1 This splitter works by trying a sequence of defined separators (like newlines, then paragraphs, then spaces) to split the text, aiming to keep semantically related pieces together.

For this project, the recommended chunking parameters are a **chunk size of 1000 tokens** and a **chunk overlap of 200 tokens**.1 This specific configuration is justified by the architecture's use of modern embedding models. Since high-performance open-source embedding models, like Nomic Embed-Text, support context lengths up to 8192 tokens, relying on historical chunk size limits (like 512 tokens) is unnecessary.8 A 1000-token chunk is large enough to contain rich, self-sufficient context, enabling the LLM to synthesize a detailed, accurate response without relying on adjacent chunks.15 The 200-token overlap ensures semantic continuity, preventing critical transitional information from being accidentally split across the boundaries of two chunks, thereby improving retrieval accuracy.

### **C. Implementation of ChromaDB for Persistent Vector Storage**

To manage the embeddings for 300+ documents reliably, ChromaDB is initialized as a persistent vector database.5 Persistence is critical, guaranteeing that the indexed embeddings and documents remain stored on disk and are instantly accessible when the application restarts.7

The integration requires specifying the local storage path and the chosen embedding model. For this project, the chosen open-source model, Nomic Embed-Text, must first be made available locally via Ollama. The embedding generation process then converts the 1000-token chunks into dense vector representations, which are written to a designated collection within the persistent ChromaDB instance.7

## **III. Deep Dive into Embedding and Vector Techniques**

The efficacy of RAG hinges entirely on the quality and robustness of its search mechanism, which requires leveraging both classical and modern vector techniques.

### **A. Classical vs. Semantic Embeddings: The Hybrid Advantage**

The retrieval component must combine the strengths of two fundamentally different vector types:

1. **Semantic (Dense) Embeddings:** These are high-dimensional vectors (e.g., 384 or 768 dimensions) generated by deep learning models like Nomic Embed-Text or E5-Small-V2.18 Dense embeddings capture the nuanced semantic meaning of the text. This allows the system to retrieve documents that are conceptually related to a query, even if they share no exact keywords. For example, a query about "pet care" could retrieve a document discussing "animal welfare".19  
2. **Classical (Sparse/Lexical) Embeddings:** Methods like BM25 (Best Matching 25\) are based on keyword frequency and rarity (similar to TF-IDF). Sparse vectors excel at lexical matching, ensuring high precision when a query contains specific terminology, jargon, or proper nouns.19

The reliance solely on semantic embeddings introduces a weakness: dense models can sometimes fail to find documents that contain specific, rare keywords if the overall semantic representation of the document is dominated by other themes. This necessitates the adoption of **Hybrid Retrieval**, which combines both semantic and lexical search results to ensure the system is robust across a wide range of query types, covering both conceptual meaning and precise keyword matching.19

### **B. Dense Vector Embeddings: Choosing an Open-Source Model**

Nomic Embed-Text is the recommended model for the vector store, specifically because it addresses the limitations inherent in many older open-source models. While models like E5-Small-V2 are highly proficient (often accessible via the sentence-transformers library) 18, they traditionally operate with shorter context lengths, often around 512 tokens.8 Nomic Embed-Text v1, with its 8192 context length, is optimized for general retrieval tasks and facilitates the use of the previously justified 1000-token chunk size, ensuring superior performance and context capture.8

Deployment of Nomic Embed-Text in this project is simplified by using Ollama, which allows the model to be pulled and run locally with simple terminal commands such as ollama pull nomic-embed-text.11

## **IV. Optimizing Retrieval: Hybrid Search and Re-Ranking**

After the initial indexing of the 300+ documents, the focus shifts to the efficiency and accuracy of retrieval during the inference phase. This involves Hybrid Retrieval, which generates a large initial set of candidate chunks, followed by a critical filtering step using a re-ranking algorithm.

### **A. Mechanism of Hybrid Retrieval (Dense \+ Sparse)**

Hybrid Retrieval begins when the user submits a query. The system simultaneously executes two distinct search operations against the indexed documents:

1. **Dense Retrieval:** The query is embedded using Nomic Embed-Text, and ChromaDB returns the top  most semantically similar chunks (e.g., 50 chunks).  
2. **Sparse Retrieval:** The query is run against the BM25 index, returning the top  most lexically relevant chunks (e.g., 50 chunks).

This process pools a large, comprehensive set of candidate chunks (up to 100 in this example) that represent the best matches according to both conceptual meaning and keyword precision.19

### **B. The Role of Re-Ranking in Relevance Filtering**

While the combined retrieval step is robust, the resultant list of 100 documents is often far too large to pass into the LLM's context window, which typically has space for only a handful of highly relevant documents (Top K, usually  to ).1 The primary function of re-ranking is to analyze the pooled list of 100 chunks and select only the absolute most relevant Top K documents for generation.

Selecting an appropriate re-ranking method is vital for a local project. While highly accurate re-rankers powered by external large language models (LLM-based re-rankers) exist, they introduce cost and latency into the local pipeline.22 Therefore, the recommended method is **Reciprocal Rank Fusion (RRF)**. RRF is an efficient, algorithm-based approach demonstrated in major RAG frameworks like LlamaIndex and LangChain.23 It provides highly effective fusion and re-ranking without requiring expensive external models or excessive computation.22

### **C. Practical Implementation of Reciprocal Rank Fusion (RRF)**

RRF works by aggregating the ranks of a document across multiple retrieval results (in this case, Dense and BM25) into a single, combined score. A document ranked high by both methods receives a substantially higher RRF score than a document ranked high by only one method.

The RRF score for a document  is calculated using the following general formula 23:

Where  is the set of retrievers (Dense, BM25),  is the rank of document  in the results of retriever , and  is a damping constant (typically 60\) used to ensure that documents slightly further down the list still receive a meaningful score.23

#### **Dummy Example: Ranking 100 Retrieved Documents to Top K**

In a practical application, if 100 unique documents are returned (50 from each retriever, with some overlap), RRF efficiently merges and scores them. The following table illustrates how RRF favors documents consistently ranked highly across both sparse and dense searches, effectively reducing the candidate pool from 100 to the essential Top 5 documents required for the LLM context.

Table 3: Reciprocal Rank Fusion (RRF) Calculation Example (C=60)

| Document ID | Rank (Semantic R1) | Rank (BM25 R2) | RRF Score (Formula: 1/(60+R1) \+ 1/(60+R2)) | Final Rank |
| :---- | :---- | :---- | :---- | :---- |
| Document A | 1 | 3 |  | 1 |
| Document B | 10 | 1 |  | 2 |
| Document C | 2 | 15 |  | 3 |
| Document D | 30 | 2 |  | 4 |
| Document E | 4 | 45 |  | 5 |
| Document F | 90 | 50 |  | 98 (Discarded) |

The final ranked list is then truncated to include only the Top K documents (e.g., Document A through E), which are subsequently used to augment the prompt sent to Llama 3\.

## **V. LLM Inference and Local Deployment Strategy**

To achieve a cost-effective and transparent RAG system, the Large Language Model must be deployed locally. This project utilizes Ollama for simplified local LLM execution.

### **A. Utilizing Free Tier/External LLMs (Alternative Context)**

It is useful to understand external LLM inference options for comparison. Platforms such as Groq offer extremely low First Token Latency (FTL), often delivering responses in 0.13 to 0.14 seconds for short prompts, demonstrating highly optimized hardware infrastructure.24 However, these services inherently involve recurring costs, which vary significantly (e.g., comparing Google’s Gemini 2.5 Flash at $0.30 per 1M input tokens versus Gemini 2.5 Pro at $1.25 per 1M input tokens).25 Local deployment with Ollama eliminates these API costs entirely.

### **B. The Power of Local LLMs: Introduction to Ollama**

Ollama simplifies the process of running LLMs directly on a personal computer, supporting Windows, Mac, and Linux distributions.26 Installation is straightforward, often requiring only a single command line installer or a curl command.26

For this project, the **Llama 3 8B** model is selected. Once Ollama is installed, the model can be downloaded and initiated via the terminal commands: ollama pull llama3 followed by ollama serve & (to run the server in the background).26 The RAG pipeline then connects to this local server for inference.

### **C. Hardware Assessment and Installation Guide for Ollama**

Running a model in the 8 billion parameter class (Llama 3 8B) locally necessitates specific hardware to ensure responsive performance. While the base installation of Ollama may work on systems with lower specifications, achieving low-latency inference depends heavily on the presence of dedicated GPU memory (VRAM).

The minimum budget-friendly hardware configuration for smoothly running a 7B/8B model involves a modern 6-core CPU, 16GB of DDR4 RAM, and, most critically, a dedicated NVIDIA GPU with at least 8GB of VRAM.27

The VRAM requirement is critical because modern LLMs use quantization (reducing model precision, often to 4-bit) to decrease the memory footprint. If the quantized model weights cannot be fully loaded onto the GPU's VRAM, the system must resort to swapping data to the much slower system RAM, causing severe bottlenecks and significantly increasing token generation latency.27 An 8GB VRAM card, such as an older GTX 1070 or a newer RTX 4060, is generally sufficient to avoid this performance collapse and deliver the necessary speed for a usable chat application.27

Table 4 outlines the minimum hardware requirements:

Table 4: Ollama Minimum Hardware Requirements for Llama 3 8B

| Component | Minimum Recommendation (Budget-Friendly) | Role in LLM Inference | Performance Impact |
| :---- | :---- | :---- | :---- |
| **CPU** | Modern 6-core (Intel or AMD) | Manages OS and orchestrates model loading/offloading.27 | Limits speed if VRAM is insufficient. |
| **RAM** | 16GB DDR4 | Holds model weights when VRAM is exceeded or unavailable.27 | Minimum requirement for a 7B/8B model class. |
| **VRAM (GPU)** | 8GB NVIDIA (e.g., GTX 1070/RTX 4060\) | **Critical:** Stores quantized model weights for fast parallel processing.27 | Directly dictates token generation speed and latency. |

## **VI. Chatbot UI, Flow Control, and Safety Guardrails**

The final component is the user-facing interface, which must be responsive, maintain conversation history, and incorporate essential safety features (guardrails).

### **A. Building the Streamlit Chatbot Interface and Session State Management**

Streamlit enables the rapid creation of the chat interface using Python. The application structure typically utilizes st.chat\_message to visually distinguish between user inputs and AI outputs, and st.write to display the content.28

To support multi-turn conversation, the application must manage memory. Streamlit’s built-in **st.session\_state** is used to store the running history of the conversation, treating every message as a dictionary with a role and content.28 This history is paramount; it is included in the LLM prompt at each turn, providing the context Llama 3 needs to understand follow-up questions.1

### **B. RAG Flow at Runtime: From Query Validation to Response Generation**

The inference process must integrate safety checks to ensure responsible operation:

1. **User Query Input & Validation:** The flow is initiated by a user query.  
2. **Input Guardrail/Moderation Check:** The raw query is passed through a moderation filter (e.g., profanity check). If restricted content is detected, the query is blocked, and a re-prompt message is generated.  
3. **Retrieval Step:** If the query is safe, Hybrid Retrieval (Dense \+ BM25) is executed against the ChromaDB index.  
4. **Re-ranking:** The RRF algorithm is applied to the retrieved chunks, reducing the set to the Top K most relevant documents.  
5. **Prompt Construction:** The system combines the conversation history, the strict contextual system prompt, and the Top K retrieved context documents.  
6. **LLM Generation:** Llama 3 8B generates the response, grounded by the retrieved information.  
7. **Output Guardrail Check:** The final response may be checked for compliance (e.g., PII leakage or harmful content) before being displayed.

### **C. Implementing Moderation and Contextual Guardrails**

Safety features mitigate risks such as sensitive data disclosure, prompt injection, and hallucination.30

#### **1\. Input Validation and Profanity Filtering**

For basic input moderation, the Python library alt-profanity-check provides a simple solution.32 This library allows the system to quickly detect offensive language using methods like predict, which returns a Boolean (True/False) indicating the presence of profanity.32 If the query is flagged, the RAG flow is terminated immediately, preventing the potentially malicious or inappropriate input from reaching the LLM or influencing retrieval.

#### **2\. Contextual Guardrails via System Prompting**

The most critical safety feature in a RAG system is a strict **contextual guardrail** enforced through the LLM’s system prompt. This mechanism ensures that the LLM limits its answers solely to the information retrieved from the knowledge base, dramatically reducing the risk of hallucinations and mitigating prompt injection attempts designed to override system instructions.31

A high-fidelity system prompt enforces this constraint by explicitly instructing the model: "If the answer is not contained within the provided context, state that the information is not available in the knowledge base." Advanced system design further instructs the model to utilize specific formatting (e.g., XML tags) for its internal processes. For instance, the model can be commanded to first write down exact quotes relevant to the query inside \<thinking\>\</thinking\> tags before formulating the final user-facing response within \<answer\>\</answer\> tags.33 This "internal thinking log" approach forces the LLM to ground its response, providing a robust, simple defense against the model generalizing or hallucinating based on its pre-trained knowledge, a common vulnerability in complex retrieval systems.34

### **D. Designing for Conversation: Managing History and Re-Prompting Queries**

To deliver a conversational experience, the application must manage the continuous exchange of information. As noted, the conversation history (session state) is critical for multi-turn interactions, allowing the LLM to process contextually dependent follow-up questions.1

**Re-prompting** is an essential user experience feature.35 If a user query fails the input moderation check, or if the LLM cannot confidently generate an answer based on the retrieved context (due to the contextual guardrail), the system issues a polite, constructive re-prompt. This guides the user back to an acceptable format or topic without leaving them with an abrupt failure, ensuring a smoother user journey.

## **VII. Conclusion and Project Implementation Roadmap**

The construction of a RAG chatbot capable of handling 300+ documents reliably requires integrating advanced retrieval techniques with a robust local inference architecture. The blueprint provided specifies the use of **LangChain** and **ChromaDB** for orchestration and persistent storage, high-performing open-source models (**Nomic Embed-Text** and **Llama 3 8B**) deployed via **Ollama**, and retrieval optimization using **Hybrid Search** followed by efficient **RRF re-ranking**. Finally, necessary safety is addressed via simple **input moderation** and stringent **contextual system prompting**.

The combination of Hybrid Retrieval and RRF is not an optional optimization but a necessary foundation to ensure consistent relevance across diverse queries. Deploying Llama 3 8B locally provides transparency and cost savings, provided the system meets the minimum VRAM requirement of 8GB for optimal inference speed.

The following roadmap provides the necessary implementation steps:

1. **Environment Preparation:** Install Python, create a virtual environment, and install core dependencies: pip install streamlit langchain-community chromadb sentence-transformers alt-profanity-check.  
2. **Ollama Setup and Model Pull:** Install the Ollama runner for your operating system. Open the terminal and download the necessary models:  
   * ollama pull llama3  
   * ollama pull nomic-embed-text  
   * Start the local Ollama server: ollama serve &  
3. **Indexing Pipeline (ingestion.py):**  
   * Load the 300+ documents.  
   * Implement LangChain's Recursive Character Text Splitter (chunk size: 1000, overlap: 200).  
   * Initialize the ChromaDB Persistent Client.  
   * Embed the chunks using Nomic Embed-Text via the Ollama client interface in LangChain, and store the resulting vectors in ChromaDB.  
4. **Inference Pipeline (app.py):**  
   * Design the Streamlit UI with st.session\_state for chat history.  
   * Implement the profanity check on user input (alt-profanity-check).  
   * Configure Hybrid Retrieval (Dense and BM25) and apply the RRF algorithm to rank the top 5-10 documents.  
   * Define the strict contextual guardrail system prompt, ensuring the LLM is instructed to answer *only* from the retrieved context.  
   * Connect to the local Llama 3 8B model via the Ollama client.  
   * Execute the RAG chain and display the response.

#### **Works cited**

1. Build a Retrieval Augmented Generation (RAG) App: Part 1 ..., accessed October 12, 2025, [https://python.langchain.com/docs/tutorials/rag/](https://python.langchain.com/docs/tutorials/rag/)  
2. RAG Architecture Made Simple: A Guide to Indexing and Inference ..., accessed October 13, 2025, [https://lathashreeh.medium.com/rag-architecture-keeping-it-simple-17677ee3ade9](https://lathashreeh.medium.com/rag-architecture-keeping-it-simple-17677ee3ade9)  
3. Automate advanced agentic RAG pipeline with Amazon SageMaker AI | Artificial Intelligence, accessed October 13, 2025, [https://aws.amazon.com/blogs/machine-learning/automate-advanced-agentic-rag-pipeline-with-amazon-sagemaker-ai/](https://aws.amazon.com/blogs/machine-learning/automate-advanced-agentic-rag-pipeline-with-amazon-sagemaker-ai/)  
4. RAG in Production: Deployment Strategies & Practical Considerations \- Coralogix, accessed October 13, 2025, [https://coralogix.com/ai-blog/rag-in-production-deployment-strategies-and-practical-considerations/](https://coralogix.com/ai-blog/rag-in-production-deployment-strategies-and-practical-considerations/)  
5. A Comprehensive Beginner's Guide to ChromaDB | by Syeedmdtalha \- Medium, accessed October 12, 2025, [https://medium.com/@syeedmdtalha/a-comprehensive-beginners-guide-to-chromadb-eb2fa22ee22f](https://medium.com/@syeedmdtalha/a-comprehensive-beginners-guide-to-chromadb-eb2fa22ee22f)  
6. ChromaDB: The Ultimate Beginner's Guide to Vector Databases for AI Applications | by Harsh Gupta | JavaScript in Plain English, accessed October 12, 2025, [https://javascript.plainenglish.io/chromadb-the-ultimate-beginners-guide-to-vector-databases-for-ai-applications-5dc59efd153b](https://javascript.plainenglish.io/chromadb-the-ultimate-beginners-guide-to-vector-databases-for-ai-applications-5dc59efd153b)  
7. Embeddings and Vector Databases With ChromaDB \- Real Python, accessed October 12, 2025, [https://realpython.com/chromadb-vector-database/](https://realpython.com/chromadb-vector-database/)  
8. Nomic Embed: Training a Reproducible Long Context Text Embedder \- arXiv, accessed October 13, 2025, [https://arxiv.org/html/2402.01613v2](https://arxiv.org/html/2402.01613v2)  
9. Introducing Nomic Embed: A Truly Open Embedding Model, accessed October 13, 2025, [https://www.nomic.ai/blog/posts/nomic-embed-text-v1](https://www.nomic.ai/blog/posts/nomic-embed-text-v1)  
10. RAG using Llama3, Langchain and ChromaDB \- Kaggle, accessed October 13, 2025, [https://www.kaggle.com/code/gpreda/rag-using-llama3-langchain-and-chromadb](https://www.kaggle.com/code/gpreda/rag-using-llama3-langchain-and-chromadb)  
11. RAG(Retrieval-Augmented Generation) using LLama3 \- GeeksforGeeks, accessed October 13, 2025, [https://www.geeksforgeeks.org/artificial-intelligence/rag-using-llama3/](https://www.geeksforgeeks.org/artificial-intelligence/rag-using-llama3/)  
12. Build a simple RAG chatbot with LangChain and Streamlit \- GitHub, accessed October 12, 2025, [https://github.com/Faridghr/Simple-RAG-Chatbot](https://github.com/Faridghr/Simple-RAG-Chatbot)  
13. RAG Project: Build an AI Onboarding Chatbot with Streamlit, LangChain, and ChromaDB, accessed October 12, 2025, [https://www.youtube.com/watch?v=WUUujm1MRQg](https://www.youtube.com/watch?v=WUUujm1MRQg)  
14. Chunking Strategies to Improve Your RAG Performance \- Weaviate, accessed October 13, 2025, [https://weaviate.io/blog/chunking-strategies-for-rag](https://weaviate.io/blog/chunking-strategies-for-rag)  
15. Mastering Chunking Strategies for RAG: Best Practices & Code Examples \- Databricks Community, accessed October 13, 2025, [https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)  
16. Visualizing Chunking Impacts in Agentic RAG with Agno, Qdrant, RAGAS and LlamaIndex: Part-1 | by M K Pavan Kumar | Towards Dev \- Medium, accessed October 13, 2025, [https://medium.com/towardsdev/visualizing-chunking-impacts-in-agentic-rag-with-agno-qdrant-ragas-and-llamaindex-part-1-7fe1ee31f5a5](https://medium.com/towardsdev/visualizing-chunking-impacts-in-agentic-rag-with-agno-qdrant-ragas-and-llamaindex-part-1-7fe1ee31f5a5)  
17. Building Performant RAG Applications for Production | LlamaIndex Python Documentation, accessed October 13, 2025, [https://developers.llamaindex.ai/python/framework/optimizing/production\_rag/](https://developers.llamaindex.ai/python/framework/optimizing/production_rag/)  
18. Use embedding models with Vertex AI RAG Engine \- Google Cloud, accessed October 12, 2025, [https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/use-embedding-models](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/use-embedding-models)  
19. BM25 Retriever | LlamaIndex Python Documentation, accessed October 13, 2025, [https://developers.llamaindex.ai/python/examples/retrievers/bm25\_retriever/](https://developers.llamaindex.ai/python/examples/retrievers/bm25_retriever/)  
20. BM25 as a retrieval method? : r/Rag \- Reddit, accessed October 13, 2025, [https://www.reddit.com/r/Rag/comments/1h159fo/bm25\_as\_a\_retrieval\_method/](https://www.reddit.com/r/Rag/comments/1h159fo/bm25_as_a_retrieval_method/)  
21. intfloat/e5-small-v2 \- Hugging Face, accessed October 12, 2025, [https://huggingface.co/intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2)  
22. Reciprocal Rerank Fusion Retriever | LlamaIndex Python ..., accessed October 12, 2025, [https://developers.llamaindex.ai/python/examples/retrievers/reciprocal\_rerank\_fusion/](https://developers.llamaindex.ai/python/examples/retrievers/reciprocal_rerank_fusion/)  
23. RAG \- Reworking Reranking Algorithms | by Rjnclarke \- Medium, accessed October 12, 2025, [https://medium.com/@rjnclarke/rag-reworking-reranking-182ff0d04755](https://medium.com/@rjnclarke/rag-reworking-reranking-182ff0d04755)  
24. Top 5 AI Gateways for OpenAI: OpenRouter Alternatives, accessed October 12, 2025, [https://research.aimultiple.com/ai-gateway/](https://research.aimultiple.com/ai-gateway/)  
25. LLM Pricing Comparison (2025): Live Rates \+ Cost Calculator \- Binary Verse AI, accessed October 12, 2025, [https://binaryverseai.com/llm-pricing-comparison/](https://binaryverseai.com/llm-pricing-comparison/)  
26. How to Run a Local LLM: Complete Guide to Setup & Best Models (2025) \- n8n Blog, accessed October 12, 2025, [https://blog.n8n.io/local-llm/](https://blog.n8n.io/local-llm/)  
27. Ollama Hardware Guide: CPU, GPU & RAM for Local LLMs \- Arsturn, accessed October 12, 2025, [https://www.arsturn.com/blog/ollama-hardware-guide-what-you-need-to-run-llms-locally](https://www.arsturn.com/blog/ollama-hardware-guide-what-you-need-to-run-llms-locally)  
28. RAG Chatbot With HuggingFace And Streamlit: Complete Tutorial \- Codecademy, accessed October 12, 2025, [https://www.codecademy.com/article/aichatbot-using-huggingface-rag-streamlit](https://www.codecademy.com/article/aichatbot-using-huggingface-rag-streamlit)  
29. RAG Based Conversational Chatbot Using Streamlit | by Ashish Malhotra | Medium, accessed October 12, 2025, [https://medium.com/@mrcoffeeai/rag-based-conversational-chatbot-using-streamlit-364c4c02c2f1](https://medium.com/@mrcoffeeai/rag-based-conversational-chatbot-using-streamlit-364c4c02c2f1)  
30. AI Security \- LLM Prompt & Response Guardrails \- Pangea, accessed October 12, 2025, [https://pangea.cloud/docs/ai-security/langchain-python-inference-guardrails](https://pangea.cloud/docs/ai-security/langchain-python-inference-guardrails)  
31. LLM Guardrails for Data Leakage, Prompt Injection, and More \- Confident AI, accessed October 13, 2025, [https://www.confident-ai.com/blog/llm-guardrails-the-ultimate-guide-to-safeguard-llm-systems](https://www.confident-ai.com/blog/llm-guardrails-the-ultimate-guide-to-safeguard-llm-systems)  
32. Handling profanity in text data with Python. | by Lubna Khan | Medium, accessed October 12, 2025, [https://lubna2004.medium.com/profanity-to-be-or-not-to-be-dd32d53648f7](https://lubna2004.medium.com/profanity-to-be-or-not-to-be-dd32d53648f7)  
33. New RAG template (with guardrails) \- AWS Prescriptive Guidance, accessed October 13, 2025, [https://docs.aws.amazon.com/prescriptive-guidance/latest/llm-prompt-engineering-best-practices/enhanced-template.html](https://docs.aws.amazon.com/prescriptive-guidance/latest/llm-prompt-engineering-best-practices/enhanced-template.html)  
34. RAG Makes Guardrails Unsafe? Investigating Robustness of Guardrails under RAG-style Contexts \- arXiv, accessed October 13, 2025, [https://arxiv.org/html/2510.05310v1](https://arxiv.org/html/2510.05310v1)  
35. Create a Teams AI Bot with RAG \- Microsoft Learn, accessed October 12, 2025, [https://learn.microsoft.com/en-us/microsoftteams/platform/toolkit/build-a-rag-bot-in-teams](https://learn.microsoft.com/en-us/microsoftteams/platform/toolkit/build-a-rag-bot-in-teams)