import streamlit as st
from src.document_reader import load_all_documents
from src.Chunking import chunk_documents
from src.Embedding import build_or_load_faiss
from src.similarity_reranker import similarity_reranker
from src.querry_rewrite import rewrite_query
from src.Agentic_RAG import agentic_rag
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PDF/Video RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=100)
    st.title("⚙️ Configuration")
    
    # Document path input
    st.subheader("📁 Document Source")
    default_path = r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\PDF_Video Reader\PDF_Samples"
    doc_path = st.text_input(
        "Document Directory Path",
        value=default_path,
        help="Enter the path to your documents folder"
    )
    
    # Load documents button
    if st.button("🔄 Load/Reload Documents", use_container_width=True):
        with st.spinner("Loading and processing documents..."):
            try:
                path = Path(doc_path)
                if not path.exists():
                    st.error("❌ Path does not exist!")
                else:
                    # Load documents
                    documents = load_all_documents(path)
                    st.success(f"✅ Loaded {len(documents)} documents")
                    
                    # Chunk documents
                    chunked_docs = chunk_documents(documents)
                    st.success(f"✅ Created {len(chunked_docs)} chunks")
                    
                    # Build/load vectorstore
                    st.session_state.vectorstore = build_or_load_faiss(chunked_docs)
                    st.session_state.documents_loaded = True
                    st.success("✅ Vector store ready!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Document loading error: {e}")
    
    # Status indicator
    st.divider()
    st.subheader("📊 System Status")
    if st.session_state.documents_loaded:
        st.success("🟢 Documents Loaded")
    else:
        st.warning("🟡 No Documents Loaded")
    
    # Clear history button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        faiss_k = st.slider("FAISS Top K", 5, 50, 20, help="Number of initial retrieval results")
        rerank_n = st.slider("Rerank Top N", 1, 20, 5, help="Number of results after reranking")
        max_attempts = st.slider("Max RAG Attempts", 1, 5, 3, help="Maximum agentic RAG iterations")

# Main content
st.markdown('<div class="main-header">📚 Intelligent Document Q&A System</div>', unsafe_allow_html=True)
st.markdown("Ask questions about your documents and get AI-powered answers with agentic RAG")

# Create tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Query Analysis", "ℹ️ About"])

with tab1:
    # Chat interface
    if not st.session_state.documents_loaded:
        st.info("👈 Please load documents from the sidebar to begin")
    else:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, chat in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(chat["question"])
                with st.chat_message("assistant"):
                    st.write(chat["answer"])
                    if "rewritten_query" in chat:
                        with st.expander("🔄 View Rewritten Query"):
                            st.code(chat["rewritten_query"])
        
        # Question input
        question = st.chat_input("Ask a question about your documents...")
        
        if question:
            with st.chat_message("user"):
                st.write(question)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Similarity reranking
                        status_text.text("🔍 Retrieving relevant documents...")
                        progress_bar.progress(25)
                        reranked_context = similarity_reranker(
                            st.session_state.vectorstore, 
                            question,
                            faiss_k=faiss_k,
                            top_n=rerank_n
                        )
                        
                        # Step 2: Query rewriting
                        status_text.text("✍️ Rewriting query for better results...")
                        progress_bar.progress(50)
                        rewritten_query = rewrite_query(question, reranked_context)
                        
                        # Step 3: Agentic RAG
                        status_text.text("🤖 Generating answer with agentic RAG...")
                        progress_bar.progress(75)
                        final_answer = agentic_rag(rewritten_query, st.session_state.vectorstore)
                        
                        progress_bar.progress(100)
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Display answer
                        st.write(final_answer)
                        
                        # Store in history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": final_answer,
                            "rewritten_query": rewritten_query
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        logger.error(f"Query processing error: {e}")

with tab2:
    st.subheader("🔍 Query Analysis & Testing")
    
    if not st.session_state.documents_loaded:
        st.info("👈 Please load documents from the sidebar first")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            test_query = st.text_area(
                "Enter a test query:",
                placeholder="e.g., Explain the time series analysis",
                height=100
            )
            
            if st.button("🔬 Analyze Query", use_container_width=True):
                if test_query:
                    with st.spinner("Analyzing..."):
                        try:
                            # Get reranked context
                            reranked_docs = similarity_reranker(
                                st.session_state.vectorstore, 
                                test_query,
                                faiss_k=faiss_k,
                                top_n=rerank_n
                            )
                            
                            # Rewrite query
                            rewritten = rewrite_query(test_query, reranked_docs)
                            
                            # Display results
                            st.success("✅ Analysis Complete!")
                            
                            st.markdown("**Original Query:**")
                            st.info(test_query)
                            
                            st.markdown("**Rewritten Query:**")
                            st.success(rewritten)
                            
                            st.markdown("**Retrieved Documents:**")
                            for i, doc in enumerate(reranked_docs, 1):
                                with st.expander(f"Document {i}"):
                                    st.write(doc.page_content)
                                    if hasattr(doc, 'metadata'):
                                        st.json(doc.metadata)
                                        
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("Please enter a query to analyze")

with tab3:
    st.subheader("ℹ️ About This System")
    
    st.markdown("""
    ### 🎯 Features
    
    - **📄 Multi-format Support**: Process PDFs, videos, and other document types
    - **🔍 Semantic Search**: Uses FAISS vector store with sentence transformers
    - **🎨 Query Rewriting**: Improves search accuracy through intelligent query reformulation
    - **🤖 Agentic RAG**: Iterative retrieval for comprehensive answers
    - **📊 Similarity Reranking**: Enhanced result relevance using cross-encoders
    
    ### 🛠️ Technology Stack
    
    - **Frontend**: Streamlit
    - **LLM**: Ollama (Llama 3.1)
    - **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
    - **Vector Store**: FAISS
    - **Framework**: LangChain
    
    ### 📖 How It Works
    
    1. **Document Loading**: Reads documents from specified directory
    2. **Chunking**: Splits documents into manageable pieces
    3. **Embedding**: Converts chunks to vector representations
    4. **Retrieval**: Finds relevant chunks using semantic similarity
    5. **Reranking**: Re-scores results for better relevance
    6. **Query Rewriting**: Reformulates query for improved retrieval
    7. **Agentic RAG**: Iteratively retrieves until sufficient context is found
    8. **Answer Generation**: LLM generates final answer from context
    
    ### 💡 Tips for Best Results
    
    - Use specific, well-formed questions
    - Load relevant documents only
    - Adjust retrieval parameters in advanced settings
    - Check query analysis to understand system behavior
    """)
    
    st.divider()
    st.caption("Built with ❤️ using Streamlit and LangChain")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Documents Loaded", "Yes" if st.session_state.documents_loaded else "No")
with col2:
    st.metric("Chat History", len(st.session_state.chat_history))
with col3:
    st.metric("Status", "🟢 Ready" if st.session_state.documents_loaded else "🟡 Waiting")