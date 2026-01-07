import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests
import json
from io import BytesIO

# Page config
st.set_page_config(
    page_title="PDF RAG System",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Helper Functions
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap
    return chunks

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_chunks(question, model, index, chunks, top_k=3):
    q_embedding = model.encode([question])
    distances, indices = index.search(q_embedding, top_k)
    return [chunks[i] for i in indices[0]], distances[0]

def query_llm(prompt, model_name, api_key=None):
    """Query different LLM APIs"""
    
    if model_name == "Groq - Llama 3.3 70B":
        if not api_key:
            return "⚠️ Please provide a Groq API key in the sidebar."
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    elif model_name == "Hugging Face - Mistral 7B":
        if not api_key:
            return "⚠️ Please provide a Hugging Face API key in the sidebar."
        
        url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"inputs": prompt, "parameters": {"max_new_tokens": 1024, "temperature": 0.7}}
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()[0]['generated_text'].replace(prompt, "").strip()
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    elif model_name == "Together AI - Llama 3.1 8B":
        if not api_key:
            return "⚠️ Please provide a Together AI API key in the sidebar."
        
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"❌ Error: {str(e)}"

# UI Layout
st.title("📄 PDF RAG System with Multiple LLMs")
st.markdown("Upload a PDF, ask questions, and get AI-powered answers!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    llm_model = st.selectbox(
        "Select LLM Model",
        [
            "Groq - Llama 3.3 70B",
            "Hugging Face - Mistral 7B",
            "Together AI - Llama 3.1 8B"
        ]
    )
    
    # API Key input
    st.markdown("---")
    st.markdown("### 🔑 API Keys")
    
    if "Groq" in llm_model:
        api_key = st.text_input("Groq API Key", type="password", help="Get free API key at https://console.groq.com")
        st.markdown("[Get Groq API Key](https://console.groq.com)")
    elif "Hugging Face" in llm_model:
        api_key = st.text_input("Hugging Face API Key", type="password", help="Get free API key at https://huggingface.co/settings/tokens")
        st.markdown("[Get HF API Key](https://huggingface.co/settings/tokens)")
    elif "Together" in llm_model:
        api_key = st.text_input("Together AI API Key", type="password", help="Get API key at https://api.together.xyz/settings/api-keys")
        st.markdown("[Get Together AI Key](https://api.together.xyz/settings/api-keys)")
    
    # Retrieval settings
    st.markdown("---")
    st.markdown("### 🔍 Retrieval Settings")
    top_k = st.slider("Number of chunks to retrieve", 1, 5, 3)
    chunk_size = st.slider("Chunk size (words)", 200, 1000, 500, step=100)
    overlap = st.slider("Chunk overlap (words)", 0, 200, 100, step=50)
    
    # Clear history button
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            # Extract text
            raw_text = extract_text_from_pdf(uploaded_file)
            
            # Show preview
            with st.expander("📄 Preview extracted text"):
                st.text_area("First 1000 characters", raw_text[:1000], height=200)
            
            # Chunk text
            st.session_state.chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
            st.success(f"✅ Created {len(st.session_state.chunks)} chunks")
            
            # Create embeddings
            if st.session_state.model is None:
                st.session_state.model = load_embedding_model()
            
            embeddings = st.session_state.model.encode(st.session_state.chunks, show_progress_bar=False)
            st.session_state.embeddings = np.array(embeddings)
            
            # Create FAISS index
            st.session_state.index = create_faiss_index(st.session_state.embeddings)
            
            st.success("✅ PDF processed and indexed successfully!")

with col2:
    st.header("💬 Ask Questions")
    
    if st.session_state.chunks is not None:
        # Question input
        question = st.text_input("Enter your question:", placeholder="What is this document about?")
        
        if st.button("🔍 Search & Answer", type="primary"):
            if question:
                with st.spinner("Searching and generating answer..."):
                    # Retrieve relevant chunks
                    retrieved_chunks, distances = retrieve_chunks(
                        question, 
                        st.session_state.model, 
                        st.session_state.index, 
                        st.session_state.chunks, 
                        top_k=top_k
                    )
                    
                    # Create context
                    context = "\n\n".join(retrieved_chunks)
                    
                    # Create prompt
                    prompt = f"""Answer the question using ONLY the context below. If the answer cannot be found in the context, say "I cannot find the answer in the provided document."

Context:
{context}

Question: {question}

Answer:"""
                    
                    # Get answer from LLM
                    answer = query_llm(prompt, llm_model, api_key)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "chunks": retrieved_chunks,
                        "distances": distances
                    })
            else:
                st.warning("⚠️ Please enter a question.")
    else:
        st.info("👈 Please upload a PDF file first.")

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.header("📜 Chat History")
    
    for idx, chat in enumerate(reversed(st.session_state.chat_history), 1):
        with st.expander(f"Q{len(st.session_state.chat_history) - idx + 1}: {chat['question']}", expanded=(idx==1)):
            st.markdown("**🤖 Answer:**")
            st.markdown(chat['answer'])
            
            st.markdown("**📚 Retrieved Chunks:**")
            for i, (chunk, dist) in enumerate(zip(chat['chunks'], chat['distances']), 1):
                st.markdown(f"*Chunk {i} (distance: {dist:.4f})*")
                st.text_area(f"chunk_{idx}_{i}", chunk[:300] + "...", height=100, key=f"chunk_{idx}_{i}", disabled=True)

# Footer
st.markdown("---")
st.markdown("""
### 📖 How to Use:
1. **Upload a PDF** document in the left panel
2. **Enter your API key** for the selected LLM model in the sidebar
3. **Type your question** in the question box
4. **Click 'Search & Answer'** to get AI-powered responses
5. View retrieved context chunks and similarity scores

### 🔑 Free API Keys:
- **Groq**: Fast inference, generous free tier - [Get Key](https://console.groq.com)
- **Hugging Face**: Free tier available - [Get Key](https://huggingface.co/settings/tokens)
- **Together AI**: $25 free credits - [Get Key](https://api.together.xyz/settings/api-keys)
""")