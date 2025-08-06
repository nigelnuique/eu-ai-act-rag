import streamlit as st
import tiktoken
from openai import OpenAI
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Retriever imports
from retriever import Retriever, RetrievalConfig

# LangChain imports for FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="EU AI Act Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

@st.cache_resource
def load_retriever(api_key):
    """Load the retriever with structural chunking."""
    try:
        print("🔧 Loading retriever with structural chunking...")
        
        # Import the retriever
        from retriever import Retriever, RetrievalConfig
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings
        
        # Initialize with structural chunking enabled
        config = RetrievalConfig(
            use_structural_chunking=True,
            chunk_size=1500,
            chunk_overlap=200,
            final_top_k=5
        )
        retriever = Retriever(api_key, config)
        
        # Load the vector store
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=api_key
        )
        
        try:
            vector_store = FAISS.load_local("faiss_vectordb", embeddings, allow_dangerous_deserialization=True)
            retriever.vector_store = vector_store
            
            # Rebuild BM25 and ensemble retrievers from the loaded vector store
            print("🔄 Rebuilding BM25 retriever for hybrid search...")
            
            # Extract documents from the vector store
            docs = []
            for i in range(len(vector_store.docstore._dict)):
                doc_id = list(vector_store.docstore._dict.keys())[i]
                doc = vector_store.docstore._dict[doc_id]
                docs.append(doc)
            
            # Rebuild BM25 retriever
            from langchain_community.retrievers import BM25Retriever
            from langchain.retrievers import EnsembleRetriever
            
            retriever.bm25_retriever = BM25Retriever.from_documents(docs)
            retriever.bm25_retriever.k = retriever.config.top_k_bm25
            
            # Rebuild ensemble retriever
            retriever.ensemble_retriever = EnsembleRetriever(
                retrievers=[
                    vector_store.as_retriever(search_kwargs={"k": retriever.config.top_k_semantic}), 
                    retriever.bm25_retriever
                ],
                weights=[retriever.config.semantic_weight, retriever.config.bm25_weight]
            )
                
            print("✅ Retriever loaded successfully!")
            return retriever
            
        except Exception as e:
            print(f"⚠️ Could not load existing vector store: {e}")
            print("💡 Please run `python retriever.py` to create a new enhanced vector database")
            return None
            
    except Exception as e:
        print(f"❌ Error loading retriever: {e}")
        return None

@st.cache_resource  
def load_fallback_vector_store(api_key):
    """Load fallback FAISS vector store if enhanced version not available."""
    try:
        embeds = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
        vector_store = FAISS.load_local("faiss_vectordb", embeds, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        st.error(f"Failed to load fallback vector database: {str(e)}")
        return None

def find_relevant_documents(query: str, retriever, top_k: int = 5) -> List[Dict[str, Any]]:
    """Find relevant documents using the structural retriever."""
    try:
        if hasattr(retriever, 'retrieve_with_reranking'):
            # Use the retrieval method with reranking
            results = retriever.retrieve_with_reranking(query)
            return results[:top_k]
        else:
            # Fallback to basic retrieval
            if not retriever.vector_store:
                return []
            
            docs = retriever.vector_store.similarity_search(query, k=top_k)
            return [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": 1.0  # Default score
                }
                for doc in docs
            ]
    except Exception as e:
        print(f"❌ Error in document retrieval: {e}")
        return []

def find_relevant_documents_fallback(query: str, vector_store: FAISS, top_k: int = 5) -> List[Dict[str, Any]]:
    """Fallback document retrieval using basic FAISS."""
    try:
        docs = vector_store.similarity_search(query, k=top_k)
        
        relevant_docs = []
        for doc in docs:
            doc_data = {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            relevant_docs.append(doc_data)
        
        return relevant_docs
    except Exception as e:
        st.error(f"Error in fallback retrieval: {str(e)}")
        return []

def generate_answer(query: str, relevant_docs: List[Dict[str, Any]], api_key: str) -> str:
    """Generate answer with better context handling."""
    if not api_key:
        return "Please provide an OpenAI API key to generate answers."
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Prepare context
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            metadata = doc.get('metadata', {})
            
            # Build context with metadata
            context_header = f"\n--- Document {i+1} ---"
            if metadata.get('article'):
                context_header += f"\nArticle: {metadata['article']}"
            if metadata.get('content_type'):
                context_header += f"\nType: {metadata['content_type']}"
            if doc.get('relevance_score'):
                context_header += f"\nRelevance Score: {doc['relevance_score']:.2f}"
            
            context_parts.append(f"{context_header}\n{doc['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt with better instructions
        prompt = f"""Based on the following context from the EU AI Act, please provide a comprehensive and accurate answer to the question.

Context from EU AI Act:
{context}

Question: {query}

Instructions:
1. Provide a detailed answer based on the context provided
2. Reference specific articles when mentioned in the context
3. If the context mentions different types of content (definitions, requirements, prohibitions, etc.), organize your answer accordingly
4. If the context doesn't contain sufficient information, clearly state what information is missing
5. Use clear, professional language appropriate for legal/regulatory content

Answer:"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Better model for complex legal text
            messages=[
                {"role": "system", "content": "You are an expert on the EU AI Act. Provide accurate, comprehensive answers based on the provided context. Always reference specific articles and sections when available. Organize your response clearly and professionally."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,  # More tokens for comprehensive answers
            temperature=0.3  # Lower temperature for more consistent legal responses
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Main app
st.title("EU AI Act Assistant")
st.markdown("**RAG-powered assistant for the European Union's Artificial Intelligence Act**")

# Add info about the EU AI Act
st.info("""
**About the EU AI Act:**

The European Union's Artificial Intelligence Act is the world's first comprehensive AI regulation, adopted in 2024. 
It establishes harmonized rules for AI systems based on their risk levels, sets obligations for AI providers and deployers, 
and aims to ensure AI development and use is safe, transparent, and respects fundamental rights.

This assistant helps you navigate and understand the complex legal framework through intelligent search and question answering.
""")

# Check if system is ready
if st.session_state.api_key_set:
    # Try to load retriever first
    if not st.session_state.retriever:
        st.session_state.retriever = load_retriever(st.session_state.api_key)
    
    # Fallback to basic vector store if retriever not available
    if not st.session_state.retriever and not st.session_state.vector_store:
        st.session_state.vector_store = load_fallback_vector_store(st.session_state.api_key)
    
    # Check what's available
    has_retriever = st.session_state.retriever is not None
    has_fallback = st.session_state.vector_store is not None
    
    if has_retriever or has_fallback:
        # Sidebar for API key
        with st.sidebar:
            st.header("System Status")
            
            if has_retriever:
                st.success("System Ready")
                st.markdown("Full search capabilities active")
            elif has_fallback:
                st.success("System Ready")
                st.markdown("Basic search active")
            
            st.header("API Key")
            st.success("API key configured")
            if st.button("Change API Key"):
                st.session_state.api_key_set = False
                st.session_state.api_key = ""
                st.session_state.retriever = None
                st.session_state.vector_store = None
                st.rerun()
        
        # Main interface
        # Example questions
        st.subheader("Example Questions")
        example_questions = [
            "How is the AI Act enforced and what are the AI Office's responsibilities?",
            "What constitutes a high-risk AI system and what are the specific requirements?",
            "What AI practices are completely prohibited under the Act?",
            "What are the key definitions in the AI Act for artificial intelligence?",
            "What documentation and compliance obligations do AI providers have?",
            "What are the penalties and sanctions for non-compliance?",
            "How does the Act address general-purpose AI models?",
            "What transparency obligations exist for AI system deployers?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(question, key=f"example_{i}"):
                    st.session_state.current_query = question
        
        # Query input
        query = st.text_input(
            "Your Question:", 
            placeholder="Ask anything about the EU AI Act...",
            value=st.session_state.get('current_query', ''),
            help="The system will automatically expand your query and find the most relevant content"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                show_metadata = st.checkbox("Show document metadata", value=True)
                show_scores = st.checkbox("Show relevance scores", value=has_retriever)
            with col2:
                max_results = st.slider("Max results to show", 3, 8, 5)
        
        if query and st.button("Get Answer", type="primary"):
            with st.spinner("Searching..."):
                # Use retrieval if available
                if has_retriever:
                    relevant_docs = find_relevant_documents(query, st.session_state.retriever)
                else:
                    relevant_docs = find_relevant_documents_fallback(query, st.session_state.vector_store, max_results)
                
                if relevant_docs:
                    # Generate answer first
                    answer = generate_answer(query, relevant_docs, st.session_state.api_key)
                    
                    # Display answer
                    st.subheader("Answer")
                    st.markdown(answer)
                    
                    # Display relevant documents
                    st.subheader("Source Documents")
                    
                    for i, doc in enumerate(relevant_docs[:max_results]):
                        metadata = doc.get('metadata', {})
                        
                        # Build section title
                        section_title = f"Section {i+1}"
                        if metadata.get('chunk_id'):
                            section_title += f" (Chunk {metadata['chunk_id']})"
                        
                        with st.expander(section_title, expanded=i < 2):  # Expand first 2 results
                            # Show metadata if requested
                            if show_metadata:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if metadata.get('article'):
                                        st.markdown(f"**Article:** {metadata['article']}")
                                with col2:
                                    if metadata.get('content_type'):
                                        st.markdown(f"**Type:** {metadata['content_type']}")
                                with col3:
                                    if show_scores and doc.get('relevance_score'):
                                        st.markdown(f"**Score:** {doc['relevance_score']:.2f}")
                                
                                if metadata.get('chunk_strategy'):
                                    st.markdown(f"**Strategy:** {metadata['chunk_strategy']}")
                            
                            # Show document text
                            st.write(doc['text'])
                else:
                    st.warning("No relevant documents found. Try rephrasing your question or using different keywords.")
    else:
        st.error("Failed to load retrieval system. Please check your setup and API key.")

else:
    # Sidebar for API key input
    with st.sidebar:
        st.header("API Key Setup")
        
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password", 
            placeholder="sk-..."
        )
        
        if st.button("Set API Key"):
            if api_key and api_key.strip():
                st.session_state.api_key = api_key
                st.session_state.api_key_set = True
                st.success("API key configured")
                st.rerun()
            else:
                st.error("Please enter a valid API key")
    
    st.info("Please enter your OpenAI API key in the sidebar to start using the assistant.")
    
    # Show getting started info
    st.markdown("""
    ### Getting Started:
    
    1. **Enter API Key**: Add your OpenAI API key in the sidebar
    2. **Start Asking Questions**: The system is ready to help you understand the EU AI Act
    
    ### What You Can Ask:
    - Questions about AI system classifications and requirements
    - Information on compliance obligations and enforcement
    - Details about prohibited AI practices
    - Explanations of key definitions and legal concepts
    """)

# Footer
st.markdown("---")
st.markdown("*RAG-powered assistant for EU AI Act guidance*") 