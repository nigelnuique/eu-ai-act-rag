#!/usr/bin/env python3
"""
RAG Retriever for EU AI Act with hybrid search capabilities.
Features:
- Hybrid search (semantic + keyword)
- Query expansion and preprocessing
- Reranking with structural awareness
- Structural chunking strategies
- Multiple retrieval strategies
"""

import os
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import numpy as np
from dataclasses import dataclass

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Import our structural chunker
from structural_chunker import StructuralChunker, ChunkConfig

# Load environment variables
load_dotenv()

@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    chunk_size: int = 1500  # Larger chunks for legal documents
    chunk_overlap: int = 300  # More overlap to preserve context
    top_k_semantic: int = 10  # More candidates for reranking
    top_k_bm25: int = 10  # BM25 candidates
    final_top_k: int = 5  # Final results after reranking
    semantic_weight: float = 0.7  # Weight for semantic search
    bm25_weight: float = 0.3  # Weight for BM25 search
    use_structural_chunking: bool = True  # Use structural chunking


class Retriever:
    """Retriever with multiple search strategies."""
    
    def __init__(self, api_key: str, config: RetrievalConfig = None):
        self.api_key = api_key
        self.config = config or RetrievalConfig()
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",  # Better model for legal text
            openai_api_key=api_key
        )
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
    def preprocess_query(self, query: str) -> List[str]:
        """Expand and preprocess query for better retrieval."""
        # Base query
        queries = [query]
        
        # Add variations for legal terminology
        legal_expansions = {
            "enforce": ["enforcement", "enforced", "enforcing", "compliance", "implementation"],
            "AI system": ["artificial intelligence system", "AI systems", "artificial intelligence systems"],
            "provider": ["providers", "developer", "developers", "deployer", "deployers"],
            "high-risk": ["high risk", "high-risk AI", "high risk AI", "risky AI systems"],
            "prohibited": ["forbidden", "banned", "not allowed", "restricted"],
            "obligation": ["obligations", "requirement", "requirements", "duty", "duties"],
            "penalty": ["penalties", "fine", "fines", "sanction", "sanctions", "punishment"],
            "definition": ["definitions", "means", "defined as", "shall mean"],
        }
        
        query_lower = query.lower()
        for term, expansions in legal_expansions.items():
            if term in query_lower:
                for expansion in expansions:
                    expanded_query = query_lower.replace(term, expansion)
                    if expanded_query not in [q.lower() for q in queries]:
                        queries.append(expanded_query)
        
        # Add context-specific variations
        if "enforce" in query_lower:
            queries.extend([
                query + " AI Office responsibilities",
                query + " compliance monitoring",
                query + " supervisory authorities"
            ])
        
        if "definition" in query_lower or "what is" in query_lower:
            queries.extend([
                query + " Article definition",
                query + " legal meaning"
            ])
            
        return queries[:5]  # Limit to top 5 variations
    
    def create_chunks(self, clean_text: str) -> List[Document]:
        """Create chunks with enhanced structural strategies for legal documents."""
        print(f"🔧 Creating chunks with structural awareness...")
        
        if self.config.use_structural_chunking:
            # Use the new structural chunker
            chunk_config = ChunkConfig(
                max_chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                min_chunk_size=100
            )
            structural_chunker = StructuralChunker(chunk_config)
            documents = structural_chunker.chunk_document(clean_text)
        else:
            # Fallback to original chunking methods
            documents = []
            
            # Strategy 1: Article-aware splitting
            article_docs = self._split_by_articles(clean_text)
            documents.extend(article_docs)
            
            # Strategy 2: Semantic splitting with larger chunks
            semantic_docs = self._split_semantically(clean_text)
            documents.extend(semantic_docs)
            
            # Remove duplicates and very short chunks
            documents = self._deduplicate_documents(documents)
        
        print(f"✅ Created {len(documents)} chunks")
        return documents
    
    def _split_by_articles(self, text: str) -> List[Document]:
        """Split text by articles to preserve legal structure (legacy method)."""
        documents = []
        
        # Find article boundaries
        article_pattern = r'\n\s*=== Article \d+[A-Za-z]?.*? ===\s*\n'
        articles = re.split(article_pattern, text)
        
        for i, article_text in enumerate(articles):
            if len(article_text.strip()) < 100:
                continue
                
            # Extract article number if present
            article_match = re.search(r'Article (\d+[A-Za-z]?)', article_text)
            article_num = article_match.group(1) if article_match else f"section_{i}"
            
            # Further split large articles
            if len(article_text) > self.config.chunk_size * 2:
                sub_chunks = self._split_large_article(article_text, article_num)
                documents.extend(sub_chunks)
            else:
                doc = Document(
                    page_content=article_text.strip(),
                    metadata={
                        "chunk_id": f"article_{article_num}",
                        "article": article_num,
                        "content_type": self._classify_content(article_text),
                        "char_length": len(article_text),
                        "source": "EU_AI_Act",
                        "chunk_strategy": "article_aware"
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _split_large_article(self, article_text: str, article_num: str) -> List[Document]:
        """Split large articles into smaller chunks while preserving context."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
            length_function=len,
        )
        
        chunks = splitter.split_text(article_text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk.strip(),
                metadata={
                    "chunk_id": f"article_{article_num}_part_{i+1}",
                    "article": article_num,
                    "part": i+1,
                    "content_type": self._classify_content(chunk),
                    "char_length": len(chunk),
                    "source": "EU_AI_Act",
                    "chunk_strategy": "article_split"
                }
            )
            documents.append(doc)
        
        return documents
    
    def _split_semantically(self, text: str) -> List[Document]:
        """Split text semantically with larger chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=[
                "\n\n=== ",  # Article headers
                "\n\n--- ",  # Section headers  
                "\n\n",      # Paragraph breaks
                "\n",        # Line breaks
                ". ",        # Sentence ends
                "! ",        # Exclamation ends
                "? ",        # Question ends
                "; ",        # Semicolon breaks
                " ",         # Word breaks
                ""           # Character breaks
            ],
            length_function=len,
        )
        
        chunks = splitter.split_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 100:  # Skip very short chunks
                continue
                
            doc = Document(
                page_content=chunk.strip(),
                metadata={
                    "chunk_id": f"semantic_{i}",
                    "article": self._extract_article_number(chunk),
                    "chapter": self._extract_chapter_number(chunk),
                    "section": self._extract_section_number(chunk),
                    "content_type": self._classify_content(chunk),
                    "char_length": len(chunk),
                    "source": "EU_AI_Act",
                    "chunk_strategy": "semantic"
                }
            )
            documents.append(doc)
        
        return documents
    
    def _classify_content(self, text: str) -> str:
        """Classify the type of content in the chunk."""
        text_lower = text.lower()
        
        if re.search(r'\bdefinition\b|\bmeans\b|\bshall mean\b', text_lower):
            return "definition"
        elif re.search(r'\bshall\b|\bmust\b|\brequired\b|\bobligation\b', text_lower):
            return "requirement"
        elif re.search(r'\bprohibited\b|\bforbidden\b|\bnot allowed\b|\bbanned\b', text_lower):
            return "prohibition"
        elif re.search(r'\bpenalty\b|\bfine\b|\bsanction\b|\bpunishment\b', text_lower):
            return "enforcement"
        elif re.search(r'\bhigh-risk\b|\bhigh risk\b', text_lower):
            return "high_risk"
        elif re.search(r'\bai office\b|\bsupervisory\b|\bcompliance\b', text_lower):
            return "governance"
        else:
            return "general"
    
    def _extract_article_number(self, text: str) -> str:
        """Extract article number from text."""
        match = re.search(r'Article\s+(\d+[A-Za-z]?)', text, re.IGNORECASE)
        return match.group(1) if match else ""
    
    def _extract_chapter_number(self, text: str) -> str:
        """Extract chapter number from text."""
        match = re.search(r'Chapter\s+([IVX]+|\d+)', text, re.IGNORECASE)
        return match.group(1) if match else ""
    
    def _extract_section_number(self, text: str) -> str:
        """Extract section number from text."""
        match = re.search(r'Section\s+(\d+)', text, re.IGNORECASE)
        return match.group(1) if match else ""
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content similarity."""
        unique_docs = []
        seen_hashes = set()
        
        for doc in documents:
            # Create a hash of the normalized content
            normalized_content = re.sub(r'\s+', ' ', doc.page_content.lower().strip())
            content_hash = hash(normalized_content)
            
            if content_hash not in seen_hashes and len(doc.page_content.strip()) >= 100:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    def build_vectorstore(self, documents: List[Document]) -> FAISS:
        """Build FAISS vector store."""
        print(f"🔧 Building FAISS vector store with {len(documents)} documents...")
        
        # Create FAISS vector store
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Also create BM25 retriever for hybrid search
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = self.config.top_k_bm25
        
        # Create ensemble retriever combining both
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_store.as_retriever(search_kwargs={"k": self.config.top_k_semantic}), 
                       self.bm25_retriever],
            weights=[self.config.semantic_weight, self.config.bm25_weight]
        )
        
        print("✅ Vector store and hybrid retriever built!")
        return vector_store
    
    def retrieve_with_reranking(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve documents with query expansion and structural reranking."""
        if not self.vector_store or not self.ensemble_retriever:
            raise ValueError("Vector store not initialized. Call build_vectorstore first.")
        
        # Expand query
        expanded_queries = self.preprocess_query(query)
        
        # Collect candidates from all query variations
        all_candidates = []
        seen_content = set()
        
        for expanded_query in expanded_queries:
            # Use ensemble retriever for hybrid search
            docs = self.ensemble_retriever.get_relevant_documents(expanded_query)
            
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_candidates.append({
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "query_used": expanded_query
                    })
        
        # Reranking with structural awareness
        reranked_docs = self._rerank_documents_with_structure(query, all_candidates)
        
        return reranked_docs[:self.config.final_top_k]
    
    def _rerank_documents_with_structure(self, original_query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reranking that considers structural metadata."""
        query_terms = set(original_query.lower().split())
        
        for doc in candidates:
            score = 0
            text_lower = doc["text"].lower()
            metadata = doc["metadata"]
            
            # Base term frequency score
            for term in query_terms:
                score += text_lower.count(term) * 1.0
            
            # Enhanced content type bonuses
            content_type = metadata.get("content_type", "general")
            if "definition" in original_query.lower() and content_type == "definition":
                score += 10.0  # Strong boost for definition queries
            elif "enforce" in original_query.lower() and content_type == "enforcement":
                score += 8.0
            elif "prohibit" in original_query.lower() and content_type == "prohibition":
                score += 8.0
            elif "requirement" in original_query.lower() and content_type == "requirement":
                score += 6.0
            elif "high-risk" in original_query.lower() and content_type == "high_risk":
                score += 6.0
            
            # Structural level bonuses
            structural_level = metadata.get("structural_level", "")
            if structural_level == "recital":
                # Recitals are good for background/context questions
                if any(word in original_query.lower() for word in ["why", "background", "purpose", "context"]):
                    score += 5.0
                else:
                    score += 1.0  # Slight preference for recitals
            elif structural_level == "article":
                # Articles are the main operative provisions
                score += 4.0
            elif structural_level == "annex":
                # Annexes contain detailed lists and specifications
                if any(word in original_query.lower() for word in ["list", "annex", "detailed", "specific"]):
                    score += 6.0
                else:
                    score += 2.0
            
            # Document section bonuses
            doc_section = metadata.get("document_section", "")
            if doc_section == "preamble" and any(word in original_query.lower() for word in ["whereas", "recital", "background"]):
                score += 5.0
            elif doc_section == "enacting_terms":
                score += 3.0  # Main content gets preference
            
            # Chapter and article context
            if metadata.get("chapter"):
                score += 2.0  # Has clear chapter context
            if metadata.get("article"):
                score += 2.0  # Has clear article context
            
            # Chunk strategy bonus
            chunk_strategy = metadata.get("chunk_strategy", "")
            if chunk_strategy.startswith("structural_"):
                score += 3.0  # Prefer structurally-aware chunks
            
            # Length penalties for very long or very short chunks
            char_length = metadata.get("char_length", 0)
            if char_length < 200:
                score -= 2.0  # Too short
            elif char_length > 3000:
                score -= 1.0  # Too long
            
            doc["relevance_score"] = score
        
        # Sort by relevance score
        return sorted(candidates, key=lambda x: x["relevance_score"], reverse=True)
    
    def save_vectorstore(self, vector_store: FAISS, path: str = "faiss_vectordb"):
        """Save the vector store."""
        vector_store.save_local(path)
        print(f"✅ Vector store saved to {path}")


def create_vectordb():
    """Create vector database with superior retrieval quality and structural chunking."""
    print("🚀 Creating EU AI Act Vector Database with Structural Chunking")
    print("=" * 70)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OpenAI API key not found!")
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            print("❌ API key is required. Exiting.")
            return
    
    # Initialize retriever with structural chunking enabled
    config = RetrievalConfig(use_structural_chunking=True)
    retriever = Retriever(api_key, config)
    
    # Download and extract content (reuse existing function)
    from create_html_vectordb_langchain import download_html_document, extract_clean_content
    
    html_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"
    html_content = download_html_document(html_url)
    
    if not html_content:
        print("❌ Failed to download HTML document")
        return
    
    clean_text = extract_clean_content(html_content)
    if not clean_text:
        print("❌ Failed to extract clean content")
        return
    
    # Create chunks using structural approach
    documents = retriever.create_chunks(clean_text)
    
    # Build vector store
    vector_store = retriever.build_vectorstore(documents)
    
    # Save vector store
    retriever.save_vectorstore(vector_store)
    
    # Test retrieval with structural awareness
    print("\n🔍 Testing structural retrieval...")
    test_queries = [
        "What is the purpose of this regulation?",  # Should find recital (1)
        "How is the AI Act enforced?",
        "What are the definitions in the AI Act?", 
        "What AI systems are prohibited?",
        "What are the obligations for AI providers?",
        "What are high-risk AI systems?",
        "What annexes are included?",  # Should find annex content
    ]
    
    # Set the vector store for the retriever
    retriever.vector_store = vector_store
    
    for query in test_queries:
        print(f"\n📋 Results for: '{query}'")
        results = retriever.retrieve_with_reranking(query)
        
        for i, doc in enumerate(results):
            metadata = doc['metadata']
            print(f"\n{i+1}. Score: {doc.get('relevance_score', 0):.2f}")
            print(f"   Section: {metadata.get('document_section', 'N/A')}")
            print(f"   Level: {metadata.get('structural_level', 'N/A')}")
            print(f"   Article: {metadata.get('article', 'N/A')}")
            print(f"   Chapter: {metadata.get('chapter', 'N/A')}")
            print(f"   Recital: {metadata.get('recital_number', 'N/A')}")
            print(f"   Type: {metadata.get('content_type', 'general')}")
            print(f"   Strategy: {metadata.get('chunk_strategy', 'unknown')}")
            print(f"   Text: {doc['text'][:200]}...")
    
    print("\n" + "=" * 70)
    print("🎉 Vector database created successfully!")
    print("✅ Features: Structural chunking, hybrid search, query expansion, reranking")
    
    return retriever, vector_store


if __name__ == "__main__":
    create_vectordb() 