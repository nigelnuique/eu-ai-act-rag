#!/usr/bin/env python3
"""
Structural Document Chunker for EU AI Act RAG System.
This module implements structure-aware chunking based on the document's logical organization:
- Preamble: Split by individual recitals
- Enacting terms: Split by articles within chapters
- Concluding formulas: As separate sections
- Annexes: As distinct sections
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class ChunkConfig:
    """Configuration for structural chunking."""
    max_chunk_size: int = 2000  # Maximum characters per chunk
    chunk_overlap: int = 200    # Overlap between chunks
    min_chunk_size: int = 100   # Minimum viable chunk size
    preserve_structure: bool = True  # Keep structural boundaries


class StructuralChunker:
    """Enhanced chunker that respects document structure."""
    
    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
        
    def chunk_document(self, text: str) -> List[Document]:
        """Main method to chunk the entire document structurally."""
        print("🔧 Starting structural document chunking...")
        
        # Parse document structure
        structure = self._parse_document_structure(text)
        
        # Create chunks based on structure
        documents = []
        
        # Process preamble recitals
        if structure.get('preamble'):
            recital_docs = self._chunk_preamble_recitals(structure['preamble'])
            documents.extend(recital_docs)
            print(f"✅ Created {len(recital_docs)} recital chunks")
        
        # Process enacting terms (chapters and articles)
        if structure.get('enacting_terms'):
            article_docs = self._chunk_enacting_terms(structure['enacting_terms'])
            documents.extend(article_docs)
            print(f"✅ Created {len(article_docs)} article chunks")
        
        # Process concluding formulas
        if structure.get('concluding_formulas'):
            concluding_docs = self._chunk_concluding_formulas(structure['concluding_formulas'])
            documents.extend(concluding_docs)
            print(f"✅ Created {len(concluding_docs)} concluding formula chunks")
        
        # Process annexes
        if structure.get('annexes'):
            annex_docs = self._chunk_annexes(structure['annexes'])
            documents.extend(annex_docs)
            print(f"✅ Created {len(annex_docs)} annex chunks")
        
        print(f"🎉 Total structural chunks created: {len(documents)}")
        return documents
    
    def _parse_document_structure(self, text: str) -> Dict[str, Any]:
        """Parse the document to identify structural elements."""
        print("📋 Parsing document structure...")
        
        structure = {
            'preamble': None,
            'enacting_terms': {},
            'concluding_formulas': None,
            'annexes': {}
        }
        
        # Find major structural boundaries
        
        # 1. Find preamble (from "Whereas:" to "HAVE ADOPTED THIS REGULATION:")
        preamble_match = re.search(
            r'Whereas:\s*(.*?)\s*HAVE ADOPTED THIS REGULATION:', 
            text, 
            re.DOTALL | re.IGNORECASE
        )
        if preamble_match:
            structure['preamble'] = preamble_match.group(1).strip()
        
        # 2. Find enacting terms (from "HAVE ADOPTED THIS REGULATION:" to annexes or end)
        enacting_start = text.find("HAVE ADOPTED THIS REGULATION:")
        if enacting_start != -1:
            # Look for start of annexes
            annex_start_match = re.search(r'\bANNEX\s+[IVX]+', text[enacting_start:])
            if annex_start_match:
                enacting_end = enacting_start + annex_start_match.start()
            else:
                # Look for concluding formulas pattern
                concluding_match = re.search(r'This Regulation shall be binding|Done at Brussels', text[enacting_start:])
                if concluding_match:
                    enacting_end = enacting_start + concluding_match.start()
                else:
                    enacting_end = len(text)
            
            enacting_text = text[enacting_start:enacting_end]
            structure['enacting_terms'] = self._parse_enacting_terms(enacting_text)
        
        # 3. Find annexes (before concluding formulas)
        annexes = self._parse_annexes(text)
        structure['annexes'] = annexes
        
        # 4. Find concluding formulas (usually at the very end, after annexes)
        # Look for the final adoption text
        concluding_pattern = r'(This Regulation shall be binding.*?)$'
        concluding_match = re.search(concluding_pattern, text, re.DOTALL | re.IGNORECASE)
        if concluding_match:
            concluding_text = concluding_match.group(1).strip()
            # Only include if it's reasonably short (actual concluding formulas)
            if len(concluding_text) < 5000:  # Reasonable limit for concluding formulas
                structure['concluding_formulas'] = concluding_text
        
        return structure
    
    def _parse_enacting_terms(self, enacting_text: str) -> Dict[str, Any]:
        """Parse the enacting terms into chapters and articles."""
        chapters = {}
        
        # Split by chapters
        chapter_pattern = r'\n\s*(CHAPTER\s+[IVX]+[A-Z\s]*)\n'
        chapter_splits = re.split(chapter_pattern, enacting_text)
        
        current_chapter = None
        for i, section in enumerate(chapter_splits):
            if re.match(r'CHAPTER\s+[IVX]+', section.strip()):
                current_chapter = section.strip()
                chapters[current_chapter] = []
            elif current_chapter and section.strip():
                # Parse articles within this chapter
                articles = self._parse_articles(section)
                chapters[current_chapter].extend(articles)
        
        return chapters
    
    def _parse_articles(self, chapter_text: str) -> List[Dict[str, Any]]:
        """Parse articles within a chapter."""
        articles = []
        
        # Split by articles
        article_pattern = r'\n\s*(Article\s+\d+[a-z]?[A-Za-z\s]*)\n'
        article_splits = re.split(article_pattern, chapter_text)
        
        current_article = None
        for i, section in enumerate(article_splits):
            if re.match(r'Article\s+\d+', section.strip()):
                current_article = section.strip()
            elif current_article and section.strip():
                articles.append({
                    'title': current_article,
                    'content': section.strip()
                })
                current_article = None
        
        return articles
    
    def _parse_annexes(self, text: str) -> Dict[str, str]:
        """Parse annexes from the document."""
        annexes = {}
        
        # Find all annex boundaries
        annex_pattern = r'\n\s*(ANNEX\s+[IVX]+[A-Za-z\s]*)\n'
        annex_matches = list(re.finditer(annex_pattern, text))
        
        for i, match in enumerate(annex_matches):
            annex_title = match.group(1).strip()
            start_pos = match.end()
            
            # Find end of this annex (start of next annex or end of document)
            if i + 1 < len(annex_matches):
                end_pos = annex_matches[i + 1].start()
            else:
                end_pos = len(text)
            
            annex_content = text[start_pos:end_pos].strip()
            if annex_content:
                annexes[annex_title] = annex_content
        
        return annexes
    
    def _chunk_preamble_recitals(self, preamble_text: str) -> List[Document]:
        """Chunk preamble by individual recitals."""
        documents = []
        
        # Split by recital numbers (1), (2), etc. - handle both start of text and newlines
        recital_pattern = r'(?:^|\n)\s*\((\d+)\)\s*'
        recital_splits = re.split(recital_pattern, preamble_text)
        
        # The first element might be empty or contain text before the first recital
        if recital_splits and not recital_splits[0].strip():
            recital_splits = recital_splits[1:]
        
        current_recital_num = None
        for i, section in enumerate(recital_splits):
            if section.isdigit():
                current_recital_num = section
            elif current_recital_num and section.strip():
                # Clean up the recital text
                recital_content = section.strip()
                
                # Add recital number prefix for context
                full_recital = f"({current_recital_num}) {recital_content}"
                
                # If recital is too long, split it further
                if len(full_recital) > self.config.max_chunk_size:
                    sub_chunks = self._split_long_text(
                        full_recital, 
                        f"recital_{current_recital_num}"
                    )
                    for j, sub_chunk in enumerate(sub_chunks):
                        doc = Document(
                            page_content=sub_chunk,
                            metadata={
                                "chunk_id": f"recital_{current_recital_num}_part_{j+1}",
                                "document_section": "preamble",
                                "recital_number": current_recital_num,
                                "part": j+1,
                                "content_type": "recital",
                                "structural_level": "recital",
                                "char_length": len(sub_chunk),
                                "source": "EU_AI_Act",
                                "chunk_strategy": "structural_recital"
                            }
                        )
                        documents.append(doc)
                else:
                    doc = Document(
                        page_content=full_recital,
                        metadata={
                            "chunk_id": f"recital_{current_recital_num}",
                            "document_section": "preamble",
                            "recital_number": current_recital_num,
                            "content_type": "recital",
                            "structural_level": "recital",
                            "char_length": len(full_recital),
                            "source": "EU_AI_Act",
                            "chunk_strategy": "structural_recital"
                        }
                    )
                    documents.append(doc)
                
                current_recital_num = None
        
        return documents
    
    def _chunk_enacting_terms(self, enacting_terms: Dict[str, Any]) -> List[Document]:
        """Chunk enacting terms by articles within chapters."""
        documents = []
        
        for chapter_title, articles in enacting_terms.items():
            # Extract chapter number/identifier
            chapter_match = re.match(r'CHAPTER\s+([IVX]+)', chapter_title)
            chapter_num = chapter_match.group(1) if chapter_match else "UNKNOWN"
            
            for article in articles:
                article_title = article['title']
                article_content = article['content']
                
                # Extract article number
                article_match = re.match(r'Article\s+(\d+[a-z]?)', article_title)
                article_num = article_match.group(1) if article_match else "UNKNOWN"
                
                # Combine title and content
                full_content = f"{article_title}\n\n{article_content}"
                
                # If article is too long, split it further
                if len(full_content) > self.config.max_chunk_size:
                    sub_chunks = self._split_long_text(
                        full_content, 
                        f"article_{article_num}"
                    )
                    for j, sub_chunk in enumerate(sub_chunks):
                        doc = Document(
                            page_content=sub_chunk,
                            metadata={
                                "chunk_id": f"chapter_{chapter_num}_article_{article_num}_part_{j+1}",
                                "document_section": "enacting_terms",
                                "chapter": chapter_num,
                                "chapter_title": chapter_title,
                                "article": article_num,
                                "article_title": article_title,
                                "part": j+1,
                                "content_type": self._classify_content(sub_chunk),
                                "structural_level": "article",
                                "char_length": len(sub_chunk),
                                "source": "EU_AI_Act",
                                "chunk_strategy": "structural_article"
                            }
                        )
                        documents.append(doc)
                else:
                    doc = Document(
                        page_content=full_content,
                        metadata={
                            "chunk_id": f"chapter_{chapter_num}_article_{article_num}",
                            "document_section": "enacting_terms",
                            "chapter": chapter_num,
                            "chapter_title": chapter_title,
                            "article": article_num,
                            "article_title": article_title,
                            "content_type": self._classify_content(full_content),
                            "structural_level": "article",
                            "char_length": len(full_content),
                            "source": "EU_AI_Act",
                            "chunk_strategy": "structural_article"
                        }
                    )
                    documents.append(doc)
        
        return documents
    
    def _chunk_concluding_formulas(self, concluding_text: str) -> List[Document]:
        """Chunk concluding formulas."""
        documents = []
        
        if len(concluding_text) > self.config.max_chunk_size:
            sub_chunks = self._split_long_text(concluding_text, "concluding")
            for j, sub_chunk in enumerate(sub_chunks):
                doc = Document(
                    page_content=sub_chunk,
                    metadata={
                        "chunk_id": f"concluding_part_{j+1}",
                        "document_section": "concluding_formulas",
                        "part": j+1,
                        "content_type": "concluding_formula",
                        "structural_level": "concluding",
                        "char_length": len(sub_chunk),
                        "source": "EU_AI_Act",
                        "chunk_strategy": "structural_concluding"
                    }
                )
                documents.append(doc)
        else:
            doc = Document(
                page_content=concluding_text,
                metadata={
                    "chunk_id": "concluding_formulas",
                    "document_section": "concluding_formulas",
                    "content_type": "concluding_formula",
                    "structural_level": "concluding",
                    "char_length": len(concluding_text),
                    "source": "EU_AI_Act",
                    "chunk_strategy": "structural_concluding"
                }
            )
            documents.append(doc)
        
        return documents
    
    def _chunk_annexes(self, annexes: Dict[str, str]) -> List[Document]:
        """Chunk annexes as separate sections."""
        documents = []
        
        for annex_title, annex_content in annexes.items():
            # Extract annex identifier
            annex_match = re.match(r'ANNEX\s+([IVX]+)', annex_title)
            annex_id = annex_match.group(1) if annex_match else "UNKNOWN"
            
            # Combine title and content
            full_content = f"{annex_title}\n\n{annex_content}"
            
            # If annex is too long, split it further
            if len(full_content) > self.config.max_chunk_size:
                sub_chunks = self._split_long_text(full_content, f"annex_{annex_id}")
                for j, sub_chunk in enumerate(sub_chunks):
                    doc = Document(
                        page_content=sub_chunk,
                        metadata={
                            "chunk_id": f"annex_{annex_id}_part_{j+1}",
                            "document_section": "annexes",
                            "annex": annex_id,
                            "annex_title": annex_title,
                            "part": j+1,
                            "content_type": "annex",
                            "structural_level": "annex",
                            "char_length": len(sub_chunk),
                            "source": "EU_AI_Act",
                            "chunk_strategy": "structural_annex"
                        }
                    )
                    documents.append(doc)
            else:
                doc = Document(
                    page_content=full_content,
                    metadata={
                        "chunk_id": f"annex_{annex_id}",
                        "document_section": "annexes",
                        "annex": annex_id,
                        "annex_title": annex_title,
                        "content_type": "annex",
                        "structural_level": "annex",
                        "char_length": len(full_content),
                        "source": "EU_AI_Act",
                        "chunk_strategy": "structural_annex"
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _split_long_text(self, text: str, base_id: str) -> List[str]:
        """Split long text while trying to preserve semantic boundaries."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.max_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence ends
                "; ",    # Semicolon breaks
                ", ",    # Comma breaks
                " ",     # Word breaks
                ""       # Character breaks
            ],
            length_function=len,
        )
        
        return splitter.split_text(text)
    
    def _classify_content(self, text: str) -> str:
        """Classify the type of content based on keywords and patterns."""
        text_lower = text.lower()
        
        # Check for specific content types
        if re.search(r'\bdefinition\b|\bmeans\b|\bshall mean\b', text_lower):
            return "definition"
        elif re.search(r'\bprohibited\b|\bforbidden\b|\bnot allowed\b|\bbanned\b', text_lower):
            return "prohibition"
        elif re.search(r'\bshall\b|\bmust\b|\brequired\b|\bobligation\b', text_lower):
            return "requirement"
        elif re.search(r'\bpenalty\b|\bfine\b|\bsanction\b|\bpunishment\b', text_lower):
            return "enforcement"
        elif re.search(r'\bhigh-risk\b|\bhigh risk\b', text_lower):
            return "high_risk"
        elif re.search(r'\bai office\b|\bsupervisory\b|\bcompliance\b', text_lower):
            return "governance"
        elif re.search(r'\bgeneral.purpose\b|\bgpai\b', text_lower):
            return "general_purpose_ai"
        elif re.search(r'\btransparency\b|\bdisclosure\b', text_lower):
            return "transparency"
        else:
            return "general"


def test_structural_chunker():
    """Test the structural chunker with the extracted document."""
    print("🧪 Testing structural chunker...")
    
    # Load the extracted document
    try:
        with open("test_extraction.txt", "r", encoding="utf-8") as f:
            document_text = f.read()
    except FileNotFoundError:
        print("❌ test_extraction.txt not found. Run the HTML extraction first.")
        return
    
    # Create chunker
    config = ChunkConfig(
        max_chunk_size=1500,
        chunk_overlap=200,
        min_chunk_size=100
    )
    chunker = StructuralChunker(config)
    
    # Chunk the document
    documents = chunker.chunk_document(document_text)
    
    # Show results
    print(f"\n📊 Chunking Results:")
    print(f"Total chunks: {len(documents)}")
    
    # Group by section
    sections = {}
    for doc in documents:
        section = doc.metadata.get('document_section', 'unknown')
        if section not in sections:
            sections[section] = []
        sections[section].append(doc)
    
    for section, docs in sections.items():
        print(f"  {section}: {len(docs)} chunks")
    
    # Show sample chunks from each section
    for section, docs in sections.items():
        if docs:
            print(f"\n📋 Sample from {section}:")
            sample = docs[0]
            print(f"  Chunk ID: {sample.metadata.get('chunk_id', 'N/A')}")
            print(f"  Content Type: {sample.metadata.get('content_type', 'N/A')}")
            print(f"  Length: {sample.metadata.get('char_length', 0)} chars")
            print(f"  Preview: {sample.page_content[:200]}...")
    
    return documents


if __name__ == "__main__":
    test_structural_chunker() 