"""
Semantic chunking infrastructure using LangChain's SemanticChunker and torch-based embeddings.
Provides intelligent text splitting with fallback to character-based chunking.
"""

import logging
from typing import List
from tqdm.auto import tqdm

try:
    from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
    from langchain_core.embeddings import Embeddings
    from sentence_transformers import SentenceTransformer
    LANGCHAIN_SEMANTIC_AVAILABLE = True
except ImportError:
    LANGCHAIN_SEMANTIC_AVAILABLE = False
    # Fallback dummy classes
    class Embeddings:
        pass


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for SentenceTransformer models to work with LangChain."""
    
    def __init__(self, model_name: str = 'nomic-ai/nomic-embed-text-v1.5'):
        """Initialize the SentenceTransformer embeddings wrapper.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        print(f"🤖 Initializing SentenceTransformer with model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            print(f"✅ Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"⚠️ Failed to load {model_name}, falling back to BAAI/bge-large-en-v1.5: {e}")
            self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs with progress tracking.
        
        Args:
            texts: List of text to embed.
            
        Returns:
            List of embeddings.
        """
        print(f"🔍 Embedding {len(texts)} text segments")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding.
        """
        embedding = self.model.encode([text])
        return embedding[0].tolist()


class SemanticChunker:
    """Semantic chunker using LangChain's SemanticChunker and torch-based embeddings."""

    def __init__(self, chunk_size: int = 8192, threshold: float = 0.75):
        """Initialize the semantic chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters
            threshold: Semantic similarity threshold for splitting
        """
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.semantic_chunker = None

        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        print(f"🔧 Initializing SemanticChunker (chunk_size={chunk_size:,}, threshold={threshold})")
        self._initialize_semantic_chunker()
    
    def chunk_text(self, text: str, source_file: str) -> List[str]:
        """Split text into semantic chunks with progress tracking.
        
        Args:
            text: Text content to chunk
            source_file: Name of source file for logging
            
        Returns:
            List of text chunks
        """
        print(f"\n🔄 === Chunking {source_file} ===")
        print(f"📄 Text length: {len(text):,} characters")
        print(f"🤖 Semantic chunker: {'Available' if self.semantic_chunker else 'Not Available'}")
        
        if self.semantic_chunker:
            try:
                return self._semantic_chunk(text, source_file)
            except Exception as e:
                print(f"❌ Semantic chunking failed: {e}")
                print(f"🔄 Falling back to character-based chunking")
                return self._fallback_chunk(text, source_file)
        else:
            print("📝 Using character-based chunking")
            return self._fallback_chunk(text, source_file)

    # HELPER FUNCTIONS
    
    def _initialize_semantic_chunker(self):
        """Initialize the semantic chunker with error handling."""
        try:
            if LANGCHAIN_SEMANTIC_AVAILABLE:
                print("🚀 Creating semantic chunking pipeline...")
                
                # Create embeddings wrapper
                embeddings = SentenceTransformerEmbeddings('nomic-ai/nomic-embed-text-v1.5')
                
                # Calculate min_chunk_size based on chunk_size
                min_chunk_size = max(500, self.chunk_size // 4)
                print(f"📏 Using min_chunk_size: {min_chunk_size:,} characters")
                
                # Initialize LangChain SemanticChunker
                self.semantic_chunker = LangChainSemanticChunker(
                    embeddings=embeddings, 
                    breakpoint_threshold_amount=self.threshold,
                    min_chunk_size=min_chunk_size
                )
                print("🎉 Semantic chunker initialized successfully!")
                self.logger.info(f"Semantic chunker ready (chunk_size={self.chunk_size}, threshold={self.threshold})")
            else:
                print("⚠️ LangChain experimental not available - using fallback chunking")
                self.semantic_chunker = None
        except Exception as e:
            print(f"❌ ERROR initializing semantic chunker: {e}")
            print(f"🔍 Exception type: {type(e).__name__}")
            self.semantic_chunker = None
            self.logger.warning(f"Semantic chunker failed to initialize: {e}")
    
    def _semantic_chunk(self, text: str, source_file: str) -> List[str]:
        """Perform semantic chunking with progress tracking."""
        print("🧠 Applying semantic analysis...")
        
        with tqdm(total=100, desc=f"🔍 Analyzing {source_file}", unit="%") as pbar:
            pbar.update(10)  # Start
            
            # Create semantic chunks
            docs = self.semantic_chunker.create_documents([text])
            pbar.update(70)  # Processing
            
            chunks = [d.page_content for d in docs]
            pbar.update(20)  # Finalize
        
        if not chunks:
            print("⚠️ No chunks created - falling back")
            return self._fallback_chunk(text, source_file)
        
        # Log statistics
        chunk_lengths = [len(chunk) for chunk in chunks]
        print(f"🎯 Semantic chunking completed:")
        print(f"   📦 Total chunks: {len(chunks)}")
        print(f"   📏 Min: {min(chunk_lengths):,} chars")
        print(f"   📏 Max: {max(chunk_lengths):,} chars")
        print(f"   📏 Avg: {sum(chunk_lengths)/len(chunk_lengths):,.0f} chars")
        
        return chunks
    
    def _fallback_chunk(self, text: str, source_file: str) -> List[str]:
        """Fallback character-based chunking with progress."""
        print(f"📝 === Character-based chunking for {source_file} ===")
        
        chunk_size = self.chunk_size
        overlap = 500
        chunks = []
        
        # Calculate iterations for progress
        total_iterations = (len(text) // (chunk_size - overlap)) + 1
        
        with tqdm(total=total_iterations, desc=f"📝 Fixed chunking {source_file}", unit="chunks") as pbar:
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
                pbar.update(1)
        
        print(f"📦 Created {len(chunks)} fixed-size chunks")
        return chunks