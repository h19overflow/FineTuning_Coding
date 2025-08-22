"""
Chunk processor for loading and managing semantic chunks.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

class ChunkProcessor:
    """Processes semantic chunks for training data generation."""
    
    def __init__(self, chunks_dir: str = "data/chunks"):
        """Initialize the chunk processor.
        
        Args:
            chunks_dir: Directory containing chunk JSON files
        """
        self.chunks_dir = Path(chunks_dir)
        self.logger = logging.getLogger(__name__)
        
    def load_chunk_files(self) -> List[str]:
        """Load all chunk file paths from the chunks directory.
        
        Returns:
            List of chunk file paths
        """
        if not self.chunks_dir.exists():
            self.logger.warning(f"Chunks directory not found: {self.chunks_dir}")
            return []
            
        chunk_files = list(self.chunks_dir.glob("*_chunks.json"))
        self.logger.info(f"Found {len(chunk_files)} chunk files")
        return [str(f) for f in chunk_files]
    
    def load_chunks_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load chunks from a single JSON file.
        
        Args:
            file_path: Path to the chunk file
            
        Returns:
            List of chunk dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            self.logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to load chunks from {file_path}: {e}")
            return []
    
    def get_source_name(self, file_path: str) -> str:
        """Extract source name from chunk file path.
        
        Args:
            file_path: Path to the chunk file
            
        Returns:
            Source name for the chunks
        """
        filename = Path(file_path).stem
        # Remove '_chunks' suffix if present
        if filename.endswith('_chunks'):
            filename = filename[:-7]
        return filename
    
    def prepare_chunks_for_processing(self) -> Dict[str, List[Dict[str, Any]]]:
        """Prepare all chunks organized by source.
        
        Returns:
            Dict mapping source names to their chunks
        """
        chunk_files = self.load_chunk_files()
        all_chunks = {}
        
        for file_path in chunk_files:
            source_name = self.get_source_name(file_path)
            chunks = self.load_chunks_from_file(file_path)
            
            if chunks:
                all_chunks[source_name] = chunks
                
        self.logger.info(f"Prepared chunks from {len(all_chunks)} sources")
        return all_chunks