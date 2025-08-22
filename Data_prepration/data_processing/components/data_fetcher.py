"""
Data fetching component for downloading and saving text content from URLs.
Handles HTTP requests, file operations, and error management.
"""

import requests
import json
from pathlib import Path
from typing import List, Dict


class DataFetcher:
    """Handles fetching text content from URLs and saving to files."""
    
    def __init__(self, output_dir: str = "data"):
        """Initialize the data fetcher.
        
        Args:
            output_dir: Directory to save downloaded files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def fetch_text_from_url(self, url: str) -> str:
        """Fetch text content from a URL with error handling.
        
        Args:
            url: The URL to fetch text from
            
        Returns:
            str: The text content from the URL
            
        Raises:
            requests.RequestException: If the request fails
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to fetch content from {url}: {e}")
    
    def save_text_as_markdown(self, text: str, filename: str) -> str:
        """Save text content as a markdown file.
        
        Args:
            text: The text content to save
            filename: Name of the file (without extension)
            
        Returns:
            str: Full path to the saved file
        """
        file_path = self.output_dir / f"{filename}.md"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return str(file_path)
    
    def save_chunks_as_json(self, chunks: List[str], source_name: str) -> str:
        """Save text chunks as JSON with source metadata.
        
        Args:
            chunks: List of text chunks
            source_name: Name of the source file
            
        Returns:
            str: Full path to the saved file
        """
        chunks_dir = self.output_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        # Create metadata for each chunk
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                "chunk_id": i + 1,
                "source": source_name,
                "content": chunk,
                "chunk_length": len(chunk)
            })
        
        # Save as JSON
        json_filename = f"{source_name}_chunks.json"
        file_path = chunks_dir / json_filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    def fetch_and_save_multiple(self, urls: List[str], filenames: List[str]) -> Dict[str, str]:
        """Fetch multiple URLs and save as markdown files.
        
        Args:
            urls: List of URLs to fetch
            filenames: Corresponding filenames for each URL
            
        Returns:
            Dict mapping source names to file paths
        """
        file_paths = {}
        
        for url, filename in zip(urls, filenames):
            text_content = self.fetch_text_from_url(url)
            file_path = self.save_text_as_markdown(text_content, filename)
            file_paths[filename] = file_path
        
        return file_paths