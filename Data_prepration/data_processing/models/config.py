"""
Configuration data structures for the semantic chunking pipeline.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class PipelineConfig:
    """Configuration for the data processing pipeline."""
    
    # Data sources
    urls: List[str]
    filenames: List[str]
    
    # Chunking parameters
    chunk_size: int = 8192
    threshold: float = 0.75
    
    # Output settings
    data_dir: str = "data"
    
    @classmethod
    def create_default(cls) -> 'PipelineConfig':
        """Create default configuration with predefined data sources."""
        return cls(
            urls=[
                "https://context7.com/langchain-ai/langgraph/llms.txt?tokens=600000",
                "https://context7.com/pydantic/pydantic-ai/llms.txt?tokens=100000", 
                "https://context7.com/llmstxt/python_langchain_llms_txt/llms.txt?tokens=100000",
                "https://context7.com/pydantic/pydantic/llms.txt?tokens=100000"
            ],
            filenames=[
                "langgraph_llms_data",
                "pydantic_ai_llms_data",
                "python_langchain_llms_data"
                , "pydantic_llms_data"
            ],
            chunk_size=8192,
            threshold=0.75,
            data_dir="data"
        )
    
    def validate(self) -> bool:
        """Validate the configuration parameters.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if len(self.urls) != len(self.filenames):
            raise ValueError("Number of URLs must match number of filenames")
        
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        if not 0 <= self.threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        return True
    