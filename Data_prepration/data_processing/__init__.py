"""
Semantic chunking data processing pipeline.
A comprehensive data processing pipeline that fetches documentation from multiple sources 
and applies intelligent semantic chunking using PyTorch-based embeddings.
"""

from Data_prepration.data_processing.components import DataFetcher, SemanticChunker, data_processing_pipeline
from Data_prepration.data_processing.models import PipelineConfig
from Data_prepration.data_processing.utils import ChunkAnalyzer

__version__ = "1.0.0"
__all__ = [
    'DataFetcher',
    'SemanticChunker', 
    'data_processing_pipeline',
    'PipelineConfig',
    'ChunkAnalyzer'
]