"""
Data structures for Q&A training example generation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TrainingExample(BaseModel):
    """Single Q&A training example."""
    instruction: str = Field(description="The question or instruction for the model")
    input: str = Field(default="", description="Additional context (usually empty for Q&A)")
    output: str = Field(description="The expected response from the model")

class TrainingBatch(BaseModel):
    """Batch of training examples from a single chunk."""
    source: str = Field(description="Source identifier for the chunk")
    examples: List[TrainingExample] = Field(description="Generated Q&A examples")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class GenerationConfig(BaseModel):
    """Configuration for training data generation."""
    examples_per_chunk: int = 100
    input_dir: str = "data/chunks"
    output_dir: str = "data/training"
    merge_output_file: str = "data/training_dataset.json"