"""
Training data manager for saving and merging Q&A examples.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

from Data_prepration.training_data_generation.agents.qa_generation_schema import TrainingBatch, TrainingExample

class TrainingDataManager:
    """Manages saving and merging of training data."""
    
    def __init__(self, output_dir: str = "data/training"):
        """Initialize the training data manager.
        
        Args:
            output_dir: Directory to save training data files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def save_training_batch(self, batch: TrainingBatch, source_name: str) -> str:
        """Save a training batch to JSON file.
        
        Args:
            batch: Training batch to save
            source_name: Source identifier for the batch
            
        Returns:
            Path to the saved file
        """
        filename = f"{source_name}_training_examples.json"
        file_path = self.output_dir / filename
        
        try:
            # Convert to dict for JSON serialization
            batch_data = {
                "source": batch.source,
                "examples": [example.model_dump() for example in batch.examples],
                "metadata": {
                    **batch.metadata,
                    "generated_at": datetime.utcnow().isoformat(),
                    "examples_count": len(batch.examples)
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Saved {len(batch.examples)} examples to {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save training batch for {source_name}: {e}")
            raise
    
    def load_training_files(self) -> List[str]:
        """Load all training data file paths.
        
        Returns:
            List of training data file paths
        """
        training_files = list(self.output_dir.glob("*_training_examples.json"))
        return [str(f) for f in training_files]
    
    def merge_training_data(self, output_file: str = "data/training_dataset.json") -> Dict[str, Any]:
        """Merge all training data files into a single dataset.
        
        Args:
            output_file: Path for the merged dataset file
            
        Returns:
            Metadata about the merged dataset
        """
        training_files = self.load_training_files()
        all_examples = []
        sources_info = {}
        
        for file_path in training_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                source = data.get("source", "unknown")
                examples = data.get("examples", [])
                
                all_examples.extend(examples)
                sources_info[source] = {
                    "examples_count": len(examples),
                    "file_path": file_path
                }
                
                self.logger.info(f"Loaded {len(examples)} examples from {source}")
                
            except Exception as e:
                self.logger.error(f"Failed to load training file {file_path}: {e}")
                continue
        
        # Create merged dataset
        merged_dataset = {
            "dataset_info": {
                "total_examples": len(all_examples),
                "sources_count": len(sources_info),
                "created_at": datetime.utcnow().isoformat(),
                "sources": sources_info
            },
            "examples": all_examples
        }
        
        # Save merged dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_dataset, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Merged dataset saved to {output_path}")
        self.logger.info(f"Total examples: {len(all_examples)} from {len(sources_info)} sources")
        
        return merged_dataset["dataset_info"]