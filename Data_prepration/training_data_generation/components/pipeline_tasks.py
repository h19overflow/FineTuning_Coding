"""
Prefect workflow tasks for training data generation pipeline.
"""

import asyncio
from typing import List, Dict, Any
from tqdm.auto import tqdm
from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

from Data_prepration.training_data_generation.components.chunk_processor import ChunkProcessor
from Data_prepration.training_data_generation.components.training_data_manager import TrainingDataManager
from Data_prepration.training_data_generation.agents.qa_generation_agent import QAGenerationAgent

@task(name="load-chunks", retries=1)
def load_chunks_task(chunks_dir: str = "data/chunks") -> Dict[str, List[Dict[str, Any]]]:
    """Load semantic chunks from all chunk files.
    
    Args:
        chunks_dir: Directory containing chunk files
        
    Returns:
        Dict mapping source names to their chunks
    """
    logger = get_run_logger()
    logger.info(f"🔍 Loading chunks from {chunks_dir}")
    
    processor = ChunkProcessor(chunks_dir)
    chunks_data = processor.prepare_chunks_for_processing()
    
    total_chunks = sum(len(chunks) for chunks in chunks_data.values())
    logger.info(f"📦 Loaded {total_chunks} chunks from {len(chunks_data)} sources")
    
    return chunks_data

@task(name="generate-qa-examples", retries=2, retry_delay_seconds=10)
async def generate_qa_examples_task(
    source_name: str, 
    chunks: List[Dict[str, Any]], 
    examples_per_chunk: int = 100
) -> Dict[str, Any]:
    """Generate Q&A examples for a single source.
    
    Args:
        source_name: Name of the source
        chunks: List of chunks for this source
        examples_per_chunk: Number of examples to generate per chunk
        
    Returns:
        Dict with generation results
    """
    logger = get_run_logger()
    logger.info(f"🤖 Generating Q&A examples for {source_name}")
    logger.info(f"📊 Processing {len(chunks)} chunks, {examples_per_chunk} examples each")
    
    agent = QAGenerationAgent()
    all_examples = []
    
    for i, chunk in enumerate(tqdm(chunks, desc=f"Processing {source_name}")):
        try:
            logger.info(f"🔄 Processing chunk {i+1}/{len(chunks)} for {source_name}")
            
            batch = await agent.generate_training_examples(
                chunk_content=chunk.get("content", ""),
                chunk_source=f"{source_name}_chunk_{chunk.get('chunk_id', i)}",
                examples_count=examples_per_chunk
            )
            
            all_examples.extend([ex.model_dump() for ex in batch.examples])
            logger.info(f"✅ Generated {len(batch.examples)} examples from chunk {i+1}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process chunk {i+1} for {source_name}: {e}")
            continue
    
    logger.info(f"🎉 Generated {len(all_examples)} total examples for {source_name}")
    
    return {
        "source_name": source_name,
        "examples": all_examples,
        "chunks_processed": len(chunks),
        "total_examples": len(all_examples)
    }

@task(name="save-training-data", retries=1)
def save_training_data_task(
    generation_results: List[Dict[str, Any]], 
    output_dir: str = "data/training"
) -> List[str]:
    """Save generated training data to individual files.
    
    Args:
        generation_results: List of generation results from each source
        output_dir: Directory to save training files
        
    Returns:
        List of saved file paths
    """
    logger = get_run_logger()
    logger.info(f"💾 Saving training data to {output_dir}")
    
    manager = TrainingDataManager(output_dir)
    saved_files = []
    
    for result in generation_results:
        try:
            source_name = result["source_name"]
            examples = result["examples"]
            
            # Create a simple batch structure
            from Data_prepration.training_data_generation.agents.qa_generation_schema import TrainingBatch, TrainingExample
            
            training_examples = [TrainingExample(**ex) for ex in examples]
            batch = TrainingBatch(
                source=source_name,
                examples=training_examples,
                metadata={
                    "chunks_processed": result["chunks_processed"],
                    "total_examples": result["total_examples"]
                }
            )
            
            file_path = manager.save_training_batch(batch, source_name)
            saved_files.append(file_path)
            
            logger.info(f"✅ Saved {len(examples)} examples for {source_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save training data for {result.get('source_name', 'unknown')}: {e}")
            continue
    
    logger.info(f"💾 Saved training data to {len(saved_files)} files")
    return saved_files

@task(name="merge-datasets", retries=1)
def merge_datasets_task(
    saved_files: List[str], 
    output_file: str = "data/training_dataset.json"
) -> Dict[str, Any]:
    """Merge all training data files into a single dataset.
    
    Args:
        saved_files: List of training data file paths
        output_file: Path for the merged dataset
        
    Returns:
        Metadata about the merged dataset
    """
    logger = get_run_logger()
    logger.info(f"🔗 Merging {len(saved_files)} training files into {output_file}")
    
    manager = TrainingDataManager()
    merge_info = manager.merge_training_data(output_file)
    
    logger.info(f"🎉 Merged dataset created with {merge_info['total_examples']} examples")
    return merge_info

@flow(name="training-data-generation-pipeline", task_runner=ConcurrentTaskRunner())
async def training_data_generation_pipeline(
    chunks_dir: str = "data/chunks",
    output_dir: str = "data/training", 
    output_file: str = "data/training_dataset.json",
    examples_per_chunk: int = 100
) -> Dict[str, Any]:
    """Main Prefect flow for training data generation.
    
    Args:
        chunks_dir: Directory containing semantic chunks
        output_dir: Directory to save individual training files  
        output_file: Path for the merged training dataset
        examples_per_chunk: Number of examples to generate per chunk
        
    Returns:
        Pipeline execution results
    """
    logger = get_run_logger()
    logger.info("🚀 === Training Data Generation Pipeline Started ===")
    
    # Stage 1: Load chunks
    logger.info("📦 Stage 1: Loading Semantic Chunks")
    chunks_data = load_chunks_task(chunks_dir)
    
    # Stage 2: Generate Q&A examples (concurrent processing)
    logger.info("🤖 Stage 2: Generating Q&A Examples")
    generation_tasks = []
    
    for source_name, chunks in chunks_data.items():
        task = generate_qa_examples_task(source_name, chunks, examples_per_chunk)
        generation_tasks.append(task)
    
    generation_results = await asyncio.gather(*generation_tasks)
    
    # Stage 3: Save training data
    logger.info("💾 Stage 3: Saving Training Data")
    saved_files = save_training_data_task(generation_results, output_dir)
    
    # Stage 4: Merge datasets
    logger.info("🔗 Stage 4: Merging Training Datasets")
    merge_info = merge_datasets_task(saved_files, output_file)
    
    logger.info("🎉 === Training Data Generation Pipeline Completed ===")
    
    return {
        "status": "completed",
        "sources_processed": len(chunks_data),
        "files_saved": len(saved_files),
        "total_examples": merge_info["total_examples"],
        "output_file": output_file,
        "merge_info": merge_info
    }