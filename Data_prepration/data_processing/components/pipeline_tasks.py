"""
Prefect workflow pipeline tasks for orchestrated data processing.
Handles task orchestration, retries, and error management with progress tracking.
"""

import os
from typing import List, Dict, Any
from tqdm.auto import tqdm
from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

from Data_prepration.data_processing.components.data_fetcher import DataFetcher
from Data_prepration.data_processing.components.semantic_chunker import SemanticChunker


@task(name="fetch-data", retries=2, retry_delay_seconds=5)
def fetch_data_task(urls: List[str], filenames: List[str]) -> Dict[str, str]:
    """Prefect task to fetch data from multiple URLs.
    
    Args:
        urls: List of URLs to fetch data from
        filenames: Corresponding filenames for each URL
        
    Returns:
        Dict mapping source names to file paths
    """
    logger = get_run_logger()
    logger.info(f"🚀 Starting data fetch for {len(urls)} URLs")
    
    # Initialize data fetcher (always save to surface level)
    fetcher = DataFetcher("data")
    file_paths = {}
    
    # Fetch with progress tracking
    for url, filename in tqdm(zip(urls, filenames), total=len(urls), desc="🌐 Fetching URLs"):
        logger.info(f"📥 Fetching: {filename} from {url}")
        
        try:
            text_content = fetcher.fetch_text_from_url(url)
            logger.info(f"✅ Fetched {len(text_content):,} characters")
            
            file_path = fetcher.save_text_as_markdown(text_content, filename)
            file_paths[filename] = file_path
            logger.info(f"💾 Saved: {file_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch {filename}: {e}")
            raise
    
    logger.info(f"🎉 Data fetch completed: {len(file_paths)} files")
    return file_paths


@task(name="chunk-data", retries=1, retry_delay_seconds=10)
def chunk_data_task(file_paths: Dict[str, str], chunk_size: int = 8192, threshold: float = 0.75) -> Dict[str, str]:
    """Prefect task to perform semantic chunking on fetched data.
    
    Args:
        file_paths: Dict mapping source names to markdown file paths
        chunk_size: Maximum chunk size in characters
        threshold: Semantic similarity threshold for splitting
        
    Returns:
        Dict mapping source names to chunk file paths
    """
    logger = get_run_logger()
    logger.info(f"🔄 Starting chunking for {len(file_paths)} files")
    
    # Initialize chunker and fetcher
    chunker = SemanticChunker(chunk_size=chunk_size, threshold=threshold)
    fetcher = DataFetcher("data")  # For saving chunks to surface level
    chunk_files = {}
    
    # Process each file
    for source_name, file_path in tqdm(file_paths.items(), desc="📝 Processing files"):
        logger.info(f"🔍 Processing {source_name}: {file_path}")
        
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            logger.info(f"📄 File size: {len(text_content):,} characters")
            
            # Create chunks
            chunks = chunker.chunk_text(text_content, source_name)
            logger.info(f"✂️ Created {len(chunks)} chunks")
            
            # Save chunks
            chunk_file = fetcher.save_chunks_as_json(chunks, source_name)
            chunk_files[source_name] = chunk_file
            
            # Log statistics
            if chunks:
                chunk_lengths = [len(chunk) for chunk in chunks]
                avg_length = sum(chunk_lengths) / len(chunk_lengths)
                logger.info(f"📊 Stats - Avg: {avg_length:.0f}, Min: {min(chunk_lengths)}, Max: {max(chunk_lengths)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {source_name}: {e}")
            raise
    
    logger.info(f"🎉 Chunking completed: {len(chunk_files)} chunk files")
    return chunk_files


@flow(name="data-processing-pipeline", task_runner=ConcurrentTaskRunner())
def data_processing_pipeline(
    urls: List[str],
    filenames: List[str], 
    chunk_size: int = 8192,
    threshold: float = 0.75
) -> Dict[str, Any]:
    """Main Prefect flow for the complete data processing pipeline.
    
    Args:
        urls: List of URLs to fetch data from
        filenames: Corresponding filenames for each URL
        chunk_size: Maximum chunk size for semantic chunking
        threshold: Semantic similarity threshold
        
    Returns:
        Dict with pipeline results and metadata
    """
    logger = get_run_logger()
    logger.info("🚀 === Data Processing Pipeline Started ===")
    logger.info(f"📋 Processing {len(urls)} sources with semantic chunking")
    
    # Stage 1: Data Fetching
    logger.info("📥 Stage 1: Data Fetching")
    file_paths = fetch_data_task(urls, filenames)
    
    # Stage 2: Semantic Chunking
    logger.info("✂️ Stage 2: Semantic Chunking")
    chunk_files = chunk_data_task(file_paths, chunk_size, threshold)
    
    # Pipeline completion
    logger.info("🎉 === Pipeline Completed Successfully ===")
    
    return {
        "status": "completed",
        "files_processed": len(file_paths),
        "chunks_created": len(chunk_files),
        "file_paths": file_paths,
        "chunk_files": chunk_files,
        "pipeline_config": {
            "chunk_size": chunk_size,
            "threshold": threshold
        }
    }