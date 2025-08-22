"""
Main orchestrator for the semantic chunking data processing pipeline.
Coordinates components without containing business logic.
"""

import traceback
import sys
import os

# Add parent directory to Python path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Data_prepration.data_processing.models.config import PipelineConfig
from Data_prepration.data_processing.components.pipeline_tasks import data_processing_pipeline
from Data_prepration.data_processing.utils.analysis import ChunkAnalyzer


def main():
    """Execute the complete data processing pipeline."""
    
    # Create configuration
    config = PipelineConfig.create_default()
    
    print("🚀 Launching Data Processing Pipeline")
    print("=" * 60)
    print(f"📊 Sources: {len(config.urls)}")
    print(f"⚙️ Chunk size: {config.chunk_size:,} characters")
    print(f"🎯 Threshold: {config.threshold}")
    print(f"📂 Output: data")
    print("=" * 60)
    
    try:
        # Validate configuration
        config.validate()
        
        # Execute pipeline
        results = data_processing_pipeline(
            urls=config.urls,
            filenames=config.filenames,
            chunk_size=config.chunk_size,
            threshold=config.threshold
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"📊 Results Summary:")
        print(f"   📁 Files processed: {results['files_processed']}")
        print(f"   📦 Chunk files created: {results['chunks_created']}")
        print(f"   ⚙️ Configuration: {config.chunk_size:,} chars, {config.threshold} threshold")
        
        print(f"\n📋 Output Files:")
        for source, chunk_file in results['chunk_files'].items():
            print(f"   ✅ {source}: {chunk_file}")
        
        print(f"\n🔍 Next Steps:")
        print(f"   📂 Check data/chunks/ for chunked data")
        print(f"   📊 Analyze chunk statistics in the JSON files")
        print(f"   🚀 Ready for embedding generation or vector storage")
        
        # Run analysis
        print("\n" + "=" * 60)
        analyzer = ChunkAnalyzer("data/chunks")
        analyzer.print_summary()
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        print("\n📜 Full error trace:")
        traceback.print_exc()
        
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Check internet connectivity for URL fetching")
        print(f"   2. Verify all dependencies are installed (einops, sentence-transformers)")
        print(f"   3. Check disk space for file operations")
        print(f"   4. Review logs above for specific error details")


if __name__ == "__main__":
    main()