"""
Main orchestrator for the training data generation pipeline.
Coordinates components without containing business logic.
"""

import asyncio
import traceback
import sys
import os

# Add parent directory to Python path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Data_prepration.training_data_generation.components.pipeline_tasks import training_data_generation_pipeline

async def main():
    """Execute the complete training data generation pipeline."""
    
    print("🤖 Launching Training Data Generation Pipeline")
    print("=" * 60)
    print("📊 Input: data/chunks/")
    print("💾 Output: data/training/")
    print("🔗 Merged: data/training_dataset.json")
    print("📈 Examples per chunk: 100")
    print("=" * 60)
    
    try:
        # Execute pipeline with absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        chunks_dir = os.path.join(base_dir, "data", "chunks")
        output_dir = os.path.join(base_dir, "data", "training")
        output_file = os.path.join(base_dir, "data", "training_dataset.json")
        
        print(f"📁 Using chunks directory: {chunks_dir}")
        
        results = await training_data_generation_pipeline(
            chunks_dir=chunks_dir,
            output_dir=output_dir,
            output_file=output_file, 
            examples_per_chunk=5  # Start small for testing
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("🎉 TRAINING DATA GENERATION COMPLETED!")
        print("=" * 60)
        
        print(f"📊 Pipeline Results:")
        print(f"   📁 Sources processed: {results['sources_processed']}")
        print(f"   💾 Training files saved: {results['files_saved']}")
        print(f"   📈 Total examples generated: {results['total_examples']:,}")
        print(f"   🔗 Merged dataset: {results['output_file']}")
        
        print(f"\n📋 Dataset Summary:")
        merge_info = results['merge_info']
        print(f"   🎯 Total examples: {merge_info['total_examples']:,}")
        print(f"   📚 Sources: {merge_info['sources_count']}")
        
        print(f"\n🔍 Next Steps:")
        print(f"   📂 Check data/training/ for individual source files")
        print(f"   📊 Review data/training_dataset.json for merged dataset")
        print(f"   🚀 Ready for LLM fine-tuning with Unsloth")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        print("\n📜 Full error trace:")
        traceback.print_exc()
        
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Check data/chunks/ directory exists with chunk files")
        print(f"   2. Verify API key is set for Gemini 2.5 Flash Lite")
        print(f"   3. Check internet connectivity for AI model access")
        print(f"   4. Review logs above for specific error details")

if __name__ == "__main__":
    asyncio.run(main())