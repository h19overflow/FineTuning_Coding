"""
Unified pipeline orchestrator for both data processing and training data generation.
"""

import asyncio
import sys
import os
from enum import Enum
from dataclasses import dataclass

# Ensure proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PipelineMode(Enum):
    DATA_ONLY = "data_only"
    TRAINING_ONLY = "training_only" 
    FULL_PIPELINE = "full_pipeline"

@dataclass
class PipelineConfig:
    mode: PipelineMode = PipelineMode.FULL_PIPELINE
    examples_per_chunk: int = 5
    skip_data_processing: bool = False

async def run_data_processing():
    """Run the data processing pipeline."""
    print("🔧 Running Data Processing Pipeline...")
    from Data_prepration.data_processing.main import main as data_main
    # Run in executor since it's sync
    import concurrent.futures
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, data_main)
    print("✅ Data processing completed!")

async def run_training_generation(examples_per_chunk: int = 5):
    """Run the training data generation pipeline."""
    print("🤖 Running Training Data Generation Pipeline...")
    
    # Set examples per chunk in the main module
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_file = os.path.join(base_dir, "Data_prepration", "training_data_generation", "main.py")
    
    # Read and modify the examples count
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Replace examples_per_chunk value
    content = content.replace("examples_per_chunk=5", f"examples_per_chunk={examples_per_chunk}")
    
    with open(main_file, 'w') as f:
        f.write(content)
    
    from Data_prepration.training_data_generation.main import main as training_main
    await training_main()
    print("✅ Training data generation completed!")

async def main():
    """Main unified pipeline orchestrator."""
    print("🚀 UNIFIED DATA PROCESSING & TRAINING PIPELINE")
    print("=" * 60)
    
    config = PipelineConfig()
    
    try:
        if config.mode == PipelineMode.FULL_PIPELINE:
            if not config.skip_data_processing:
                await run_data_processing()
            await run_training_generation(config.examples_per_chunk)
            
        elif config.mode == PipelineMode.DATA_ONLY:
            await run_data_processing()
            
        elif config.mode == PipelineMode.TRAINING_ONLY:
            await run_training_generation(config.examples_per_chunk)
        
        print("\n🎉 UNIFIED PIPELINE COMPLETED SUCCESSFULLY!")
        print("📊 Check data/training_dataset.json for final results")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("STARTING UNIFIED PIPELINE ORCHESTRATOR...")
    asyncio.run(main())