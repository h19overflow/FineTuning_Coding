#!/usr/bin/env python3
"""
UNIFIED PIPELINE ORCHESTRATOR - Simple version that works
"""

import asyncio
import sys
import os

# Add to path
sys.path.insert(0, os.getcwd())

async def main():
    print("🚀 UNIFIED PIPELINE ORCHESTRATOR STARTING...")
    print("=" * 50)
    
    # Stage 1: Data Processing (skip since chunks exist)
    print("✅ Stage 1: Using existing chunks from data/chunks/")
    
    # Stage 2: Training Data Generation  
    print("🤖 Stage 2: Starting Q&A Training Data Generation...")
    try:
        from Data_prepration.training_data_generation.main import main as training_main
        await training_main()
        print("✅ Training data generation completed!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    print("\n🎉 UNIFIED PIPELINE COMPLETED!")
    print("📊 Results:")
    print("  - Training files: data/training/")  
    print("  - Merged dataset: data/training_dataset.json")
    print("  - Ready for Gemma 3 fine-tuning!")

if __name__ == "__main__":
    asyncio.run(main())