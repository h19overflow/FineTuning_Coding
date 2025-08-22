"""
Analysis utilities for chunking results and pipeline outputs.
Provides statistical analysis and reporting capabilities.
"""

import json
from pathlib import Path
from typing import Dict, Any


class ChunkAnalyzer:
    """Analyzes chunking results and provides statistical reports."""
    
    def __init__(self, chunk_dir: str = "data/chunks"):
        """Initialize the chunk analyzer.
        
        Args:
            chunk_dir: Directory containing chunk JSON files
        """
        self.chunk_dir = Path(chunk_dir)
    
    def analyze_chunks(self) -> Dict[str, Any]:
        """Analyze the chunking results and return statistics.
        
        Returns:
            Dict containing analysis results and statistics
        """
        if not self.chunk_dir.exists():
            return {
                "error": f"Chunk directory not found: {self.chunk_dir}",
                "total_chunks": 0,
                "files_processed": 0
            }
        
        print("📊 Chunk Analysis Results")
        print("=" * 50)
        
        results = {
            "files": {},
            "total_chunks": 0,
            "files_processed": 0,
            "overall_stats": {}
        }
        
        all_lengths = []
        
        for json_file in self.chunk_dir.glob("*_chunks.json"):
            print(f"\n📄 {json_file.name}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                if chunks:
                    lengths = [chunk['chunk_length'] for chunk in chunks]
                    all_lengths.extend(lengths)
                    
                    file_stats = {
                        "total_chunks": len(chunks),
                        "min_length": min(lengths),
                        "max_length": max(lengths),
                        "avg_length": sum(lengths) / len(lengths),
                        "total_text": sum(lengths)
                    }
                    
                    results["files"][json_file.stem] = file_stats
                    results["total_chunks"] += len(chunks)
                    results["files_processed"] += 1
                    
                    print(f"   📦 Total chunks: {len(chunks)}")
                    print(f"   📏 Min length: {min(lengths):,} chars")
                    print(f"   📏 Max length: {max(lengths):,} chars")
                    print(f"   📏 Avg length: {file_stats['avg_length']:,.0f} chars")
                    print(f"   📊 Total text: {sum(lengths):,} chars")
                else:
                    print(f"   ⚠️ No chunks found in file")
                    results["files"][json_file.stem] = {"error": "No chunks found"}
                    
            except Exception as e:
                print(f"   ❌ Error reading {json_file.name}: {e}")
                results["files"][json_file.stem] = {"error": str(e)}
        
        # Calculate overall statistics
        if all_lengths:
            results["overall_stats"] = {
                "min_length": min(all_lengths),
                "max_length": max(all_lengths),
                "avg_length": sum(all_lengths) / len(all_lengths),
                "total_text": sum(all_lengths)
            }
        
        print(f"\n🎯 Overall Summary:")
        print(f"   📦 Total chunks across all sources: {results['total_chunks']:,}")
        print(f"   📁 Files processed: {results['files_processed']}")
        
        if results["overall_stats"]:
            stats = results["overall_stats"]
            print(f"   📏 Overall avg length: {stats['avg_length']:,.0f} chars")
            print(f"   📊 Total text processed: {stats['total_text']:,} chars")
        
        return results
    
    def print_summary(self) -> None:
        """Print a summary of chunk analysis results."""
        results = self.analyze_chunks()
        
        if "error" in results:
            print(f"📝 Analysis will be available after running the pipeline: {results['error']}")
            return
        
        print(f"\n📋 Quick Summary:")
        print(f"   ✅ Successfully processed {results['files_processed']} files")
        print(f"   📦 Generated {results['total_chunks']} total chunks")
        
        if results["overall_stats"]:
            avg_length = results["overall_stats"]["avg_length"]
            print(f"   📏 Average chunk size: {avg_length:,.0f} characters")
    
    def get_file_analysis(self, filename: str) -> Dict[str, Any]:
        """Get analysis for a specific file.
        
        Args:
            filename: Name of the file to analyze (without extension)
            
        Returns:
            Dict containing file-specific analysis
        """
        results = self.analyze_chunks()
        return results["files"].get(filename, {"error": "File not found"})