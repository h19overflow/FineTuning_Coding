"""
Q&A Generation Agent for creating training data from semantic chunks using Pydantic AI.
"""

import os
from pydantic_ai import Agent, RunContext  
from pydantic import BaseModel  
from typing import List, Dict, Any

from Data_prepration.training_data_generation.agents.qa_generation_schema import TrainingExample, TrainingBatch
from Data_prepration.training_data_generation.agents.qa_generation_prompt import QA_GENERATION_PROMPT

from dotenv import load_dotenv
load_dotenv()

class QAGenerationDeps(BaseModel):
    """Dependencies for the Q&A generation agent."""
    chunk_content: str
    chunk_source: str
    examples_count: int = 100

qa_generation_agent = Agent(
    model='gemini-2.5-flash-lite',
    output_type=TrainingBatch,
    deps_type=QAGenerationDeps, 
)

@qa_generation_agent.system_prompt
def dynamic_system_prompt(ctx: RunContext[QAGenerationDeps]) -> str:
    """Create custom instructions for the Q&A generation agent."""
    return QA_GENERATION_PROMPT.format(
        chunk_content=ctx.deps.chunk_content,
        examples_count=ctx.deps.examples_count,
        source=ctx.deps.chunk_source
    )

class QAGenerationAgent:
    """Agent for generating Q&A training examples from semantic chunks."""
    
    def __init__(self):
        """Initialize the Q&A generation agent."""
        os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        
    async def generate_training_examples(
        self, 
        chunk_content: str, 
        chunk_source: str, 
        examples_count: int = 100
    ) -> TrainingBatch:
        """Generate training Q&A examples from a semantic chunk."""
        try:
            deps = QAGenerationDeps(
                chunk_content=chunk_content,
                chunk_source=chunk_source,
                examples_count=examples_count
            )
            
            result = await qa_generation_agent.run(
                "Generate high-quality Q&A training examples from this chunk", 
                deps=deps
            )
            
            return result.output
            
        except Exception as e:
            # Return empty batch on error
            return TrainingBatch(
                source=chunk_source,
                examples=[],
                metadata={
                    "error": str(e),
                    "examples_requested": examples_count,
                    "examples_generated": 0
                }
            )
    
    async def generate_batch_examples(self, chunks_data: List[Dict[str, Any]]) -> List[TrainingBatch]:
        """Generate training examples for multiple chunks."""
        batches = []
        
        for chunk in chunks_data:
            batch = await self.generate_training_examples(
                chunk_content=chunk.get("content", ""),
                chunk_source=chunk.get("source", "unknown"),
                examples_count=100
            )
            batches.append(batch)
            
        return batches