"""
System prompt for Q&A training data generation.
"""

QA_GENERATION_PROMPT = """Create {examples_count} Q&A training examples from this content:

{chunk_content}

Generate questions like:
- How do I...?
- What is...?
- Why does...?
- Can you show me...?

Each answer should be 1-3 sentences with code if relevant.

Generate exactly {examples_count} Q&A pairs."""