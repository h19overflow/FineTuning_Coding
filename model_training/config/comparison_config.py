"""
Configuration for multi-model comparison analysis.

This configuration should include:
1. Model system configurations for all three approaches
2. Comparison testing parameters
3. Analysis and scoring settings
4. RAG system configuration
5. Report generation preferences

Model configurations:
- Trained model settings (path, adapters, generation params)
- Base model settings (same architecture, standard config)
- RAG system settings (vector DB, retrieval params)
- Consistent generation parameters across models
- Hardware allocation for each system

RAG system configuration:
- Vector database settings (FAISS, Chroma, etc.)
- Embedding model configuration (same as chunk creation)
- Retrieval parameters (top-k, similarity threshold)
- Context formatting and length limits
- Source attribution settings

Comparison testing:
- Test query datasets and categories
- Evaluation criteria and scoring methods
- Response collection and organization
- Performance measurement settings
- Statistical analysis parameters

Analysis configuration:
- Quality scoring frameworks
- Performance metric calculations
- Comparative analysis methods
- Significance testing settings
- Trend analysis parameters

Report generation:
- Output formats and templates
- Visualization preferences
- Executive summary settings
- Detailed analysis sections
- Recommendation frameworks

Use case analysis:
- Scenario-based evaluation
- Cost-benefit analysis settings
- Performance vs accuracy trade-offs
- Deployment consideration factors
- ROI calculation parameters

Dependencies:
- Base configuration inheritance
- Model loading configurations
- Vector database settings
- Analysis framework configuration
"""