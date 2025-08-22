"""
Main orchestrator for model comparison analysis.

This service should:
1. Set up three different model systems for comparison
2. Execute parallel evaluation on the same test queries
3. Compare response quality, accuracy, and performance
4. Generate comprehensive comparison reports
5. Identify strengths and weaknesses of each approach

Comparison targets:
- Trained Model: Fine-tuned Gemma on coding data
- RAG System: Base Gemma + retrieval from data/chunks/
- Base Model: Unmodified Gemma without additional context

Evaluation framework:
- Same test questions sent to all three systems
- Response quality scoring and comparison
- Factual accuracy assessment
- Response time and resource usage comparison
- User preference simulation

Analysis capabilities:
- Side-by-side response comparison
- Performance metrics dashboard
- Quality score distributions
- Use case suitability analysis
- ROI analysis for training vs RAG

Dependencies:
- All three model system integrations
- Evaluation dataset management
- Comparative analysis tools
- Report generation utilities
"""