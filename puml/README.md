# PlantUML Diagrams

This directory contains self-contained PlantUML diagrams explaining the complete Coding LLM pipeline architecture.

## 📋 Diagram Overview

### Data Preparation Pipelines

1. **`data_processing_pipeline.puml`**
   - Shows how raw documentation is fetched and processed
   - Semantic chunking with Nomic embeddings
   - Output: JSON chunk files

2. **`training_data_generation.puml`**
   - Converts semantic chunks into Q&A training examples
   - Uses Pydantic AI with Gemini 2.5 Flash Lite
   - Output: Training dataset ready for fine-tuning

### Model Training Pipelines

3. **`model_training_pipeline.puml`**
   - Fine-tunes Gemma model using LoRA adapters
   - Shows training data flow and checkpoint management
   - Output: Trained Gemma model

4. **`model_testing_pipeline.puml`**
   - Model evaluation and performance testing
   - Benchmark execution and metrics calculation
   - Output: Performance reports and quality metrics

### Comparison Pipeline

5. **`comparison_service_pipeline.puml`**
   - Three-way comparison: Trained vs RAG vs Base model
   - Shows how same query gets processed by different approaches
   - Output: Comprehensive comparison analysis

### Complete Overview

6. **`complete_pipeline_overview.puml`**
   - High-level view of the entire system
   - Shows relationships between all components
   - End-to-end data flow visualization

## 🔧 How to View

### Online Viewers
- [PlantUML Online Server](http://www.plantuml.com/plantuml/uml/)
- [PlantText](https://www.planttext.com/)

### IDE Extensions
- **VS Code**: PlantUML extension
- **IntelliJ**: PlantUML integration plugin
- **Sublime Text**: PlantUML package

### Command Line
```bash
# Install PlantUML
npm install -g node-plantuml

# Generate PNG from PUML
puml generate diagram.puml
```

## 📊 Diagram Features

- **Self-contained**: Each diagram explains one specific pipeline
- **Clear flow**: Shows data transformation at each step
- **Annotated**: Includes notes explaining key configurations
- **Consistent**: Uses standard PlantUML syntax and theming

These diagrams provide a visual understanding of how the coding LLM training system works from raw documentation to final model comparison.