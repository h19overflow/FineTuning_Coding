# Data Processing Pipeline

A modular Python implementation of the semantic chunking data processing pipeline, converted from Jupyter notebook format.

## Structure

```
data_processing/
├── components/              # Business logic components
│   ├── data_fetcher.py     # URL fetching and file operations
│   ├── semantic_chunker.py # Semantic chunking with embeddings
│   └── pipeline_tasks.py   # Prefect workflow tasks
├── models/                 # Data structures
│   └── config.py          # Pipeline configuration
├── utils/                  # Helper utilities
│   └── analysis.py        # Chunk analysis and reporting
├── main.py                # Main orchestrator (with Prefect)
├── simple_main.py         # Simple orchestrator (no Prefect)
└── __init__.py           # Package exports
```

## Usage

### Simple Mode (Recommended for testing)
```bash
python simple_main.py
```

### Prefect Mode (Requires Prefect server)
```bash
python main.py
```

## Features

- **Data Fetching**: Downloads documentation from URLs with error handling
- **Semantic Chunking**: AI-powered text splitting using torch embeddings (fallback to character-based)
- **Progress Tracking**: Visual progress bars throughout the pipeline  
- **Error Handling**: Comprehensive error reporting and fallback mechanisms
- **Metadata Preservation**: Source tracking and chunk statistics
- **Analysis Tools**: Built-in chunk analysis and reporting

## Configuration

Default configuration fetches:
- LangGraph documentation (600k tokens)
- Pydantic AI documentation (100k tokens)  
- Python LangChain documentation (100k tokens)

Modify `PipelineConfig.create_default()` in `models/config.py` to customize sources.

## Output

- Raw markdown files in `data/`
- Chunked JSON files in `data/chunks/`
- Detailed progress logs and statistics

## Dependencies

- requests: HTTP requests
- prefect: Workflow orchestration  
- tqdm: Progress bars
- sentence-transformers: Embeddings (optional)
- langchain-experimental: Semantic chunking (optional)