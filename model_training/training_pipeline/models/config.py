"""
Configuration data structures for the model training pipeline.

This file should define:
1. TrainingConfig: Main configuration class with all training parameters
2. ModelConfig: Gemma model-specific settings (model size, quantization, LoRA params)
3. DataConfig: Dataset configuration (batch size, sequence length, train/val split)
4. OptimizationConfig: Training optimization settings (learning rate, scheduler, etc.)

Key configurations to include:
- Base model selection (gemma-2b, gemma-7b, etc.)
- LoRA/QLoRA adapter configuration
- Training hyperparameters (learning rate, epochs, batch size)
- Data preprocessing settings
- Checkpoint and logging configurations
- GPU/hardware optimization settings

Should provide:
- create_default() class method for standard configurations
- validate() method to ensure configuration consistency
- Support for loading from JSON/YAML configuration files
- Environment-specific overrides (dev, staging, production)
"""