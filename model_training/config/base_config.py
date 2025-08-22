"""
Base configuration classes and common settings.

This module should define:
1. BaseConfig: Abstract base class for all configurations
2. Common configuration patterns and validation
3. Environment variable integration
4. Configuration inheritance mechanisms
5. Shared utility methods

Key features:
- Configuration validation and type checking
- Environment-specific overrides
- Hierarchical configuration inheritance
- JSON/YAML configuration file support
- Runtime configuration updates

Common settings:
- Logging configuration
- Resource limits (memory, GPU)
- File paths and directories
- API endpoints and credentials
- Hardware optimization settings

Validation features:
- Type checking for all configuration values
- Range validation for numeric parameters
- File/directory existence checks
- Dependency compatibility verification
- Environment-specific validation rules

Dependencies:
- pydantic for configuration validation
- os/pathlib for file system integration
- typing for type annotations
- abc for abstract base classes
"""