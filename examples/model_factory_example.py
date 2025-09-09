#!/usr/bin/env python
"""Example demonstrating the model factory usage."""

from pathlib import Path
import tempfile
import yaml

from transfer_learning_publication.models.model_factory import ModelFactory


def main():
    """Demonstrate model factory capabilities."""
    
    # 1. List available models
    print("Available models:")
    models = ModelFactory.list_available()
    for model in models:
        print(f"  - {model}")
    print()
    
    # 2. Create a temporary YAML configuration
    config_yaml = {
        "sequence": {
            "input_length": 30,
            "output_length": 7,
        },
        "features": {
            "forcing": ["streamflow", "precipitation", "temperature"],
            "static": ["area", "elevation"],
            "future": ["precipitation", "temperature"],
            "target": "streamflow",
        },
        "model": {
            "hidden_size": 64,
            "dropout": 0.1,
        },
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_yaml, f)
        config_path = Path(f.name)
    
    try:
        # 3. Create models using the factory
        print("Creating models from YAML configuration:")
        print(f"Config file: {config_path}")
        print()
        
        for model_name in ["tide", "ealstm", "tsmixer"]:
            print(f"Creating {model_name}...")
            model = ModelFactory.create(model_name, config_path)
            
            # Display model configuration
            print(f"  Model type: {type(model).__name__}")
            print(f"  Input length: {model.config.input_len}")
            print(f"  Output length: {model.config.output_len}")
            print(f"  Input size: {model.config.input_size}")
            print(f"  Static size: {model.config.static_size}")
            
            # Model-specific params
            if hasattr(model.config, 'hidden_size'):
                print(f"  Hidden size: {model.config.hidden_size}")
            
            print()
        
        # 4. Demonstrate error handling
        print("Demonstrating error handling:")
        try:
            ModelFactory.create("unknown_model", config_path)
        except ValueError as e:
            print(f"  Expected error: {e}")
        
    finally:
        # Clean up temporary file
        config_path.unlink()
    
    print("\nFactory demonstration complete!")


if __name__ == "__main__":
    main()