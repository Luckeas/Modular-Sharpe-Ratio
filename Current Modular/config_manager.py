"""
Configuration Manager - Centralizes configuration handling.

This module provides a ConfigManager class to load, validate, and manage
configuration settings for the backtesting framework.
"""

import os
import json
import logging
import yaml
import copy
from typing import Dict, List, Optional, Any, Union

# Configure logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages configuration settings for the backtesting framework.
    
    This class provides methods to load, validate, and manage configuration
    settings for the backtesting framework. It supports nested configurations,
    environment-specific overrides, and validation.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.config = {}
        self.default_config = {}
        self.loaded_files = []
    
    def load_default_config(self, config_dict: Dict) -> None:
        """
        Load default configuration from a dictionary.
        
        Args:
            config_dict: Default configuration dictionary
        """
        self.default_config = copy.deepcopy(config_dict)
        self.config = copy.deepcopy(config_dict)
        logger.info("Loaded default configuration")
    
    def load_config_file(self, file_path: str, override: bool = True) -> None:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file (JSON or YAML)
            override: Whether to override existing settings
        """
        if not os.path.exists(file_path):
            logger.error(f"Configuration file not found: {file_path}")
            return
        
        try:
            # Determine file type
            _, ext = os.path.splitext(file_path)
            
            # Load the file
            if ext.lower() in ['.yaml', '.yml']:
                with open(file_path, 'r') as file:
                    config_data = yaml.safe_load(file)
            elif ext.lower() == '.json':
                with open(file_path, 'r') as file:
                    config_data = json.load(file)
            else:
                logger.error(f"Unsupported file type: {ext}")
                return
            
            # Merge with existing config
            if override:
                self._merge_configs(self.config, config_data)
            else:
                new_config = copy.deepcopy(self.default_config)
                self._merge_configs(new_config, self.config)
                self._merge_configs(new_config, config_data)
                self.config = new_config
            
            self.loaded_files.append(file_path)
            logger.info(f"Loaded configuration from {file_path}")
        
        except Exception as e:
            logger.error(f"Error loading configuration file {file_path}: {str(e)}")
    
    def update_config(self, config_dict: Dict, override: bool = True) -> None:
        """
        Update configuration with a dictionary.
        
        Args:
            config_dict: Configuration dictionary
            override: Whether to override existing settings
        """
        if override:
            self._merge_configs(self.config, config_dict)
        else:
            new_config = copy.deepcopy(self.default_config)
            self._merge_configs(new_config, self.config)
            self._merge_configs(new_config, config_dict)
            self.config = new_config
        
        logger.info("Updated configuration")
    
    def get_config(self, namespace: Optional[str] = None) -> Dict:
        """
        Get the current configuration.
        
        Args:
            namespace: Optional configuration namespace
            
        Returns:
            Configuration dictionary
        """
        if namespace:
            return self.config.get(namespace, {})
        return self.config
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by key path.
        
        Args:
            key_path: Dot-separated key path (e.g., 'market_type.hmm.lookback_days')
            default: Default value if key path not found
            
        Returns:
            Configuration value or default
        """
        # Split the key path
        keys = key_path.split('.')
        
        # Navigate through the config dictionary
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set_value(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value by key path.
        
        Args:
            key_path: Dot-separated key path (e.g., 'market_type.hmm.lookback_days')
            value: Value to set
        """
        # Split the key path
        keys = key_path.split('.')
        
        # Navigate through the config dictionary
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            elif not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        logger.info(f"Set configuration value: {key_path} = {value}")
    
    def reset_to_default(self) -> None:
        """Reset configuration to default values."""
        self.config = copy.deepcopy(self.default_config)
        self.loaded_files = []
        logger.info("Reset configuration to default values")
    
    def save_config(self, file_path: str) -> None:
        """
        Save configuration to a file.
        
        Args:
            file_path: Path to save the configuration file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Determine file type
            _, ext = os.path.splitext(file_path)
            
            # Save the file
            if ext.lower() in ['.yaml', '.yml']:
                with open(file_path, 'w') as file:
                    yaml.dump(self.config, file, sort_keys=False, default_flow_style=False)
            elif ext.lower() == '.json':
                with open(file_path, 'w') as file:
                    json.dump(self.config, file, indent=4)
            else:
                logger.error(f"Unsupported file type: {ext}")
                return
            
            logger.info(f"Saved configuration to {file_path}")
        
        except Exception as e:
            logger.error(f"Error saving configuration file {file_path}: {str(e)}")
    
    def _merge_configs(self, base_config: Dict, override_config: Dict) -> None:
        """
        Recursively merge override_config into base_config.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary
        """
        for key, value in override_config.items():
            if (key in base_config and isinstance(base_config[key], dict)
                    and isinstance(value, dict)):
                self._merge_configs(base_config[key], value)
            else:
                base_config[key] = value

# Create a global configuration manager instance
config_manager = ConfigManager()
