"""
Configuration Manager

Handles loading, validating, and applying YAML configuration files.
Supports base config + preset overrides for different protein types.
"""

import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """
    Manages configuration for PocketFinder with preset support.
    
    Hierarchy:
    1. Load base config (config.yaml)
    2. If preset specified, load and merge preset file
    3. Validate all required parameters
    4. Return unified configuration dict
    """
    
    def __init__(self, base_config: str = "config.yaml", verbose: bool = False):
        """
        Initialize ConfigManager.
        
        Args:
            base_config: Path to base configuration file
            verbose: Print debug messages during loading
        """
        self.base_config_path = base_config
        self.verbose = verbose
        self.config = {}
    
    def load_base_config(self) -> Dict[str, Any]:
        """Load the base configuration file."""
        if not os.path.exists(self.base_config_path):
            raise FileNotFoundError(f"Base config not found: {self.base_config_path}")
        
        with open(self.base_config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        if self.verbose:
            print(f"✓ Loaded base config from: {self.base_config_path}")
        
        return self.config
    
    def load_preset(self, preset_name: str) -> Dict[str, Any]:
        """
        Load a preset configuration and merge with base config.
        
        Args:
            preset_name: Name of preset (e.g., 'small_enzymes', 'large_proteins')
                        Can be full path or just name (searches in 'presets/' folder)
        
        Returns:
            Merged configuration dict
        """
        # Construct preset path
        if os.path.exists(preset_name):
            preset_path = preset_name
        elif os.path.exists(f"presets/{preset_name}.yaml"):
            preset_path = f"presets/{preset_name}.yaml"
        elif os.path.exists(f"presets/presets_{preset_name}.yaml"):
            preset_path = f"presets/presets_{preset_name}.yaml"
        else:
            raise FileNotFoundError(
                f"Preset '{preset_name}' not found. "
                f"Tried: {preset_name}, presets/{preset_name}.yaml, presets/presets_{preset_name}.yaml"
            )
        
        # Load preset
        with open(preset_path, 'r') as f:
            preset = yaml.safe_load(f)
        
        if self.verbose:
            print(f"✓ Loaded preset from: {preset_path}")
        
        # Deep merge: preset values override base config
        self._deep_merge(self.config, preset)
        
        if self.verbose:
            print(f"✓ Merged preset into base configuration")
        
        return self.config
    
    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """Recursively merge override dict into base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def validate(self) -> bool:
        """
        Validate that all required configuration keys exist.
        
        Returns:
            True if valid, raises ValueError if not
        """
        required_sections = [
            'grid', 'geometry', 'clustering', 'conservation',
            'master_score', 'chemical_groups', 'residue_assignment', 'logging'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section in config: {section}")
        
        required_grid = ['spacing_angstrom', 'buffer_angstrom']
        for key in required_grid:
            if key not in self.config['grid']:
                raise ValueError(f"Missing required 'grid' parameter: {key}")
        
        if self.verbose:
            print("✓ Configuration validation passed")
        
        return True
    
    def get(self, key: str, default=None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Example:
            config.get('geometry.min_distance_angstrom')  # → 2.0
            config.get('master_score.weight_volume')      # → 0.30
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.config.get(section, {})
    
    def print_summary(self) -> None:
        """Print a human-readable summary of active configuration."""
        print("\n" + "="*60)
        print("POCKETFINDER CONFIGURATION SUMMARY")
        print("="*60)
        
        print("\n[GRID]")
        print(f"  Grid spacing: {self.get('grid.spacing_angstrom')} Å")
        print(f"  Buffer zone: {self.get('grid.buffer_angstrom')} Å")
        
        print("\n[GEOMETRY DETECTION]")
        print(f"  Min distance (clash): {self.get('geometry.min_distance_angstrom')} Å")
        print(f"  Max distance (surface): {self.get('geometry.max_distance_angstrom')} Å")
        print(f"  Surface threshold: {self.get('geometry.surface_threshold_angstrom')} Å")
        print(f"  Neighbors: {self.get('geometry.min_neighbors')}-{self.get('geometry.max_neighbors')}")
        print(f"  Enclosure check: {self.get('geometry.enclosure_check.enabled')}")
        print(f"  Min enclosure: {self.get('geometry.enclosure_check.min_enclosure_fraction')}")
        
        print("\n[CLUSTERING (DBSCAN)]")
        print(f"  Epsilon (eps): {self.get('clustering.eps_angstrom')} Å")
        print(f"  Min samples: {self.get('clustering.min_samples')}")
        print(f"  Size range: {self.get('clustering.min_points')}-{self.get('clustering.max_points')} points")
        
        print("\n[CONSERVATION]")
        print(f"  Enabled: {self.get('conservation.enabled')}")
        print(f"  Jackhmmer CPUs: {self.get('conservation.jackhmmer_cpus')}")
        
        print("\n[MASTER SCORE WEIGHTS]")
        print(f"  Volume: {self.get('master_score.weight_volume')*100:.0f}%")
        print(f"  Hydrophobicity: {self.get('master_score.weight_hydrophobicity')*100:.0f}%")
        print(f"  Conservation: {self.get('master_score.weight_conservation')*100:.0f}%")
        
        print("\n[LOGGING]")
        print(f"  Level: {self.get('logging.level')}")
        print("="*60 + "\n")


def load_config(config_file: str = "config.yaml", 
                preset: Optional[str] = None,
                verbose: bool = False) -> Dict[str, Any]:
    """
    Convenience function to load configuration with optional preset.
    
    Args:
        config_file: Path to base configuration
        preset: Name of preset to merge (optional)
        verbose: Print debug messages
    
    Returns:
        Configuration dictionary
    
   """
    manager = ConfigManager(config_file, verbose=verbose)
    manager.load_base_config()
    
    if preset:
        manager.load_preset(preset)
    
    manager.validate()
    
    if verbose:
        manager.print_summary()
    
    return manager.config
