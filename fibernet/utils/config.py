"""
YAML-based configuration system for reproducible experiments.

Provides tools for:
- Loading/saving experiment configurations
- Validating configuration schemas
- Managing experiment parameters
- Version control for configs
- Automatic documentation generation

Example config structure:
```yaml
experiment:
  name: "fiber_network_study"
  version: "1.0"
  description: "Study of fiber orientation effects"

network:
  generator: "random_straight_2d"
  parameters:
    num_fibers: 100
    fiber_length: 10.0
    box_size: [50, 50]
  seed: 42

material:
  type: "carbon_fiber"
  grade: "standard"

simulation:
  type: "mechanical"
  parameters:
    strain: 0.01
    axis: 0
    
output:
  directory: "./results"
  formats: ["vtk", "pandas"]
  save_plots: true
```
"""

from __future__ import annotations

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
import copy


@dataclass
class ExperimentConfig:
    """
    Configuration for a fiber network experiment.
    
    Attributes
    ----------
    experiment : dict
        Experiment metadata (name, version, description)
    network : dict
        Network generation parameters
    material : dict
        Material specification
    simulation : dict
        Simulation parameters
    analysis : dict
        Analysis to perform
    output : dict
        Output configuration
    metadata : dict
        Additional metadata (timestamp, hash, etc.)
    """
    experiment: Dict[str, Any] = field(default_factory=dict)
    network: Dict[str, Any] = field(default_factory=dict)
    material: Dict[str, Any] = field(default_factory=dict)
    simulation: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize metadata if not present."""
        if not self.metadata:
            self.metadata = {
                'created_at': datetime.now().isoformat(),
                'config_version': '1.0',
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(
            experiment=data.get('experiment', {}),
            network=data.get('network', {}),
            material=data.get('material', {}),
            simulation=data.get('simulation', {}),
            analysis=data.get('analysis', {}),
            output=data.get('output', {}),
            metadata=data.get('metadata', {}),
        )
    
    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        config = cls.from_dict(data)
        config.metadata['source_file'] = str(filepath.absolute())
        return config
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        config = cls.from_dict(data)
        config.metadata['source_file'] = str(filepath.absolute())
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_yaml(self, filepath: Union[str, Path], include_metadata: bool = True):
        """Save configuration to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        if not include_metadata:
            data.pop('metadata', None)
        
        # Add timestamp
        if include_metadata:
            data['metadata']['saved_at'] = datetime.now().isoformat()
        
        if yaml is None:
            raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, filepath: Union[str, Path], include_metadata: bool = True):
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        if not include_metadata:
            data.pop('metadata', None)
        
        # Add timestamp
        if include_metadata:
            data['metadata']['saved_at'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def compute_hash(self) -> str:
        """Compute hash of configuration (excluding metadata)."""
        data = self.to_dict()
        data.pop('metadata', None)
        
        # Sort dict for consistent hashing
        config_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns
        -------
        list of str
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        if not self.network:
            errors.append("Missing 'network' configuration")
        elif 'generator' not in self.network:
            errors.append("Missing 'network.generator' field")
        
        # Check material
        if self.material and 'type' not in self.material:
            errors.append("Material specified but missing 'type' field")
        
        # Check simulation
        if self.simulation and 'type' not in self.simulation:
            errors.append("Simulation specified but missing 'type' field")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0
    
    def copy(self) -> 'ExperimentConfig':
        """Create a deep copy of the configuration."""
        return ExperimentConfig.from_dict(copy.deepcopy(self.to_dict()))
    
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Parameters
        ----------
        updates : dict
            Dictionary of updates (nested structure)
        """
        for key, value in updates.items():
            if hasattr(self, key) and isinstance(value, dict):
                current = getattr(self, key)
                current.update(value)
            else:
                setattr(self, key, value)
    
    def get_network_params(self) -> Dict[str, Any]:
        """Extract network generation parameters."""
        return self.network.get('parameters', {})
    
    def get_simulation_params(self) -> Dict[str, Any]:
        """Extract simulation parameters."""
        return self.simulation.get('parameters', {})
    
    def __repr__(self) -> str:
        name = self.experiment.get('name', 'unnamed')
        version = self.experiment.get('version', '1.0')
        return f"ExperimentConfig(name='{name}', version='{version}')"


def create_template_config(
    filepath: Union[str, Path],
    template_type: str = 'mechanical',
) -> ExperimentConfig:
    """
    Create a template configuration file.
    
    Parameters
    ----------
    filepath : str or Path
        Output file path
    template_type : str
        Type of template ('mechanical', 'thermal', 'dma', 'parametric')
    
    Returns
    -------
    ExperimentConfig
        The created configuration
    
    Examples
    --------
    >>> config = create_template_config('my_experiment.yaml', template_type='mechanical')
    >>> # Edit the file, then load it
    >>> config = ExperimentConfig.from_yaml('my_experiment.yaml')
    """
    templates = {
        'mechanical': {
            'experiment': {
                'name': 'mechanical_simulation',
                'version': '1.0',
                'description': 'Uniaxial tensile simulation',
            },
            'network': {
                'generator': 'random_straight_2d',
                'parameters': {
                    'num_fibers': 100,
                    'fiber_length': 10.0,
                    'box_size': [50, 50],
                },
                'seed': 42,
            },
            'material': {
                'type': 'carbon_fiber',
                'grade': 'standard',
            },
            'simulation': {
                'type': 'mechanical',
                'parameters': {
                    'strain': 0.01,
                    'axis': 0,
                    'segments_per_fiber': 5,
                },
            },
            'analysis': {
                'stress_strain': True,
                'morphology': True,
            },
            'output': {
                'directory': './results',
                'formats': ['vtk', 'pandas'],
                'save_plots': True,
            },
        },
        
        'thermal': {
            'experiment': {
                'name': 'thermal_simulation',
                'version': '1.0',
                'description': 'Heat conduction simulation',
            },
            'network': {
                'generator': 'square_lattice_2d',
                'parameters': {
                    'spacing': 5.0,
                    'grid_size': [10, 10],
                },
            },
            'material': {
                'type': 'glass_fiber',
                'fiber_type': 'E-glass',
            },
            'simulation': {
                'type': 'thermal',
                'parameters': {
                    'temperature_diff': 100.0,
                    'direction': 0,
                },
            },
            'output': {
                'directory': './results_thermal',
                'formats': ['vtk'],
            },
        },
        
        'dma': {
            'experiment': {
                'name': 'dma_analysis',
                'version': '1.0',
                'description': 'Dynamic mechanical analysis',
            },
            'material': {
                'type': 'custom',
                'properties': {
                    'E_inf': 1e9,
                    'E_i': [5e8, 3e8, 2e8],
                    'tau_i': [0.01, 0.1, 1.0],
                },
            },
            'simulation': {
                'type': 'dma',
                'sweep_type': 'frequency',
                'parameters': {
                    'freq_range': [0.01, 100],
                    'num_points': 50,
                    'temperature': 298.15,
                },
            },
            'output': {
                'directory': './results_dma',
                'formats': ['pandas'],
                'save_plots': True,
            },
        },
        
        'parametric': {
            'experiment': {
                'name': 'parametric_study',
                'version': '1.0',
                'description': 'Parameter sweep study',
            },
            'network': {
                'generator': 'random_straight_2d',
                'base_parameters': {
                    'fiber_length': 10.0,
                    'box_size': [50, 50],
                },
                'seed': 42,
            },
            'parametric': {
                'parameters': {
                    'num_fibers': [50, 100, 150, 200],
                },
                'parallel': True,
                'max_workers': 4,
            },
            'simulation': {
                'type': 'mechanical',
                'parameters': {
                    'strain': 0.01,
                    'axis': 0,
                },
            },
            'output': {
                'directory': './results_parametric',
                'formats': ['pandas'],
            },
        },
    }
    
    if template_type not in templates:
        raise ValueError(f"Unknown template type: {template_type}. "
                        f"Available: {list(templates.keys())}")
    
    config = ExperimentConfig.from_dict(templates[template_type])
    
    # Add metadata
    config.metadata['template_type'] = template_type
    config.metadata['config_hash'] = config.compute_hash()
    
    # Save to file
    filepath = Path(filepath)
    if filepath.suffix == '.json':
        config.to_json(filepath)
    else:
        config.to_yaml(filepath)
    
    return config


def run_from_config(config: Union[str, Path, ExperimentConfig]) -> Dict[str, Any]:
    """
    Run experiment from configuration.
    
    Parameters
    ----------
    config : str, Path, or ExperimentConfig
        Configuration file path or object
    
    Returns
    -------
    dict
        Experiment results
    
    Examples
    --------
    >>> results = run_from_config('experiment.yaml')
    >>> print(results['stress_strain'].youngs_modulus)
    """
    # Load config if it's a file path
    if isinstance(config, (str, Path)):
        filepath = Path(config)
        if filepath.suffix == '.json':
            config = ExperimentConfig.from_json(filepath)
        else:
            config = ExperimentConfig.from_yaml(filepath)
    
    # Validate
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration:\n" + "\n".join(errors))
    
    results = {
        'config': config,
        'config_hash': config.compute_hash(),
        'timestamp': datetime.now().isoformat(),
    }
    
    # Import here to avoid circular imports
    from .. import gen
    from ..core.material import Material
    from ..materials import get_material
    
    # Generate network
    gen_name = config.network.get('generator')
    gen_func = getattr(gen, gen_name)
    gen_params = config.get_network_params()
    seed = config.network.get('seed')
    
    if seed is not None:
        gen_params['seed'] = seed
    
    # Get material if specified
    if config.material:
        mat_type = config.material.get('type')
        mat_params = {k: v for k, v in config.material.items() if k != 'type'}
        
        if mat_type == 'custom':
            # Create custom material from properties
            props = config.material.get('properties', {})
            material = Material(**props)
        else:
            material = get_material(mat_type, **mat_params)
        
        gen_params['material'] = material
    
    # Generate network
    print(f"Generating network with {gen_name}...")
    network = gen_func(**gen_params)
    results['network'] = network
    print(f"  Generated {network.num_fibers} fibers")
    
    # Run simulation if specified
    if config.simulation:
        sim_type = config.simulation.get('type')
        sim_params = config.get_simulation_params()
        
        print(f"Running {sim_type} simulation...")
        
        if sim_type == 'mechanical':
            from ..sim import FiberFEM
            fem = FiberFEM(network, segments_per_fiber=sim_params.get('segments_per_fiber', 5))
            result = fem.apply_uniaxial_strain(
                strain=sim_params.get('strain', 0.01),
                axis=sim_params.get('axis', 0)
            )
            results['mechanical'] = result
            print(f"  Energy: {result.energy:.4e} J")
        
        elif sim_type == 'thermal':
            from ..sim import ThermalSolver
            solver = ThermalSolver(network)
            result = solver.steady_state(
                temperature_diff=sim_params.get('temperature_diff', 100),
                direction=sim_params.get('direction', 0)
            )
            results['thermal'] = result
            print(f"  Thermal conductivity: {result.effective_conductivity:.4e} W/(m·K)")
        
        elif sim_type == 'dma':
            from ..sim import GeneralizedMaxwell, frequency_sweep, temperature_sweep
            
            # Get model parameters
            if config.material and config.material.get('type') == 'custom':
                props = config.material.get('properties', {})
                model = GeneralizedMaxwell(
                    E_inf=props.get('E_inf', 1e9),
                    E_i=props.get('E_i', [5e8]),
                    tau_i=props.get('tau_i', [0.1])
                )
            else:
                raise ValueError("DMA requires custom material with Prony series parameters")
            
            sweep_type = config.simulation.get('sweep_type', 'frequency')
            
            if sweep_type == 'frequency':
                result = frequency_sweep(
                    model,
                    freq_range=sim_params.get('freq_range', (0.01, 100)),
                    num_points=sim_params.get('num_points', 50),
                    temperature=sim_params.get('temperature', 298.15)
                )
            else:
                result = temperature_sweep(
                    model,
                    temp_range=sim_params.get('temp_range', (250, 350)),
                    num_points=sim_params.get('num_points', 50),
                    frequency=sim_params.get('frequency', 1.0)
                )
            
            results['dma'] = result
            print(f"  Storage modulus range: {result.storage_modulus.min()/1e9:.2f} - {result.storage_modulus.max()/1e9:.2f} GPa")
    
    # Run analysis if specified
    if config.analysis:
        if config.analysis.get('stress_strain'):
            from ..analysis import extract_stress_strain
            print("Extracting stress-strain curve...")
            curve = extract_stress_strain(
                network,
                strain_range=(0, sim_params.get('strain', 0.01)),
                num_steps=10
            )
            results['stress_strain'] = curve
            print(f"  Young's modulus: {curve.youngs_modulus/1e9:.2f} GPa")
        
        if config.analysis.get('morphology'):
            from ..analysis import MorphologyAnalyzer
            print("Analyzing morphology...")
            analyzer = MorphologyAnalyzer(network)
            results['morphology'] = {
                'nematic_order': analyzer.nematic_order_parameter(),
                'porosity': analyzer.porosity(),
                'mean_length': analyzer.mean_fiber_length(),
            }
            print(f"  Nematic order: {results['morphology']['nematic_order']:.3f}")
    
    # Save outputs if specified
    if config.output:
        output_dir = Path(config.output.get('directory', './results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        formats = config.output.get('formats', [])
        
        if 'vtk' in formats and 'network' in results:
            from ..io import to_vtk
            vtk_path = output_dir / 'network.vtk'
            to_vtk(results['network'], str(vtk_path))
            print(f"  Saved VTK: {vtk_path}")
        
        if 'pandas' in formats and 'stress_strain' in results:
            df = results['stress_strain'].to_dataframe()
            csv_path = output_dir / 'stress_strain.csv'
            df.to_csv(csv_path, index=False)
            print(f"  Saved CSV: {csv_path}")
        
        # Save config with results
        config.to_yaml(output_dir / 'config.yaml')
        print(f"  Saved config: {output_dir / 'config.yaml'}")
        
        if config.output.get('save_plots') and 'stress_strain' in results:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            results['stress_strain'].plot(ax=ax)
            plot_path = output_dir / 'stress_strain.png'
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved plot: {plot_path}")
    
    print("\n✓ Experiment complete!")
    return results
