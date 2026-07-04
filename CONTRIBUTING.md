# Contributing to FiberNet

Thank you for your interest in contributing to FiberNet!

## How to Contribute

### Reporting Bugs
- Use GitHub Issues to report bugs
- Include a minimal reproducer and expected behavior

### Suggesting Features
- Open an issue with the "enhancement" label
- Describe the use case and expected API

### Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest tests/`)
5. Commit with descriptive messages
6. Push to your fork and submit a Pull Request

### Code Style
- Follow PEP 8
- Use type hints for public functions
- Add docstrings in NumPy format
- Keep functions focused and composable

### Adding Generators
New generators should:
- Accept a `seed` parameter for reproducibility
- Return a `FiberNetwork` instance
- Have unit tests in `tests/test_generators.py`
- Be documented with parameters and examples

### Adding Simulation Models
New simulation models should:
- Accept a `FiberNetwork` as input
- Return a result dataclass
- Have unit tests covering basic functionality
- Document physical assumptions

## Development Setup

```bash
pip install -e ".[dev]"
pytest tests/
```

## License
By contributing, you agree that your contributions will be licensed under the MIT License.
