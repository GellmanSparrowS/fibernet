# Contributing to FiberNet

We love contributions! This document provides guidelines for contributing to FiberNet.

## Code of Conduct

This project adheres to the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.

## How Can I Contribute?

### Reporting Bugs

1. **Check existing issues** first
2. **Use the latest version**
3. **Create a detailed report** with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python, OS, dependencies)
   - Minimal reproducible example

### Suggesting Features

1. **Check existing issues and PRs**
2. **Describe the use case** clearly
3. **Propose an implementation** if possible

### Submitting Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest tests/ -v`
6. Format code: `black fibernet tests`
7. Commit with clear messages
8. Push and submit a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/fibernet.git
cd fibernet

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Build documentation
cd docs && make html
```

## Code Style

- Follow PEP 8
- Use `black` for formatting
- Use type hints where helpful
- Write docstrings in NumPy format
- Keep functions focused and small
- Add tests for new functionality

## Testing

All new features must include tests:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=fibernet --cov-report=html

# Run specific test file
pytest tests/test_core.py -v
```

## Documentation

Update documentation when adding features:

- Add API reference to `docs/api/`
- Update tutorials in `tutorials/`
- Update `CHANGELOG.md`

Build docs locally:
```bash
cd docs
make html
open _build/html/index.html
```

## Pull Request Process

1. Update `CHANGELOG.md` with details of your change
2. Update `README.md` if necessary
3. Ensure CI passes (tests, docs build)
4. Request review from maintainers
5. Address review comments

## Optional Dependencies

FiberNet is designed to work with only `numpy` and `scipy`. When adding features that require optional dependencies:

- Make imports conditional with try/except
- Provide clear error messages when dependencies are missing
- Document the optional dependency in docstrings

Example:
```python
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

def my_function():
    if not HAS_NETWORKX:
        raise ImportError("networkx required. Install with: pip install fibernet[graph]")
    # ... use networkx
```

## Questions?

- Open a GitHub on GitHub
- Email the ML-BioMat lab

Thank you for contributing!
