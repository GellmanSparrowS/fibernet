Contributing
============

Thank you for your interest in contributing to FiberNet!

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Install in development mode: ``pip install -e ".[dev]"``
4. Create a feature branch: ``git checkout -b feature/my-feature``
5. Make your changes
6. Run tests: ``pytest tests/``
7. Commit and push: ``git push origin feature/my-feature``
8. Open a Pull Request

Code Style
----------

- Follow PEP 8
- Use type hints for all public functions
- Add NumPy-format docstrings
- Keep functions focused and composable

Adding Generators
-----------------

New generators should:

- Accept a ``seed`` parameter for reproducibility
- Return a ``FiberNetwork`` instance
- Have unit tests in ``tests/test_generators.py``
- Be documented with parameters and examples

Adding Simulation Models
------------------------

New simulation models should:

- Accept a ``FiberNetwork`` as input
- Return a result dataclass
- Have unit tests covering basic functionality
- Document physical assumptions

Development Commands
--------------------

.. code-block:: bash

   # Run tests
   pytest tests/ -v

   # Run with coverage
   pytest tests/ --cov=fibernet

   # Build docs
   cd docs && make html

   # Check formatting
   black --check fibernet/ tests/
