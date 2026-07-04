Contributing
============

We welcome contributions to FiberNet! This guide will help you get started.

Development Setup
-----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/GellmanSparrowS/fibernet.git
      cd fibernet

2. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

3. Run the tests:

   .. code-block:: bash

      pytest tests/

Code Style
----------

FiberNet follows PEP 8 style guidelines. We use ``black`` for code formatting:

.. code-block:: bash

   black fibernet tests

Running Tests
-------------

Run all tests:

.. code-block:: bash

   pytest tests/ -v

Run specific test file:

.. code-block:: bash

   pytest tests/test_core.py -v

Run with coverage:

.. code-block:: bash

   pytest tests/ --cov=fibernet --cov-report=html

Adding New Features
-------------------

1. **Generators**: Add to ``fibernet/gen/`` and update ``__init__.py``
2. **Simulators**: Add to ``fibernet/sim/`` and update ``__init__.py``
3. **Analyzers**: Add to ``fibernet/analysis/`` and update ``__init__.py``

Writing Tests
-------------

All new features should include tests. Place tests in ``tests/test_*.py``:

.. code-block:: python

   import pytest
   from fibernet import gen

   def test_my_feature():
       net = gen.random_straight_2d(num_fibers=10, seed=42)
       assert net.num_fibers == 10

Documentation
-------------

Update documentation in ``docs/`` when adding features:

- Add API reference to ``docs/api/``
- Update tutorials in ``tutorials/``
- Update changelog in ``docs/changelog.rst``

Build documentation:

.. code-block:: bash

   cd docs
   make html

Submitting Changes
------------------

1. Fork the repository
2. Create a feature branch: ``git checkout -b feature/my-feature``
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: ``pytest tests/``
6. Commit with a clear message
7. Push and submit a Pull Request

Reporting Issues
----------------

Open an issue on GitHub with:

- Clear description of the problem
- Minimal reproducible example
- Environment details (Python version, OS, dependencies)
- Expected vs actual behavior

Contact
-------

For questions, email the ML-BioMat lab or open a GitHub Discussion.
