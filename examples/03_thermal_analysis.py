"""Unsupported historical CTE/thermal-deformation example.

Original source is preserved in examples/_archived/03_thermal_analysis.py.
Importing this notice performs no simulation and imports no numerical backend.
"""


def main():
    print(
        'This historical CTE/thermal-deformation example is unsupported.\n'
        'Original source: examples/_archived/03_thermal_analysis.py (historical only).\n'
        'For a DIFFERENT explicit transient conduction model, run:\n'
        '  python -m examples.transient_thermal_explicit\n'
        'That model requires explicit heat capacities/conductances and does not compute CTE or deformation.'
    )
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
