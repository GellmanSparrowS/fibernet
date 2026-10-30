"""Documentation/AST checks without importing configuration or solvers."""
import ast
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'fibernet/utils/config.py'
def parsed(path):
    return ast.parse(path.read_text())
def test_documented_configuration_record_and_refusal():
    tree = parsed(CANDIDATE)
    docs = [ast.get_docstring(tree)] + [ast.get_docstring(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('create_template_config', 'run_from_config')]
    assert len(docs) == 3
    for doc in docs:
        assert 'mechanical' in doc and 'stress-strain' in doc
        assert 'small-deformation linear FEM' in doc
        assert 'examples/open_source_workflow.py' in doc
        assert 'not an equivalent' in doc
    assert "results['stress_strain'].youngs_modulus" not in docs[-1]
def test_referenced_example_explicit_model_source():
    tree = parsed(ROOT / 'examples/open_source_workflow.py')
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'simulate']
    assert len(calls) == 1
    kws = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords if k.arg in ('backend','nonlinear')}
    assert kws == {'backend': 'fem', 'nonlinear': False}
    assert 'small-deformation linear analysis' in ast.get_docstring(tree)
