"""Report semantics only: no FiberNet imports, data generation or training."""
import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'examples/physical_learning_workflow.py'


def helper():
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == '_target_report')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['_target_report']


@pytest.mark.parametrize('data_path', [None, Path('custom.npz')])
def test_target_names_and_scope(data_path):
    report = helper()(data_path)
    assert report['targets'] == ['target_0', 'target_1', 'target_2']
    scope = report['target_scope']
    if data_path is None:
        assert 'Synthetic' in scope and 'not physical labels' in scope
    else:
        assert 'definitions and units unspecified' in scope
        assert 'no target metadata consumed' in scope
    assert not any(name in report['targets'] for name in ('stiffness', 'toughness'))
    report['targets'][0] = 'changed'
    assert helper()(data_path)['targets'][0] == 'target_0'


def test_run_summary_uses_report_and_keeps_metric_names():
    tree = ast.parse(SOURCE.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == 'PhysicalLearningWorkflow')
    run = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    summary = next(n.value for n in run.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'summary' for t in n.targets))
    keys = [k.value for k in summary.keys if isinstance(k, ast.Constant)]
    assert keys == ['source', 'schema', 'feature_count', 'model', 'seed',
                    'samples', 'validation_rows', 'rmse_by_target', 'best_step']
    additions = [v for k, v in zip(summary.keys, summary.values) if k is None]
    assert len(additions) == 1
    assert ast.unparse(additions[0]) == '_target_report(self.data_path)'
