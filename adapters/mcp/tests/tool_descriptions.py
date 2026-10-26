"""Advertised MCP descriptions, schema invariance and a metadata-only SDK session.

Run ``python -B adapters/mcp/tests/tool_descriptions.py`` from a matching checkout.
Optional ``--work-root DIR`` retains receipts in a fresh/empty directory;
``--static-only`` omits the session. No tool is called and no native code,
scientific provider or worker is imported/launched. Existing SDK dependencies only.
"""
import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import unittest

from distribution_metadata import CHILD, TOOLS, assert_receipt, canonical, work_area

ADAPTER = Path(__file__).resolve().parents[1]
CORE = ADAPTER.parents[1]
SOURCE = ADAPTER / 'src/fibernet_mcp'
SCHEMAS = SOURCE / 'tool_schemas.json'
WORK_ROOT = None
# Main cd4c4e9 contract, omitting ONLY the five descriptions being improved.
# This pins tool names/order, all schemas/annotations and the other descriptions;
# deliberate future contract changes should update this sentinel explicitly.
from snapshot_compatibility import legacy_seven

CONTRACT_SHA256 = 'c704df4cf26dc71114bc036dde2e59c810f5e5ef302ed2d558cfb624877d5097'
HINTS = {
    'fibernet_catalog': ('graph_id', 'length_unit', 'operator-asserted', 'session', 'structuredContent'),
    'fibernet_validate': ('graph_id', 'validation_status', 'invalid', 'solver readiness', 'isError'),
    'fibernet_analyze_topology': ('graph_id', 'artifact_id', 'dimensionless', 'physical-edge', 'manifest'),
    'fibernet_artifact_info': ('artifact_id', 'byte_size', 'SHA256', 'manifest', 'shutdown'),
    'fibernet_artifact_read': ('offset', '32768', 'base64', 'next_offset', 'eof', 'sha256', 'ephemeral'),
}


def schemas():
    return json.loads(SCHEMAS.read_bytes())


def file_hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def refs(value):
    if isinstance(value, dict):
        if '$ref' in value:
            yield value['$ref'].removeprefix('#/$defs/')
        for key, child in value.items():
            if key != '$defs':
                yield from refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from refs(child)


def advertised_schema(value):
    """Independently select the transitive local definitions needed on the wire."""
    result = copy.deepcopy(value)
    definitions = result.pop('$defs', {})
    required = set(refs(result))
    pending = list(required)
    while pending:
        for name in refs(definitions[pending.pop()]):
            if name not in required:
                required.add(name)
                pending.append(name)
    result['$defs'] = {name: definitions[name] for name in required}
    return result


class DescriptionContract(unittest.TestCase):
    def test_only_first_five_descriptions_are_outside_frozen_contract(self):
        value = schemas()
        self.assertEqual(tuple(t['name'] for t in value['tools']), TOOLS)
        value['tools']=legacy_seven(value['tools'])
        for tool in value['tools'][:5]:
            del tool['description']
        self.assertEqual(hashlib.sha256(canonical(value)).hexdigest(), CONTRACT_SHA256)

    def test_first_five_descriptions_explain_inputs_and_results(self):
        for tool in schemas()['tools'][:5]:
            with self.subTest(tool=tool['name']):
                description = tool['description']
                self.assertGreater(len(description), 120)
                for hint in HINTS[tool['name']]:
                    self.assertIn(hint, description)

    def test_metadata_regression_is_in_bounded_ci_suite(self):
        tree = ast.parse((ADAPTER / 'tests/bounded_suite.py').read_text())
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'SCRIPTS'
                               for target in node.targets)]
        self.assertEqual(len(assignments), 1)
        selected = ast.literal_eval(assignments[0].value)
        self.assertEqual(selected.count(Path(__file__).name), 1)

    def test_all_input_and_output_schemas_remain_valid(self):
        from jsonschema import Draft202012Validator
        for tool in schemas()['tools']:
            for key in ('inputSchema', 'outputSchema'):
                with self.subTest(tool=tool['name'], schema=key):
                    Draft202012Validator.check_schema(tool[key])


class MetadataSession(unittest.TestCase):
    def test_sdk_advertises_descriptions_without_native_work(self):
        import anyio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self.assertEqual(sys.platform, 'linux')
        self.assertTrue(sys.dont_write_bytecode, 'run with Python -B')
        self.assertEqual(importlib.metadata.version('mcp'), '1.29.0')
        before = file_hashes(SOURCE)
        manifest_raw = (SOURCE / 'core_manifest.json').read_bytes()
        manifest = json.loads(manifest_raw)
        actual = {p.relative_to(CORE / 'fibernet').as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (CORE / 'fibernet').rglob('*.py')}
        self.assertEqual(len(manifest), 247)
        self.assertIn("examples/linear_static_network.py", manifest)
        self.assertEqual(actual, manifest)
        expected = schemas()['tools']
        expected = [dict(tool, inputSchema=advertised_schema(tool['inputSchema']),
                         outputSchema=advertised_schema(tool['outputSchema'])) for tool in expected]
        environment = {k: v for k, v in os.environ.items()
                       if k not in ('PYTHONPATH', 'FIBERNET_CORE_PATH')}
        environment.update(PYTHONPATH=os.pathsep.join((str(ADAPTER / 'src'), str(CORE))),
                           PYTHONDONTWRITEBYTECODE='1')
        with work_area(WORK_ROOT) as area:
            output = area / 'output'
            output.mkdir()
            # Serialized minimal input only; do not construct/generate a native graph.
            graph = area / 'graph.json'
            graph.write_bytes(canonical({'version': '3.1', 'dimension': 3, 'tolerance': 1e-6,
                'nodes': [{'id': 0, 'position': [0, 0, 0]}], 'edges': []}))
            operator = area / 'operator.json'
            operator.write_bytes(canonical({'source_tree': '6eabb154cd37d95c7c447a568b1b737c3ec883c8',
                'graphs': [{'path': str(graph), 'sha256': hashlib.sha256(graph.read_bytes()).hexdigest(),
                            'length_unit': 'unspecified', 'frame_id': 'unknown'}]}))
            receipt = area / 'server-receipt.json'
            params = StdioServerParameters(command=sys.executable, args=['-B', '-c', CHILD,
                'server', str(CORE), str(ADAPTER / 'src'), str(operator), str(output), str(receipt)],
                env=environment, cwd=str(area))

            async def forbidden_call(*args, **kwargs):
                raise AssertionError('tool calls are forbidden in this metadata test')

            async def session_check():
                with anyio.fail_after(20):
                    async with stdio_client(params) as (reader, writer):
                        async with ClientSession(reader, writer) as session:
                            session.call_tool = forbidden_call
                            initialized = await session.initialize()
                            self.assertEqual(initialized.protocolVersion, '2025-11-25')
                            listing = await session.list_tools()
                            actual_tools = [tool.model_dump(by_alias=True, exclude_none=True)
                                            for tool in listing.tools]
                            self.assertEqual(actual_tools, expected)
                            (area / 'advertised-tools.json').write_bytes(canonical(actual_tools) + b'\n')
                            (area / 'initialized.json').write_text(initialized.model_dump_json() + '\n')
            anyio.run(session_check)
            assert_receipt(receipt)
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(file_hashes(SOURCE), before)
            self.assertEqual((SOURCE / 'core_manifest.json').read_bytes(), manifest_raw)
            after = {p.relative_to(CORE / 'fibernet').as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (CORE / 'fibernet').rglob('*.py')}
            self.assertEqual(after, manifest)
            (area / 'result.json').write_bytes(canonical({'sdk_sessions': 1,
                'initialize_requests': 1, 'list_tools_requests': 1, 'listed_tools': len(expected),
                'call_tool_requests': 0, 'scientific_workers': 0, 'source_hashes_verified': len(manifest),
                'binding_unchanged': True, 'output_empty_after_close': True}) + b'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path)
    parser.add_argument('--static-only', action='store_true')
    args = parser.parse_args()
    WORK_ROOT = args.work_root
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DescriptionContract)
    if not args.static_only:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(MetadataSession))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
