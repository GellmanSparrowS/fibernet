"""Portable CLI-only publication contract; no native engine is used."""
import ast
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from fibernet.integrations import LAMMPSContractError, RuntimeSpec
from fibernet.integrations import lammps_backend as backend

ROOT=Path(__file__).resolve().parents[1]


def load_source(name,path):
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


class PublicationTests(unittest.TestCase):
    def test_python39_syntax_of_added_modules_and_example(self):
        for relative in ('fibernet/integrations/lammps_backend.py','fibernet/integrations/_lammps_worker.py',
                         'examples/22_lammps_harmonic_network.py','fibernet/integrations/__init__.py',
                         'tests/test_lammps_backend_contract.py','tests/test_lammps_type_compaction.py',
                         'tests/test_lammps_publication.py'):
            ast.parse((ROOT/relative).read_text(),feature_version=(3,9))

    def test_linked_engine_choice_rejected_before_io(self):
        with patch('builtins.open',side_effect=AssertionError('must reject before I/O')):
            for mode in ('python','auto','unknown'):
                with self.subTest(mode=mode),self.assertRaisesRegex(LAMMPSContractError,'only CLI'):
                    RuntimeSpec(mode,sys.executable).checked()

    def test_unsupported_platform_and_architecture_rejected(self):
        with patch.object(backend.sys,'platform','win32'):
            with self.assertRaisesRegex(LAMMPSContractError,'Linux x86_64'):
                RuntimeSpec('cli',sys.executable).checked()
        with patch.object(backend.sys,'platform','linux'),patch.object(backend.platform,'machine',return_value='aarch64'):
            with self.assertRaisesRegex(LAMMPSContractError,'Linux x86_64'):
                RuntimeSpec('cli',sys.executable).checked()

    def test_worker_has_no_engine_import_or_entry_point_discovery(self):
        source=(ROOT/'fibernet/integrations/_lammps_worker.py').read_text()
        tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):self.assertNotIn('lammps',[n.name for n in node.names])
            if isinstance(node,ast.ImportFrom):self.assertNotEqual(node.module,'lammps')
        self.assertNotIn('entry_points',source)

    def test_worker_rejects_python_without_exec_or_resource_changes(self):
        worker=load_source('lammps_worker_no_run',ROOT/'fibernet/integrations/_lammps_worker.py')
        fake_resource=types.SimpleNamespace(setrlimit=lambda *a: (_ for _ in ()).throw(AssertionError('limits must not change')))
        with patch.object(worker.sys,'platform','linux'),patch.dict(sys.modules,{'resource':fake_resource}), \
             patch.object(worker.sys,'argv',['worker',json.dumps({'mode':'python'})]), \
             patch.object(worker.os,'execve',side_effect=AssertionError('native launch forbidden')):
            with self.assertRaisesRegex(ValueError,'only CLI'):worker.main()

    def test_documented_minimal_prepare_example(self):
        text=(ROOT/'docs/integrations/LAMMPS_HARMONIC_CLI.md').read_text()
        snippet=text.split('```python\n',1)[1].split('```',1)[0]
        namespace={}
        with patch('subprocess.Popen',side_effect=AssertionError('native launch forbidden')),patch('builtins.print'):
            exec(compile(snippet,'documented-prepare-example','exec'),namespace)
        self.assertEqual(namespace['prepared'].steps,100)
        self.assertEqual(len(namespace['prepared'].node_ids),2)

    def test_cli_worker_applies_limits_then_exact_native_argv(self):
        worker=load_source('lammps_worker_mock_cli',ROOT/'fibernet/integrations/_lammps_worker.py')
        calls=[]
        fake_resource=types.SimpleNamespace(RLIMIT_CPU=1,RLIMIT_AS=2,RLIMIT_FSIZE=3,RLIMIT_CORE=4,
                                           setrlimit=lambda *args:calls.append(args))
        config=dict(mode='cli',cpu_seconds=3,memory_bytes=1024**3,file_size_bytes=8*1024**2,
                    file_sha256=[],executable='/reviewed/native/lmp')
        def fake_exec(path,argv,env):
            self.assertEqual(calls,[(1,(3,3)),(2,(1024**3,1024**3)),(3,(8*1024**2,8*1024**2)),(4,(0,0))])
            self.assertEqual(path,'/reviewed/native/lmp')
            self.assertEqual(argv,[path,'-in','in.model','-log','log.lammps','-screen','none'])
            self.assertIs(env,worker.os.environ)
        with patch.object(worker.sys,'platform','linux'),patch.dict(sys.modules,{'resource':fake_resource}), \
             patch.object(worker.sys,'argv',['worker',json.dumps(config)]),patch.object(worker.os,'execve',side_effect=fake_exec) as launch:
            worker.main()
        launch.assert_called_once()

    def test_representative_example_preserves_qualified_prepared_bytes(self):
        example=load_source('lammps_example_no_run',ROOT/'examples/22_lammps_harmonic_network.py')
        with patch('subprocess.Popen',side_effect=AssertionError('native launch forbidden')):
            prepared=example.prepare_example()
        self.assertEqual(prepared.identity,'b4ab1d63c9f49dfb4c09198d615df3b67fa4293249460a1f8dc3853af4706e19')
        self.assertEqual(len(prepared.node_ids),64);self.assertEqual(len(prepared.edge_ids),441)
        self.assertEqual(prepared.steps,100)
        contract=json.loads(prepared.contract_json)
        self.assertEqual(len(contract['type_to_mass_kg']),3)
        self.assertEqual(contract['maximum_bond_length_m'],.003)


if __name__=='__main__':unittest.main()
