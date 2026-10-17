"""Pure exact-mass type-table tests; synthetic dumps are not native evidence."""
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_lammps_backend_contract import graph, prepared, fixture
from fibernet.integrations import (prepare_harmonic_beads, HarmonicBond, RunSpec,
    LAMMPSTask, RuntimeSpec, ResourceLimits)
from fibernet.integrations.lammps_backend import validate_prepared, parse_outputs, LAMMPSContractError


class ExactMassTypesTests(unittest.TestCase):
    def test_equal_mass_compacts_without_changing_ids(self):
        p=prepared(masses_kg={10:1e-9,40:1e-9});c=json.loads(p.contract_json)
        self.assertIn('1 atom types\n',p.data_text)
        self.assertEqual(c['atom_to_type_id'],[[1,1],[2,1]])
        self.assertEqual(c['type_to_mass_kg'],[[1,1e-9]])
        self.assertEqual(c['type_pair_cutoff_payload_lower_bound_bytes'],64)
        self.assertEqual(p.node_ids,(10,40));self.assertEqual(p.edge_ids,(77,))
        self.assertEqual(p.manifest()['atom_to_node_id'],{'1':10,'2':40})
        self.assertEqual(p.manifest()['bond_to_edge_id'],{'1':77})
        self.assertEqual(p.manifest()['atom_to_type_id'],{'1':1,'2':1})
        self.assertEqual(p.manifest()['type_to_mass_kg'],{'1':1e-9})
        atoms=p.data_text.split('Atoms # bond\n\n')[1].split('\n\n')[0]
        self.assertEqual([row.split()[:3] for row in atoms.splitlines()],[['1','1','1'],['2','1','1']])
        result=parse_outputs(p,*fixture(p));self.assertEqual(result.trajectory[0].node_ids,(10,40))

    def test_heterogeneous_table_has_numeric_deterministic_order(self):
        p=prepared(masses_kg={10:2e-9,40:1e-9});c=json.loads(p.contract_json)
        self.assertEqual(c['atom_to_type_id'],[[1,2],[2,1]])
        self.assertEqual(c['type_to_mass_kg'],[[1,1e-9],[2,2e-9]])
        self.assertEqual(c['type_pair_cutoff_payload_lower_bound_bytes'],144)
        masses=p.data_text.split('Masses\n\n')[1].split('\n\n')[0]
        self.assertEqual([(int(row.split()[0]),float(row.split()[1])) for row in masses.splitlines()],[(1,1e-9),(2,2e-9)])
        parse_outputs(p,*fixture(p))

    def test_adjacent_binary64_masses_remain_distinct(self):
        lower=1e-9;upper=math.nextafter(lower,math.inf)
        p=prepared(masses_kg={10:lower,40:upper});c=json.loads(p.contract_json)
        self.assertEqual(len(c['type_to_mass_kg']),2)
        data=p.data_text.split('Masses\n\n')[1].split('\n\n')[0]
        self.assertEqual([float(r.split()[1]).hex() for r in data.splitlines()],[lower.hex(),upper.hex()])

    def test_declared_kg_normalization_is_exact(self):
        p=prepared(masses_kg={10:1,40:1.0})
        self.assertEqual(json.loads(p.contract_json)['type_to_mass_kg'],[[1,1.0]])
        self.assertEqual(json.loads(p.contract_json)['atom_type_key_schema'],'mass_kg.binary64.v1')

    def test_source_insertion_order_does_not_change_identity(self):
        g=graph();g.nodes.update({})
        g._nodes=dict(reversed(list(g.nodes.items())))
        args=dict(length_unit='mm',maximum_bond_length_m=.01,masses_kg={40:2e-9,10:1e-9},
            bonds={77:HarmonicBond(2,.001)},run=RunSpec(1e-6,2,1,((-.01,.01),)*3))
        self.assertEqual(prepare_harmonic_beads(g,**args),prepared())

    def test_equal_mass_preserves_different_metadata_boundaries_and_loads(self):
        g=graph();g.nodes[10].metadata['source']='left';g.nodes[40].metadata['source']='right'
        before=json.dumps(g.to_dict(),sort_keys=True)
        p=prepare_harmonic_beads(g,length_unit='mm',maximum_bond_length_m=.01,
            masses_kg={10:1e-9,40:1e-9},bonds={77:HarmonicBond(2,.001)},
            run=RunSpec(1e-6,2,1,((-.01,.01),)*3),fixed_node_ids=(10,),forces_N={40:(1,2,3)},velocities_m_s={40:(.1,.2,.3)})
        self.assertEqual(json.loads(p.contract_json)['atom_to_type_id'],[[1,1],[2,1]])
        self.assertEqual(json.loads(p.graph_json),g.to_dict())
        self.assertEqual(json.dumps(g.to_dict(),sort_keys=True),before)
        self.assertIn('group held id 1',p.input_text);self.assertIn('group loaded0 id 2',p.input_text)
        self.assertNotIn('group held type',p.input_text)
        parse_outputs(p,*fixture(p))

    def test_wrong_valid_mass_class_for_exact_atom_is_rejected(self):
        p=prepared();dump,log=fixture(p)
        bad=dump.replace('1 1 0.0','1 2 0.0').replace('2 2 0.001','2 1 0.001')
        self.assertNotEqual(dump,bad)
        with self.assertRaises(LAMMPSContractError):parse_outputs(p,bad,log)

    def test_old_type_equals_id_dump_rejected_for_compacted_atom(self):
        p=prepared(masses_kg={10:1e-9,40:1e-9});dump,log=fixture(p)
        bad=dump.replace('2 1 0.001','2 2 0.001');self.assertNotEqual(dump,bad)
        with self.assertRaises(LAMMPSContractError):parse_outputs(p,bad,log)

    def test_contract_type_or_mass_table_tampering_rejected(self):
        p=prepared()
        for key,value in [('atom_to_type_id',[[1,2],[2,1]]),('type_to_mass_kg',[[1,2e-9],[2,1e-9]]),
                          ('atom_type_key_schema','future-unknown'),('type_pair_cutoff_payload_lower_bound_bytes',0)]:
            c=json.loads(p.contract_json);c[key]=value
            with self.subTest(key=key),self.assertRaises(LAMMPSContractError):validate_prepared(replace(p,contract_json=json.dumps(c,sort_keys=True,separators=(',',':'))))

    def test_mass_data_table_tampering_rejected(self):
        p=prepared();bad=p.data_text.replace('1 1.0000000000000001e-09','1 2.0000000000000001e-09')
        self.assertNotEqual(bad,p.data_text)
        with self.assertRaises(LAMMPSContractError):validate_prepared(replace(p,data_text=bad))

    def test_missing_compaction_schema_not_silently_upgraded(self):
        p=prepared();c=json.loads(p.contract_json)
        for key in ['atom_to_type_id','type_to_mass_kg','atom_type_key_schema','type_pair_cutoff_payload_lower_bound_bytes']:del c[key]
        with self.assertRaises(LAMMPSContractError):validate_prepared(replace(p,contract_json=json.dumps(c,sort_keys=True,separators=(',',':'))))
        self.assertEqual(p.manifest()['schema'],'fibernet.lammps.explicit-beads-exact-mass-types.v3')

    def test_shared_junction_with_one_mass_class_remains_one_atom(self):
        g=graph();third=g.add_node([1,1,0]);edge=g.add_edge(40,third,n_internal=0)
        p=prepare_harmonic_beads(g,length_unit='mm',maximum_bond_length_m=.01,
            masses_kg={n:1e-9 for n in g.nodes},bonds={e:HarmonicBond(1,.001) for e in g.edges},run=RunSpec(1e-6,2,1,((-.01,.01),)*3))
        self.assertIn('3 atoms\n',p.data_text);self.assertIn('1 atom types\n',p.data_text)
        self.assertIn('1 1 1 2\n2 2 2 3',p.data_text)
        self.assertEqual(p.node_ids,(10,40,third));self.assertEqual(p.edge_ids,(77,edge))

    def test_lower_bound_rejects_before_runtime_check_or_directory(self):
        with tempfile.TemporaryDirectory() as root:
            target=Path(root)/'never-created'
            with patch.object(ResourceLimits,'checked',lambda self:self),patch.object(RuntimeSpec,'checked') as runtime_check:
                with self.assertRaises(LAMMPSContractError):LAMMPSTask(prepared(),RuntimeSpec('cli','/fake'),target,ResourceLimits(memory_bytes=100))
            runtime_check.assert_not_called();self.assertFalse(target.exists())

if __name__=='__main__':unittest.main()
