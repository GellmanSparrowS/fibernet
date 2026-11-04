"""Fail-closed writer API tests; no LAMMPS/native execution."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.io.lammps import to_lammps, from_lammps
from fibernet import io
from fibernet.api import export


def fixture(crosslinked=False):
    net=FiberNetwork(fibers=[Fiber([[0.,0.,0.],[2.,0.,0.]],radius=.25,fiber_id=0),
                            Fiber([[1.,-1.,0.],[1.,1.,0.]],radius=.25,fiber_id=1)])
    if crosslinked:net.crosslinks.append(Crosslink(0,1,.5,.5,np.array([1.,0.,0.])))
    return net


class LegacyLammpsExportBoundary(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def assert_no_write(self,call,suffix='.data'):
        for exists in (False,True):
            path=self.root/('existing'+suffix if exists else 'absent'+suffix)
            if exists:path.write_bytes(b'KEEP\x00ORIGINAL')
            elif path.exists():path.unlink()
            with self.assertRaisesRegex(NotImplementedError,'legacy to_lammps export is unsupported'):
                call(path)
            if exists:self.assertEqual(path.read_bytes(),b'KEEP\x00ORIGINAL')
            else:self.assertFalse(path.exists())

    def test_public_aliases_and_reader_still_importable(self):
        self.assertIs(io.to_lammps,to_lammps)
        self.assertIs(io.from_lammps,from_lammps)
        self.assertTrue(callable(from_lammps))

    def test_direct_straight_crosslinked_empty_and_curved_reject(self):
        curved=FiberNetwork(fibers=[Fiber([[0,0,0],[1,1,0],[2,0,0]])])
        for net in (fixture(),fixture(True),FiberNetwork(),curved):
            self.assert_no_write(lambda path:to_lammps(net,path))

    def test_api_inferred_extensions_reject(self):
        for suffix in ('.data','.lammps'):
            self.assert_no_write(lambda path:export(fixture(),path),suffix)

    def test_api_explicit_format_rejects(self):
        self.assert_no_write(lambda path:export(fixture(True),path,format='lammps'),'.custom')

    def test_options_do_not_bypass_rejection(self):
        for atom,bond,units in [('bond','harmonic','real'),('molecular','fene','si'),('full','morse','lj')]:
            self.assert_no_write(lambda path:to_lammps(fixture(),path,atom_style=atom,bond_type=bond,units=units,bead_spacing=1.))

    def test_rejection_precedes_input_access_resampling_and_io(self):
        class Untouchable:
            def __getattribute__(self,name):raise AssertionError('input must not be evaluated')
        with patch.object(Fiber,'resample',side_effect=AssertionError('must not resample')) as resample:
            with patch('builtins.open',side_effect=AssertionError('must not open')) as opened:
                with self.assertRaises(NotImplementedError):to_lammps(Untouchable(),Untouchable(),bead_spacing=0)
        resample.assert_not_called();opened.assert_not_called()

    def test_network_unchanged_and_message_actionable(self):
        net=fixture(True);before=net.to_dict()
        with self.assertRaises(NotImplementedError) as caught:to_lammps(net,self.root/'out.data')
        self.assertEqual(net.to_dict(),before)
        for text in ('save_json','StructureGraph','length_unit','masses_kg','no automatic conversion'):
            self.assertIn(text,str(caught.exception))

    def test_native_json_preserves_fixture_centerlines_crosslinks(self):
        net=fixture(True);path=self.root/'native.json';export(net,path)
        self.assertEqual(FiberNetwork.load_json(path).to_dict(),net.to_dict())

if __name__=='__main__':unittest.main()
