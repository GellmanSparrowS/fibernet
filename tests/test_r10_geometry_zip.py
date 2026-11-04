"""Synthetic ZIP inputs only; no private research geometry or solver."""
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import zipfile

from fibernet.io.r10_geometry import read_r10_geometry_zip, import_r10_geometry
from test_r10_geometry import fixture


class R10ZipGeometryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'bundle.zip'
        self.member='package/examples/synthetic/geometry.json'
        self.data,self.props=fixture()
        self.raw=json.dumps(self.data,allow_nan=False).encode()
    def write(self, data=None, member=None, compression=zipfile.ZIP_DEFLATED):
        with zipfile.ZipFile(self.path,'w',compression=compression) as z:
            z.writestr(self.member if member is None else member,self.raw if data is None else data)
    def read(self, **kwargs):
        return read_r10_geometry_zip(self.path,self.member,length_unit='mm',**kwargs)
    def test_roundtrip_valid_and_no_extraction(self):
        self.write(); before=set(Path(self.tmp.name).iterdir())
        source=self.read(); self.assertEqual(source,self.data)
        source['case_id']='changed'; self.assertEqual(self.read()['case_id'],'synthetic')
        graph,binding=import_r10_geometry(self.read(),edge_properties=self.props,length_unit='mm')
        self.assertEqual((len(graph.nodes),len(graph.edges)),(3,3))
        self.assertEqual(set(Path(self.tmp.name).iterdir()),before)
        with self.path.open('rb') as stream, self.assertRaises(ValueError):
            read_r10_geometry_zip(stream,self.member,length_unit='mm')
    def test_path_and_limits(self):
        self.write()
        for name in ('', '/abs.json','../a.json','a/../b.json','a/./b.json','a//b.json',
                     'C:/x.json','a\\b.json','a\n.json','a.txt','a/' ,'a'*4097+'.json'):
            with self.subTest(name=name),self.assertRaises(ValueError):
                read_r10_geometry_zip(self.path,name,length_unit='mm')
        for cap in (0,-1,True,1.5,8*1024*1024+1):
            with self.subTest(cap=cap),self.assertRaises(ValueError): self.read(max_bytes=cap)
        with self.assertRaisesRegex(ValueError,'byte limit'): self.read(max_bytes=len(self.raw)-1)
        self.assertEqual(self.read(max_bytes=len(self.raw)),self.data)
        with self.assertRaises(ValueError): read_r10_geometry_zip(self.path,self.member,length_unit='m')
    def test_duplicate_or_missing_members(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with zipfile.ZipFile(self.path,'w') as z:
                z.writestr(self.member,self.raw);z.writestr(self.member,self.raw)
        with self.assertRaisesRegex(ValueError,'exactly once'): self.read()
        self.write(member='elsewhere.json')
        with self.assertRaisesRegex(ValueError,'exactly once'): self.read()
    def test_invalid_json_and_duplicate_nested_keys(self):
        for data in (b'\xff', b'{', b'{"schema":"x","schema":"y"}',
                     b'{"nested":{"x":1,"x":2}}', b'{"x":NaN}', b'{}'):
            with self.subTest(data=data):
                self.write(data)
                with self.assertRaises(ValueError): self.read()
    def test_decimal_rounding_and_range(self):
        text=self.raw.decode().replace('"case_id": "synthetic"',
            '"case_id": "synthetic", "numeric_probe": 0.10000000000000001')
        self.write(text.encode()); self.assertEqual(self.read()['numeric_probe'],0.1)
        for value in ('1e-400','-1e-400','1e400','-1e400'):
            with self.subTest(value=value):
                self.write(text.replace('0.10000000000000001',value).encode())
                with self.assertRaisesRegex(ValueError,'binary64 range'): self.read()
    def test_bad_zip_unsupported_compression_and_symlink(self):
        self.path.write_bytes(b'not a zip')
        with self.assertRaises(ValueError): self.read()
        self.write(compression=zipfile.ZIP_BZIP2)
        with self.assertRaisesRegex(ValueError,'compression'): self.read()
        info=zipfile.ZipInfo(self.member);info.create_system=3
        info.external_attr=(stat.S_IFLNK|0o777)<<16
        with zipfile.ZipFile(self.path,'w') as z: z.writestr(info,'target')
        with self.assertRaisesRegex(ValueError,'regular'): self.read()
    def test_crc_corruption_and_missing_file(self):
        self.write(compression=zipfile.ZIP_STORED)
        data=bytearray(self.path.read_bytes()); idx=data.index(self.raw);data[idx]^=1
        self.path.write_bytes(data)
        with self.assertRaises(ValueError): self.read()
        self.path.unlink()
        with self.assertRaises(ValueError): self.read()
        if hasattr(os,'mkfifo') and hasattr(os,'O_NONBLOCK'):
            fifo=Path(self.tmp.name)/'pipe'; os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError,'regular'):
                read_r10_geometry_zip(fifo,self.member,length_unit='mm')
    def test_compressed_size_budget_and_encryption_admission(self):
        self.write()
        real=zipfile.ZipFile.infolist
        for attr,value,reason in (('compress_size',8*1024*1024+1,'byte limit'),
                                  ('flag_bits',1,'encrypted'),
                                  ('orig_filename',self.member+'\0hidden','altered')):
            def altered(z,attr=attr,value=value):
                entries=real(z);setattr(entries[0],attr,value);return entries
            with self.subTest(attr=attr),patch.object(zipfile.ZipFile,'infolist',altered):
                with self.assertRaisesRegex(ValueError,reason): self.read()
    def test_archive_size_and_entry_caps(self):
        self.write()
        fake=SimpleNamespace(st_mode=stat.S_IFREG,st_size=512*1024*1024+1)
        with patch('os.fstat',return_value=fake),self.assertRaisesRegex(ValueError,'archive byte limit'):
            self.read()
        with patch.object(zipfile.ZipFile,'infolist',return_value=[None]*10001):
            with self.assertRaisesRegex(ValueError,'entry limit'): self.read()

if __name__ == '__main__': unittest.main()
