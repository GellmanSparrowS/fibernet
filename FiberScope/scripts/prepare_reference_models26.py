"""Resume lightweight reference OBJ preparation; originals are read-only."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fslab.surface_mapping import load_obj


class ReferenceModels:
    def __init__(self, root):
        self.root = Path(root)
        self.output = self.root / 'assets' / 'obj'
        self.manifest = self.output / 'reference26.json'

    def run(self):
        records = json.loads(self.manifest.read_text(encoding='utf-8')) if self.manifest.exists() else {}
        for source, target in (('Shirt OBJ.obj','reference_shirt.obj'),
                               ('Shoe.obj','reference_shoe.obj'), ('paris.obj','reference_paris.obj')):
            path = self.root / '参考' / 'OBJ' / source
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            destination = self.output / target
            if records.get(target, {}).get('source_sha256') == digest and destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() == records[target]['sha256']:
                    continue
            vertices, faces, info = load_obj(path, return_info=True, target_faces=1200)
            temporary = destination.with_suffix('.obj.tmp')
            with temporary.open('w',encoding='utf-8',newline='\n') as stream:
                stream.write('# FiberScope lightweight reference; source preserved\n')
                for point in vertices:
                    stream.write('v %.7g %.7g %.7g\n' % tuple(point))
                for face in faces:
                    stream.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')
            temporary.replace(destination)
            records[target] = dict(source='参考/OBJ/'+source, source_sha256=digest,
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), bytes=destination.stat().st_size,
                source_license='not supplied', conversion=info)
            temp_manifest = self.manifest.with_suffix('.json.tmp')
            temp_manifest.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            temp_manifest.replace(self.manifest)
            print('[asset]',target,len(faces),'quads',destination.stat().st_size,'bytes',flush=True)


if __name__ == '__main__':
    ReferenceModels(Path(__file__).resolve().parents[1]).run()
