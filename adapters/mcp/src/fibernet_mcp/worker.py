"""Fixed operation subprocess; resource limits precede scientific imports."""
import os
import resource
import sys
resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
resource.setrlimit(resource.RLIMIT_FSIZE, (16*1024*1024, 16*1024*1024))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
from .safety import canonical, strict_json, read_file, write_at, open_directory, _verify_core_result, sha

def main():
    operation, source, output, expected_sha = sys.argv[1:]
    if operation not in ('validate','topology'): raise ValueError('unknown operation')
    # The source was preflighted by the parent; reparse strict bytes here.
    core_verification = _verify_core_result()
    raw=read_file(source)
    if sha(raw)!=expected_sha: raise ValueError('source integrity mismatch')
    data = strict_json(raw)
    try:
        from fibernet.core.structure_graph import StructureGraph
        from fibernet.analysis.topology_metrics import analyze_topology
        from dataclasses import asdict
        graph = StructureGraph.from_dict(data)
        metrics = asdict(analyze_topology(graph))
        result = {'valid': True, 'metrics': metrics} if operation == 'topology' else {'valid': True}
    except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError):
        result = {'valid': False}
    fd = open_directory(os.path.dirname(output))
    payload = {'core_provenance': core_verification.provenance(), 'result': result}
    try: write_at(fd, os.path.basename(output), canonical(payload))
    finally: os.close(fd)
if __name__ == '__main__': main()
