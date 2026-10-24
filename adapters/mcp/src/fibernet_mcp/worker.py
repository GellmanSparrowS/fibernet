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
    operation, source, output, expected_sha = sys.argv[1:5]
    if operation not in ('validate','topology','linear_transport','fiber_projection_report'): raise ValueError('unknown operation')
    if len(sys.argv) != (7 if operation in ('linear_transport','fiber_projection_report') else 5):
        raise ValueError('invalid worker arguments')
    # The source was preflighted by the parent; reparse strict bytes here.
    core_verification = _verify_core_result()
    raw=read_file(source)
    if sha(raw)!=expected_sha: raise ValueError('source integrity mismatch')
    data = strict_json(raw)
    if operation == 'linear_transport':
        from .electrical import MAX_TRANSPORT_REQUEST_BYTES, solve_transport
        request_raw = read_file(sys.argv[5], MAX_TRANSPORT_REQUEST_BYTES)
        if sha(request_raw) != sys.argv[6]: raise ValueError('request integrity mismatch')
        result = solve_transport(data, strict_json(request_raw))
    elif operation == 'fiber_projection_report':
        from .projection import MAX_PROJECTION_REQUEST_BYTES, MAX_PROJECTION_WORKER_BYTES, project_report
        request_raw = read_file(sys.argv[5], MAX_PROJECTION_REQUEST_BYTES)
        if sha(request_raw) != sys.argv[6]: raise ValueError('request integrity mismatch')
        result = project_report(data, strict_json(request_raw))
    else:
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
    try:
        raw = canonical(payload)
        if operation == 'fiber_projection_report' and len(raw) > MAX_PROJECTION_WORKER_BYTES:
            raise ValueError('projection worker output limit')
        write_at(fd, os.path.basename(output), raw)
    finally: os.close(fd)
if __name__ == '__main__': main()
