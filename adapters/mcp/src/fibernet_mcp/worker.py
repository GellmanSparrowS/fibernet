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
    if operation not in ('validate','topology','linear_transport','prepare_deformed_transport'): raise ValueError('unknown operation')
    if len(sys.argv) != (7 if operation in ('linear_transport','prepare_deformed_transport') else 5):
        raise ValueError('invalid worker arguments')
    # The source was preflighted by the parent; reparse strict bytes here.
    core_verification = _verify_core_result()
    if operation == 'prepare_deformed_transport':
        from .preparation import MAX_PREPARATION_SOURCE_BYTES
        raw=read_file(source,MAX_PREPARATION_SOURCE_BYTES)
    else:
        raw=read_file(source)
    if sha(raw)!=expected_sha: raise ValueError('source integrity mismatch')
    data = strict_json(raw)
    if operation == 'prepare_deformed_transport':
        from .preparation import MAX_PREPARATION_REQUEST_BYTES, prepare_transport
        request_raw=read_file(sys.argv[5],MAX_PREPARATION_REQUEST_BYTES)
        if sha(request_raw)!=sys.argv[6]: raise ValueError('request integrity mismatch')
        result=prepare_transport(data,strict_json(request_raw),core_verification.provenance(),expected_sha)
    elif operation == 'linear_transport':
        from .electrical import MAX_TRANSPORT_REQUEST_BYTES, solve_transport
        request_raw = read_file(sys.argv[5], MAX_TRANSPORT_REQUEST_BYTES)
        if sha(request_raw) != sys.argv[6]: raise ValueError('request integrity mismatch')
        result = solve_transport(data, strict_json(request_raw))
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
        raw=canonical(payload)
        if operation == 'prepare_deformed_transport':
            from .preparation import MAX_PREPARATION_ENVELOPE_BYTES
            from .safety import ContractError
            try:
                if len(raw)>MAX_PREPARATION_ENVELOPE_BYTES: raise ContractError('worker envelope limit')
                strict_json(raw)
            except ContractError:
                payload['result']={'error_code':'PREPARATION_RESULT_LIMIT','error_category':'resource'}
                raw=canonical(payload)
        write_at(fd, os.path.basename(output), raw)
    finally: os.close(fd)
if __name__ == '__main__': main()
