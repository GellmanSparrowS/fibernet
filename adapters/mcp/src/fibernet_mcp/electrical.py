"""Bounded wire translation for the public DC solver, not another physics model."""
from .safety import ContractError, canonical, integer, keys, number

MAX_TRANSPORT_NODES = 256
MAX_TRANSPORT_EDGES = 1024
MAX_TRANSPORT_REQUEST_BYTES = 256 * 1024
MAX_TRANSPORT_RESULT_BYTES = 1024 * 1024
TRANSPORT_LIMITS = {'max_nodes': MAX_TRANSPORT_NODES, 'max_edges': MAX_TRANSPORT_EDGES,
                    'max_request_bytes': MAX_TRANSPORT_REQUEST_BYTES,
                    'max_result_bytes': MAX_TRANSPORT_RESULT_BYTES}


class TransportContractError(ContractError):
    def __init__(self, code, category='contract'):
        super().__init__(code)
        self.code, self.category = code, category


def transport_request(graph_info, arguments):
    """Preserve typed stable IDs; never silently collapse duplicate edge entries."""
    if (graph_info['node_count'] > MAX_TRANSPORT_NODES or
            graph_info['edge_count'] > MAX_TRANSPORT_EDGES):
        raise TransportContractError('TRANSPORT_GRAPH_LIMIT', 'resource')
    if graph_info['length_unit'] not in ('m', 'cm', 'mm', 'um', 'nm'):
        raise TransportContractError('DECLARED_LENGTH_UNIT_REQUIRED')
    request = {k: arguments[k] for k in ('conductance_S', 'dirichlet_voltage_V')}
    request['length_unit'] = graph_info['length_unit']
    if len(canonical(request)) > MAX_TRANSPORT_REQUEST_BYTES:
        raise TransportContractError('TRANSPORT_REQUEST_LIMIT', 'resource')
    conductance = request['conductance_S']
    boundary = request['dirichlet_voltage_V']
    if (not isinstance(conductance, list) or len(conductance) > MAX_TRANSPORT_EDGES or
            not isinstance(boundary, list) or not 1 <= len(boundary) <= MAX_TRANSPORT_NODES):
        raise TransportContractError('INVALID_TRANSPORT_INPUT')
    seen = set()
    for item in conductance:
        keys(item, {'edge_id', 'value_S'}, {'edge_id', 'value_S'})
        integer(item['edge_id']); number(item['value_S'])
        if item['value_S'] < 0:
            raise TransportContractError('INVALID_TRANSPORT_INPUT')
        if item['edge_id'] in seen:
            raise TransportContractError('DUPLICATE_CONDUCTANCE_ID')
        seen.add(item['edge_id'])
    for item in boundary:
        keys(item, {'node_id', 'value_V'}, {'node_id', 'value_V'})
        integer(item['node_id']); number(item['value_V'])
    return request


def solve_transport(data, request):
    """Call the native owner once; return its complete unmodified JSON contract."""
    from fibernet.core.structure_graph import StructureGraph
    from fibernet.sim.linear_transport import (
        FloatingComponentError, TransportSolveError, solve_linear_transport)
    try:
        transport_request({'node_count': len(data['nodes']), 'edge_count': len(data['edges']),
                           'length_unit': request['length_unit']}, request)
        graph = StructureGraph.from_dict(data)
        result = solve_linear_transport(
            graph, {item['edge_id']: item['value_S'] for item in request['conductance_S']},
            [(item['node_id'], item['value_V']) for item in request['dirichlet_voltage_V']],
            length_unit=request['length_unit']).to_dict()
        if len(canonical(result)) > MAX_TRANSPORT_RESULT_BYTES:
            return {'error_code': 'TRANSPORT_RESULT_LIMIT', 'error_category': 'resource'}
        return {'electrical': result}
    except TransportContractError as exc:
        return {'error_code': exc.code, 'error_category': exc.category}
    except FloatingComponentError:
        return {'error_code': 'FLOATING_COMPONENT', 'error_category': 'domain'}
    except TransportSolveError:
        return {'error_code': 'TRANSPORT_SOLVE_FAILED', 'error_category': 'domain'}
    except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError):
        return {'error_code': 'INVALID_TRANSPORT_INPUT', 'error_category': 'domain'}
