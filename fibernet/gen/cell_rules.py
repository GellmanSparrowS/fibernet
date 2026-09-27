"""Deterministic cell transforms and bounded graph assembly."""
import numpy as np

RULES = ('translate', 'rotate', 'mirror', 'mirror_rows', 'quarter_turn',
         'checker_mirror', 'custom')


def validate_rule(pattern=None):
    """A bounded periodic table of quarter turns and local reflections."""
    pattern = pattern or [[{'turn': 0, 'flip': False}]]
    if not isinstance(pattern, list) or not 1 <= len(pattern) <= 4:
        raise ValueError('rule requires 1..4 rows')
    width = len(pattern[0])
    if not 1 <= width <= 4 or any(len(row) != width for row in pattern):
        raise ValueError('rule requires a rectangular 1..4 column table')
    result = []
    for row in pattern:
        cleaned = []
        for cell in row:
            turn = cell.get('turn', 0)
            flip = cell.get('flip', False)
            if isinstance(turn, bool) or turn not in (0, 1, 2, 3) or not isinstance(flip, bool):
                raise ValueError('turn must be 0..3; flip must be boolean')
            cleaned.append({'turn': int(turn), 'flip': flip})
        result.append(cleaned)
    return result


def transform_cell(pos, i, j, rule, size, pattern=None):
    out = np.array(pos, dtype=float, copy=True)
    if rule == 'rotate' and (i + j) % 2:
        out[:, :2] = size - out[:, :2]
    elif rule == 'mirror' and i % 2:
        out[:, 0] = size - out[:, 0]
    elif rule == 'mirror_rows' and j % 2:
        out[:, 1] = size - out[:, 1]
    elif rule in ('quarter_turn', 'checker_mirror', 'custom'):
        turn, flip = 0, False
        if rule == 'quarter_turn':
            turn = (i + j) % 4
        elif rule == 'checker_mirror':
            turn, flip = (2 if j % 2 else 0), bool((i + j) % 2)
        else:
            table = validate_rule(pattern)
            cell = table[j % len(table)][i % len(table[0])]
            turn, flip = cell['turn'], cell['flip']
        if flip:
            out[:, 0] = size - out[:, 0]
        for _ in range(turn):
            x = out[:, 0].copy()
            out[:, 0], out[:, 1] = size - out[:, 1], x
    out[:, :2] += (i * size, j * size)
    return out


def expand_graph(base, gx, gy, rule, size, max_nodes, pattern=None):
    from fibernet.core.structure_graph import StructureGraph
    if rule not in RULES:
        raise ValueError('unknown expansion rule: ' + str(rule))
    if base.num_nodes * gx * gy > max_nodes:
        raise MemoryError('expansion exceeds node budget')
    graph = StructureGraph(dimension=2, box_size=[gx * size, gy * size])
    positions = np.asarray(base.node_positions(), float)
    for j in range(gy):
        for i in range(gx):
            pos = transform_cell(positions, i, j, rule, size, pattern)
            ids = [graph.add_node(p) for p in pos]
            for edge in base._edges.values():
                internal = edge.internal_points
                if internal is not None:
                    internal = transform_cell(internal, i, j, rule, size, pattern)
                graph.add_edge(ids[edge.node_i], ids[edge.node_j],
                               radius=edge.radius, material=edge.material,
                               internal_points=internal, segments=edge.segments)
    graph._metadata.update(base._metadata)
    graph._metadata['expansion_rule'] = rule
    return graph


def graph_health(pos, edges):
    """Topology screen; geometric/process constraints require separate checks."""
    n = len(pos)
    parent = list(range(n))
    degree = np.zeros(n, int)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        a, b = int(a), int(b)
        degree[a] += 1
        degree[b] += 1
        parent[root(a)] = root(b)
    components = len({root(i) for i in range(n)})
    odd = int(np.count_nonzero(degree % 2))
    return dict(nodes=n, edges=len(edges), components=components, odd=odd,
                max_degree=int(degree.max()) if n else 0,
                continuous=components == 1 and odd in (0, 2))
