"""Private exact bookkeeping for exported binary64 discrete-stage energies.

The closure certifies scalar accounting only. It does not certify the force law,
continuous-time work, or the numerical accuracy of endpoint energy evaluation.
"""
import copy
import json
import math
import struct
from fractions import Fraction
from decimal import Context, Decimal, InvalidOperation, ROUND_HALF_EVEN
from numbers import Real

SCHEMA = 'native_axial_discrete_ledger/1'
_UNSPECIFIED = object()
CHANNELS = ('edge_kick_delta_K', 'external_kick_delta_K', 'drag_delta_K',
            'drift_delta_U', 'fixed_projection_delta_H', 'schedule_projection_delta_H')
ENERGIES = ('kinetic_energy', 'potential_energy', 'total_energy')


def _bits(value):
    return struct.unpack('>Q', struct.pack('>d', value))[0]


def _float(bits):
    return struct.unpack('>d', struct.pack('>Q', bits))[0]


def rational(value):
    """Decode IEEE binary64 without host FTZ/DAZ arithmetic after Taichi init."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('ledger scalar must be a finite real number')
    try:
        out = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError('ledger scalar is outside binary64 range') from exc
    bits = _bits(out)
    sign, exponent, significand = bits >> 63, (bits >> 52) & 2047, bits & ((1 << 52)-1)
    if exponent == 2047:
        raise ValueError('ledger scalar is outside finite binary64 range')
    if exponent:
        significand += 1 << 52
        power = exponent-1075
    else:
        power = -1074
    if sign: significand = -significand
    return Fraction(significand * 2**power) if power >= 0 else Fraction(significand,2**-power)


def _rounded(value, allow_zero=False):
    if not value:
        return 0.
    value = Fraction(value)
    sign = int(value < 0); n,d = abs(value.numerator),value.denominator
    exponent = n.bit_length()-d.bit_length()
    if (n < d << exponent) if exponent >= 0 else (n << -exponent < d):
        exponent -= 1
    scale = max(exponent,-1022)-52
    if scale >= 0: denominator=d << scale; numerator=n
    else: numerator=n << -scale; denominator=d
    mantissa,remainder=divmod(numerator,denominator)
    if 2*remainder > denominator or (2*remainder == denominator and mantissa & 1):
        mantissa += 1
    if mantissa == 0:
        if allow_zero: return 0.
        raise ValueError('nonzero ledger output is outside binary64 range')
    if mantissa >= 1 << 53:
        mantissa >>= 1;exponent += 1
    if exponent > 1023:
        raise ValueError('ledger output overflow')
    if mantissa < 1 << 52:
        bits=mantissa
    else:
        bits=((max(exponent,-1022)+1023) << 52) | (mantissa-(1 << 52))
    return _float(bits | (sign << 63))


def nearest(value):
    """One nearest/ties-even binary64 rounding, independent of host FTZ/DAZ."""
    return _rounded(value)


def parse_binary64(value):
    """Preserve JSON subnormals even when native code enabled host FTZ/DAZ."""
    try:
        decimal = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError('invalid ledger JSON number') from exc
    if not decimal.is_finite():
        raise ValueError('ledger JSON number must be finite')
    if decimal.is_zero():
        return _float(int(decimal.is_signed()) << 63)
    if not -324 <= decimal.adjusted() <= 308:
        raise ValueError('ledger JSON number is outside finite nonzero binary64 range')
    return nearest(Fraction(decimal))


def _binary64_json_number(value):
    """A finite JSON number that round-trips binary64 without FP arithmetic.

    Decode the payload before any comparison or decimal conversion. Seventeen
    significant decimal digits suffice for nearest/ties-even binary64 recovery;
    Decimal's integer arithmetic is independent of native FTZ/DAZ settings.
    """
    bits = _bits(value)
    sign, exponent, significand = bits >> 63, (bits >> 52) & 2047, bits & ((1 << 52)-1)
    if exponent == 2047:
        raise ValueError('ledger JSON numbers must be finite')
    if not exponent and not significand:
        return '-0.0' if sign else '0.0'
    if exponent:
        significand += 1 << 52
        power = exponent-1075
    else:
        power = -1074
    coefficient = significand << power if power >= 0 else significand * 5**(-power)
    if sign:
        coefficient = -coefficient
    context = Context(prec=17, rounding=ROUND_HALF_EVEN,
                      Emin=-999999, Emax=999999, clamp=0)
    decimal = Decimal(coefficient).scaleb(min(power, 0), context=context)
    return format(decimal, 'E')


def dump_binary64_json(value, stream, default=None):
    """Write the ledger-bearing result as finite JSON, preserving float bits.

    This narrow writer retains the save path's two-space indentation, standard
    string escaping, insertion order, basic JSON key types and default converter.
    It does not alter the caller's floating-point or Decimal context. Cycles and
    nonfinite numbers fail before the owned temporary can replace its target.
    """
    active = set()

    def key_text(key):
        if isinstance(key, str):
            return key
        if key is None:
            return 'null'
        if key is True:
            return 'true'
        if key is False:
            return 'false'
        if isinstance(key, int):
            return int.__repr__(key)
        if isinstance(key, float):
            if (_bits(key) >> 52) & 2047 == 2047:
                raise ValueError('ledger JSON keys must be finite')
            # Keys are JSON strings, not numeric tokens. Retain the historical
            # stdlib key spelling; ledger schema keys themselves are strings.
            return float.__repr__(key)
        raise TypeError('JSON object keys must be str, int, float, bool or None')

    def emit(item, depth):
        if item is None:
            yield 'null'
        elif item is True:
            yield 'true'
        elif item is False:
            yield 'false'
        elif isinstance(item, str):
            yield json.dumps(item)
        elif isinstance(item, int):
            yield int.__repr__(item)
        elif isinstance(item, float):
            yield _binary64_json_number(item)
        else:
            identity = id(item)
            if identity in active:
                raise ValueError('Circular reference detected')
            active.add(identity)
            try:
                if isinstance(item, (list, tuple, dict)):
                    mapping = isinstance(item, dict)
                    opening, closing = ('{', '}') if mapping else ('[', ']')
                    yield opening
                    if item:
                        for index, entry in enumerate(item.items() if mapping else item):
                            yield (',\n' if index else '\n') + '  '*(depth+1)
                            if mapping:
                                key, entry = entry
                                yield json.dumps(key_text(key)) + ': '
                            yield from emit(entry, depth+1)
                        yield '\n' + '  '*depth
                    yield closing
                elif default is not None:
                    yield from emit(default(item), depth)
                else:
                    raise TypeError(f'Cannot serialize {type(item).__name__}')
            finally:
                active.remove(identity)

    for chunk in emit(value, 0):
        stream.write(chunk)


def upward(value):
    """Least binary64 >= nonnegative rational, including half-subnormal units."""
    if value < 0:
        raise ValueError('absolute ledger bound cannot be negative')
    out = _rounded(value,allow_zero=True)
    if rational(out) < value:
        bits = _bits(out)+1
        if bits >= 0x7ff0000000000000:
            raise ValueError('ledger bound overflow')
        out = _float(bits)
    return out


def quantum(value):
    """Exact conservative half ULP; never underflow the 2**-1075 floor."""
    rational(value)
    exponent=(_bits(float(value)) >> 52) & 2047
    power=exponent-1076 if exponent else -1075
    return Fraction(2**power) if power >= 0 else Fraction(1,2**-power)


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(name + ' has missing or unknown fields')


def closure(channels, initial_h, final_h):
    _keys(channels, CHANNELS, 'ledger channels')
    residual = abs(sum((rational(channels[k]) for k in CHANNELS), Fraction()) -
                   (rational(final_h) - rational(initial_h)))
    bound = sum((quantum(channels[k]) for k in CHANNELS), Fraction()) + quantum(initial_h) + quantum(final_h)
    if residual > bound:
        raise ValueError('exported ledger closure exceeds its exact derived bound')
    return dict(residual_abs=upward(residual), bound=upward(bound),
                ratio=_rounded(residual / bound, allow_zero=True) if bound else None,
                ratio_reason=None if bound else 'zero_bound')


def validate_closure(value, channels, initial_h, final_h):
    expected = closure(channels, initial_h, final_h)
    _keys(value, expected, 'closure')
    for key in ('residual_abs', 'bound'):
        rational(value[key])
    if value['ratio'] is not None:
        rational(value['ratio'])
    if any(rational(value[k]) != rational(expected[k]) for k in ('residual_abs','bound','ratio') if expected[k] is not None) or value['ratio_reason'] != expected['ratio_reason']:
        raise ValueError('reported closure does not match exact exported-number audit')


def _endpoint(value):
    _keys(value, ENERGIES, 'endpoint energies')
    for key in ENERGIES:
        if rational(value[key]) < 0:
            raise ValueError('endpoint energy must be nonnegative')
    if rational(value['total_energy']) != rational(nearest(rational(value['kinetic_energy']) + rational(value['potential_energy']))):
        raise ValueError('endpoint H must be the binary64 K+U result')
    return {key: float(value[key]) for key in ENERGIES}


def endpoint_closure(initial, final):
    initial, final = _endpoint(initial), _endpoint(final)
    residual = abs(sum((rational(final[k])-rational(initial[k]) for k in ENERGIES[:2]), Fraction()) -
                   (rational(final['total_energy'])-rational(initial['total_energy'])))
    bound = quantum(initial['total_energy'])+quantum(final['total_energy'])
    if residual > bound:
        raise ValueError('endpoint-H addition rounding exceeds its derived bound')
    return (dict(residual_abs=upward(residual), bound=upward(bound),
                 ratio=_rounded(residual/bound,allow_zero=True), ratio_reason=None), residual/bound)


def _unavailable(reason):
    return dict(status='unavailable', value=None, reason=reason)


def _interpretation(dashpot):
    return dict(closure='exported_number_bookkeeping_only',
        edge_drift=('elastic_integration_defect_indicator' if dashpot == 0 else 'combined_edge_kick_and_drift'),
        projection='algorithmic_energy_change_not_physical_work',
        geometric='stored_force_dot_preprojection_drift_not_kick_energy',
        cutoff='evaluated_A_E_F_G_only_not_within_drift')


def _physical(dashpot):
    out = {key: _unavailable('not_collected') for key in ('external_work','constraint_work','dashpot_work','elastic_work')}
    if dashpot > 0:
        for key in ('dashpot_work','elastic_work'):
            out[key] = _unavailable('combined_kick_not_decomposed')
    return out


class ExactLedger:
    """O(N) constants and scalar rational running totals; no stage history."""
    def __init__(self, initial, external_force):
        self.initial = _endpoint(initial)
        self.previous = self.initial
        self.totals = dict.fromkeys(CHANNELS, Fraction())
        self.work = Fraction()
        self.subtraction_error = Fraction()
        # A zero force component contributes exactly zero, so no subtraction is needed.
        self.forces = [(i, j, rational(f)) for i, row in enumerate(external_force)
                       for j, f in enumerate(row) if rational(f)]
        self.worst = None
        self.worst_ratio = None
        self.near_cutoff_any = False
        self.substeps = 0

    def advance(self, step, time, kinetic, potential, x_a, x_e, near_cutoff):
        ka,kb,kc,kd,kf,kg = map(rational, kinetic)
        ua,ue,uf,ug = map(rational, potential)
        if ka != rational(self.previous['kinetic_energy']) or ua != rational(self.previous['potential_energy']):
            raise ValueError('shared ledger endpoint was not reused')
        increments = dict(zip(CHANNELS,(kb-ka,kc-kb,kd-kc,ue-ua,kf-kd+uf-ue,kg-kf+ug-uf)))
        for key in CHANNELS:
            self.totals[key] += increments[key]
        final = _endpoint(dict(kinetic_energy=nearest(kg), potential_energy=nearest(ug), total_energy=nearest(kg)+nearest(ug)))
        # No per-step channel output rounding. Only shared endpoint-H addition
        # rounding is audited here; saved cumulative channels have their own audit.
        check, ratio = endpoint_closure(self.previous, final)
        witness = dict(step=int(step), initial=self.previous, final=final,
                       criterion='largest_exact_endpoint_addition_rounding_ratio_earliest_tie', closure=check)
        if self.worst is None or ratio > self.worst_ratio:
            self.worst, self.worst_ratio = witness, ratio
        for i,j,force in self.forces:
            a,e = float(x_a[i,j]),float(x_e[i,j])
            exact_delta = rational(e)-rational(a)
            # Explicit RN binary64 subtraction, immune to host FTZ/DAZ.
            d = nearest(exact_delta)
            rd = rational(d)
            if exact_delta and not rd:
                raise ValueError('nonzero geometric subtraction rounded to zero')
            self.work += force*rd
            self.subtraction_error += abs(force)*abs(rd-exact_delta)
        self.previous = final
        self.near_cutoff_any |= bool(near_cutoff)
        self.substeps += 1

    def sample(self, step, time, endpoint, dashpot=0.):
        endpoint = _endpoint(endpoint)
        if endpoint != self.previous:
            raise ValueError('saved ledger endpoint differs from shared endpoint')
        channels = {key:nearest(value) for key,value in self.totals.items()}
        work = nearest(self.work)
        difference = nearest(self.work-self.totals['external_kick_delta_K'])
        subtraction = upward(self.subtraction_error)
        geometric = dict(value=work, subtraction_bound=subtraction,
            bound=upward(rational(subtraction)+quantum(work)),
            minus_external_kick=difference,
            difference_bound=upward(rational(subtraction)+quantum(difference)+quantum(channels['external_kick_delta_K'])))
        combined = nearest(self.totals['edge_kick_delta_K']+self.totals['drift_delta_U'])
        return dict(step=int(step),time=float(time),**endpoint, channels=channels,
            geometric_external_work=geometric, edge_drift_delta=combined,
            elastic_integration_defect=(dict(status='available',value=combined,reason=None) if dashpot == 0
                else _unavailable('combined_kick_not_decomposed')),
            closure=closure(channels,self.initial['total_energy'],endpoint['total_energy']))

    def result(self, rows, state, dashpot, drag, mass_mode, inputs=None):
        out = dict(schema=SCHEMA, representation='sampled',availability='available',
            units=copy.deepcopy(state['units']), dt=state['dt'], node_ids=state['node_ids'], masses=state['masses'],
            mass_mode=mass_mode,dashpot=float(dashpot),drag=float(drag),
            initial=self.initial,final=rows[-1],substeps=self.substeps,
            near_cutoff_any=self.near_cutoff_any,interpretation=_interpretation(dashpot),
            physical_work=_physical(dashpot),worst_step=self.worst,
            steps=list(state['steps']),times=list(state['times']),samples=rows)
        return validate_ledger(out,state,inputs=inputs)


def _validate_row(row, initial, dashpot):
    _keys(row, {'step','time',*ENERGIES,'channels','geometric_external_work','edge_drift_delta',
                'elastic_integration_defect','closure'}, 'ledger sample')
    if type(row['step']) is not int or row['step'] < 0:
        raise ValueError('ledger step must be a nonnegative integer')
    if rational(row['time']) < 0:
        raise ValueError('ledger time must be nonnegative')
    _endpoint({key:row[key] for key in ENERGIES})
    channels=row['channels']
    if not isinstance(channels,dict): raise ValueError('invalid ledger channels')
    for output in channels.values():
        if not rational(output) and _bits(float(output)) >> 63:
            raise ValueError('exact zero ledger channels must use canonical positive zero')
    validate_closure(row['closure'],channels,initial['total_energy'],row['total_energy'])
    work=row['geometric_external_work']
    _keys(work,{'value','subtraction_bound','bound','minus_external_kick','difference_bound'},'geometric work')
    for value in work.values(): rational(value)
    sub=rational(work['subtraction_bound'])
    if sub<0 or rational(work['bound'])!=rational(upward(sub+quantum(work['value']))):
        raise ValueError('invalid geometric subtraction/output bound')
    ext=channels['external_kick_delta_K']; difference=work['minus_external_kick']
    if rational(work['difference_bound'])!=rational(upward(sub+quantum(difference)+quantum(ext))):
        raise ValueError('invalid geometric difference bound')
    delta=rational(difference)-(rational(work['value'])-rational(ext))
    if abs(delta)>quantum(difference)+quantum(work['value'])+quantum(ext):
        raise ValueError('geometric difference disagrees with exported channels')
    combined=row['edge_drift_delta']; rational(combined)
    if abs(rational(combined)-rational(channels['edge_kick_delta_K'])-rational(channels['drift_delta_U'])) > quantum(combined)+quantum(channels['edge_kick_delta_K'])+quantum(channels['drift_delta_U']):
        raise ValueError('combined edge/drift value disagrees with channels')
    expected=(dict(status='available',value=combined,reason=None) if dashpot==0 else _unavailable('combined_kick_not_decomposed'))
    if dashpot == 0 and isinstance(row['elastic_integration_defect'], dict):
        if rational(row['elastic_integration_defect'].get('value')) != rational(combined):
            raise ValueError('invalid elastic-defect value')
    if row['elastic_integration_defect']!=expected:
        raise ValueError('invalid elastic-defect availability')
    if row['step'] == 0:
        if any(rational(v) for v in channels.values()) or rational(combined) or any(
                rational(work[k]) for k in ('value','subtraction_bound','minus_external_kick')):
            raise ValueError('initial/zero-step ledger diagnostics must be exactly zero')


def validate_ledger(value, state, node_ids=None, inputs=None, n_nodes=_UNSPECIFIED):
    """Validate exact fieldsets and exported arithmetic, without inventing stages."""
    if not isinstance(state,dict):
        raise ValueError('energy ledger requires an accompanying dynamics_state')
    if not isinstance(value,dict) or value.get('schema')!=SCHEMA:
        raise ValueError('unsupported algorithmic_energy_ledger schema')
    rep=value.get('representation')
    if rep not in ('sampled','summary'):
        raise ValueError('unsupported ledger representation')
    common={'schema','representation','availability','units','dt','node_ids','masses','mass_mode',
            'dashpot','drag','initial','final','substeps','near_cutoff_any','interpretation','physical_work','worst_step'}
    _keys(value, common|({'steps','times','samples'} if rep=='sampled' else set()), 'ledger')
    if value['availability']!=('available' if rep=='sampled' else 'summary_only'):
        raise ValueError('invalid ledger availability')
    if value['units']!=state['units'] or value['dt']!=state['dt'] or rep!=state['representation']:
        raise ValueError('ledger units, dt or representation disagree with state')
    if rational(value['dt'])<=0:
        raise ValueError('ledger dt must be positive')
    ids=value['node_ids']; masses=value['masses']
    if not isinstance(ids,list) or any(type(i) is not int for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('ledger node IDs must be unique strict integers')
    if not isinstance(masses,list) or len(masses)!=len(ids) or any(rational(m)<=0 for m in masses):
        raise ValueError('ledger masses must be finite positive and aligned')
    if n_nodes is not _UNSPECIFIED and (type(n_nodes) is not int or len(ids) != n_nodes):
        raise ValueError('ledger row count disagrees with result n_nodes')
    if node_ids is not None and ids!=list(node_ids):
        raise ValueError('ledger node IDs disagree with result')
    if value['mass_mode'] not in ('explicit','unit'):
        raise ValueError('unsupported mass provenance')
    if value['mass_mode']=='unit' and any(m!=1. for m in masses):
        raise ValueError('unit masses must be exactly one')
    if value['mass_mode']=='explicit' and inputs is None:
        raise ValueError('explicit ledger requires mass input provenance')
    if inputs is not None and (inputs['node_ids']!=ids or inputs['masses']!=masses or inputs['units']!=value['units']):
        raise ValueError('ledger disagrees with native mass inputs')
    if rep=='sampled' and (ids!=state['node_ids'] or masses!=state['masses']):
        raise ValueError('ledger row/mass provenance disagrees with sampled state')
    for key in ('dashpot','drag'):
        if rational(value[key])<0: raise ValueError('invalid ledger damping coefficient')
    if value['interpretation']!=_interpretation(value['dashpot']) or value['physical_work']!=_physical(value['dashpot']):
        raise ValueError('invalid ledger interpretation or physical work availability')
    initial=_endpoint(value['initial'])
    final=value['final'];_validate_row(final,initial,value['dashpot'])
    if type(value['substeps']) is not int or value['substeps']!=final['step'] or type(value['near_cutoff_any']) is not bool:
        raise ValueError('invalid ledger substep or cutoff status')
    for key in ('step','time',*ENERGIES):
        if rational(final[key])!=rational(state['final'][key]):
            raise ValueError('final ledger endpoint disagrees with state')
    if final['time']!=final['step']*value['dt']:
        raise ValueError('ledger final time differs from step*dt')
    worst=value['worst_step']
    if value['substeps']==0:
        if worst is not None or initial!={k:final[k] for k in ENERGIES} or any(final['channels'].values()):
            raise ValueError('invalid zero-step ledger')
        if value['near_cutoff_any']!=state['final']['near_cutoff']:
            raise ValueError('invalid zero-step cutoff check')
    else:
        _keys(worst,{'step','initial','final','criterion','closure'},'worst endpoint witness')
        if type(worst['step']) is not int or not 1<=worst['step']<=value['substeps']:
            raise ValueError('invalid worst-step axis')
        if worst['criterion']!='largest_exact_endpoint_addition_rounding_ratio_earliest_tie':
            raise ValueError('invalid endpoint witness selection criterion')
        expected,_ = endpoint_closure(worst['initial'],worst['final'])
        _keys(worst['closure'],expected,'endpoint witness closure')
        if any(rational(worst['closure'][k]) != rational(expected[k]) for k in ('residual_abs','bound','ratio')) or worst['closure']['ratio_reason'] is not None:
            raise ValueError('invalid endpoint witness closure')
    if state['final']['near_cutoff'] and not value['near_cutoff_any']:
        raise ValueError('ledger cutoff flag contradicts state')
    if rep=='sampled':
        if value['steps']!=state['steps'] or value['times']!=state['times']:
            raise ValueError('ledger axes differ from state')
        rows=value['samples']
        if not isinstance(rows,list) or len(rows)!=len(state['steps']) or rows[-1]!=final:
            raise ValueError('ledger samples/final mismatch')
        for i,row in enumerate(rows):
            _validate_row(row,initial,value['dashpot'])
            if row['step']!=state['steps'][i] or row['time']!=state['times'][i]:
                raise ValueError('ledger sample axis differs from state')
            for key in ENERGIES:
                if rational(row[key])!=rational(state[key][i]): raise ValueError('ledger endpoint differs from sampled state')
        if initial!={k:rows[0][k] for k in ENERGIES} or any(rows[0]['channels'].values()) or rows[0]['geometric_external_work']['value']!=0:
            raise ValueError('initial ledger channels must be zero')
        if any(state['near_cutoff']) and not value['near_cutoff_any']:
            raise ValueError('ledger cutoff flag contradicts sampled state')
    return copy.deepcopy(value)


def ledger_summary(value):
    result=copy.deepcopy(value)
    for key in ('steps','times','samples'): result.pop(key,None)
    result['representation']='summary';result['availability']='summary_only'
    return result
