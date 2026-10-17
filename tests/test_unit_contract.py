"""Metadata-only checks; factories must never run in this suite."""
import copy
import hashlib
import json
import pytest
from fibernet.gen import pattern
from fibernet.gen.unit_contract import UnitContracts, SCHEMA_VERSION, canonical_json, verify_source_bytes
from fibernet.ml._generate_contract import implementation_recipe


@pytest.fixture(autouse=True)
def restore_registry():
    originals = [dict(pattern._UNIT_FACTORIES), dict(pattern._UNIT_FACTORIES_3D)]
    yield
    for registry, original in zip([pattern._UNIT_FACTORIES, pattern._UNIT_FACTORIES_3D], originals):
        registry.clear(); registry.update(original)


def factory(**kwargs):
    pytest.fail('factory executed by metadata inspector')


def descriptor():
    return dict(descriptor_schema=SCHEMA_VERSION, plugin_id='test/unit', plugin_version='1',
                kind='unit_generator', dimensions=[2, 3], representation='StructureGraph',
                parameter_schema_version='1', parameters={'type': 'object', 'properties': {
                    'radius': {'type': 'number', 'minimum': 0, 'units': 'unscaled', 'default': .1},
                    'seed': {'type': 'integer', 'minimum': 0},
                    'box': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2},
                    'mirror': {'type': 'boolean', 'default': False},
                    'label': {'type': 'string', 'enum': ['a','b'], 'default': 'a'}},
                    'required': ['seed'], 'additionalProperties': False},
                capabilities={'parallel_edges': 'unknown', 'curves': 'unsupported'},
                randomness={'seed_parameter': 'seed', 'algorithm': 'unknown'},
                id_policy={'nodes': 'creates', 'edges': 'creates', 'orientation': 'unknown'})


def setup(dimension=2):
    register = pattern.register_unit if dimension == 2 else pattern.register_unit_3d
    register('__contract__', factory)
    catalog = UnitContracts()
    catalog.declare('__contract__', dimension, descriptor())
    return catalog


@pytest.mark.parametrize('dimension',[2,3])
def test_detached_roundtrip_defaults_and_requested_spelling(dimension):
    register = pattern.register_unit if dimension == 2 else pattern.register_unit_3d
    register('__contract__',factory)
    catalog = UnitContracts(); data = descriptor()
    first = catalog.declare('__CONTRACT__',dimension,data)
    data['randomness']['algorithm']='changed'
    first['descriptor']['id_policy']['nodes']='changed'
    got = catalog.describe('__CONTRACT__',dimension)
    assert got['requested_name']=='__CONTRACT__'
    assert got['descriptor']['randomness']['algorithm']=='unknown'
    assert got['descriptor']['id_policy']['nodes']=='creates'
    assert json.loads(canonical_json(got))==got
    raw={'seed': 7,'box':[2,3]}
    result=catalog.resolve_parameters('__contract__',dimension,raw)
    assert result['effective']==dict(raw,radius=.1,mirror=False,label='a')
    result['effective']['box'][0]=99
    assert raw['box']==[2,3] and result['raw']['box']==[2,3]
    assert not got['resume_eligible']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),{1:'a'},(1,2),object()])
def test_strict_json(value):
    with pytest.raises(ValueError):canonical_json(value)


@pytest.mark.parametrize('change',[
    lambda d:d.update(extra=1), lambda d:d.pop('plugin_id'),
    lambda d:d.update(dimensions=[True]),lambda d:d.update(dimensions=[3]),
    lambda d:d.update(dimensions=[2,2]),lambda d:d.update(kind='sampler'),
    lambda d:d.update(representation='FiberNetwork'),lambda d:d.update(plugin_id=' '),
    lambda d:d.update(descriptor_schema='2'),lambda d:d.update(capabilities={'x':True}),
    lambda d:d.update(randomness=3),lambda d:d.update(id_policy=[]),
    lambda d:d['parameters'].update(unknown=True),
    lambda d:d['parameters'].update(required=['unknown']),
    lambda d:d['parameters'].update(required=['seed','seed']),
    lambda d:d['parameters'].update(additionalProperties=1),
    lambda d:d['parameters']['properties']['radius'].update(default=True),
    lambda d:d['parameters']['properties']['radius'].update(minimum=True),
    lambda d:d['parameters']['properties']['radius'].update(minimum=2,maximum=1),
    lambda d:d['parameters']['properties']['radius'].update(default=float('inf')),
    lambda d:d['parameters']['properties']['box'].update(minItems=True),
    lambda d:d['parameters']['properties']['box'].update(minItems=3,maxItems=2),
    lambda d:d['parameters']['properties']['label'].update(enum=[]),
    lambda d:d['parameters']['properties']['label'].update(default='c'),
])
def test_bad_declaration_does_not_replace_valid_record(change):
    catalog=setup();before=catalog.describe('__contract__',2)
    data=descriptor();change(data)
    with pytest.raises(ValueError):catalog.declare('__contract__',2,data)
    assert catalog.describe('__contract__',2)==before


@pytest.mark.parametrize('raw',[{}, {'seed':True},{'seed':1,'radius':True}, {'seed':1,'extra':3},
    {'seed':1,'box':[1]}, {'seed':1,'mirror':0},{'seed':1,'label':'c'},
    {'seed':-1},{'seed':1,'radius':float('nan')},[]])
def test_bad_parameters(raw):
    catalog=setup()
    with pytest.raises(ValueError):catalog.resolve_parameters('__contract__',2,raw)


def test_undeclared_and_unknown_schema():
    pattern.register_unit('__contract__',factory)
    catalog=UnitContracts(); before=dict(pattern._UNIT_FACTORIES)
    assert catalog.describe('__contract__',2)['status']=='undeclared'
    data=descriptor();data.update(parameters=None,capabilities='unknown')
    assert catalog.declare('__contract__',2,data)['parameters'] is None
    with pytest.raises(ValueError,match='unknown'):catalog.resolve_parameters('__contract__',2,{})
    assert pattern._UNIT_FACTORIES==before


def test_missing_noncallable_and_invalid_dimension():
    catalog=UnitContracts()
    with pytest.raises(KeyError):catalog.describe('__missing_contract__',2)
    pattern._UNIT_FACTORIES['__broken__']=17
    with pytest.raises(TypeError):catalog.describe('__broken__',2)
    for d in (True,2.,4):
        with pytest.raises(ValueError):catalog.describe('__broken__',d)


@pytest.mark.parametrize('dimension',[2,3])
def test_replace_stale_restore_exact_original(dimension):
    catalog=setup(dimension)
    register=pattern.register_unit if dimension==2 else pattern.register_unit_3d
    class Other:
        def __call__(self):pytest.fail('executed')
    register('__contract__',Other())
    with pytest.raises(ValueError,match='stale'):catalog.describe('__contract__',dimension)
    with pytest.raises(ValueError,match='stale'):catalog.resolve_parameters('__contract__',dimension,{'seed':1})
    register('__contract__',factory)
    assert catalog.describe('__contract__',dimension)['status']=='declared'


def test_closures_and_resume_guard():
    def make(x):
        def closure(**kwargs):return x
        return closure
    a,b=make(1),make(2)
    assert a.__code__ is b.__code__
    pattern.register_unit('__contract__',a)
    catalog=UnitContracts();catalog.declare('__contract__',2,descriptor())
    assert not catalog.describe('__contract__',2)['resume_eligible']
    with pytest.raises(ValueError):implementation_recipe(pattern,['__contract__'])
    pattern.register_unit('__contract__',b)
    with pytest.raises(ValueError,match='stale'):catalog.describe('__contract__',2)
    catalog.declare('__contract__',2,descriptor())
    with pytest.raises(ValueError):implementation_recipe(pattern,['__contract__'])


def test_bytes_only_verification():
    data=b'explicit bytes';digest=hashlib.sha256(data).hexdigest()
    result=verify_source_bytes(data,digest)
    assert not result['resume_eligible'] and not result['callable_link_verified']
    with pytest.raises(ValueError,match='mismatch'):verify_source_bytes(b'other',digest)
    with pytest.raises(TypeError):verify_source_bytes('/some/path',digest)
    with pytest.raises(ValueError):verify_source_bytes(data,'bad')


def test_nested_defaults_and_open_parameters():
    catalog=setup();data=descriptor()
    data['parameters']['additionalProperties']=True
    data['parameters']['properties']['nested']={'type':'object','properties':{
        'enabled':{'type':'boolean','default':True}},'additionalProperties':False,'default':{}}
    catalog.declare('__contract__',2,data)
    got=catalog.resolve_parameters('__contract__',2,{'seed':1,'other':None})
    assert got['effective']['nested']=={'enabled':True} and got['effective']['other'] is None


@pytest.mark.parametrize('kind',['dict','list'])
def test_cycles_reject_clearly(kind):
    value={} if kind=='dict' else []
    if kind=='dict':value['self']=value
    else:value.append(value)
    with pytest.raises(ValueError,match='cyclic'):canonical_json(value)
    catalog=setup()
    with pytest.raises(ValueError,match='cyclic'):catalog.declare('__contract__',2,value)
    with pytest.raises(ValueError,match='cyclic'):catalog.resolve_parameters('__contract__',2,value)


def test_deep_json_and_shared_noncycle():
    value=0
    for _ in range(100):value=[value]
    canonical_json(value)
    with pytest.raises(ValueError,match='100'):canonical_json([value])
    shared={'v':[1,2]}
    assert json.loads(canonical_json([shared,shared]))==[{'v':[1,2]},{'v':[1,2]}]
