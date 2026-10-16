"""Registry validation only: no graph generation, no callback execution."""
import pytest
import fibernet.gen.pattern as pattern

@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('factory',[None,17,'factory',{},[]])
@pytest.mark.parametrize('replace',[False,True])
def test_noncallable_registration_preserves_entire_registry(dimension,factory,replace):
    registry=pattern._UNIT_FACTORIES if dimension==2 else pattern._UNIT_FACTORIES_3D
    register=pattern.register_unit if dimension==2 else pattern.register_unit_3d
    before=dict(registry)
    name=next(iter(registry)) if replace else '__new_invalid_fixture__'
    with pytest.raises(TypeError,match='factory must be callable'):
        register(name,factory)
    assert registry==before
    assert all(registry[k] is v for k,v in before.items())

@pytest.mark.parametrize('dimension',[2,3])
def test_valid_callable_override_and_case_behavior_unchanged(dimension):
    registry=pattern._UNIT_FACTORIES if dimension==2 else pattern._UNIT_FACTORIES_3D
    register=pattern.register_unit if dimension==2 else pattern.register_unit_3d
    before=dict(registry);calls=[]
    def factory(**kwargs):calls.append(kwargs)
    class Factory:
        def __call__(self,**kwargs):calls.append(kwargs)
    try:
        name=next(iter(registry))
        register(name.upper(),factory)
        assert registry[name] is factory
        instance=Factory();register(name,instance)
        assert registry[name] is instance
        register('__NEW_VALID_FIXTURE__',factory)
        assert registry['__new_valid_fixture__'] is factory
        assert not calls
    finally:
        registry.clear();registry.update(before)
    assert registry==before
