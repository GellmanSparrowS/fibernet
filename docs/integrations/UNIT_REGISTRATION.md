# Unit factory registration: small validation boundary

`fibernet.gen.pattern.register_unit(name, factory)` and `register_unit_3d` require
`factory` to be callable. Noncallable input raises `TypeError` before changing any
registry entry, including when the name already exists. Callable functions and
callable instances retain the previous registration behavior: names are lowercased
and an existing entry may be replaced. Registration does not invoke the factory.

This is only an early input check. It does not validate the callable signature,
returned graph, seed determinism, parameter schema, capabilities or physical model.
A callable closure is not certified by its function name or source hash. Native
`FiberNetDataset.generate(..., save_dir=...)` continues to reject unrecognized
custom factories; valid registration alone does not confer safe checkpoint resume.
Fresh custom generation remains available under its existing contract.

The generate checkpoint recipe hashes pattern.py conservatively. Updating this
registration code therefore changes the source binding and can reject an older
checkpoint even when its chosen built-in graph would be unchanged. Use a new
save directory; no digest exception or automatic migration is introduced.
