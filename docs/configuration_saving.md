# Saving experiment configurations

`ExperimentConfig.to_json(path)` and `to_yaml(path)` serialize the configuration before opening the destination or creating parent directories. If serialization fails, an existing configuration file is preserved, and a new destination and its missing parent directories are not created.

Both methods serialize a detached copy. Including metadata adds `saved_at` to the saved data without modifying the original configuration; `include_metadata=False` omits metadata from the output. JSON keeps its two-space indentation, and YAML keeps block-style output and insertion order. YAML requires the optional PyYAML dependency.

This protection covers serialization errors. Saving is not an atomic filesystem replacement: disk/write failures or interruption after opening the destination can still leave partial output. Existing path, permissions, encoding and overwrite behavior are otherwise unchanged.
