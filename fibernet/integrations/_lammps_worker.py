"""Internal CLI subprocess entrypoint, reached only by explicit task execution."""
import hashlib
import json
import os
import sys


def main():
    if sys.platform != 'linux':
        raise RuntimeError('native LAMMPS execution requires Linux')
    import resource
    config = json.loads(sys.argv[1])
    if config['mode'] != 'cli':
        raise ValueError('only CLI execution is supported')
    cpu = config['cpu_seconds']
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_AS, (config['memory_bytes'], config['memory_bytes']))
    resource.setrlimit(resource.RLIMIT_FSIZE, (config['file_size_bytes'], config['file_size_bytes']))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for path, expected in config['file_sha256']:
        h = hashlib.sha256()
        with open(path,'rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
        if h.hexdigest() != expected: raise RuntimeError('runtime changed before launch: '+path)
    executable = config['executable']
    os.execve(executable, [executable, '-in', 'in.model', '-log', 'log.lammps', '-screen', 'none'], os.environ)


if __name__ == '__main__':
    main()
