"""Run the original byte-sealed checkout assertions with isolated Git timing evidence."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time

TEST_SHA256 = '61f88f090a1e99eed7b860abb6509b72b47e63e2d445086f4958c513948cbf43'
TEST_CRLF_SHA256 = '8ad1733a514ae48de19d8b1360b767f7b8436e7c18d83b655c6ce43cdbc2df29'
TEST_BLOB = '732fc6e56177dc35c8ebe214fa9b48cfa9f69f3e'
PROTECTED_SHA256 = '86adbb9dae2df3eb38e3d37a13d86cb24c5adbfceb95362bd2bd968fe75e50fd'
CORE_MANIFEST_SHA256 = '74e299325033f49a5f81ac0b54ef5083bdcd89e29bc3ee80d690b42738cc087e'
ATTRIBUTES_BLOB = '6d9988eb09ca0c700fded7871a81319d6ff2cd6e'
TRACE_FIELDS = {'event', 'time', 'category', 'label', 't_abs', 't_rel', 'code'}
MAX_TRACE_BYTES = 2 * 1024 * 1024
MAX_TRACE_EVENTS = 2000


def check_test_identity(raw, repository_blob):
    """Accept only the committed original test with its exact LF/CRLF expansion."""
    digest = hashlib.sha256(raw).hexdigest()
    if repository_blob != TEST_BLOB or digest not in (TEST_SHA256, TEST_CRLF_SHA256):
        raise ValueError('original byte-sealed test identity changed')
    return digest


def filtered_trace(path):
    """Retain typed timing metadata; parseable never means the process completed."""
    events = []
    if not path.exists():
        return events, 'absent'
    if path.stat().st_size > MAX_TRACE_BYTES:
        return events, 'oversize'
    text = path.read_text(encoding='utf-8', errors='replace')
    if not text.strip():
        return events, 'empty'
    partial = False
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except (ValueError, TypeError):
            partial = True
            continue
        if not isinstance(raw, dict):
            partial = True
            continue
        safe = {}
        for key in ('event', 'category', 'label'):
            value = raw.get(key)
            if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_:.-]{1,100}', value):
                safe[key] = value
        stamp = raw.get('time')
        if isinstance(stamp, str) and re.fullmatch(
                r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z', stamp):
            safe['time'] = stamp
        for key in ('t_abs', 't_rel'):
            value = raw.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    if math.isfinite(value):
                        safe[key] = value
                except OverflowError:
                    pass
        if isinstance(raw.get('code'), int) and not isinstance(raw['code'], bool):
            safe['code'] = raw['code']
        if safe:
            events.append(safe)
        if len(events) >= MAX_TRACE_EVENTS:
            return events, 'event_limit'
    return events, 'partial' if partial else 'parseable'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    test = root / 'tests/test_source_checkout_bytes.py'
    test_blob = subprocess.run(
        ['git', 'rev-parse', 'HEAD:tests/test_source_checkout_bytes.py'], cwd=root,
        check=True, capture_output=True, text=True, timeout=15).stdout.strip()
    check_test_identity(test.read_bytes(), test_blob)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('original_checkout_test', test)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protected = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                 for name in module.PROTECTED}
    canonical_hash = lambda value: hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (canonical_hash(protected) != PROTECTED_SHA256 or
            canonical_hash(module.CORE_MANIFEST) != CORE_MANIFEST_SHA256):
        raise ValueError('diagnostic protected source inventory differs from d12bc35')
    original_git = module._git
    phases = []
    original_staging_size = module._staging_size
    active_phase = None

    def timed_staging_size(repo, names):
        # Delegate exactly once; reuse its admitted total instead of walking twice.
        started = time.monotonic()
        try:
            total = original_staging_size(repo, names)
            if active_phase is not None:
                active_phase.update(files=len(names), bytes=total)
            return total
        finally:
            if active_phase is not None:
                active_phase['admission_elapsed_s'] = time.monotonic() - started

    module._staging_size = timed_staging_size

    def timed_git(repo, env, autocrlf, *git_args):
        nonlocal active_phase
        # The original test removes inherited GIT_* variables. This explicit
        # diagnostic addition is the only change to its isolated Git environment.
        diagnostic_env = dict(env)
        trace = repo.parent / ('trace-{}.jsonl'.format(len(phases)))
        diagnostic_env['GIT_TRACE2_EVENT'] = str(trace.resolve())
        phase = {'autocrlf': autocrlf, 'phase': git_args[0]}
        active_phase = phase
        started = time.monotonic()
        try:
            result = original_git(repo, diagnostic_env, autocrlf, *git_args)
            phase.update(status='passed', code=0)
            return result
        except Exception as error:
            phase.update(status='failed', exception=type(error).__name__)
            if isinstance(error, subprocess.TimeoutExpired):
                phase['timeout_s'] = error.timeout
            elif isinstance(error, subprocess.CalledProcessError):
                phase['code'] = error.returncode
            raise
        finally:
            phase['outer_total_s'] = time.monotonic() - started
            active_phase = None
            try:
                phase['trace'], phase['trace_status'] = filtered_trace(trace)
            except Exception as error:
                phase.update(trace=[], trace_status='unavailable',
                             trace_exception=type(error).__name__)
            phases.append(phase)

    module._git = timed_git
    cases = []
    for label, autocrlf in (('inventory', None), ('checkout_false', 'false'),
                            ('checkout_true', 'true')):
        started = time.monotonic()
        case = {'case': label}
        captured_out, captured_err = io.StringIO(), io.StringIO()
        try:
            # Original failure diagnostics are captured, not echoed or uploaded.
            # No failed repository is reused; subsequent cases get fresh roots.
            with contextlib.redirect_stdout(captured_out), contextlib.redirect_stderr(captured_err):
                if autocrlf is None:
                    module.test_checkout_inventory_covers_the_native_core_manifest()
                else:
                    with tempfile.TemporaryDirectory(prefix='fibernet-checkout-diagnostic-') as temporary:
                        module.test_byte_sealed_checkout_preserves_raw_hashes(Path(temporary), autocrlf)
            case['status'] = 'passed'
        except Exception as error:
            case.update(status='failed', exception=type(error).__name__)
        case['elapsed_s'] = time.monotonic() - started
        case['captured_output_chars'] = {'stdout': len(captured_out.getvalue()),
                                         'stderr': len(captured_err.getvalue())}
        cases.append(case)
        print(json.dumps(case), flush=True)

    sources = {}
    for relative in ('tests/test_source_checkout_bytes.py', '.gitattributes',
                     'adapters/mcp/src/fibernet_mcp/core_manifest.json'):
        data = (root / relative).read_bytes()
        sources[relative] = {'sha256': hashlib.sha256(data).hexdigest(),
            'raw_git_blob': hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest(),
            'bytes': len(data)}
    attributes_blob = subprocess.run(
        ['git', 'rev-parse', 'HEAD:.gitattributes'], cwd=root, check=True,
        capture_output=True, text=True, timeout=15).stdout.strip()
    if attributes_blob != ATTRIBUTES_BLOB:
        raise ValueError('diagnostic attributes Git blob differs from d12bc35')
    sources['.gitattributes']['repository_git_blob'] = attributes_blob
    sources['tests/test_source_checkout_bytes.py']['repository_git_blob'] = test_blob
    report = {'source_identities': sources,
        'protected_inventory_sha256': canonical_hash(protected),
        'protected_files': len(protected),
        'core_manifest_canonical_sha256': canonical_hash(module.CORE_MANIFEST),
        'python': sys.version,
        'platform': sys.platform, 'architecture': platform.machine(),
        'image_os': os.environ.get('ImageOS'), 'image_version': os.environ.get('ImageVersion'),
        'git_version': subprocess.run(['git', '--version'], check=True, capture_output=True,
                                      text=True, timeout=15).stdout.strip(),
        'method': 'original inventory and full false/true checkout assertions; original validation, argv, Git config and deadlines; explicit GIT_TRACE2_EVENT diagnostic environment addition; fresh disposable repositories; only filtered trace events retained',
        'comparison_note': 'false then true retain original order and isolated repositories but share OS/AV caches; later success does not establish autocrlf causality; attributes working bytes may legitimately be CRLF while Git blob remains fixed',
        'timing_note': 'outer_total_s includes original staging admission and process setup/teardown; admission_elapsed_s delegates original validation once; original subprocess timeout does not bound spawn/kill/drain; absent trace or index-lock snapshot cannot establish cause',
        'cases': cases, 'git_phases': phases}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if any(case['status'] != 'passed' for case in cases):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
