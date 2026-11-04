"""Linux fixed-test supervisor with terminal CPU accounting and child reaping.

RSS and intermediate disk/CPU observations are sampled, not a cgroup sandbox.
Successful runs also require exact waited-child CPU plus supervisor CPU,
terminal disk/time checks and no surviving descendants. Worker OS limits and
workflow wall timeout are separate safeguards.
"""
import ctypes
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import uuid
import psutil

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ('provenance_contract.py', 'tool_descriptions.py', 'monitor_checks.py', 'bounded_smoke.py', 'core_binding.py',
           'adversarial.py', 'transport_edges.py', 'cleanup_failures.py', 'test_electrical.py',
           'test_preparation.py', 'test_shared_lifecycle.py', 'test_thermoelastic.py',
           'test_projection_fixtures.py', 'test_projection.py', 'test_projection_lifecycle.py')
DEFAULT_LIMITS = {'cpu_seconds': 180, 'wall_seconds': 300,
                  'rss_bytes': 1024 ** 3, 'disk_bytes': 60 * 1024 ** 2}
REPORT_RESERVE = 16384
FINISHING_CPU_RESERVE = 1.0


def disk_bytes(root):
    total = 0
    pending = [os.fspath(root)]
    while pending:
        try:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except FileNotFoundError:
                        pass
        except FileNotFoundError:
            pass
    return total


def usage_cpu(who):
    value = resource.getrusage(who)
    return value.ru_utime + value.ru_stime


def subreaper(value=None):
    if sys.platform != 'linux':
        raise RuntimeError('Linux child-subreaper support required')
    library = ctypes.CDLL(None, use_errno=True)
    call = library.prctl
    call.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    call.restype = ctypes.c_int
    if value is None:
        result = ctypes.c_int()
        if call(37, ctypes.addressof(result), 0, 0, 0):
            raise OSError(ctypes.get_errno(), 'cannot read child-subreaper state')
        return result.value
    if call(36, value, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'cannot set child-subreaper state')


def run_bounded(command, directory, disk_root, limits=None, interval=.05):
    limits = dict(DEFAULT_LIMITS if limits is None else limits)
    directory.mkdir(parents=True, exist_ok=False)
    supervisor = psutil.Process()
    if supervisor.children():
        raise RuntimeError('supervisor must start without existing children')
    start = time.monotonic()
    self_start = usage_cpu(resource.RUSAGE_SELF)
    children_start = usage_cpu(resource.RUSAGE_CHILDREN)
    process = None
    previous = None
    enabled = False
    caught = None
    failures, reaped, leftovers = [], [], []
    observed = {}
    peak_rss = peak_disk = sampled_cpu = 0

    def owned():
        found = []
        for child in supervisor.children(recursive=True):
            try:
                born = child.create_time()
                observed[(child.pid, born)] = child
                found.append((child, born))
            except psutil.NoSuchProcess:
                pass
        return found

    def reap():
        while True:
            try:
                pid, status, used = os.wait4(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if not pid:
                break
            code = os.waitstatus_to_exitcode(status)
            reaped.append({'pid': pid, 'returncode': code,
                           'cpu_seconds': used.ru_utime + used.ru_stime})
            if process is not None and pid == process.pid:
                process.returncode = code

    def cleanup():
        deadline = time.monotonic() + 5
        while True:
            reap()
            current = owned()
            if not current:
                return []
            for child, born in current:
                try:
                    if child.create_time() == born:
                        child.kill()
                except psutil.NoSuchProcess:
                    pass
            if time.monotonic() >= deadline:
                reap()
                return [{'pid': child.pid, 'created': born} for child, born in owned()]
            time.sleep(.02)

    try:
        previous = subreaper()
        subreaper(1)
        enabled = True
        if subreaper() != 1:
            raise RuntimeError('child-subreaper state was not established')
        with (directory / 'worker.log').open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            while process.returncode is None:
                reap()
                current = owned()
                rss = supervisor.memory_info().rss
                live_cpu = 0
                for child, born in current:
                    try:
                        if child.create_time() != born:
                            continue
                        rss += child.memory_info().rss
                        used = child.cpu_times()
                        # Reaped descendants are represented in their live
                        # parent's children_* fields, never a stale PID cache.
                        live_cpu += (used.user + used.system + used.children_user
                                     + used.children_system)
                    except psutil.NoSuchProcess:
                        pass
                now_cpu = (usage_cpu(resource.RUSAGE_SELF) - self_start
                           + usage_cpu(resource.RUSAGE_CHILDREN) - children_start
                           + live_cpu)
                sampled_cpu = max(sampled_cpu, now_cpu)
                peak_rss = max(peak_rss, rss)
                peak_disk = max(peak_disk, disk_bytes(disk_root))
                if now_cpu > limits['cpu_seconds']:
                    failures.append('sampled_cpu_limit')
                if rss > limits['rss_bytes']:
                    failures.append('sampled_rss_limit')
                if peak_disk + REPORT_RESERVE > limits['disk_bytes']:
                    failures.append('sampled_disk_limit')
                if time.monotonic() - start > limits['wall_seconds']:
                    failures.append('wall_limit')
                if process.returncode is not None or failures:
                    break
                time.sleep(interval)
    except BaseException as error:
        caught = error
        failures.append('supervisor_exception:' + type(error).__name__)
    finally:
        if enabled:
            reap()
            if owned() and process is not None and process.returncode is not None:
                failures.append('descendants_survived_worker_exit')
            try:
                leftovers = cleanup()
            except BaseException as error:
                failures.append('cleanup_exception:' + type(error).__name__)
                leftovers = [{'pid': child.pid, 'created': born} for child, born in owned()]
            if leftovers:
                failures.append('cleanup_not_verified')
            reap()
            try:
                subreaper(previous)
                if subreaper() != previous:
                    raise RuntimeError('child-subreaper state was not restored')
            except BaseException as error:
                failures.append('subreaper_restore:' + type(error).__name__)
        terminal_disk = disk_bytes(disk_root)
        exact_child_cpu = usage_cpu(resource.RUSAGE_CHILDREN) - children_start
        supervisor_cpu = usage_cpu(resource.RUSAGE_SELF) - self_start
        terminal_cpu = exact_child_cpu + supervisor_cpu
        elapsed = time.monotonic() - start
        if process is None or process.returncode != 0:
            failures.append('worker_did_not_succeed')
        if terminal_cpu + FINISHING_CPU_RESERVE > limits['cpu_seconds']:
            failures.append('terminal_cpu_limit')
        if elapsed > limits['wall_seconds']:
            failures.append('terminal_wall_limit')
        if terminal_disk + REPORT_RESERVE > limits['disk_bytes']:
            failures.append('terminal_disk_limit')
        # Each user/system timeval is truncated to microseconds independently.
        # Allow their finite rounding accumulation, not missing process usage.
        reconciliation_allowance = 2e-6 * (len(reaped) + 1)
        reconciliation_difference = abs(sum(item['cpu_seconds'] for item in reaped)
                                        - exact_child_cpu)
        if process is not None and reconciliation_difference > reconciliation_allowance:
            failures.append('child_cpu_accounting_mismatch')
        report = {'returncode': None if process is None else process.returncode,
                  'failures': sorted(set(failures)), 'elapsed_s': elapsed,
                  'waited_child_cpu_seconds': exact_child_cpu,
                  'supervisor_cpu_seconds': supervisor_cpu,
                  'terminal_accounted_cpu_seconds': terminal_cpu,
                  'finishing_cpu_reserve_seconds': FINISHING_CPU_RESERVE,
                  'wait4_reconciliation_difference_seconds': reconciliation_difference,
                  'wait4_rounding_allowance_seconds': reconciliation_allowance,
                  'wait4_reconciliation_applicable': process is not None,
                  'launch_failure_note': None if process is not None else 'Popen may internally reap failed exec; aggregate RUSAGE still includes it and launch fails closed.',
                  'sampled_peak_cpu_seconds': sampled_cpu,
                  'sampled_peak_rss_bytes': peak_rss, 'sampled_peak_disk_bytes': peak_disk,
                  'terminal_disk_before_report_bytes': terminal_disk,
                  'report_disk_reserve_bytes': REPORT_RESERVE,
                  'remaining_descendants': leftovers, 'wait4_records': reaped,
                  'observed_process_identities': [{'pid': pid, 'created': born}
                                                  for pid, born in observed],
                  'limits': limits,
                  'scope': 'Terminal kernel waited-child CPU plus supervisor CPU snapshot, with one CPU second reserved for finishing. RSS and intermediate observations sampled. No OS aggregate-memory sandbox.'}
        raw = json.dumps(report, indent=2) + '\n'
        if len(raw.encode()) > REPORT_RESERVE:
            raise RuntimeError('supervisor report exceeds reserved output budget')
        (directory / 'resource-report.json').write_text(raw)
    if caught is not None:
        raise caught
    return report


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        directory = Path(sys.argv[2])
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 ** 2, 16 * 1024 ** 2))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        stages = []
        for name in SCRIPTS:
            start = time.monotonic()
            child_cpu_start = usage_cpu(resource.RUSAGE_CHILDREN)
            with (directory / (name + '.log')).open('wb') as log:
                command = [sys.executable, '-B', str(ROOT / 'tests' / name)]
                if name in ('test_electrical.py', 'test_projection_fixtures.py', 'test_projection.py', 'test_projection_lifecycle.py'):
                    command = [sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider',
                               '-o', 'addopts=', '--basetemp', str(directory / (name + '-pytest')),
                               '--junitxml', str(directory / (name + '.junit.xml')),
                               '-q', str(ROOT / 'tests' / name)]
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            stages.append({'script': name, 'returncode': result.returncode,
                           'elapsed_s': time.monotonic() - start,
                           'waited_child_cpu_seconds':
                               usage_cpu(resource.RUSAGE_CHILDREN) - child_cpu_start})
            (directory / 'stages.json').write_text(json.dumps(stages, indent=2))
            if result.returncode:
                raise SystemExit(result.returncode)
        return
    assert len(sys.argv) == 1, 'no arbitrary test commands are accepted'
    (ROOT / 'evidence').mkdir(exist_ok=True)
    directory = ROOT / 'evidence' / ('suite-' + uuid.uuid4().hex[:8])
    report = run_bounded([sys.executable, str(Path(__file__).resolve()),
                          '--worker', str(directory)], directory, ROOT)
    print(json.dumps(report, indent=2))
    stages_path = directory / 'stages.json'
    stages = json.loads(stages_path.read_text()) if stages_path.exists() else []
    incomplete = (SCRIPTS[len(stages)] if len(stages) < len(SCRIPTS)
                  and all(stage['returncode'] == 0 for stage in stages) else None)
    print(json.dumps({'completed_stages': stages, 'incomplete_stage': incomplete}, indent=2))
    assert not report['failures'], 'bounded adapter suite failed'


if __name__ == '__main__':
    main()
