"""Short-lived CPU, terminal limits and new-session descendant regressions."""
import json
import os
from pathlib import Path
import sys
import uuid
from unittest.mock import patch

from bounded_suite import ROOT, disk_bytes, run_bounded, subreaper

AREA = ROOT / 'evidence' / ('monitor-' + uuid.uuid4().hex[:8])
AREA.mkdir()
LIMITS = {'cpu_seconds': 5, 'wall_seconds': 5,
          'rss_bytes': 256 * 1024 ** 2, 'disk_bytes': 4 * 1024 ** 2}
before = subreaper()
records = []
inventory = AREA / 'disk_inventory'
inventory.mkdir()
(inventory / 'nested').mkdir()
(inventory / 'regular').write_bytes(b'abc')
(inventory / 'nested/payload').write_bytes(b'defgh')
(inventory / 'file_link').symlink_to(inventory / 'regular')
(inventory / 'directory_link').symlink_to(inventory / 'nested', target_is_directory=True)
(inventory / 'missing_link').symlink_to(inventory / 'missing')
os.mkfifo(inventory / 'fifo')
assert disk_bytes(inventory) == 8
assert disk_bytes(inventory / 'missing') == 0
# Scanning must tolerate deletion between directory discovery and traversal.
scandir = os.scandir
def remove_before_scan(path):
    if Path(path) == inventory / 'nested':
        (inventory / 'nested/payload').unlink()
        (inventory / 'nested').rmdir()
    return scandir(path)
with patch('bounded_suite.os.scandir', side_effect=remove_before_scan):
    assert disk_bytes(inventory) == 3
records.append({'case': 'disk_inventory', 'regular_file_bytes': 3,
                'symlinks_and_fifo_excluded': True, 'vanished_directory_tolerated': True})
burn = ('import time; start=time.process_time(); '
        'exec("while time.process_time()-start < .08: pass")')
short = ('import subprocess,sys; '
         'subprocess.run([sys.executable,"-c",' + repr(burn) + '],check=True)')


def run(name, source, limits=None):
    case = AREA / name
    case.mkdir()
    result = run_bounded([sys.executable, '-c', source], case / 'report', case,
                         LIMITS if limits is None else limits, interval=.5)
    assert not result['remaining_descendants']
    assert subreaper() == before
    records.append({'case': name, 'report': result})
    return result, case


result, _ = run('short_lived_cpu', short)
assert not result['failures']
assert result['waited_child_cpu_seconds'] >= .08
assert result['supervisor_cpu_seconds'] > 0
assert abs(sum(item['cpu_seconds'] for item in result['wait4_records'])
           - result['waited_child_cpu_seconds']) <= result['wait4_rounding_allowance_seconds']

result, _ = run('terminal_cpu_limit', short, {**LIMITS, 'cpu_seconds': .05})
assert set(result['failures']) == {'sampled_cpu_limit', 'terminal_cpu_limit'}
assert result['waited_child_cpu_seconds'] >= .05

lingering = ('import subprocess,sys; '
             'subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],'
             'start_new_session=True,stdin=subprocess.DEVNULL,'
             'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)')
result, _ = run('new_session_survives_worker', lingering)
assert result['returncode'] == 0
assert result['failures'] == ['descendants_survived_worker_exit']
assert len(result['wait4_records']) == 2
assert any(item['returncode'] == -9 for item in result['wait4_records'])

case = AREA / 'terminal_disk_limit'
case.mkdir()
payload = case / 'large'
result = run_bounded([sys.executable, '-c',
                      'from pathlib import Path; Path(' + repr(str(payload))
                      + ').write_bytes(b"x"*(2*1024**2))'],
                     case / 'report', case, {**LIMITS, 'disk_bytes': 1024 ** 2}, interval=.5)
assert set(result['failures']) == {'sampled_disk_limit', 'terminal_disk_limit'}
assert result['terminal_disk_before_report_bytes'] >= 2 * 1024 ** 2
assert not result['remaining_descendants']
records.append({'case': 'terminal_disk_limit', 'report': result})
payload.unlink()  # Test-owned oversized artifact; report is retained.

case = AREA / 'setup_refusal'
case.mkdir()
with patch('bounded_suite.subreaper', side_effect=OSError('injected unavailable subreaper')):
    try:
        run_bounded([sys.executable, '-c', 'raise AssertionError("must not launch")'],
                    case / 'report', case, LIMITS)
    except OSError as error:
        assert str(error) == 'injected unavailable subreaper'
    else:
        raise AssertionError('subreaper setup failure accepted')
result = json.loads((case / 'report/resource-report.json').read_text())
assert set(result['failures']) == {'supervisor_exception:OSError', 'worker_did_not_succeed'}
assert result['returncode'] is None and not result['wait4_records']
assert not result['remaining_descendants'] and subreaper() == before
records.append({'case': 'setup_refusal', 'report': result})

case = AREA / 'launch_exception'
case.mkdir()
try:
    run_bounded([str(case / 'nonexistent-command')], case / 'report', case, LIMITS)
except FileNotFoundError:
    pass
else:
    raise AssertionError('missing executable accepted')
result = json.loads((case / 'report/resource-report.json').read_text())
assert set(result['failures']) == {'supervisor_exception:FileNotFoundError', 'worker_did_not_succeed'}
assert not result['remaining_descendants'] and subreaper() == before
records.append({'case': 'launch_exception', 'report': result})

(AREA / 'summary.json').write_text(json.dumps(records, indent=2))
print(json.dumps({'passed': len(records), 'cases': [item['case'] for item in records],
                  'subreaper_state_restored': True}, indent=2))
