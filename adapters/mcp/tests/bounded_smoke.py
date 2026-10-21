"""Measure aggregate live process RSS and CPU; kill process tree at budget."""
import json,os,subprocess,sys,time,uuid
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'evidence').mkdir(exist_ok=True)
start=time.monotonic();peak=0;cpu={}
run_id=uuid.uuid4().hex[:8]
log=ROOT/'evidence'/('acceptance-'+run_id+'.log')
with log.open('wb') as f:
 p=subprocess.Popen([sys.executable,str(ROOT/'tests/acceptance.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 while p.poll() is None:
  processes=[psutil.Process(p.pid)]+psutil.Process(p.pid).children(recursive=True)
  rss=0
  for q in processes:
   try:
    rss+=q.memory_info().rss;t=q.cpu_times();cpu[(q.pid,q.create_time())]=t.user+t.system
   except psutil.Error:pass
  peak=max(peak,rss)
  if rss>1024**3 or sum(cpu.values())>120 or time.monotonic()-start>300:
   os.killpg(p.pid,9);raise RuntimeError('smoke resource budget exceeded')
  time.sleep(.01)
report={'returncode':p.returncode,'aggregate_peak_rss_bytes':peak,'observed_cpu_seconds':sum(cpu.values()),'elapsed_s':time.monotonic()-start,'disk_bytes':sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file() and not x.is_symlink())}
(ROOT/'evidence'/('resource-report-'+run_id+'.json')).write_text(json.dumps(report,indent=2));print(report)
assert p.returncode==0 and report['disk_bytes']<60*1024**2
