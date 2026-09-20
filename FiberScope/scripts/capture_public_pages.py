"""Capture populated Chinese/English desktop pages without exposing local settings.

python scripts/capture_public_pages.py --output PATH
"""
from pathlib import Path
import os,sys,time,json,argparse,tempfile
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QT_SCALE_FACTOR','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase,QFont
from studio.main import MainWindow
from studio.ai_assistant import _ToolRequest
from studio import workflow,ml_tab

class PageCapture:
    def run(self,output):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        app=QApplication.instance() or QApplication([])
        for font in ('msyh.ttc','msyhbd.ttc','segoeui.ttf','arial.ttf'):
            QFontDatabase.addApplicationFont(str(Path('C:/Windows/Fonts')/font))
        app.setFont(QFont('Microsoft YaHei',10))
        win=MainWindow(lang='zh',mode='light');win.resize(1600,1000);win.show()
        records=[]
        def settle(seconds=.5):
            end=time.monotonic()+seconds
            while time.monotonic()<end:app.processEvents();time.sleep(.01)
        with tempfile.TemporaryDirectory(prefix='fiberscope-capture-') as directory:
            base=Path(directory);(base/'ml').mkdir()
            with patch.object(workflow,'writable_data_dir',lambda _:str(base/'workflow')),patch.object(ml_tab,'_cache_dir',lambda:str(base/'ml')):
                runner=win.ai_panel.workflow
                runner.start(samples=60,iterations=24,demo_unit='square',target='J',curved=True,open_slicer=False,create_demo=True,surface_model='demo_pyramid.obj',print_size=150.)
                deadline=time.monotonic()+600
                while runner.running:
                    settle(.03)
                    if getattr(runner,'_preview_until',None) is not None:runner._preview_until=0
                    if time.monotonic()>deadline:runner.cancel();raise TimeoutError('capture computation')
                assert runner.record['state']=='complete',runner.record
                win.tab_sim._play_timer.stop()
                names=['structure','simulation','features','learning','inverse','surface','manufacturing']
                for lang in ['zh','en']:
                    win.lang_combo.setCurrentIndex(0 if lang=='zh' else 1);settle()
                    win.ai_panel.hide();win.tab_design.replay_toggle.setChecked(False)
                    for index,name in enumerate(names):
                        win.tabs.setCurrentIndex(index)
                        if name=='features':win.tab_features.refresh()
                        if name=='surface':
                            paths=[Path(p).name for p in win.tab_surface._obj_files]
                            win.tab_surface.obj_combo.setCurrentIndex(paths.index('4_Heart_quad_500.obj'))
                            win.tab_surface._recompute()
                        settle(.6)
                        target=output/f'{name}-{lang}.png';assert win.grab().save(str(target))
                        records.append({'file':target.name,'language':lang,'page':name,'width':win.width(),'height':win.height(),'real_computation':True})
                    panel=win.ai_panel;panel._log=[('user','请切换到制造并检查当前结构。' if lang=='zh' else 'Switch to Manufacturing and inspect the current structure.')]
                    panel.refresh_theme();panel.show();win.split.setSizes([1060,540])
                    for name,args in [('set_tab',{'tab':'manufacturing'}),('get_app_state',{})]:
                        panel._on_note(name);req=_ToolRequest(name,args);panel._run_tool(req)
                        assert 'error' not in json.loads(req.result)
                    panel.status.setText('就绪' if lang=='zh' else 'Ready')
                    settle(.8)
                    target=output/f'assistant-{lang}.png';assert win.grab().save(str(target))
                    records.append({'file':target.name,'language':lang,'page':'assistant','width':win.width(),'height':win.height(),'real_computation':True})
                win.close();settle(.5)
        (output/'manifest.json').write_text(json.dumps({'samples':60,'inverse_evaluations':24,'pages':records},ensure_ascii=False,indent=2),encoding='utf-8')
        print('[capture] 16 populated bilingual desktop screenshots captured')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    PageCapture().run(parser.parse_args().output)
