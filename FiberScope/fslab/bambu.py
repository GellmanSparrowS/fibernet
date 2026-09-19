"""Open a local model in Bambu Studio; never send a printer job or G-code."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


def find_bambu():
    candidates = [os.environ.get('FIBERSCOPE_BAMBU_STUDIO'), shutil.which('bambu-studio'), shutil.which('BambuStudio')]
    for base in (os.environ.get('ProgramFiles'), os.environ.get('ProgramFiles(x86)'), os.environ.get('LOCALAPPDATA')):
        if base:
            candidates.extend(str(Path(base)/folder/'bambu-studio.exe') for folder in ('Bambu Studio','BambuStudio','Programs/Bambu Studio'))
    if os.name == 'nt':
        for letter in 'CDEFGHIJKLMNOPQRSTUVWXYZ':
            candidates.extend(str(Path(letter+':/')/folder/'bambu-studio.exe')
                              for folder in ('Bambu/Bambu Studio','Bambu Studio','Program Files/Bambu Studio'))
        import winreg
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(hive, r'SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\bambu-studio.exe') as key:
                    candidates.append(winreg.QueryValue(key, None))
            except OSError:
                pass
    return next((str(Path(p).resolve()) for p in candidates if p and Path(p).is_file()), None)


def open_in_bambu(model_path, executable=None):
    path = Path(model_path).resolve()
    executable = executable or find_bambu()
    if not executable or not Path(executable).is_file():
        raise FileNotFoundError('Bambu Studio executable not found')
    if path.suffix.lower() not in ('.obj', '.stl') or not path.is_file():
        raise ValueError('a saved STL or OBJ model is required')
    environment = dict(os.environ)
    root = getattr(sys, '_MEIPASS', None)
    kernel, previous = None, None
    if root:
        normalized = os.path.normcase(os.path.abspath(root))
        def bundled(value):
            value = os.path.normcase(os.path.abspath(value.strip('"')))
            return value == normalized or value.startswith(normalized + os.sep)
        environment['PATH'] = os.pathsep.join(p for p in environment.get('PATH', '').split(os.pathsep) if p and not bundled(p))
        for key in ('QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
            if any(bundled(p) for p in environment.get(key, '').split(os.pathsep) if p):
                environment.pop(key, None)
        if os.name == 'nt':
            import ctypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.SetDllDirectoryW.argtypes = [ctypes.c_wchar_p]
            buffer = ctypes.create_unicode_buffer(32768)
            kernel.GetDllDirectoryW(len(buffer), buffer)
            previous = buffer.value
            if not kernel.SetDllDirectoryW(None):
                raise ctypes.WinError(ctypes.get_last_error())
    try:
        return subprocess.Popen([str(executable), str(path)], shell=False,
                                env=environment, cwd=str(Path(executable).resolve().parent))
    finally:
        if kernel is not None:
            kernel.SetDllDirectoryW(previous or None)
