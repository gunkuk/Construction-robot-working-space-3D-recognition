"""Windows-safe atomic records; status locks must not discard model progress."""
import ctypes
import errno
import json
import os
from pathlib import Path
import time
import uuid

STATUS_NAMES = {'state.json', 'supervisor.json', 'tiered_cache_state.json', 'storage_budget.json', 'cache_timing.json', 'forecast.json'}


def read_text_shared(path):
    path = Path(path)
    if os.name != 'nt': return path.read_text(encoding='utf-8-sig')
    import msvcrt
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                               ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    api.CreateFileW.restype = ctypes.c_void_p
    api.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = api.CreateFileW(str(path), 0x80000000, 1|2|4, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
    try: fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        api.CloseHandle(handle); raise
    with os.fdopen(fd, 'r', encoding='utf-8-sig') as stream: return stream.read()


def replace_file(source, target, timeout=8.):
    deadline = time.monotonic()+timeout
    delay = .02
    while True:
        try:
            os.replace(source, target); return
        except OSError as error:
            transient = getattr(error, 'winerror', None) in (5, 32, 33) or error.errno in (errno.EACCES, errno.EPERM)
            if not transient or time.monotonic() >= deadline: raise
            time.sleep(min(delay, max(0., deadline-time.monotonic())))
            delay = min(delay*2, .2)


def _generations(path):
    return list(path.parent.glob(path.name+'.status.*.json')) if path.name in STATUS_NAMES else []


def _prune(path, keep=2):
    candidates = []
    for item in _generations(path):
        try: candidates.append((item.stat().st_mtime_ns, item))
        except FileNotFoundError: pass
    for _, item in sorted(candidates, reverse=True)[keep:]:
        try: item.unlink()
        except (FileNotFoundError, PermissionError): pass


def read_json(path, default=None):
    path = Path(path)
    candidates = []
    for item in [path, *_generations(path)]:
        try: candidates.append((item.stat().st_mtime_ns, item))
        except FileNotFoundError: pass
    if not candidates: return default
    last_error = None
    for _, item in sorted(candidates, reverse=True):
        try: return json.loads(read_text_shared(item))
        except (OSError, json.JSONDecodeError) as error: last_error = error
    raise last_error


def atomic_json(path, value, timeout=None):
    path = Path(path)
    status = path.name in STATUS_NAMES
    timeout = (1. if status else 8.) if timeout is None else timeout
    partial = path.with_name(path.name+f'.part.{os.getpid()}.{uuid.uuid4().hex}')
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    try:
        partial.write_text(payload, encoding='utf8')
        try:
            replace_file(partial, path, timeout)
        except PermissionError as error:
            if not status: raise
            # A unique, bounded status generation bypasses a reader locking the fixed filename.
            # Readers select the newest complete JSON. Model/config/checkpoint errors never use this fallback.
            generation = path.with_name(path.name+f'.status.{time.time_ns()}.{os.getpid()}.json')
            generation.write_text(payload, encoding='utf8')
            with (path.parent/'status_io_recovery.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps({'time': time.time(), 'path': str(path), 'fallback': str(generation),
                                         'winerror': getattr(error, 'winerror', None)})+'\n')
            _prune(path)
        else:
            _prune(path, keep=0)
    finally:
        try: partial.unlink(missing_ok=True)
        except PermissionError: pass
