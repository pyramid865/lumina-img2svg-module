"""Native Windows drag & drop for Tk (no third-party DLLs).

`windnd`/`tkinterdnd2` both failed inside the frozen build, so the WM_DROPFILES
handling is done here directly:

* ``DragAcceptFiles`` marks the FLTK/Tk toplevel as a drop target (removes the
  "forbidden" cursor),
* the window procedure is subclassed to catch ``WM_DROPFILES``,
* the ``WNDPROC`` callback object is kept on the instance - letting it be
  garbage collected is what silently breaks most ctypes implementations.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
shell32 = ctypes.WinDLL("shell32", use_last_error=True)

WM_DROPFILES = 0x0233
GWL_WNDPROC = -4
GA_ROOT = 2

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT,
                             wintypes.WPARAM, wintypes.LPARAM)

_set_proc = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
_set_proc.restype = LRESULT
_set_proc.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
user32.CallWindowProcW.restype = LRESULT
user32.CallWindowProcW.argtypes = [LRESULT, wintypes.HWND, wintypes.UINT,
                                   wintypes.WPARAM, wintypes.LPARAM]
user32.GetAncestor.restype = wintypes.HWND
user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
shell32.DragQueryFileW.restype = wintypes.UINT
# HWND/HDROP must be pointer-sized; wintypes.HANDLE is 32-bit on some builds,
# which made every real drop die with "int too long to convert".
shell32.DragQueryFileW.argtypes = [ctypes.c_void_p, wintypes.UINT,
                                   wintypes.LPWSTR, wintypes.UINT]
shell32.DragFinish.restype = None
shell32.DragFinish.argtypes = [ctypes.c_void_p]
shell32.DragAcceptFiles.restype = None
shell32.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]


class DropTarget:
    """Subclass the Tk content window and accept dropped files.

    Tk creates a container window plus a content window (``winfo_id``); the
    drop message goes to the window under the cursor, so the content window is
    the one that matters (this is what the old windnd-based build hooked).
    """

    def __init__(self, tk_widget, callback, log=None):
        self.callback = callback
        self.log = log or (lambda _m: None)
        self.ok = False
        self._hooks = []
        try:
            child = tk_widget.winfo_id()
            parent = user32.GetParent(child)
            for hwnd in (child, parent):
                if not hwnd:
                    continue
                holder = {"old": None}

                def make(h, box):
                    def handler(_h, msg, wparam, lparam):
                        if msg == WM_DROPFILES:
                            try:
                                self._handle_drop(wparam)
                            except Exception as exc:  # noqa: BLE001
                                self.log("drop handler error %r" % exc)
                            return 0
                        return user32.CallWindowProcW(box["old"], _h, msg, wparam, lparam)

                    return handler

                proc = WNDPROC(make(hwnd, holder))
                shell32.DragAcceptFiles(hwnd, True)
                old = _set_proc(hwnd, GWL_WNDPROC, ctypes.cast(proc, ctypes.c_void_p))
                holder["old"] = old
                if old:
                    self._hooks.append((hwnd, old, proc, holder))
                    self.log("dnd hooked hwnd=%s" % hwnd)
            self.ok = bool(self._hooks)
            self.hwnd = self._hooks[0][0] if self._hooks else 0
        except Exception as exc:  # noqa: BLE001
            self.log("dnd setup failed %r" % exc)
            self.ok = False

    def _handle_drop(self, hdrop) -> None:
        hdrop = ctypes.c_void_p(int(hdrop))
        count = shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
        files = []
        for i in range(count):
            buf = ctypes.create_unicode_buffer(1024)
            if shell32.DragQueryFileW(hdrop, i, buf, 1024):
                files.append(buf.value)
        shell32.DragFinish(hdrop)
        self.log("drop received: %d file(s)" % len(files))
        if files:
            self.callback(files)

    def close(self) -> None:
        for hwnd, old, _proc, _box in self._hooks:
            try:
                _set_proc(hwnd, GWL_WNDPROC, ctypes.c_void_p(old))
                shell32.DragAcceptFiles(hwnd, False)
            except Exception:  # noqa: BLE001
                pass
        self._hooks = []
