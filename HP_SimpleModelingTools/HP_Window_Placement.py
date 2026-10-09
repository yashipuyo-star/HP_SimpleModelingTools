"""Move only the newly created Blender window to another Windows monitor.

Blender's Window coordinates are read-only. Win32 is used only on Windows,
with process ownership and before/after HWND identity checks.
"""
import os
import sys
import ctypes
from ctypes import wintypes


def _windows_api():
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = (callback, wintypes.LPARAM)
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    user32.SetWindowPos.argtypes = (wintypes.HWND,wintypes.HWND,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,wintypes.UINT)
    monitor_callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
    user32.EnumDisplayMonitors.argtypes = (wintypes.HDC, ctypes.POINTER(wintypes.RECT), monitor_callback, wintypes.LPARAM)
    class MonitorInfo(ctypes.Structure):
        _fields_ = [('cbSize',wintypes.DWORD),('rcMonitor',wintypes.RECT),('rcWork',wintypes.RECT),('dwFlags',wintypes.DWORD)]
    user32.GetMonitorInfoW.argtypes = (wintypes.HANDLE,ctypes.POINTER(MonitorInfo))
    return user32, callback, monitor_callback, MonitorInfo


def _rect_tuple(rect):
    return rect.left,rect.top,rect.right,rect.bottom


def choose_other_monitor(work_areas, source_rect):
    """Choose the nearest other display, including monitors left/above primary."""
    if len(work_areas) < 2:
        return None
    cx=(source_rect[0]+source_rect[2])/2
    cy=(source_rect[1]+source_rect[3])/2
    def distance(rect):
        return max(rect[0]-cx,0,cx-rect[2])**2 + max(rect[1]-cy,0,cy-rect[3])**2
    source=min(range(len(work_areas)),key=lambda i:distance(work_areas[i]))
    others=[i for i in range(len(work_areas)) if i!=source]
    return work_areas[min(others,key=lambda i:distance(work_areas[i]))]


def placement_in_work_area(work_area, window_rect):
    left,top,right,bottom=work_area
    padding=min(32,max(0,min(right-left,bottom-top)//20))
    width=min(max(1,window_rect[2]-window_rect[0]),max(1,right-left-padding*2))
    height=min(max(1,window_rect[3]-window_rect[1]),max(1,bottom-top-padding*2))
    return left+padding,top+padding,width,height


def snapshot_windows():
    if sys.platform != 'win32':
        return None
    try:
        api,callback,_,_= _windows_api()
        handles={}
        @callback
        def collect(hwnd, _):
            pid=wintypes.DWORD()
            api.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
            rect=wintypes.RECT()
            if pid.value == os.getpid() and api.IsWindowVisible(hwnd) and api.GetWindowRect(hwnd,ctypes.byref(rect)):
                handles[int(hwnd)] = _rect_tuple(rect)
            return True
        api.EnumWindows(collect,0)
        foreground=api.GetForegroundWindow()
        source=handles.get(int(foreground)) if foreground else None
        return dict(handles=handles,source=source)
    except (AttributeError,OSError,ValueError):
        return None


def move_new_window(snapshot):
    """Return moved/no_other_monitor/pending/unavailable; never guess an HWND."""
    if snapshot is None or snapshot['source'] is None:
        return 'unavailable'
    current=snapshot_windows()
    if current is None:
        return 'unavailable'
    added=set(current['handles'])-set(snapshot['handles'])
    if len(added)!=1:
        return 'pending'
    hwnd=added.pop()
    try:
        api,_,monitor_callback,MonitorInfo=_windows_api()
        work=[]
        @monitor_callback
        def collect(monitor, _hdc, _rect, _data):
            info=MonitorInfo()
            info.cbSize=ctypes.sizeof(info)
            if api.GetMonitorInfoW(monitor,ctypes.byref(info)):
                work.append(_rect_tuple(info.rcWork))
            return True
        api.EnumDisplayMonitors(None,None,collect,0)
        target=choose_other_monitor(work,snapshot['source'])
        if target is None:
            return 'no_other_monitor'
        # Confirm the candidate still belongs to this Blender process.
        pid=wintypes.DWORD()
        api.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
        if pid.value != os.getpid():
            return 'unavailable'
        x,y,width,height=placement_in_work_area(target,current['handles'][hwnd])
        return 'moved' if api.SetWindowPos(hwnd,None,x,y,width,height,0x0004 | 0x0010) else 'unavailable'
    except (AttributeError,OSError,ValueError):
        return 'unavailable'


def schedule_other_monitor(snapshot, still_owned):
    import bpy
    attempts=0
    def move():
        nonlocal attempts
        if not still_owned():
            return None
        result=move_new_window(snapshot)
        attempts+=1
        if result=='pending' and attempts<10:
            return 0.1
        if result=='unavailable':
            print('[HP Section] Automatic monitor placement was unavailable; the window can be moved manually.')
        return None
    bpy.app.timers.register(move,first_interval=0.1)
