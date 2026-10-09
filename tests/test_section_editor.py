"""Run with Blender --background --factory-startup --python-exit-code 1 --python tests/test_section_editor.py.
For real window/renderer tests, run without --background under a display (or Xvfb).
"""
import sys
import math
import ctypes
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import bpy
import bmesh
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import HP_SimpleModelingTools as addon
from HP_SimpleModelingTools import HP_Section_MiniEditor as section
from HP_SimpleModelingTools import HP_Curve_MiniEditor as curve
from HP_SimpleModelingTools import HP_Window_Placement as placement
addon.register()


def event(kind, value='PRESS', x=-100, y=-100, **kwargs):
    fields = dict(type=kind, value=value, mouse_region_x=x, mouse_region_y=y,
                  ctrl=False, alt=False, shift=False, is_repeat=False)
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def make_ring():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    mesh = bpy.data.meshes.new('TestSection')
    points = [(math.cos(i*math.tau/8), math.sin(i*math.tau/8), 0) for i in range(8)]
    mesh.from_pydata(points, [(i, (i+1)%8) for i in range(8)], [])
    obj = bpy.data.objects.new('TestSection', mesh)
    bpy.context.collection.objects.link(obj)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    for v in bm.verts:
        v.select_set(True)
    for e in bm.edges:
        e.select_set(True)
    bmesh.update_edit_mesh(mesh)
    return obj


class BrushTests(unittest.TestCase):
    def test_swept_brush_hits_midpoint_of_fast_drag(self):
        points = [Vector((50,0)), Vector((50,100))]
        result = section._hp_brush_repel_2d(points, (100,0), (100,0), radius=20)
        self.assertGreater(result[0].x, 50)
        self.assertEqual(result[1], points[1])

    def test_event_density_does_not_change_drag(self):
        points = [Vector((20,5)), Vector((40,0)), Vector((60,-5))]
        fast = section._hp_brush_repel_2d(points, (80,0), (80,0), radius=30)
        slow = points
        for x in range(4, 81, 4):
            slow = section._hp_brush_repel_2d(slow, (x,0), (4,0), radius=30)
        for a,b in zip(fast,slow):
            self.assertLess((a-b).length, 0.0001)

    def test_soft_boundary_and_zero_strength(self):
        points = [Vector((0,20)), Vector((0,19.999)), Vector((0,0))]
        result = section._hp_brush_repel_2d(points, (0.1,0), (0.1,0), radius=20)
        self.assertEqual(result[0], points[0])
        self.assertLess((result[1]-points[1]).length, 1e-5)
        self.assertEqual(section._hp_brush_repel_2d(points, (10,0), (10,0), strength=0), points)

    def test_curve_brush_uses_same_continuous_path(self):
        points = [Vector((50,0)), Vector((50,100))]
        self.assertEqual(curve._hp_brush_repel_2d(points,(100,0),(100,0),radius=20),
                         section._hp_brush_repel_2d(points,(100,0),(100,0),radius=20))

    def test_pen_stabilizer_preserves_open_endpoints(self):
        raw = [Vector((i*5, math.sin(i)*2)) for i in range(20)]
        result = section._stabilize_2d(raw, level=4, closed=False)
        self.assertEqual(result[0], raw[0])
        self.assertEqual(result[-1], raw[-1])


class MonitorPlacementTests(unittest.TestCase):
    def test_other_monitor_when_main_window_is_on_either_display(self):
        monitors=[(0,0,1920,1040),(1920,0,3840,1040)]
        self.assertEqual(placement.choose_other_monitor(monitors,(100,100,1500,900)),monitors[1])
        self.assertEqual(placement.choose_other_monitor(monitors,(2000,100,3500,900)),monitors[0])

    def test_negative_monitor_coordinates_and_window_size_are_preserved(self):
        target=(-2560,0,0,1400)
        self.assertEqual(placement.choose_other_monitor([(0,0,1920,1040),target],(100,100,1500,900)),target)
        x,y,width,height=placement.placement_in_work_area(target,(0,0,1322,876))
        self.assertLess(x,0)
        self.assertEqual((width,height),(1322,876))
        self.assertGreaterEqual(y,0)

    def test_single_monitor_is_left_unchanged(self):
        self.assertIsNone(placement.choose_other_monitor([(0,0,1920,1040)],(100,100,1500,900)))

    def test_smaller_display_clamps_window_to_work_area(self):
        x,y,w,h=placement.placement_in_work_area((1920,-200,3200,800),(0,0,2000,1400))
        self.assertGreaterEqual(x,1920)
        self.assertGreaterEqual(y,-200)
        self.assertLessEqual(x+w,3200)
        self.assertLessEqual(y+h,800)

    def test_native_move_targets_only_the_new_owned_window(self):
        from ctypes import wintypes
        class Info(ctypes.Structure):
            _fields_=[('cbSize',wintypes.DWORD),('rcWork',wintypes.RECT)]
        work=[(0,0,1920,1040),(-1920,0,0,1040)]
        def monitors(_hdc,_rect,callback,_data):
            for i in range(len(work)):callback(i,None,None,0)
        def monitor_info(i,info):
            info._obj.rcWork=wintypes.RECT(*work[i])
            return True
        native=SimpleNamespace(
            GetWindowThreadProcessId=lambda hwnd,pid:setattr(pid._obj,'value',os.getpid()),
            EnumDisplayMonitors=monitors,GetMonitorInfoW=monitor_info,SetWindowPos=Mock(return_value=True))
        before=dict(handles={1:(100,100,1500,900)},source=(100,100,1500,900))
        after=dict(handles={1:(100,100,1500,900),2:(100,100,1500,900)},source=(100,100,1500,900))
        with patch.object(placement,'snapshot_windows',return_value=after), patch.object(placement,'_windows_api',return_value=(native,None,lambda cb:cb,Info)):
            self.assertEqual(placement.move_new_window(before),'moved')
        native.SetWindowPos.assert_called_once_with(2,None,-1888,32,1400,800,0x14)

    def test_ambiguous_native_window_identity_never_moves_a_window(self):
        before=dict(handles={1:(0,0,100,100)},source=(0,0,100,100))
        after=dict(handles={1:(0,0,100,100),2:(0,0,100,100),3:(0,0,100,100)},source=(0,0,100,100))
        with patch.object(placement,'snapshot_windows',return_value=after), patch.object(placement,'_windows_api') as native:
            self.assertEqual(placement.move_new_window(before),'pending')
            native.assert_not_called()


@unittest.skipIf(bpy.app.background, 'Modal editing requires a window manager; run the GUI command')
class EditorTests(unittest.TestCase):
    def setUp(self):
        self.window = bpy.context.window or bpy.context.window_manager.windows[0]
        self.area = next(a for a in self.window.screen.areas if a.type == 'VIEW_3D')
        self.region = next(r for r in self.area.regions if r.type == 'WINDOW')
        self.override = bpy.context.temp_override(window=self.window, area=self.area, region=self.region)
        self.override.__enter__()
        self.obj = make_ring()

    def tearDown(self):
        if section._ACTIVE_SECTION_EDITOR:
            section._ACTIVE_SECTION_EDITOR._finish(bpy.context, release_workspace=True)
        if curve._ACTIVE_CURVE_EDITOR:
            curve._ACTIVE_CURVE_EDITOR._finish(bpy.context, release_workspace=True)
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        self.override.__exit__(None, None, None)

    def open_inline(self):
        self.assertEqual(bpy.ops.hp.section_mini_editor('INVOKE_DEFAULT'), {'RUNNING_MODAL'})
        return section._ACTIVE_SECTION_EDITOR

    def test_curve_editor_is_manual_compact_and_passes_outside_input(self):
        self.assertFalse(bpy.app.timers.is_registered(curve._curve_autostart))
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.curve.primitive_bezier_circle_add()
        bpy.ops.object.mode_set(mode='EDIT')
        self.assertEqual(bpy.ops.hp.curve_mini_editor('INVOKE_DEFAULT'), {'RUNNING_MODAL'})
        editor = curve._ACTIVE_CURVE_EDITOR
        self.assertLessEqual(editor._panel_w,400)
        self.assertLessEqual(editor._panel_h,300)
        self.assertEqual(editor.modal(bpy.context,event('F')),{'PASS_THROUGH'})
        self.assertFalse(editor._pen_mode)

    def test_orbit_release_outside_panel_clears_navigation(self):
        editor = self.open_inline()
        x,y=editor._panel_points[0]
        editor._modal_impl(bpy.context,event('MIDDLEMOUSE',x=x,y=y))
        self.assertTrue(editor._viewport_navigation_active)
        editor._modal_impl(bpy.context,event('MIDDLEMOUSE',value='RELEASE'))
        self.assertFalse(editor._viewport_navigation_active)

    def test_redo_keyboard_outside_panels_passes_to_blender(self):
        editor = self.open_inline()
        editor._selected = {0}
        before = [v.co.copy() for v in editor._bm.verts]
        for kind, opts in [('F', {}), ('Z', {'ctrl':True}), ('LEFTMOUSE', {}), ('E', {})]:
            self.assertEqual(editor._modal_impl(bpy.context, event(kind, **opts)), {'PASS_THROUGH'})
        self.assertEqual([v.co.copy() for v in editor._bm.verts], before)
        self.assertFalse(editor._pen_mode)

    def test_idle_and_target_reselection_do_not_close(self):
        editor = self.open_inline()
        bm = bmesh.from_edit_mesh(self.obj.data)
        for v in bm.verts: v.select_set(False)
        for e in bm.edges: e.select_set(False)
        bmesh.update_edit_mesh(self.obj.data)
        editor._modal_impl(bpy.context, event('TIMER'))
        self.assertTrue(editor._idle)
        self.assertFalse(editor._finished)
        for v in bm.verts: v.select_set(True)
        for e in bm.edges: e.select_set(True)
        bmesh.update_edit_mesh(self.obj.data)
        editor._modal_impl(bpy.context, event('TIMER'))
        self.assertFalse(editor._idle)
        self.assertEqual(len(editor._ordered),8)

    def test_can_open_before_any_chain_is_selected(self):
        bpy.ops.object.mode_set(mode='OBJECT')
        editor = self.open_inline()
        self.assertTrue(editor._idle)
        self.assertFalse(editor._finished)

    def test_same_topology_on_new_object_rebuilds_target(self):
        editor = self.open_inline()
        first = editor._obj
        second = make_ring()
        editor._modal_impl(bpy.context,event('TIMER'))
        self.assertIsNot(editor._obj,first)
        self.assertEqual(editor._obj,second)
        self.assertFalse(editor._idle)

    def test_mirror_clipping_still_holds(self):
        editor = self.open_inline()
        modifier = self.obj.modifiers.new('TestMirror','MIRROR')
        modifier.use_clip = True
        modifier.use_axis = (True,False,False)
        self.assertAlmostEqual(editor._apply_world_locks(Vector((0,0,0)),Vector((2,0,0))).x,0)

    def test_layout_keeps_two_panels_inside_region(self):
        editor = self.open_inline()
        self.assertLessEqual(editor._secondary_x()+editor._panel_w, self.region.width)
        self.assertLessEqual(editor._panel_y+editor._panel_h,self.region.height)
        for stage in range(3):
            editor._set_panel_stage(stage)
            editor._layout_panels(bpy.context)
            self.assertLessEqual(editor._secondary_x()+editor._panel_w,self.region.width)

    def test_brush_cancel_restores_vertices_and_history(self):
        editor = self.open_inline()
        editor._selected = set(range(8))
        before = [v.co.copy() for v in editor._bm.verts]
        self.assertTrue(editor._begin_smooth_drag(bpy.context,'A',editor._panel_points[0].x,editor._panel_points[0].y))
        point = editor._panel_points[0]
        editor._update_smooth_drag(bpy.context,point.x+20,point.y)
        self.assertTrue(any((v.co-c).length > 1e-6 for v,c in zip(editor._bm.verts,before)))
        editor._finish_smooth_drag(bpy.context,cancel=True)
        for v,c in zip(editor._bm.verts,before): self.assertLess((v.co-c).length,1e-6)
        self.assertEqual(len(editor._history),0)

    def test_pen_brush_cancel_restores_uncommitted_stroke(self):
        editor = self.open_inline()
        raw = [Vector((i*5, 0)) for i in range(20)]
        editor._pen_stroke = [p.copy() for p in raw]
        self.assertTrue(editor._begin_pen_smooth_drag(bpy.context,'A',0,0))
        editor._update_pen_smooth_drag(bpy.context,30,0)
        self.assertNotEqual(editor._pen_stroke,raw)
        editor._finish_pen_smooth_drag(bpy.context,cancel=True)
        self.assertEqual(editor._pen_stroke,raw)

    def open_preview(self):
        editor = self.open_inline()
        editor.detached = True
        return editor

    def test_default_panel_size_is_medium(self):
        editor = self.open_inline()
        self.assertEqual(editor._panel_stage,1)
        self.assertEqual(editor._panel_sizes[1],(480,360))

    def test_preview_click_selects_shared_point_without_undo_noise(self):
        editor = self.open_preview()
        point = editor._preview_points(bpy.context)[0]
        self.assertIsNotNone(point)
        editor._preview_event(bpy.context,event('LEFTMOUSE',x=point.x,y=point.y),False)
        self.assertEqual(editor._selected,{0})
        editor._preview_event(bpy.context,event('LEFTMOUSE','RELEASE',x=point.x,y=point.y),False)
        self.assertIsNone(editor._preview_state)
        self.assertEqual(len(editor._history),0)

    def test_preview_move_commit_and_undo_preserve_unselected_points(self):
        editor = self.open_preview()
        editor._selected = {0}
        before = [v.co.copy() for v in editor._bm.verts]
        editor._preview_event(bpy.context,event('G',x=600,y=600),False)
        editor._preview_event(bpy.context,event('MOUSEMOVE',x=620,y=615),False)
        self.assertGreater((editor._bm.verts[editor._ordered[0]].co-before[editor._ordered[0]]).length,1e-5)
        for i in editor._ordered[1:]: self.assertEqual(editor._bm.verts[i].co,before[i])
        editor._preview_event(bpy.context,event('RET',x=620,y=615),False)
        editor._preview_event(bpy.context,event('Z',ctrl=True),False)
        for i,co in enumerate(before):self.assertLess((editor._bm.verts[i].co-co).length,1e-6)

    def test_preview_axis_move_and_cancel(self):
        editor = self.open_preview()
        editor._selected = {0,1}
        before = [v.co.copy() for v in editor._bm.verts]
        editor._preview_event(bpy.context,event('G',x=600,y=600),False)
        editor._preview_event(bpy.context,event('X',x=600,y=600),False)
        editor._preview_event(bpy.context,event('MOUSEMOVE',x=625,y=600),False)
        for i in editor._selected:
            co=editor._bm.verts[editor._ordered[i]].co
            self.assertAlmostEqual(co.y,before[editor._ordered[i]].y,places=5)
            self.assertAlmostEqual(co.z,before[editor._ordered[i]].z,places=5)
        editor._preview_event(bpy.context,event('ESC'),False)
        for i,co in enumerate(before):self.assertLess((editor._bm.verts[i].co-co).length,1e-6)
        self.assertEqual(len(editor._history),0)

    def test_preview_scale_and_rotate(self):
        editor = self.open_preview()
        editor._selected = {0,2}
        a,b=[editor._bm.verts[editor._ordered[i]].co.copy() for i in (0,2)]
        editor._preview_event(bpy.context,event('S',x=600,y=600),False)
        editor._preview_event(bpy.context,event('RET',x=640,y=600),False)
        c,d=[editor._bm.verts[editor._ordered[i]].co.copy() for i in (0,2)]
        self.assertAlmostEqual((c-d).length,(a-b).length*math.exp(.4),places=5)
        editor._preview_event(bpy.context,event('R',x=600,y=600),False)
        editor._preview_event(bpy.context,event('Z',x=600,y=600),False)
        editor._preview_event(bpy.context,event('RET',x=600,y=640),False)
        e,f=[editor._bm.verts[editor._ordered[i]].co.copy() for i in (0,2)]
        self.assertAlmostEqual((e-f).length,(c-d).length,places=5)
        self.assertGreater((e-c).length,1e-4)

    def test_preview_brush_cancel_restores_mesh(self):
        editor = self.open_preview()
        editor._selected = {0}
        point = editor._preview_points(bpy.context)[0]
        before=[v.co.copy() for v in editor._bm.verts]
        editor._preview_event(bpy.context,event('E',x=point.x,y=point.y),False)
        editor._preview_event(bpy.context,event('MOUSEMOVE',x=point.x+20,y=point.y),False)
        self.assertGreater((editor._bm.verts[editor._ordered[0]].co-before[editor._ordered[0]]).length,1e-5)
        editor._preview_event(bpy.context,event('RIGHTMOUSE'),False)
        for i,co in enumerate(before):self.assertLess((editor._bm.verts[i].co-co).length,1e-6)

    def test_preview_box_selection_matches_projected_points(self):
        editor = self.open_preview()
        point=editor._preview_points(bpy.context)[0]
        a,b=point-Vector((15,15)),point+Vector((15,15))
        editor._preview_state=dict(mode='BOX',start=a,end=a,additive=False)
        editor._preview_event(bpy.context,event('LEFTMOUSE','RELEASE',x=b.x,y=b.y),False)
        expected={i for i,p in enumerate(editor._preview_points(bpy.context))
                  if p is not None and a.x<=p.x<=b.x and a.y<=p.y<=b.y}
        self.assertEqual(editor._selected,expected)
        self.assertIn(0,expected)

    def test_preview_pen_brush_cancel_preserves_uncommitted_line(self):
        editor = self.open_preview()
        raw=[Vector((500+i*5,500)) for i in range(10)]
        editor._preview_state=dict(mode='PEN',stroke=[p.copy() for p in raw],drawing=False)
        editor._preview_event(bpy.context,event('E',x=500,y=500),False)
        editor._preview_event(bpy.context,event('MOUSEMOVE',x=520,y=500),False)
        self.assertNotEqual(editor._preview_state['stroke'],raw)
        editor._preview_event(bpy.context,event('ESC'),False)
        self.assertEqual(editor._preview_state['stroke'],raw)
        self.assertFalse(editor._preview_state['brush'])
        self.assertEqual(editor._preview_state['mode'],'PEN')

    def test_preview_pen_changes_mesh_only_on_confirmation(self):
        editor = self.open_preview()
        editor._selected = {0,1,2}
        points=editor._preview_points(bpy.context)
        order,_=editor._pen_selected_order()
        start,end=points[order[0]],points[order[-1]]
        before=[v.co.copy() for v in editor._bm.verts]
        editor._preview_event(bpy.context,event('F'),False)
        editor._preview_event(bpy.context,event('LEFTMOUSE',x=start.x,y=start.y+30),False)
        mid=(start+end)/2
        editor._preview_event(bpy.context,event('MOUSEMOVE',x=mid.x,y=mid.y+30),False)
        editor._preview_event(bpy.context,event('LEFTMOUSE','RELEASE',x=end.x,y=end.y+30),False)
        self.assertEqual([v.co.copy() for v in editor._bm.verts],before)
        editor._preview_event(bpy.context,event('RET'),False)
        self.assertIsNone(editor._preview_state)
        self.assertTrue(any((v.co-c).length>1e-5 for v,c in zip(editor._bm.verts,before)))
        editor._preview_event(bpy.context,event('Z',ctrl=True),False)
        for i,co in enumerate(before):self.assertLess((editor._bm.verts[i].co-co).length,1e-6)

    def test_right_click_in_panel_reaches_context_menu_with_selected_points(self):
        editor = self.open_inline()
        editor._selected={0,1}
        point=editor._panel_points[0]
        self.assertEqual(editor._modal_impl(bpy.context,event('RIGHTMOUSE',x=point.x,y=point.y)),{'PASS_THROUGH'})
        self.assertFalse(editor._finished)
        self.assertEqual(editor._selected,{0,1})

    def test_sidebar_is_hidden_by_default(self):
        self.assertFalse(section.HP_PT_section_tools.poll(bpy.context))

    def test_inline_to_detached_and_back_preserves_selection_settings_and_undo(self):
        old=self.open_inline()
        old._selected={0,2}
        old._hp_settings['follow_strength_a']=.5
        old._push_history()
        windows_before={w.as_pointer() for w in bpy.context.window_manager.windows}
        self.assertEqual(bpy.ops.hp.section_window(),{'FINISHED'})
        detached=section._ACTIVE_SECTION_EDITOR
        owner=detached._owner_window_ptr
        self.assertTrue(old._finished)
        self.assertTrue(detached.detached)
        self.assertEqual(detached._selected,{0,2})
        self.assertEqual(detached._hp_settings['follow_strength_a'],.5)
        self.assertEqual(len(detached._history),1)
        self.assertEqual(bpy.ops.hp.section_window(),{'FINISHED'})
        self.assertEqual(section._ACTIVE_SECTION_EDITOR._owner_window_ptr,owner)
        self.assertEqual(len(bpy.context.window_manager.windows),len(windows_before)+1)
        window,area,region=section._view_context(owner,detached._owner_area_ptr)
        with bpy.context.temp_override(window=window,area=area,region=region):
            self.assertEqual(bpy.ops.hp.section_inline(),{'FINISHED'})
        inline=section._ACTIVE_SECTION_EDITOR
        self.assertFalse(inline.detached)
        self.assertEqual(inline._owner_window_ptr,self.window.as_pointer())
        self.assertEqual(inline._selected,{0,2})
        self.assertEqual(len(inline._history),1)
        # Deferred native close runs in the real event loop; finish this test's
        # duplicate explicitly to avoid retaining a window across other tests.
        with bpy.context.temp_override(window=window):
            bpy.ops.wm.window_close()

    def test_close_detached_schedules_its_owned_window_for_closing(self):
        self.assertEqual(bpy.ops.hp.section_window(),{'FINISHED'})
        editor=section._ACTIVE_SECTION_EDITOR
        editor._selected={0,2}
        window,area,region=section._view_context(editor._owner_window_ptr,editor._owner_area_ptr)
        with bpy.context.temp_override(window=window,area=area,region=region):
            self.assertEqual(bpy.ops.hp.section_close(),{'FINISHED'})
        self.assertTrue(editor._finished)
        self.assertIsNone(section._ACTIVE_SECTION_EDITOR)
        # The final event-loop check below confirms that this native window
        # actually disappears, rather than merely losing its overlay.

    def test_settings_survive_switch_while_idle_without_mesh_selection(self):
        editor=self.open_inline()
        editor._smooth_brush_strength=.8
        bpy.ops.object.mode_set(mode='OBJECT')
        editor._modal_impl(bpy.context,event('TIMER'))
        self.assertTrue(editor._idle)
        self.assertEqual(bpy.ops.hp.section_window(),{'FINISHED'})
        detached=section._ACTIVE_SECTION_EDITOR
        self.assertTrue(detached._idle)
        self.assertAlmostEqual(detached._smooth_brush_strength,.8)
        owner=detached._owner_window_ptr
        window,area,region=section._view_context(owner,detached._owner_area_ptr)
        with bpy.context.temp_override(window=window,area=area,region=region):
            self.assertEqual(bpy.ops.hp.section_inline(),{'FINISHED'})
        self.assertFalse(section._ACTIVE_SECTION_EDITOR.detached)
        self.assertAlmostEqual(section._ACTIVE_SECTION_EDITOR._smooth_brush_strength,.8)
        with bpy.context.temp_override(window=window):
            bpy.ops.wm.window_close()

    def test_full_scene_mode_keeps_geometry_camera_and_native_navigation(self):
        view=self.area.spaces.active.region_3d
        old_perspective=view.view_perspective
        view.view_perspective='ORTHO'
        rotation=view.view_rotation.copy()
        location=view.view_location.copy()
        distance=view.view_distance
        before=len(bpy.context.window_manager.windows)
        self.assertEqual(bpy.ops.hp.section_window(full_scene=True),{'FINISHED'})
        full=section._ACTIVE_SECTION_EDITOR
        owner=full._owner_window_ptr
        window,area,region=section._view_context(owner,full._owner_area_ptr)
        with bpy.context.temp_override(window=window,area=area,region=region):
            self.assertFalse(full.preview_only)
            self.assertEqual(area.spaces.active.region_3d.view_perspective,'ORTHO')
            self.assertTrue(area.spaces.active.show_object_viewport_mesh)
            self.assertTrue(area.spaces.active.overlay.show_overlays)
            rv3d=area.spaces.active.region_3d
            self.assertLess(rotation.rotation_difference(rv3d.view_rotation).angle,1e-5)
            self.assertLess((location-rv3d.view_location).length,1e-5)
            self.assertAlmostEqual(distance,rv3d.view_distance,places=5)
            for kind in ('NUMPAD_1','NUMPAD_3','NUMPAD_7','MIDDLEMOUSE','WHEELUPMOUSE','G'):
                self.assertEqual(full._modal_impl(bpy.context,event(kind)),{'PASS_THROUGH'})
            self.assertIsNone(full._preview_state)
            full._selected={0,2}
            self.assertEqual(bpy.ops.hp.section_window(full_scene=False),{'FINISHED'})
            limited=section._ACTIVE_SECTION_EDITOR
            self.assertTrue(limited.preview_only)
            self.assertEqual(rv3d.view_perspective,'PERSP')
            self.assertEqual(limited._owner_window_ptr,owner)
            self.assertFalse(area.spaces.active.show_object_viewport_mesh)
            self.assertEqual(bpy.ops.hp.section_window(full_scene=True),{'FINISHED'})
            restored=section._ACTIVE_SECTION_EDITOR
            self.assertFalse(restored.preview_only)
            self.assertEqual(rv3d.view_perspective,'ORTHO')
            self.assertEqual(restored._selected,{0,2})
            self.assertEqual(len(bpy.context.window_manager.windows),before+1)
            self.assertTrue(area.spaces.active.show_object_viewport_mesh)
            self.assertLess((rv3d.view_location-location).length,1e-5)
            self.assertAlmostEqual(rv3d.view_distance,distance,places=5)
            restored._finish(bpy.context,release_workspace=True)
            bpy.ops.wm.window_close()
        view.view_perspective=old_perspective

    def test_close_command_does_not_require_deselecting(self):
        editor=self.open_inline()
        editor._selected={0,2}
        self.assertEqual(bpy.ops.hp.section_close(),{'FINISHED'})
        self.assertTrue(editor._finished)
        self.assertIsNone(section._ACTIVE_SECTION_EDITOR)

    @unittest.skipIf(bpy.app.background, 'Requires an actual window manager')
    def test_detached_window_and_visibility_restoration(self):
        previous = {w.as_pointer() for w in bpy.context.window_manager.windows}
        self.assertEqual(bpy.ops.hp.section_window(), {'FINISHED'})
        editor = section._ACTIVE_SECTION_EDITOR
        self.assertTrue(editor.detached)
        window = next(w for w in bpy.context.window_manager.windows if w.as_pointer() not in previous)
        area = next(a for a in window.screen.areas if a.type=='VIEW_3D')
        region = next(r for r in area.regions if r.type=='WINDOW')
        self.assertTrue(self.area.spaces.active.show_object_viewport_mesh)
        with bpy.context.temp_override(window=window,area=area,region=region):
            self.assertEqual(bpy.context.edit_object,self.obj)
            self.assertFalse(area.spaces.active.show_object_viewport_mesh)
            editor._selected = {0,2}
            editor._modal_impl(bpy.context,event('TIMER'))
            self.assertFalse(editor._idle)
            self.assertTrue(editor._target_available(bpy.context))
            editor._finish(bpy.context,release_workspace=True)
            self.assertTrue(area.spaces.active.show_object_viewport_mesh)
            bpy.ops.wm.window_close()
        self.assertIsNone(section._ACTIVE_SECTION_EDITOR)


suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])

def run():
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if bpy.app.background:
        addon.unregister()
        if not result.wasSuccessful(): raise RuntimeError('Regression tests failed')
    else:
        sys.stdout.flush()
        if not result.wasSuccessful():
            os._exit(1)
        def finish():
            if len(bpy.context.window_manager.windows) != 1:
                print('FAIL: deferred section-window closure left extra windows',flush=True)
                os._exit(1)
            print('PASS: deferred section-window closure',flush=True)
            with bpy.context.temp_override(window=bpy.context.window_manager.windows[0]):
                addon.unregister()
                bpy.ops.wm.quit_blender()
        bpy.app.timers.register(finish, first_interval=0.5)

if bpy.app.background:
    run()
else:
    bpy.app.timers.register(run, first_interval=2.0)
