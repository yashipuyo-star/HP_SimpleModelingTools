"""Run with Blender --background --factory-startup --python-exit-code 1 --python tests/test_section_editor.py.
For real window/renderer tests, run without --background under a display (or Xvfb).
"""
import sys
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
import bpy
import bmesh
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import HP_SimpleModelingTools as addon
from HP_SimpleModelingTools import HP_Section_MiniEditor as section
from HP_SimpleModelingTools import HP_Curve_MiniEditor as curve
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
            import os
            os._exit(1)
        def finish():
            with bpy.context.temp_override(window=bpy.context.window_manager.windows[0]):
                addon.unregister()
                bpy.ops.wm.quit_blender()
        bpy.app.timers.register(finish, first_interval=0.5)

if bpy.app.background:
    run()
else:
    bpy.app.timers.register(run, first_interval=2.0)
