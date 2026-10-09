"""Screen-space editing of the detached section's real BMesh vertices."""
import math
import bpy
import bmesh
import blf
from mathutils import Vector, Quaternion
from bpy_extras import view3d_utils


class SectionPreviewMixin:
    def _preview_points(self, context):
        mw = self._obj.matrix_world
        return [view3d_utils.location_3d_to_region_2d(
            context.region, context.space_data.region_3d, mw @ self._bm.verts[vi].co
        ) for vi in self._ordered]

    def _preview_pick(self, context, mouse):
        candidates = []
        rv3d = context.space_data.region_3d
        for i, point in enumerate(self._preview_points(context)):
            if point is None or (point - mouse).length > 12:
                continue
            world = self._obj.matrix_world @ self._bm.verts[self._ordered[i]].co
            depth = (rv3d.view_matrix @ world).z
            candidates.append(((point-mouse).length, -depth, i))
        # At coincident projections prefer the vertex nearest the viewer.
        if not candidates:
            return None
        nearest = min(c[0] for c in candidates)
        return min((c for c in candidates if c[0] <= nearest + 1), key=lambda c:(c[1],c[0]))[2]

    def _preview_begin_edit(self, context, mode, mouse):
        if not self._selected:
            return False
        follow = self._hp_capture_direct_follow()
        history = list(self._history)
        if not self._push_history(follow):
            return False
        mw = self._obj.matrix_world
        world = [mw @ self._bm.verts[vi].co.copy() for vi in self._ordered]
        indices = sorted(self._selected)
        self._preview_state = dict(mode=mode, start=mouse.copy(), base=world,
            pivot=sum((world[i] for i in indices), Vector()) / len(indices),
            selected=indices, axis=None, follow=follow, history=history)
        self._direct_follow_state = follow
        return True

    def _preview_write(self, context, targets):
        inv = self._obj.matrix_world.inverted_safe()
        for i, world in targets.items():
            vert = self._bm.verts[self._ordered[i]]
            old = self._obj.matrix_world @ vert.co
            vert.co = inv @ self._apply_world_locks(old, world)
        self._hp_apply_direct_follow()
        bmesh.update_edit_mesh(self._obj.data, loop_triangles=False, destructive=False)
        self._sync_primary_from_mesh()

    def _preview_finish_edit(self, context, cancel=False):
        state = self._preview_state
        if state and 'base' in state and cancel and self._bmesh_ready():
            inv = self._obj.matrix_world.inverted_safe()
            for vi, world in zip(self._ordered, state['base']):
                self._bm.verts[vi].co = inv @ world
            if state['follow'] is not None:
                for vi, co in state['follow']['base'].items():
                    self._bm.verts[vi].co = co.copy()
            self._history = state['history']
            bmesh.update_edit_mesh(self._obj.data, loop_triangles=False, destructive=False)
            self._sync_primary_from_mesh()
        if state and 'base' in state and not cancel and self._bmesh_ready():
            mw = self._obj.matrix_world
            if all((mw @ self._bm.verts[vi].co - world).length < 1e-8
                   for vi,world in zip(self._ordered,state['base'])):
                self._history = state['history']
        self._direct_follow_state = None
        self._preview_state = None

    def _preview_transform(self, context, mouse):
        state = self._preview_state
        rv3d = context.space_data.region_3d
        pivot = state['pivot']
        start = state['start']
        axis_name = state['axis']
        axis = Vector((axis_name == 'X', axis_name == 'Y', axis_name == 'Z')) if axis_name else None
        if state['mode'] in {'DRAG', 'G'}:
            begin = view3d_utils.region_2d_to_location_3d(context.region, rv3d, start, pivot)
            end = view3d_utils.region_2d_to_location_3d(context.region, rv3d, mouse, pivot)
            delta = end - begin
            if axis is not None:
                # Recover world-axis displacement from its visible screen projection.
                p = view3d_utils.location_3d_to_region_2d(context.region,rv3d,pivot)
                q = view3d_utils.location_3d_to_region_2d(context.region,rv3d,pivot + axis)
                projected = q-p if p is not None and q is not None else Vector((0,0))
                amount = (mouse-start).dot(projected) / projected.length_squared if projected.length_squared > 1 else delta.dot(axis)
                delta = axis * amount
            targets = {i:state['base'][i] + delta for i in state['selected']}
        elif state['mode'] == 'S':
            factor = math.exp(max(-8,min(8,(mouse.x-start.x)*0.01)))
            targets = {}
            for i in state['selected']:
                relative = state['base'][i]-pivot
                if axis is None:
                    relative *= factor
                else:
                    relative += axis * relative.dot(axis) * (factor-1)
                targets[i] = pivot + relative
        else:
            axis = axis if axis is not None else rv3d.view_rotation @ Vector((0,0,1))
            center = view3d_utils.location_3d_to_region_2d(context.region,rv3d,pivot)
            a, b = (start-center, mouse-center) if center is not None else (Vector(),Vector())
            angle = math.atan2(a.x*b.y-a.y*b.x,a.dot(b)) if a.length > 8 and b.length > 8 else (mouse.x-start.x)*0.01
            rotation = Quaternion(axis,angle)
            targets = {i:pivot + rotation @ (state['base'][i]-pivot) for i in state['selected']}
        self._preview_write(context,targets)

    def _preview_apply_pen(self, context):
        from .HP_Section_MiniEditor import _stabilize_2d, _hp_pen_samples
        state = self._preview_state
        if len(state['stroke']) < 2:
            return
        order, closed = self._pen_selected_order()
        if order is None or len(order) < 2:
            self.report({'WARNING'}, 'Pen: 連続した点の範囲を選択してください')
            return
        points = self._preview_points(context)
        if any(points[i] is None for i in order):
            self.report({'WARNING'}, 'Pen: 対象の点をビュー内に表示してください')
            return
        if closed and (state['stroke'][-1]-state['stroke'][0]).length > 20:
            self.report({'WARNING'}, 'Pen: ループ全体への適用には閉じた線を描いてください')
            return
        stroke = _stabilize_2d(state['stroke'],level=self._pen_stabilizer,closed=closed)
        samples = _hp_pen_samples(stroke,[points[i] for i in order],closed,0,False,self._pen_anchor_mode)
        self._preview_state = None
        selected = set(self._selected)
        self._selected = set(order)
        if not self._preview_begin_edit(context,'PEN_APPLY',Vector()):
            self._selected = selected
            self._preview_state = state
            return
        world = self._preview_state['base']
        targets = {i:view3d_utils.region_2d_to_location_3d(context.region,context.space_data.region_3d,p,world[i])
                   for i,p in zip(order,samples)}
        self._preview_write(context,targets)
        self._preview_finish_edit(context)
        self._selected = selected

    def _preview_brush_wheel(self, event):
        sign = 1 if event.type == 'WHEELUPMOUSE' else -1
        if event.alt:
            self._smooth_brush_strength = max(.05,min(1,self._smooth_brush_strength + sign*.08))
        else:
            self._smooth_brush_radius = max(20,min(240,self._smooth_brush_radius + sign*10))

    def _preview_event(self, context, event, in_panels):
        if not self.detached or not self.preview_only:
            return None
        state = self._preview_state
        if in_panels and state is None:
            return None
        mouse = Vector((event.mouse_region_x,event.mouse_region_y))
        navigation = (event.type == 'MIDDLEMOUSE' or event.type.startswith('NDOF_')
                      or event.type.startswith('TRACKPAD') or event.type == 'MOUSEROTATE')
        if navigation or self._viewport_navigation_active:
            return {'RUNNING_MODAL'} if state and (state['mode'] != 'PEN' or state['stroke']) else None
        if event.type == 'TIMER':
            return None
        if state is not None:
            if event.value == 'PRESS' and (event.type in {'ESC','RIGHTMOUSE'} or (event.type=='Z' and event.ctrl)):
                if state['mode'] == 'PEN' and state.get('brush'):
                    state['stroke'] = state['brush_backup']
                    state['brush'] = False
                else:
                    self._preview_finish_edit(context,cancel=True)
                return {'RUNNING_MODAL'}
            mode = state['mode']
            if mode == 'PEN':
                if state.get('brush'):
                    if event.type == 'E' and event.value == 'RELEASE':
                        state['brush'] = False
                    elif event.type in {'MOUSEMOVE','INBETWEEN_MOUSEMOVE'}:
                        from .HP_Section_MiniEditor import _hp_brush_repel_2d
                        state['stroke'] = _hp_brush_repel_2d(state['stroke'],mouse,mouse-state['previous'],
                            self._smooth_brush_radius,self._smooth_brush_strength)
                        state['previous'] = mouse.copy()
                    elif event.type in {'WHEELUPMOUSE','WHEELDOWNMOUSE'} and event.value == 'PRESS':
                        self._preview_brush_wheel(event)
                    return {'RUNNING_MODAL'}
                if event.type == 'E' and event.value == 'PRESS' and not state['drawing'] and len(state['stroke']) >= 2:
                    state['brush'] = True
                    state['brush_backup'] = [p.copy() for p in state['stroke']]
                    state['previous'] = mouse.copy()
                elif event.type == 'LEFTMOUSE':
                    if event.value == 'PRESS':
                        state['stroke'] = [mouse.copy()]
                        state['drawing'] = True
                    elif event.value == 'RELEASE':
                        state['stroke'].append(mouse.copy())
                        state['drawing'] = False
                elif event.type in {'MOUSEMOVE','INBETWEEN_MOUSEMOVE'} and state['drawing']:
                    if (mouse-state['stroke'][-1]).length >= 1.5:
                        state['stroke'].append(mouse.copy())
                elif event.value == 'PRESS' and event.type in {'F','RET','NUMPAD_ENTER'} and not state['drawing']:
                    self._preview_apply_pen(context)
                return {'RUNNING_MODAL'}
            if mode == 'BOX':
                if event.type in {'MOUSEMOVE','INBETWEEN_MOUSEMOVE'}:
                    state['end'] = mouse.copy()
                if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                    lo = Vector((min(state['start'].x,mouse.x),min(state['start'].y,mouse.y)))
                    hi = Vector((max(state['start'].x,mouse.x),max(state['start'].y,mouse.y)))
                    picked = {i for i,p in enumerate(self._preview_points(context)) if p is not None and lo.x<=p.x<=hi.x and lo.y<=p.y<=hi.y}
                    self._selected = self._selected | picked if state['additive'] else picked
                    self._preview_state = None
                return {'RUNNING_MODAL'}
            if event.value == 'PRESS' and event.type in {'X','Y','Z'} and mode in {'G','S','R'}:
                state['axis'] = None if state['axis'] == event.type else event.type
                self._preview_transform(context,mouse)
            elif event.type in {'MOUSEMOVE','INBETWEEN_MOUSEMOVE'}:
                if mode == 'BRUSH':
                    from .HP_Section_MiniEditor import _hp_brush_repel_2d
                    points = self._preview_points(context)
                    visible = [i for i,p in enumerate(points) if p is not None]
                    moved = _hp_brush_repel_2d([points[i] for i in visible],mouse,mouse-state['previous'],
                                              self._smooth_brush_radius,self._smooth_brush_strength)
                    state['previous'] = mouse.copy()
                    targets = {i:view3d_utils.region_2d_to_location_3d(context.region,context.space_data.region_3d,p,
                               self._obj.matrix_world @ self._bm.verts[self._ordered[i]].co)
                               for i,p in zip(visible,moved) if i in self._selected}
                    self._preview_write(context,targets)
                else:
                    self._preview_transform(context,mouse)
            elif mode == 'BRUSH' and event.type in {'WHEELUPMOUSE','WHEELDOWNMOUSE'} and event.value == 'PRESS':
                self._preview_brush_wheel(event)
            elif (mode=='DRAG' and event.type=='LEFTMOUSE' and event.value=='RELEASE') or (mode=='BRUSH' and event.type=='E' and event.value=='RELEASE'):
                if mode=='DRAG': self._preview_transform(context,mouse)
                self._preview_finish_edit(context)
            elif mode in {'G','S','R'} and event.value=='PRESS' and event.type in {'LEFTMOUSE','RET','NUMPAD_ENTER'}:
                self._preview_transform(context,mouse)
                self._preview_finish_edit(context)
            return {'RUNNING_MODAL'}
        if event.type in {'WHEELUPMOUSE','WHEELDOWNMOUSE'}:
            return None
        if event.value != 'PRESS':
            return None
        if event.ctrl and event.type=='Z':
            self._local_undo(context)
            return {'RUNNING_MODAL'}
        if event.type=='A':
            self._selected = set() if event.alt else set(range(len(self._ordered)))
        elif event.type=='LEFTMOUSE':
            picked = self._preview_pick(context,mouse)
            if picked is None:
                self._preview_state = dict(mode='BOX',start=mouse.copy(),end=mouse.copy(),additive=event.shift)
            elif event.shift and picked in self._selected:
                self._selected.remove(picked)
            else:
                self._selected = self._selected | {picked} if event.shift or picked in self._selected else {picked}
                self._preview_begin_edit(context,'DRAG',mouse)
        elif event.type in {'G','S','R','E'}:
            if self._preview_begin_edit(context,'BRUSH' if event.type=='E' else event.type,mouse):
                self._preview_state['previous'] = mouse.copy()
        elif event.type=='F':
            self._preview_state = dict(mode='PEN',stroke=[],drawing=False)
        elif event.type in {'X','DEL','BACK_SPACE'}:
            # Native mesh selection is the entire source loop, not these points.
            return {'RUNNING_MODAL'}
        else:
            return None
        return {'RUNNING_MODAL'}

    def _draw_preview_hud(self):
        if not self.detached or not self.preview_only:
            return
        blf.size(0,13)
        blf.color(0,.85,.9,1,1)
        blf.position(0,24,bpy.context.region.height-70,0)
        blf.draw(0,'3D: Click / Shift+Click / Box | G S R + X Y Z | E Brush | F Pen | Ctrl+Z')
        state = self._preview_state
        if state is None:
            return
        if state['mode']=='PEN' and len(state['stroke'])>1:
            self._draw_line(state['stroke'],(1,.42,.08,1),3)
            if state.get('brush'):
                self._draw_circle(state['previous'],self._smooth_brush_radius,(.6,.9,1,.6))
        elif state['mode']=='BOX':
            a,b=state['start'],state['end']
            self._draw_rect_outline(a.x,a.y,b.x,b.y,(.3,.8,1,.95))
        elif state['mode']=='BRUSH':
            self._draw_circle(state['previous'],self._smooth_brush_radius,(.6,.9,1,.6))
