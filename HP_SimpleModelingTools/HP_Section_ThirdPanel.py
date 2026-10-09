"""Embedded independent 3D panel, sharing section selection and editing."""
import math
import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Quaternion, Matrix
from types import SimpleNamespace


class ThirdPanel:
    def __init__(self):
        self.rotation=Quaternion((1,0,0,0))
        self.center=Vector()
        self.distance=4.0
        self.perspective=False
        self.navigation=None
        self.signature=None
        self.offscreen=None
        self.scene_rendered=False
        self.detached_scene=None
        self.button_pressed=False
        self.rect=(0,0,1,1)

    def close(self):
        if self.offscreen is not None:
            self.offscreen.free()
            self.offscreen=None

    def fit(self, editor, whole_object=None, reset_rotation=True):
        if editor._idle or not editor._target_available(bpy.context):
            return
        if whole_object is None:
            whole_object=self.scene_view(editor)
        indices=range(len(editor._bm.verts)) if whole_object else editor._ordered
        points=[editor._obj.matrix_world @ editor._bm.verts[i].co for i in indices]
        self.center=sum(points,Vector())/len(points)
        aspect=self.rect[2]/max(1,self.rect[3])
        self.distance=max(.1,max((p-self.center).length for p in points)*3/max(.1,min(1,aspect)))
        if reset_rotation:
            self.rotation=bpy.context.space_data.region_3d.view_rotation.copy()
        self.signature=(editor._obj,editor._signature)

    def scene_view(self, editor):
        if editor.detached:
            return editor.preview_only if self.detached_scene is None else self.detached_scene
        return False

    def buttons(self, editor):
        x,y,w,h=self.rect
        cell=(w-16)/4
        top=y+h-39
        labels=(('scene','シーン'),('section','断面')) if editor.detached else (('full','全体'),)
        return [(key,label,(x+8+i*cell,top-28,cell-2,28)) for i,(key,label) in enumerate(labels)]

    def button_at(self, editor, mouse):
        for key,_,(x,y,w,h) in self.buttons(editor):
            if x<=mouse.x<=x+w and y<=mouse.y<=y+h:
                return key
        return None

    def matrices(self):
        _,_,w,h=self.rect
        view=Matrix.Translation(Vector((0,0,-self.distance))) @ self.rotation.to_matrix().transposed().to_4x4() @ Matrix.Translation(-self.center)
        aspect=w/max(1,h)
        near,far=max(.0001,self.distance*.001),max(1000,self.distance*100)
        if self.perspective:
            f=1/math.tan(math.radians(25))
            projection=Matrix(((f/aspect,0,0,0),(0,f,0,0),(0,0,-(far+near)/(far-near),-2*far*near/(far-near)),(0,0,-1,0)))
        else:
            extent=self.distance*.5
            projection=Matrix(((1/(extent*aspect),0,0,0),(0,1/extent,0,0),(0,0,-2/(far-near),-(far+near)/(far-near)),(0,0,0,1)))
        return view,projection

    def view(self):
        return SimpleNamespace(view_rotation=self.rotation,view_matrix=self.matrices()[0])

    def project(self, world):
        view,projection=self.matrices()
        point=projection @ view @ Vector((*world,1))
        if point.w <= 0:
            return None
        x,y,w,h=self.rect
        return Vector((x+(point.x/point.w+1)*w*.5,y+(point.y/point.w+1)*h*.5))

    def unproject(self, point, depth):
        view,_=self.matrices()
        local=view @ depth
        x,y,w,h=self.rect
        extent=-local.z*math.tan(math.radians(25)) if self.perspective else self.distance*.5
        return view.inverted_safe() @ Vector((((point.x-x)/w*2-1)*extent*w/h,
                                               ((point.y-y)/h*2-1)*extent,local.z))

    def inside(self,x,y):
        left,bottom,w,h=self.rect
        return left<=x<=left+w and bottom<=y<=bottom+h

    def event(self,editor,context,event):
        mouse=Vector((event.mouse_region_x,event.mouse_region_y))
        if self.button_pressed and event.type=='LEFTMOUSE' and event.value=='RELEASE':
            self.button_pressed=False
            return {'RUNNING_MODAL'}
        if self.navigation is not None:
            if event.type=='MIDDLEMOUSE' and event.value=='RELEASE':
                self.navigation=None
            elif event.type in {'MOUSEMOVE','INBETWEEN_MOUSEMOVE'}:
                delta=mouse-self.navigation
                if event.shift:
                    self.center-=self.rotation @ Vector((delta.x,delta.y,0))*self.distance/max(1,self.rect[3])
                elif event.ctrl:
                    self.distance=max(.001,self.distance*math.exp(delta.y*.01))
                else:
                    self.rotation=Quaternion((0,0,1),-delta.x*.01) @ self.rotation @ Quaternion((1,0,0),delta.y*.01)
                self.navigation=mouse
            return {'RUNNING_MODAL'}
        if editor._preview_state is not None and not editor._third_editing:
            return None
        active=editor._third_editing and editor._preview_state is not None
        if not active and not self.inside(*mouse):
            return None
        if not active and event.type=='LEFTMOUSE' and event.value=='PRESS':
            button=self.button_at(editor,mouse)
            if button is not None:
                self.button_pressed=True
                if button=='full':
                    self.button_pressed=False
                    bpy.ops.hp.section_window(full_scene=False)
                    return {'FINISHED'} if editor._finished else {'RUNNING_MODAL'}
                elif button in {'scene','section'}:
                    self.detached_scene=button=='scene'
                    self.fit(editor,whole_object=self.detached_scene,reset_rotation=False)
                return {'RUNNING_MODAL'}
        if not active and event.type=='MIDDLEMOUSE' and event.value=='PRESS':
            self.navigation=mouse
            return {'RUNNING_MODAL'}
        if not active and event.value=='PRESS':
            if event.type in {'WHEELUPMOUSE','WHEELDOWNMOUSE'}:
                self.distance=max(.001,self.distance*(.9 if event.type=='WHEELUPMOUSE' else 1/.9))
                return {'RUNNING_MODAL'}
            if event.type in {'NUMPAD_1','NUMPAD_3','NUMPAD_7'}:
                rotations={'NUMPAD_1':(math.pi/2,0,0),'NUMPAD_3':(math.pi/2,0,math.pi/2),'NUMPAD_7':(0,0,0)}
                from mathutils import Euler
                self.rotation=Euler(rotations[event.type]).to_quaternion()
                self.perspective=False
                return {'RUNNING_MODAL'}
            if event.type=='NUMPAD_5':
                self.perspective=not self.perspective
                return {'RUNNING_MODAL'}
            if event.type in {'HOME','NUMPAD_PERIOD'}:
                self.fit(editor,whole_object=self.scene_view(editor),reset_rotation=False)
                return {'RUNNING_MODAL'}
        if editor._idle or not editor._target_available(context):
            return {'RUNNING_MODAL'} if event.type=='LEFTMOUSE' else None
        editor._third_editing=True
        try:
            return editor._preview_event(context,event,False)
        finally:
            editor._third_editing=editor._preview_state is not None

    def draw(self,editor,context):
        x,y,w,h=self.rect
        viewport=gpu.state.viewport_get()
        gpu.state.scissor_set(viewport[0]+int(x),viewport[1]+int(y),int(w),int(h))
        gpu.state.scissor_test_set(True)
        try:
            self._draw_contents(editor,context)
        finally:
            gpu.state.scissor_test_set(False)

    def _draw_contents(self,editor,context):
        x,y,w,h=self.rect
        scene_view=self.scene_view(editor)
        gpu.state.blend_set('ALPHA')
        editor._draw_rect(x,y,w,h,(.025,.025,.025,.82))
        if scene_view:
            self.draw_scene(editor,context)
        step=24
        for xx in range(int(x),int(x+w),step):
            editor._draw_line([Vector((xx,y)),Vector((xx,y+h))],(.3,.3,.3,.25),1)
        for yy in range(int(y),int(y+h),step):
            editor._draw_line([Vector((x,yy)),Vector((x+w,yy))],(.3,.3,.3,.25),1)
        editor._draw_rect_outline(x,y,x+w,y+h,(.65,.65,.65,.42))
        if not editor._idle and editor._target_available(context):
            points=[self.project(editor._obj.matrix_world @ editor._bm.verts[i].co) for i in editor._ordered]
            previous=None
            for point in points+([points[0]] if editor._closed else []):
                if point is not None and previous is not None:
                    editor._draw_line([previous,point],(.3,.7,1,1),2)
                previous=point
            editor._draw_points([p for p in points if p is not None],(.3,.7,1,1),5)
            for i in sorted(editor._selected):
                if i<len(points) and points[i] is not None:
                    p=points[i]
                    editor._draw_points([p],(1,.55,.08,1),8)
                    blf.size(0,12);blf.color(0,1,.65,.15,1);blf.position(0,p.x+7,p.y+7,0);blf.draw(0,str(i+1))
        editor._draw_rect(x,y+h-28,w,28,(.085,.085,.085,.9))
        blf.size(0,13);blf.color(0,1,1,1,1);blf.position(0,x+10,y+h-19,0)
        blf.draw(0,'C : SCENE' if scene_view else 'C : SECTION')
        for key,label,(bx,by,bw,bh) in self.buttons(editor):
            active=(scene_view and key=='scene') or (not scene_view and key=='section')
            color=(.12,.40,.32,.96) if active else (.14,.16,.19,.96)
            editor._draw_rect(bx,by,bw,bh,color)
            blf.size(0,12)
            blf.color(0,1,1,1,1)
            blf.position(0,bx+7,by+9,0)
            editor._draw_panel_text(label,bx+7,width=bw-12)
        if editor._third_editing:
            editor._draw_preview_hud()

    def scene_objects(self, editor, context):
        if editor._idle or not editor._target_available(context):
            return ()
        obj=editor._obj
        if obj.hide_viewport or obj.hide_get(view_layer=context.view_layer):
            return ()
        return (obj,)

    def draw_scene(self,editor,context):
        x,y,w,h=self.rect
        size=(max(1,int(w)),max(1,int(h)))
        if self.offscreen is None or (self.offscreen.width,self.offscreen.height)!=size:
            self.close()
            self.offscreen=gpu.types.GPUOffScreen(*size)
        view,projection=self.matrices()
        old_depth=gpu.state.depth_test_get()
        old_blend=gpu.state.blend_get()
        gpu.state.scissor_test_set(False)
        with self.offscreen.bind():
            framebuffer=gpu.state.active_framebuffer_get()
            framebuffer.clear(color=(.025,.025,.025,1),depth=1)
            with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                gpu.matrix.load_matrix(view)
                gpu.matrix.load_projection_matrix(projection)
                gpu.state.depth_test_set('LESS_EQUAL')
                gpu.state.blend_set('ALPHA')
                shader=gpu.shader.from_builtin('UNIFORM_COLOR')
                depsgraph=context.evaluated_depsgraph_get()
                for obj in self.scene_objects(editor,context):
                    if obj.hide_viewport or obj.hide_get(view_layer=context.view_layer):
                        continue
                    if obj.type not in {'MESH','CURVE','SURFACE','FONT','META'}:
                        continue
                    evaluated=obj.evaluated_get(depsgraph)
                    mesh=evaluated.to_mesh()
                    if mesh is None:
                        continue
                    try:
                        mesh.calc_loop_triangles()
                        vertices=[evaluated.matrix_world @ v.co for v in mesh.vertices]
                        triangles=[tuple(t.vertices) for t in mesh.loop_triangles]
                        if triangles:
                            shader.bind()
                            shader.uniform_float('color',(.35,.38,.42,.75))
                            batch_for_shader(shader,'TRIS',{'pos':vertices},indices=triangles).draw(shader)
                        shader.bind()
                        shader.uniform_float('color',(.12,.14,.16,1))
                        edges=[tuple(e.vertices) for e in mesh.edges]
                        if edges:
                            batch_for_shader(shader,'LINES',{'pos':vertices},indices=edges).draw(shader)
                    finally:
                        evaluated.to_mesh_clear()
        viewport=gpu.state.viewport_get()
        gpu.state.scissor_set(viewport[0]+int(x),viewport[1]+int(y),int(w),int(h))
        gpu.state.scissor_test_set(True)
        gpu.state.depth_test_set(old_depth)
        gpu.state.blend_set(old_blend)
        gpu.state.blend_set('ALPHA')
        shader=gpu.shader.from_builtin('IMAGE')
        shader.bind();shader.uniform_sampler('image',self.offscreen.texture_color)
        batch_for_shader(shader,'TRI_FAN',{'pos':[(x,y),(x+w,y),(x+w,y+h),(x,y+h)],'texCoord':[(0,0),(1,0),(1,1),(0,1)]}).draw(shader)
        self.scene_rendered=True
