bl_info = {
    "name": "HP Simple Modeling Tools",
    "author": "OpenAI + yashi",
    "version": (0, 27, 5),
    "blender": (4, 3, 0),
    "location": "3D View",
    "description": "HP curve path drawing, scalp paths, stroke fit, and mini editors.",
    "category": "3D View",
}

import bpy
from bpy.props import BoolProperty


class HP_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__
    move_to_other_monitor: BoolProperty(
        name="別ウィンドウを別モニターに配置（Windows）",default=True)
    show_section_sidebar: BoolProperty(
        name="Nパネルにも断面エディター操作を表示",default=False)

    def draw(self, context):
        self.layout.prop(self,'move_to_other_monitor')
        self.layout.prop(self,'show_section_sidebar')


from . import HP_Section_MiniEditor
from . import HP_Curve_MiniEditor
from . import HP_StrokeFit


_modules = (
    HP_Section_MiniEditor,
    HP_Curve_MiniEditor,
    HP_StrokeFit,
)


def register():
    bpy.utils.register_class(HP_AddonPreferences)
    for module in _modules:
        module.register()


def unregister():
    for module in reversed(_modules):
        module.unregister()
    bpy.utils.unregister_class(HP_AddonPreferences)
