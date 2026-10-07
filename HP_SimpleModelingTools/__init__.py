bl_info = {
    "name": "HP Simple Modeling Tools",
    "author": "OpenAI + yashi",
    "version": (0, 27, 0),
    "blender": (4, 3, 0),
    "location": "3D View",
    "description": "HP curve path drawing, scalp paths, stroke fit, and mini editors.",
    "category": "3D View",
}

from . import HP_Section_MiniEditor
from . import HP_Curve_MiniEditor
from . import HP_StrokeFit


_modules = (
    HP_Section_MiniEditor,
    HP_Curve_MiniEditor,
    HP_StrokeFit,
)


def register():
    for module in _modules:
        module.register()


def unregister():
    for module in reversed(_modules):
        module.unregister()
