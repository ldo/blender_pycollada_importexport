# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# Script copyright (C) Tim Knip, floorplanner.com
# Contributors: Tim Knip (tim@floorplanner.com)

bl_info = \
    {
        "name" : "COLLADA format",
        "author" : "Tim Knip, Dusan Maliarik, Lawrence D’Oliveiro",
        "version" : (0, 9, 0),
        "blender" : (2, 83, 0),
        "location" : "File > Import, File > Export",
        "description" : "Import/Export COLLADA",
        "warning" : "",
        "wiki_url" : "https://github.com/skrat/blender-pycollada/wiki",
        "tracker_url" : "https://github.com/skrat/blender-pycollada/issues",
        "support" : "TESTING",
        "category" : "Import-Export",
    }

if "bpy" in locals() :
    import imp
    if "import_collada" in locals() :
        imp.reload(import_collada)
    #end if
    if "export_collada" in locals() :
        imp.reload(export_collada)
    #end if
#end if

import os
import bpy
from bpy.props import \
    BoolProperty, \
    CollectionProperty, \
    EnumProperty, \
    StringProperty
from bpy_extras.io_utils import \
    ImportHelper, \
    ExportHelper

class IMPORT_OT_collada(bpy.types.Operator, ImportHelper) :
    "COLLADA import operator."

    bl_idname = "import_scene.collada"
    bl_label = "Import COLLADA"
    bl_options = {"UNDO"}

    filter_glob : StringProperty \
      (
        default = "*.dae;*.kmz",
        options = {"HIDDEN"},
      )
    files : CollectionProperty \
      (
        name = "File Path",
        type = bpy.types.OperatorFileListElement,
      )
    directory : StringProperty \
      (
        subtype = "DIR_PATH",
      )

    transparent_shadows : BoolProperty \
      (
        default = False,
        name = "Transparent shadows",
        description = "Import all materials receiving transparent shadows",
      )
    raytrace_transparency : BoolProperty \
      (
        default = False,
        name = "Raytrace transparency",
        description = "Raytrace transparent materials",
      )
    transparency_flip : BoolProperty \
      (
        default = False,
        name = "Transparency Flip",
        description = "Invert sense of interpretation of transparency values",
      )
    transformation : EnumProperty \
      (
        name = "Transformations",
        items =
            (
                ("MUL", "Multiply", ""),
                ("PARENT", "Parenting", ""),
                ("APPLY", "Apply", ""),
            ),
        default = "MUL"
      )

    def execute(self, context) :
        from . import import_collada
        kwargs = self.as_keywords(ignore = ("filter_glob", "files"))
        if not os.path.isfile(kwargs["filepath"]) :
            self.report \
              (
                {"ERROR"},
                "COLLADA import failed, not a file: %s" % repr(kwargs["filepath"])
              )
            return {"CANCELLED"}
        #end if
        return import_collada.load(self, context, **kwargs)
    #end execute

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {"RUNNING_MODAL"}
    #end invoke

#end IMPORT_OT_collada

class EXPORT_OT_collada(bpy.types.Operator, ExportHelper) :
    "COLLADA export operator."

    bl_idname = "export_scene.collada"
    bl_label = "Export COLLADA"
    bl_options = {"UNDO"}

    filename_ext = ".dae"
    filter_glob : StringProperty \
      (
        default = "*.dae;*.kmz",
        options = {"HIDDEN"},
      )
    directory : StringProperty \
      (
        subtype = "DIR_PATH",
      )

    # TODO: none of these optins are actually implemented in export_collada.py!
    export_as : EnumProperty \
      (
        name= "Export as",
        items =
            (
                ("dae_only", "DAE only", ""),
                ("dae_textures", "DAE and textures", ""),
                ("kmz", "KMZ with textures", ""),
            ),
        default = "dae_only",
      )
    axis_up : EnumProperty \
      (
        name = "Up",
        items =
            (
                ("X", "X Up", ""),
                ("Y", "Y Up", ""),
                ("Z", "Z Up", ""),
                ("-X", "-X Up", ""),
                ("-Y", "-Y Up", ""),
                ("-Z", "-Z Up", ""),
            ),
        default = "Z",
      )
    use_selection : BoolProperty \
      (
        name = "Selection Only",
        description = "Export selected objects only",
        default = False,
      )

    def execute(self, context):
        from . import export_collada
        kwargs = self.as_keywords(ignore = ("filter_glob",))
        if os.path.exists(self.filepath) and not os.path.isfile(self.filepath) :
            self.report \
              (
                {"ERROR"},
                "COLLADA export failed, not a file: %s" % repr(kwargs["filepath"])
              )
            return {"CANCELLED"}
        #end if
        return export_collada.save(self, context, **kwargs)
    #end execute

#end EXPORT_OT_collada

_classes_ = \
    (
        IMPORT_OT_collada,
        EXPORT_OT_collada,
    )

def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_collada.bl_idname, text = "COLLADA (py) (.dae, .kmz)")
#end menu_func_import

def menu_func_export(self, context):
    self.layout.operator(EXPORT_OT_collada.bl_idname, text = "COLLADA (py) (.dae, .kmz)")
#end menu_func_export

def register() :
    for çlass in _classes_ :
        bpy.utils.register_class(çlass)
    #end for
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
#end register

def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for çlass in _classes_ :
        bpy.utils.unregister_class(çlass)
    #end for
#end unregister

if __name__ == "__main__" :
    register()
#end if
