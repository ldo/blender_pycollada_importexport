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
        default = "*.dae;*.zae;*.kmz",
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

    recognize_blender_extensions : BoolProperty \
      (
        name = "Recognize Blender Extensions",
        description = "Recognize extra info specific to Blender",
        default = True,
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
        return import_collada.load(self, context, self.filepath.endswith(".zae"), **kwargs)
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
        default = "*.dae",
        options = {"HIDDEN"},
      )
    directory : StringProperty \
      (
        subtype = "DIR_PATH",
      )

    collada_version : EnumProperty \
      (
        name = "Collada Version",
        description = "version number to set in output Collada file",
        items =
            (
                ("1.4.1", "1.4.1", ""),
                ("1.5.0", "1.5.0", ""),
            ),
        default = "1.4.1",
      )
    add_blender_extensions : BoolProperty \
      (
        name = "Add Blender Extensions",
        description = "Include extra info specific to Blender",
        default = True,
      )
    export_as : EnumProperty \
      (
        name = "Export as",
        items =
            (
                ("dae", "DAE", ""),
                ("zae", "ZAE", ""),
            ),
        description = "DAE separate file or ZAE all-in-one archive",
        default = "dae",
      )
    export_textures : BoolProperty \
      (
        name = "Export Textures",
        description = "Include texture image files",
        default = False,
      )
    up_axis : EnumProperty \
      (
        name = "Up",
        items =
            (
                ("X_UP", "X Up", ""),
                ("Y_UP", "Y Up", ""),
                ("Z_UP", "Z Up", ""),
            ),
        default = "Z_UP",
      )
    use_selection : BoolProperty \
      (
        name = "Selection Only",
        description = "Export selected objects only",
        default = False,
      )

    def check(self, context) :
        filepath_changed = False
        out_ext = (".dae", ".zae")[self.export_as == "zae"]
        if not self.filepath.endswith(out_ext) :
            self.filepath = os.path.splitext(self.filepath)[0] + out_ext
            filepath_changed = True
            self.export_textures = self.export_as == "zae"
        #end if
        return filepath_changed
    #end check

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
