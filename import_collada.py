import sys
import os
import math
from numbers import \
    Real
import io
import time
import zipfile
import tempfile
import shutil

import bpy
import bmesh
from bpy_extras.image_utils import load_image
from mathutils import Matrix, Vector

from collada import Collada
from collada.camera import PerspectiveCamera, OrthographicCamera
from collada.common import DaeBrokenRefError, DaeObject, tag
from collada.light import AmbientLight, DirectionalLight, PointLight, SpotLight
from collada.material import Map
from collada.polylist import Polylist, BoundPolylist
from collada.primitive import BoundPrimitive
from collada.scene import Scene, Node, NodeNode, CameraNode, GeometryNode, LightNode
from collada.triangleset import TriangleSet, BoundTriangleSet
from collada.xmlutil import etree as ElementTree

MAX_NAME_LENGTH = 63
DEG = math.pi / 180 # angle unit conversion factor

class DATABLOCK :
    CAMERA = "CAMERA"
    EMPTY = "EMPTY"
    LAMP = "LAMP"
    MATERIAL = "MATERIAL"
    MESH = "MESH"
    SCENE = "SCENE"
#end DATABLOCK

def unurlid(uid) :
    assert uid.startswith("#")
    return uid[1:]
#end unurlid

class ColladaImport :
    "Standard COLLADA importer. Subclasses can implement a “match” method" \
    " to identify vendor-specific features they need to handle."

    def __init__(self, ctx, collada, filepath, **kwargs) :
        self.DAE_NS = {"dae": collada.xmlnode.getroot().nsmap[None]}
        basename = os.path.basename(filepath)
        self._ctx = ctx
        self._collada = collada
        self._recognize_blender_extensions = kwargs["recognize_blender_extensions"]
        self._transformation = kwargs["transformation"]
        self._name_map = {}
        self._name_revmap = {}
        self._untitledcount = 0
        self._units = collada.assetInfo.unitmeter
        if self._units == None :
            self._units = 1
        #end if
        orient = collada.assetInfo.upaxis
        if orient == "Z_UP" :
            self._orient = Matrix.Identity(4)
        elif orient == "X_UP" :
            self._orient = Matrix.Rotation(120 * DEG, 4, Vector(1, -1, 1))
        else : # "Y_UP" or unspecified
            self._orient = Matrix.Rotation(90 * DEG, 4, "X")
        #end if
        self._id_prefixes = None
        root_technique = self.get_blender_technique(True, self._collada.xmlnode.getroot())
        if root_technique != None :
            id_prefixes = root_technique.find(tag("id_prefixes"))
            if id_prefixes != None :
                self._id_prefixes = {}
                for prefix in id_prefixes.findall(tag("prefix")) :
                    name = prefix.get("name")
                    value = prefix.get("value")
                    if name != None and value != None :
                        self._id_prefixes[name] = value
                    #end if
                #end for
            #end if
        #end if
        self._collection = bpy.data.collections.new(basename)
        self._ctx.scene.collection.children.link(self._collection)
    #end __init__

    def get_blender_technique(self, as_extra, obj) :
        # experimental: add Blender-specific attributes via a custom <technique>.
        blendstuff = None
        if self._recognize_blender_extensions :
            if isinstance(obj, DaeObject) :
                obj = obj.xmlnode
            #end if
            if as_extra :
                parent = obj.find(tag("extra"))
            else :
                parent = obj
            #end if
            if parent != None :
                blendstuff = parent.find(tag("technique") + "[@profile=\"BLENDER028\"]")
            #end if
        #end if
        return blendstuff
    #end get_blender_technique

    def apply_blender_technique(self, as_extra, obj, b_data, attribs) :
        # get and apply any custom technique settings for this object.
        blendstuff = self.get_blender_technique(as_extra, obj)
        if blendstuff != None :
            for tagname, parse, attrname in attribs :
                if hasattr(b_data, attrname) :
                    subtag = blendstuff.find(tag(tagname))
                    if subtag != None :
                        try :
                            setattr(b_data, attrname, parse(subtag.text))
                        except ValueError as err :
                            sys.stderr.write \
                              (
                                    "import_collada: error setting %s attribute for %s: %s\n"
                                %
                                    (
                                        attrname,
                                        b_data.name,
                                        str(err)
                                    )
                              )
                        #end try
                    #end if
                #end if
            #end for
        #end if
        return blendstuff != None
    #end apply_blender_technique

    def name(self, prefix_name, obj) :
        "Trying to get efficient and human readable name, working around" \
        " Blender’s object name limitations."
        if hasattr(obj, "id") and obj.id != None :
            origname = obj.id
            if self._id_prefixes != None :
                prefix = self._id_prefixes.get(prefix_name)
                if prefix != None and origname.startswith(prefix) :
                    origname = origname[len(prefix):]
                #end if
            #end if
            if origname in self._name_map :
                usename = self._name_map[origname]
            else :
                usename = origname[:MAX_NAME_LENGTH]
                seq = 0
                while usename in self._name_revmap :
                    seq += 1
                    suffix = "-%0.3d" % seq
                    assert len(suffix) < MAX_NAME_LENGTH
                    usename ="%s%s" % (origname[:MAX_NAME_LENGTH - len(suffix)], suffix)
                #end while
                self._name_map[origname] = usename
                self._name_revmap[usename] = origname
            #end if
        else :
            origname = id(obj) # non-string type to avoid conflicting with any actual XML ID
            if origname in self._name_map :
                usename = self._name_map[origname]
            else :
                self._untitledcount += 1
                usename = "untitled %0.3d" % self._untitledcount
                  # space in name means it can never conflict with any actual XML ID
                self._name_map[origname] = usename
                self._name_revmap[usename] = origname
            #end if
        #end if
        return usename
    #end name

    def _transform(self, t) :
        return self._transformation == t
    #end _transform

    def _convert_units_matrix(self, mat) :
        "converts the translation part of Matrix mat from the specified" \
        " units in the Collada file to Blender units."
        mat = mat.copy()
        for i in range(3) :
            mat[i][3] *= self._units
        #end for
        return mat
    #end _convert_units_matrix

    def _convert_units_verts(self, verts) :
        "converts a sequence of vectors from the specified" \
        " units in the Collada file to Blender units."
        return \
            list(self._units * Vector(v) for v in verts)
    #end _convert_units_verts

    def camera(self, bcam) :

        def fudge_div(num, den) :
            # needed to cope with some problem files.
            try :
                result = num / den
            except ZeroDivisionError :
                result = num
            #end if
            return result
        #end fudge_div

    #begin camera
        b_name = self.name(DATABLOCK.CAMERA, bcam.original)
        # todo: shared datablocks
        b_cam = bpy.data.cameras.new(b_name)
        b_obj = bpy.data.objects.new(b_cam.name, b_cam)
        if isinstance(bcam.original, PerspectiveCamera) :
            b_cam.type = "PERSP"
            prop = b_cam.bl_rna.properties.get("lens_unit")
            if "DEGREES" in prop.enum_items :
                b_cam.lens_unit = "DEGREES"
            elif "FOV" in prop.enum_items :
                b_cam.lens_unit = "FOV"
            else :
                b_cam.lens_unit = prop.default
            #end if
            # I don’t actually support preservation of aspect ratios,
            # since in Blender that is a rendering setting, not a
            # camera setting. For now I just use the maximum of the
            # horizontal and vertical views.
            b_cam.angle = \
                max \
                  (( # “None” marks cases which shouldn’t occur
                        None, # bcam.aspect_ratio = None and bcam.yfov = None and bcam.xfov = None
                        None, # bcam.aspect_ratio = None and bcam.yfov = None and bcam.xfov ≠ None
                        None, # bcam.aspect_ratio = None and bcam.yfov ≠ None and bcam.xfov = None
                        lambda : (bcam.yfov * DEG, bcam.xfov * DEG),
                          # bcam.aspect_ratio = None and bcam.yfov ≠ None and bcam.xfov ≠ None
                        None, # bcam.aspect_ratio ≠ None and bcam.yfov = None and bcam.xfov = None
                        lambda :
                            (
                                2 * math.atan(fudge_div(math.tan(bcam.xfov * DEG / 2),  bcam.aspect_ratio)),
                                bcam.xfov * DEG
                            ),
                          # bcam.aspect_ratio ≠ None and bcam.yfov = None and bcam.xfov ≠ None
                        lambda :
                            (
                                bcam.yfov * DEG,
                                2 * math.atan(math.tan(bcam.yfov * DEG / 2) * bcam.aspect_ratio)
                            ),
                          # bcam.aspect_ratio ≠ None and bcam.yfov ≠ None and bcam.xfov = None
                        None, # bcam.aspect_ratio ≠ None and bcam.yfov ≠ None and bcam.xfov ≠ None
                  )[
                        (bcam.aspect_ratio != None) << 2
                    |
                        (bcam.yfov != None) << 1
                    |
                        (bcam.xfov != None)
                  ]()
                )
        elif isinstance(bcam.original, OrthographicCamera) :
            b_cam.type = "ORTHO"
            b_cam.ortho_scale = \
                max \
                  (( # “None” marks cases which shouldn’t occur
                        None, # bcam.aspect_ratio = None and bcam.ymag = None and bcam.xmag = None
                        None, # bcam.aspect_ratio = None and bcam.ymag = None and bcam.xmag ≠ None
                        None, # bcam.aspect_ratio = None and bcam.ymag ≠ None and bcam.xmag = None
                        lambda : (bcam.ymag, bcam.xmag),
                          # bcam.aspect_ratio = None and bcam.ymag ≠ None and bcam.xmag ≠ None
                        None, # bcam.aspect_ratio ≠ None and bcam.ymag = None and bcam.xmag = None
                        lambda : (fudge_div(bcam.xmag, bcam.aspect_ratio), bcam.xmag),
                          # bcam.aspect_ratio ≠ None and bcam.ymag = None and bcam.xmag ≠ None
                        lambda : (bcam.ymag, bcam.ymag * bcam.aspect_ratio),
                          # bcam.aspect_ratio ≠ None and bcam.ymag ≠ None and bcam.xmag = None
                        None, # bcam.aspect_ratio ≠ None and bcam.ymag ≠ None and bcam.xmag ≠ None
                  )[
                        (bcam.aspect_ratio != None) << 2
                    |
                        (bcam.ymag != None) << 1
                    |
                        (bcam.xmag != None)
                  ]()
                )
        #end if
        if bcam.znear != None :
            b_cam.clip_start = self._units * bcam.znear
        #end if
        if bcam.zfar != None :
            b_cam.clip_end = self._units * bcam.zfar
        #end if
        self._collection.objects.link(b_obj)
        return b_obj
    #end camera

    def light(self, blight) :
        result = None
        b_name = self.name(DATABLOCK.LAMP, blight.original)
        # todo: shared datablocks
        light_type = tuple \
          (
            elt
            for elt in
                (
                    (AmbientLight, "POINT"),
                    (DirectionalLight, "SUN"),
                    (PointLight, "POINT"),
                    (SpotLight, "SPOT"),
                )
            if isinstance(blight.original, elt[0])
          )
        if len(light_type) != 0 :
            light_type = light_type[0]
            b_light = bpy.data.lights.new(b_name, type = light_type[1])
            b_light.color = blight.original.color[:3]
            if isinstance(blight.original, AmbientLight) :
                # implement as a very large “point” light source
                # Alternatively, could use this to set background intensity instead.
                b_light.shadow_soft_size = 10000 # the larger, the softer the terminators
                b_light.use_shadow = False
                b_light.use_nodes = True # note: Cycles-only
                b_light.cycles.cast_shadow = False
                node_graph = b_light.node_tree
                b_shader = list(n for n in node_graph.nodes if n.type == "EMISSION")[0]
                node_x, node_y = b_shader.location
                falloff = node_graph.nodes.new("ShaderNodeLightFalloff")
                falloff.location = (node_x - 200, node_y)
                falloff.inputs["Strength"].default_value = b_shader.inputs["Strength"].default_value
                node_graph.links.new \
                  (
                    falloff.outputs["Constant"],
                    b_shader.inputs["Strength"],
                  )
            else :
                self.apply_blender_technique \
                  (
                    True,
                    blight.original,
                    b_light,
                    [
                        ("angle", float, "angle"),
                        ("power", float, "energy"),
                        ("shadow_soft_size", float, "shadow_soft_size"),
                        ("spot_blend", float, "spot_blend"),
                        ("spot_size", float, "spot_size"),
                    ]
                  )
            #end if
            b_obj = bpy.data.objects.new(b_name, b_light)
            self._collection.objects.link(b_obj)
            result = b_obj
        #end if
        return result
    #end light

    def geometry(self, bgeom) :

        def collect_from_elts(p, attrname) :
            return list(tuple(getattr(elt, attrname)) for elt in p)
        #end collect_from_elts

        def is_flat_face(normal) :
            a = Vector(normal[0])
            for n in normal[1:] :
                dp = a.dot(Vector(n))
                if dp < 0.99999 or dp > 1.00001 :
                    return False
                #end if
            #end for
            return True
        #end is_flat_face

    #begin geometry
        blendstuff = self.get_blender_technique(True, bgeom.original.xmlnode)
        b_materials = {}
        for sym, matnode in bgeom.materialnodebysymbol.items() :
            mat = matnode.target
            b_matname = self.name(DATABLOCK.MATERIAL, mat)
            if b_matname not in bpy.data.materials :
                b_matname = self.material(mat, b_matname)
            #end if
            b_materials[sym] = bpy.data.materials[b_matname]
        #end for

        if self._transform("APPLY") :
            primitives = bgeom.primitives()
            b_meshname = self.name(DATABLOCK.MESH, bgeom)
        else :
            primitives = bgeom.original.primitives
            b_meshname = self.name(DATABLOCK.MESH, bgeom.original)
        #end if
        materials = []
        new_mesh = self._transform("APPLY") or b_meshname not in bpy.data.meshes
          # FIXME: need to check mesh was one I just imported, rather than something
          # leftover in document.
        if new_mesh :
            verts = []
            vert_starts = {}
            faces = []
            smooth_shade = []
            got_normals = False
            material_assignments = []
            uvcoords = None
            uvcoord_ids = None
            for p in primitives :
                if isinstance(p, BoundPrimitive) :
                    b_mat_key = p.original.material
                else :
                    b_mat_key = p.material
                #end if
                b_mat = b_materials.get(b_mat_key, None)
                materials.append(b_mat)

                if isinstance(p, (TriangleSet, BoundTriangleSet, Polylist, BoundPolylist)) :
                    these_faces = p.vertex_index
                    if these_faces is not None and len(these_faces) != 0 :
                        collect = lambda a : collect_from_elts(p, a)
                        verts_source_id = p.sources["VERTEX"][0][2]
                          # pycollada code only looks at first source if there is
                          # more than one, so I do too
                          # (same for normals)
                        if verts_source_id in vert_starts :
                            vert_start = vert_starts[verts_source_id]
                        else :
                            vert_start = len(verts)
                            vert_starts[verts_source_id] = vert_start
                            verts.extend(tuple(v) for v in p.vertex)
                        #end if
                        these_faces = collect("indices")
                        these_smooth_shade = [False] * len(these_faces)
                        these_material_assignments = [len(materials) - 1] * len(these_faces)
                        has_normals = p.normal is not None
                        if has_normals :
                            # TODO import normals
                            these_normcoords = list(p.normal)
                            these_normindices = collect("normal_indices")
                            for i in range(len(these_faces)) :
                                these_smooth_shade[i] = not is_flat_face \
                                  (
                                    list(these_normcoords[j] for j in these_normindices[i])
                                  )
                            #end for
                            got_normals = True
                        #end if
                        if "TEXCOORD" in p.sources and len(p.sources["TEXCOORD"]) != 0 :
                            if uvcoords == None :
                                uvcoords = \
                                    [
                                        [[(0, 0)] * len(f) for f in faces]
                                          # pad out with dummies for any prior missing entries
                                        for i in range(len(p.sources["TEXCOORD"]))
                                    ]
                                uvcoord_ids = tuple(s[2] for s in p.sources["TEXCOORD"])
                            else :
                                assert len(uvcoords) == len(p.sources["TEXCOORD"]), \
                                  (
                                        "mismatch in number of UV layers between geometry"
                                        " components: %d vs %d"
                                    %
                                        (len(uvcoords), len(p.sources["TEXCOORD"]))
                                  )
                                assert uvcoord_ids == tuple(s[2] for s in p.sources["TEXCOORD"]), \
                                  (
                                        "mismatch between IDs of UV layers between geometry"
                                        " components: %s vs %s"
                                    %
                                        (uvcoord_ids, tuple(s[2] for s in p.sources["TEXCOORD"]))
                                  )
                            #end if
                            assert len(p) == len(these_faces), \
                                "mismatch in number of faces in geometry component"
                            for face in p :
                                for layer, coords in zip(uvcoords, face.texcoords) :
                                    layer.append(list(tuple(v) for v in coords))
                                #end for
                            #end for
                        elif uvcoords != None :
                            # pad out with dummies for missing entry
                            for face in p :
                                for layer in uvcoords :
                                    layer.append([(0, 0)] * len(face.vertices))
                                #end for
                            #end for
                        #end if
                        faces.extend(tuple(i + vert_start for i in f) for f in these_faces)
                        smooth_shade.extend(these_smooth_shade)
                        material_assignments.extend(these_material_assignments)
                    #end if
                else :
                    pass # can’t handle [Bound]Polygons for now
                #end if
            #end for

            b_mesh = bpy.data.meshes.new(b_meshname)
            b_mesh.from_pydata \
              (
                self._convert_units_verts(verts),
                [],
                faces
              )
            if got_normals :
                for i, f in enumerate(b_mesh.polygons) :
                    f.use_smooth = smooth_shade[i]
                #end for
            #end if
            if uvcoords != None :
                uv_layers_names = {}
                if blendstuff != None :
                    layer_names = blendstuff.find(tag("layer_names"))
                    if layer_names != None :
                        for name_entry in layer_names.findall(tag("name")) :
                            if name_entry.get("type") == "UV" :
                                layer_name = name_entry.get("name")
                                layer_refid = name_entry.get("refid")
                                if layer_name != None and layer_refid != None :
                                    uv_layers_names[layer_refid] = layer_name
                                #end if
                            #end if
                        #end for
                    #end if
                #end if
                b_mesh_loops = b_mesh.loops
                for layer, refid in zip(uvcoords, uvcoord_ids) :
                    layer_name = uv_layers_names.get(unurlid(refid))
                    uv = b_mesh.uv_layers.new()
                    if layer_name != None :
                        uv.name = layer_name
                    #end if
                    uv_data = uv.data
                    for i, face in enumerate(b_mesh.polygons) :
                        loop_start = face.loop_start
                        coords = layer[i]
                        for j in range(face.loop_total) :
                            uv_data[loop_start + j].uv = coords[j]
                        #end for
                    #end for
                #end for
            #end if
            for i, face in enumerate(b_mesh.polygons) :
                face.material_index = material_assignments[i]
            #end for
            b_mesh.update()
        else :
            b_mesh = bpy.data.meshes[b_meshname]
            for p in primitives :
                if isinstance(p, BoundPrimitive) :
                    b_mat_key = p.original.material
                else :
                    b_mat_key = p.material
                #end if
                b_mat = b_materials.get(b_mat_key, None)
                materials.append(b_mat)
            #end for
        #end if

        b_obj = bpy.data.objects.new(b_meshname, b_mesh)
        b_obj.data = b_mesh
        self._collection.objects.link(b_obj)
        self._ctx.view_layer.objects.active = b_obj
        for i, m in enumerate(materials) :
            if new_mesh :
                bpy.ops.object.material_slot_add()
            #end if
            b_obj.material_slots[i].link = "OBJECT"
            b_obj.material_slots[i].material = m
        #end for

        if self._transform("APPLY") :
            # TODO import normals
            bpy.ops.object.mode_set(mode = "EDIT")
            bpy.ops.mesh.normals_make_consistent()
            bpy.ops.object.mode_set(mode = "OBJECT")
        #end if

        return b_obj
    #end geometry

    obj_type_handlers = \
        [
            ("camera", camera, CameraNode),
            ("light", light, LightNode),
            ("geometry", geometry, GeometryNode),
        ]

    class Material :
        "interpretation of Collada material settings. Can be subclassed by" \
        " importer subclasses."

        def __init__(self, parent, mat, b_name) :
            self.parent = parent
            self.tempdir = None
            rendering = \
                { # initialize at instantiation time to allow overriding by subclasses
                    "blinn" : self.rendering_blinn,
                    "constant" : self.rendering_constant,
                    "lambert" : self.rendering_lambert,
                    "phong" : self.rendering_phong,
                }
            # self.mat = mat # not needed
            self.images = {}
            effect = mat.effect
            self.effect = effect
            b_mat = bpy.data.materials.new(b_name)
            self.b_mat = b_mat
            self.name = b_mat.name # name actually assigned by Blender
            b_mat.use_nodes = True
            b_shader = list(n for n in b_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")[0]
            self.b_shader = b_shader
            self.node_x, self.node_y = b_shader.location
            self.node_x -= 350
            self.node_y += 200
            self.tex_coords_src = None
            rendering[effect.shadingtype]()
            b_mat.use_backface_culling = not effect.double_sided
            if isinstance(effect.emission, tuple) :
                b_shader.inputs["Emission"].default_value = effect.emission
            # Map option NYI for now
            #end if
            self.rendering_transparency()
            self.rendering_reflectivity()
            self.rendering_emission()
        #end __init__

        def rendering_constant(self) :
            self.color_or_texture(self.effect.diffuse, "diffuse", "Emission")
        #end rendering_constant

        def rendering_lambert(self) :
            self.rendering_diffuse()
            inputs = self.b_shader.inputs
            inputs["Specular"].default_value = 0
            inputs["Metallic"].default_value = 0
            inputs["Roughness"].default_value = 1
        #end rendering_lambert

        def rendering_phong(self) :
            self.rendering_diffuse()
            self.rendering_specular(False)
        #end rendering_phong

        def rendering_blinn(self) :
            self.rendering_diffuse()
            self.rendering_specular(True)
        #end rendering_blinn

        def rendering_diffuse(self) :
            self.color_or_texture(self.effect.diffuse, "diffuse", "Base Color", True)
        #end rendering_diffuse

        def rendering_specular(self, blinn = False) :
            # for the difference between Blinn (actually Blinn-Phong) and Phong shaders,
            # see <https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model>
            effect = self.effect
            b_shader = self.b_shader
            if isinstance(effect.specular, tuple) :
                b_shader.inputs["Specular"].default_value = 1.0
                b_shader.inputs["Base Color"].default_value = effect.specular
                  # might clash with diffuse colour, but hey
            # Map option NYI for now
            #end if
            if isinstance(effect.shininess, Real) :
                b_shader.inputs["Roughness"].default_value = \
                    (1, 1 / 4)[blinn] / (1 + effect.shininess)
            # Map option NYI for now
            #end if
        #end rendering_specular

        def rendering_reflectivity(self) :
            effect = self.effect
            b_shader = self.b_shader
            if isinstance(effect.reflectivity, Real) and effect.reflectivity > 0 :
                b_shader.inputs["Specular"].default_value = effect.reflectivity
                if effect.reflective != None :
                    self.color_or_texture(effect.reflective, "reflective", "Base Color")
                      # might clash with diffuse colour, but hey
                #end if
            #end if
        #end rendering_reflectivity

        def rendering_transparency(self) :
            effect = self.effect
            if effect.transparency == None :
                return
            opaque_mode = effect.opaque_mode
            flip = opaque_mode in ("A_ZERO", "RGB_ZERO")
            # RGB_ONE/ZERO opacity modes NYI, treat as A_ONE/ZERO modes for now
            b_mat = self.b_mat
            b_shader = self.b_shader
            if isinstance(effect.transparency, Real) :
                alpha = effect.transparency
                if flip :
                    alpha = 1 - alpha
                #end if
                if self.parent._ctx.scene.render.engine == "CYCLES" :
                    # This setting is ignored by Eevee
                    b_shader.inputs["Transmission"].default_value = 1 - alpha
                else :
                    # This setting would affect Cycles as well,
                    # which is why I don’t do both.
                    b_shader.inputs["Alpha"].default_value = alpha
                #end if
                if alpha < 1.0 :
                    b_mat.blend_method = "BLEND"
                    b_mat.diffuse_color[3] = alpha
                      # takes effect in viewport (Workbench renderer)
                #end if
            #end if
            if isinstance(effect.index_of_refraction, Real) :
                b_shader.inputs["IOR"].default_value = effect.index_of_refraction
            #end if
        #end rendering_transparency

        def rendering_emission(self) :
            self.color_or_texture(self.effect.emission, "emission", "Emission")
        #end rendering_emission

        def color_or_texture(self, color_or_texture, tex_name, shader_input_name, set_mat_color = False) :

            def try_texture(c_image) :
                basename = os.path.basename(c_image.path)
                imgfile_name = os.path.join(self.create_tempdir(), basename)
                image = None # to begin with
                if isinstance(c_image.data, bytes) :
                    imgfile = open(imgfile_name, "wb")
                    imgfile.write(c_image.data)
                    imgfile.close()
                    imgfile = None
                    try :
                        image = bpy.data.images.load(imgfile_name)
                    except RuntimeError as fail :
                        sys.stderr.write \
                          (
                                "Error trying to load image file %s from %s: %s\n"
                            %
                                (repr(c_image.path), repr(imgfile_name), str(fail))
                          )
                    #end try
                else :
                    sys.stderr.write \
                      (
                            "No data %s for image file %s\n"
                        %
                            (repr(c_image.data), repr(c_image.path))
                      )
                #end if
                if image != None :
                    node_graph = self.b_mat.node_tree
                    image.pack()
                    # wipe all traces of original file path
                    image.filepath = "//textures/%s" % basename
                    image.filepath_raw = image.filepath
                    for item in image.packed_files :
                        item.filepath = image.filepath
                    #end for
                    # todo: use image alpha as shader alpha (diffuse texture only)
                    tex_image = node_graph.nodes.new("ShaderNodeTexImage")
                    tex_image.location = (self.node_x, self.node_y)
                    self.node_y -= 200
                    tex_image.image = image
                    if self.tex_coords_src == None :
                        tex_coords_node = node_graph.nodes.new("ShaderNodeTexCoord")
                        tex_coords_node.location = (self.node_x - 400, self.node_y)
                        fanout_node = node_graph.nodes.new("NodeReroute")
                        fanout_node.location = (self.node_x - 200, self.node_y - 200)
                        node_graph.links.new \
                          (
                            tex_coords_node.outputs["UV"],
                            fanout_node.inputs[0]
                          )
                        self.tex_coords_src = fanout_node.outputs[0]
                    #end if
                    node_graph.links.new(self.tex_coords_src, tex_image.inputs[0])
                    self.images[tex_name] = image
                    mtex = tex_image.outputs["Color"]
                else :
                    mtex = None
                #end if
                # could delete imgfile_name at this point
                return mtex
            #end try_texture

        #begin color_or_texture
            if isinstance(color_or_texture, Map) :
                image = color_or_texture.sampler.surface.image
                mtex = try_texture(image)
                if mtex == None :
                    mtex = (1, 0, 1, 1) # same hideous colour Blender uses
                #end if
            elif isinstance(color_or_texture, tuple) :
                mtex = color_or_texture
            else :
                mtex = None
            #end if
            shader_input = self.b_shader.inputs[shader_input_name]
            if isinstance(mtex, tuple) :
                shader_input.default_value = mtex
                if set_mat_color :
                    self.b_mat.diffuse_color[:3] = mtex[:3]
                #end if
            elif isinstance(mtex, bpy.types.NodeSocket) :
                self.b_mat.node_tree.links.new(mtex, shader_input)
            #end if
        #end color_or_texture

        def create_tempdir(self) :
            if self.tempdir == None :
                self.tempdir = tempfile.mkdtemp(prefix = "bpycollada-import-")
            #end if
            return self.tempdir
        #end create_tempdir

        def cleanup_tempdir(self) :
            if self.tempdir != None :
                shutil.rmtree(self.tempdir, ignore_errors = True)
                self.tempdir = None
            #end if
        #end cleanup_tempdir

    #end Material

    def material(self, mat, b_name) :
        matctx = type(self).Material(self, mat, b_name)
          # all material setup happens here
        matctx.cleanup_tempdir()
        return matctx.name
    #end material

    def parent_node(self, node, parent, node_matrix = None) :
        if isinstance(node, (Node, NodeNode)) :
            b_obj = bpy.data.objects.new(self.name(DATABLOCK.EMPTY, node), None)
            b_obj.matrix_world = self._convert_units_matrix(Matrix(node.matrix))
            if node_matrix != None :
                b_obj.matrix_world = node_matrix @ b_obj.matrix_world
            #end if
            self._collection.objects.link(b_obj)
            if parent != None :
                b_obj.parent = parent
            #end if
            parent = b_obj
        else :
            handle_type = tuple(h for h in self.obj_type_handlers if isinstance(node, h[2]))
            if len(handle_type) != 0 :
                handle_type = handle_type[0]
                bobj = list(node.objects(handle_type[0]))
                assert len(bobj) == 1
                bobj = bobj[0]
                b_obj = handle_type[1](self, bobj)
                if b_obj != None :
                    if node_matrix != None :
                        b_obj.matrix_world = node_matrix @ b_obj.matrix_world
                    #end if
                    b_obj.parent = parent
                    parent = b_obj
                #end if
            #end if
        #end if
        return parent
    #end parent_node

    @classmethod
    def match(celf, collada) :
        return True
   #end match

#end ColladaImport

class SketchUpImport(ColladaImport) :
    "SketchUp specific COLLADA import."

    SK_DAE_NS = {"dae" : "http://www.collada.org/2005/11/COLLADASchema"}
      # SketchUp only uses Collada 1.4.1, as far as I know

    class Material(ColladaImport.Material) :
        "SketchUp-specific material handling."

        def rendering_phong(self) :
            super().rendering_lambert()
        #end rendering_phong

        def rendering_transparency(self) :
            effect = self.effect
            # get opaque_mode setting direct from XML, avoiding pycollada-provided default
            transparent = effect.xmlnode.find(".//" + tag("transparent"))
            if transparent != None :
                opaque_mode = transparent.get("opaque")
            else :
                opaque_mode = None
            #end if
            # fudge for some disappearing SketchUp models
            if (
                    opaque_mode == None
                and
                    isinstance(effect.transparent, tuple)
                and
                    isinstance(effect.transparency, Real)
                and
                    tuple(effect.transparent) == (1, 1, 1, 1)
                and
                    effect.transparency == 0
            ) :
                effect.transparency = 1
            #end if
            super().rendering_transparency()
        #end rendering_transparency

        def rendering_reflectivity(self) :
            "There are no reflectivity controls in SketchUp."
            if not self.parent.match_test2(self.effect.xmlnode) :
                super().rendering_reflectivity()
            #end if
        #end rendering_reflectivity

    #end Material

    @classmethod
    def match_test2(celf, xml) :
        return \
            any \
              (
                t.get("profile") == "GOOGLEEARTH"
                for t in xml.findall(".//dae:extra/dae:technique", namespaces = celf.SK_DAE_NS)
              )
    #end match_test2

    @classmethod
    def match(celf, collada) :
        "Does this look like a Collada file from SketchUp."

        def test1(xml) :
            t1 = xml.find(".//dae:instance_visual_scene", namespaces = celf.SK_DAE_NS)
            if t1 != None :
                t1 = t1.get("url")
            #end if
            t2 = xml.find(".//dae:authoring_tool", namespaces = celf.SK_DAE_NS)
            if t2 != None :
                t2 = t2.text
            #end if
            return \
                any \
                  (
                    "SketchUp" in s
                    for s in (t1, t2)
                    if s != None
                  )
        #end test1

    #begin match
        xml = collada.xmlnode
        return test1(xml) or celf.match_test2(xml)
    #end match

#end SketchUpImport

VENDOR_SPECIFIC = \
    [
        SketchUpImport,
    ]

def get_import(collada) :
    "returns a suitable importer for the given Collada object according" \
    " to any vendor-specific features found."
    for i in VENDOR_SPECIFIC :
        if i.match(collada) :
            return i
    #end for
    return ColladaImport
#end get_import

def load(op, ctx, is_zae, filepath, **kwargs) :

    def get_obj_matrix(obj) :

        def direction_matrix(direction) :
            # calculation follows an answer from
            # <https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d>
            reference = Vector((0, 0, -1))
            direction = Vector(tuple(direction))
            direction.resize_3d()
            direction.normalize()
            cross = reference.cross(direction)
            fac = Matrix \
              (
                [
                    [0, - cross.z, cross.y, 0],
                    [cross.z, 0, - cross.x, 0],
                    [- cross.y, cross.x, 0, 0,],
                    [0, 0, 0, 1]
                ]
              )
            try :
                result = \
                  (
                        Matrix.Identity(4)
                    +
                        fac
                    +
                        1 / (1 + reference @ direction) * (fac @ fac)
                  )
            except ZeroDivisionError :
                result = Matrix.Rotation(180 * DEG, 4, "X")
                  # actually any rotation axis in plane perpendicular to reference will work
            #end try
            return result
        #end direction_matrix

    #begin get_obj_matrix
        # fixme: BoundSpotLight also has an up direction vector I should probably take into account
        if hasattr(obj, "matrix") :
            result = Matrix(obj.matrix)
        elif hasattr(obj, "position") or hasattr(obj, "direction") :
            result = Matrix.Identity(4)
            if hasattr(obj, "direction") :
                result = direction_matrix(obj.direction)
            #end if
            if hasattr(obj, "position") :
                result = Matrix.Translation(obj.position) @ result
            #end if
        else :
            result = None
        #end if
        return result
    #end get_obj_matrix

    last_update = None
    update_interval = 5
    obj_count = nr_objs = 0

    def traverse_children(self, node, action, parent) :
        nonlocal last_update, obj_count
        obj_count += 1
        now = time.time()
        if now - last_update >= update_interval :
            #sys.stderr.write("created %d/%d objects\n" % (obj_count, nr_objs))
              # nr_objs not computed accurately (see below)
            sys.stderr.write("created %d objects\n" % obj_count)
            last_update = now
        #end if
        children = ()
        empty_children = ()
        nonempty_children = ()
        node_matrix = None
        rule = tuple \
          (
            entry
            for entry in
                (
                    (Scene, lambda node : node.nodes, False),
                    (Node, lambda node : node.children, True),
                    (NodeNode, lambda node : node.node.children, True),
                )
            if isinstance(node, entry[0])
          )
        if len(rule) != 0 :
            rule = rule[0]
            children = rule[1](node)
            if rule[2] :
                empty_children = tuple \
                  ( # children which would be represented as Empty objects
                    c
                    for c in children
                    if isinstance(c, (Node, NodeNode))
                  )
                nonempty_children = tuple \
                  ( # children which would be presented as objects other than Empties
                    c
                    for c in children
                    if isinstance(c, (CameraNode, GeometryNode, LightNode)) # ControllerNode NYI
                  )
                node_matrix = self._convert_units_matrix(Matrix(node.matrix))
            #end if
        #end if
        if node_matrix != None and len(nonempty_children) == 1 :
            # make the nonempty child the parent of the other children,
            # instead of creating an Empty for this Node.
            new_parent = action(nonempty_children[0], parent, node_matrix)
            for child in empty_children :
                traverse_children(self, child, action, new_parent)
            #end for
        else :
            # create an Empty for this Node.
            new_parent = action(node, parent)
            for child in children :
                traverse_children(self, child, action, new_parent)
            #end for
        #end if
    #end traverse_children

#begin load
    start_time = time.time()
    collada_ignore = [DaeBrokenRefError]
    if is_zae :
        zip = zipfile.ZipFile(filepath)
        manifest = zip.read("manifest.xml")
        manifest = ElementTree.ElementTree(file = io.BytesIO(manifest))
        dae_root = manifest.getroot().text
          # TODO: interpret fragment part, if any
          # TODO: archive can contain nested sub-archives
        c = Collada \
          (
            filename = io.BytesIO(zip.read(dae_root)),
            aux_file_loader = zip.read,
            ignore = collada_ignore
          )
    else :
        c = Collada(filepath, ignore = collada_ignore)
    #end if
    now = time.time()
    sys.stderr.write("Time to load .dae file = %.2fs\n" % (now - start_time))
    start_time = now
    importer = get_import(c)(ctx, c, filepath, **kwargs)
    tf = importer._transformation
    if tf in ("MUL", "APPLY") :
        for handle_type in importer.obj_type_handlers :
            objs = list(c.scene.objects(handle_type[0]))
            nr_objs = len(objs)
            last_update = start_time
            for i, obj in enumerate(objs) :
                b_obj = handle_type[1](importer, obj)
                now = time.time()
                if now - last_update >= update_interval :
                    sys.stderr.write("created %s objects %d/%d\n" % (handle_type[0], i, nr_objs))
                    if i - obj_count < 5 :
                        sys.stderr.write(" obj “%s”\n" % b_obj.name)
                    #end if
                    last_update = now
                    obj_count = i
                #end if
                if tf == "MUL" :
                    tf_mat = get_obj_matrix(obj)
                    if tf_mat != None :
                        tf_mat = importer._orient @ importer._convert_units_matrix(tf_mat)
                        b_obj.matrix_world = tf_mat
                    #end if
                #end if
            #end for
        #end for
    elif tf == "PARENT" :
        nr_objs = sum \
          (
            len(c.xmlnode.findall(".//dae:%s" % t, namespaces = importer.DAE_NS))
            for t in
              (
                "node", "instance_node",
                "instance_camera", "instance_geometry", "instance_light",
                "camera", "geometry", "light",
              )
          ) # fixme: not sure how to come up with a plausible value for this
        last_update = start_time
        traverse_children(importer, c.scene, importer.parent_node, None)
    #end if
    now = time.time()
    sys.stderr.write("Time to import to Blender = %.2fs\n" % (now - start_time))
    return {"FINISHED"}
#end load
