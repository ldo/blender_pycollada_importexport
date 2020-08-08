import sys
import os
import math
from numbers import \
    Real
import tempfile
import shutil

import bpy
from bpy_extras.image_utils import load_image
from mathutils import Matrix, Vector

from collada import Collada
from collada.camera import PerspectiveCamera, OrthographicCamera
from collada.common import DaeBrokenRefError
from collada.light import AmbientLight, DirectionalLight, PointLight, SpotLight
from collada.material import Map
from collada.polylist import Polylist, BoundPolylist
from collada.primitive import BoundPrimitive
from collada.scene import Scene, Node, NodeNode, GeometryNode
from collada.triangleset import TriangleSet, BoundTriangleSet

__all__ = ["load"]

VENDOR_SPECIFIC = []
COLLADA_NS      = "http://www.collada.org/2005/11/COLLADASchema"
DAE_NS          = {"dae": COLLADA_NS}
MAX_NAME_LENGTH = 63
DEG = math.pi / 180

def load(op, ctx, filepath = None, **kwargs) :
    c = Collada(filepath, ignore = [DaeBrokenRefError])
    impclass = get_import(c)
    imp = impclass(ctx, c, os.path.dirname(filepath), **kwargs)
    tf = kwargs["transformation"]
    if tf in ("MUL", "APPLY") :
        for i, obj in enumerate(c.scene.objects("geometry")) :
            b_geoms = imp.geometry(obj)
            if tf == "MUL" :
                tf_mat = imp._convert_units_matrix(Matrix(obj.matrix))
                for b_obj in b_geoms :
                    b_obj.matrix_world = tf_mat
                #end for
            #end if
        #end for
    elif tf == "PARENT" :
        _dfs(c.scene, imp.node)
    #end if
    for i, obj in enumerate(c.scene.objects("light")) :
        imp.light(obj, i)
    #end for
    for obj in c.scene.objects("camera") :
        imp.camera(obj)
    #end for
    return {"FINISHED"}
#end load

def get_import(collada) :
    "returns a suitable importer for the given Collada object according" \
    " to any vendor-specific features found."
    for i in VENDOR_SPECIFIC :
        if i.match(collada) :
            return i
    #end for
    return ColladaImport
#end get_import

class ColladaImport :
    "Standard COLLADA importer. Subclasses can implement a “match” method" \
    " to identify vendor-specific features they need to handle."

    def __init__(self, ctx, collada, basedir, **kwargs) :
        self._ctx = ctx
        self._collada = collada
        self._kwargs = kwargs
        self._namecount = 0
        self._names = {}
        self._units = collada.assetInfo.unitmeter
        if self._units == None :
            self._units = 1
        #end if
    #end __init__

    def _convert_units_matrix(self, mat) :
        v_pos, q_rot, v_scale = mat.decompose()
        return Matrix.Translation(self._units * v_pos) @ q_rot.to_matrix().to_4x4() @ Matrix.Diagonal(v_scale).to_4x4()
    #end _convert_units_matrix

    def _convert_units_verts(self, verts) :
        return \
            list(self._units * Vector(v) for v in verts)
    #end _convert_units_verts

    def camera(self, bcam) :
        bpy.ops.object.add(type = "CAMERA")
        b_obj = self._ctx.object
        b_obj.name = self.name(bcam.original, id(bcam))
        b_obj.matrix_world = self._convert_units_matrix(Matrix(bcam.matrix))
        b_cam = b_obj.data
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
                                2 * math.atan(math.tan(bcam.xfov * DEG / 2) / bcam.aspect_ratio),
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
                        lambda : (bcam.xmag / bcam.aspect_ratio, bcam.xmag),
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
            b_cam.clip_start = bcam.znear
        #end if
        if bcam.zfar != None :
            b_cam.clip_end = bcam.zfar
        #end if
    #end camera

    def geometry(self, bgeom) :
        b_materials = {}
        for sym, matnode in bgeom.materialnodebysymbol.items() :
            mat = matnode.target
            b_matname = self.name(mat)
            if b_matname not in bpy.data.materials :
                b_matname = self.material(mat, b_matname)
            #end if
            b_materials[sym] = bpy.data.materials[b_matname]
        #end for

        primitives = bgeom.original.primitives
        if self._transform("APPLY") :
            primitives = bgeom.primitives()
        #end if

        b_geoms = []
        for i, p in enumerate(primitives) :
            if isinstance(p, BoundPrimitive) :
                b_mat_key = p.original.material
            else :
                b_mat_key = p.material
            #end if
            b_mat = b_materials.get(b_mat_key, None)
            b_meshname = self.name(bgeom.original, i)

            if isinstance(p, (TriangleSet, BoundTriangleSet)) :
                b_mesh = self.geometry_triangleset(p, b_meshname, b_mat)
            elif isinstance(p, (Polylist, BoundPolylist)) :
                b_mesh = self.geometry_triangleset(p.triangleset(), b_meshname, b_mat)
            else :
                continue
            #end if
            if not b_mesh :
                continue

            b_obj = bpy.data.objects.new(b_meshname, b_mesh)
            b_obj.data = b_mesh

            self._ctx.scene.collection.objects.link(b_obj)
            self._ctx.view_layer.objects.active = b_obj

            if len(b_obj.material_slots) == 0 :
                bpy.ops.object.material_slot_add()
            #end if
            b_obj.material_slots[0].link = "OBJECT"
            b_obj.material_slots[0].material = b_mat
            b_obj.active_material = b_mat

            if self._transform("APPLY") :
                # TODO import normals
                bpy.ops.object.mode_set(mode = "EDIT")
                bpy.ops.mesh.normals_make_consistent()
                bpy.ops.object.mode_set(mode = "OBJECT")
            #end if

            b_geoms.append(b_obj)
        #end for

        return b_geoms
    #end geometry

    def geometry_triangleset(self, triset, b_name, b_mat) :
        if not self._transform("APPLY") and b_name in bpy.data.meshes :
            # with applied transformation, mesh reuse is not possible
            return bpy.data.meshes[b_name]
        else :
            if triset.vertex_index is None or not len(triset.vertex_index) :
                return

            b_mesh = bpy.data.meshes.new(b_name)
            b_mesh.from_pydata \
              (
                self._convert_units_verts(triset.vertex),
                [],
                [((v3, v1, v2), (v1, v2, v3))[v3 != 0]
                  # is this “eekadadoodle” rearrangement really necessary?
                    for f in triset.vertex_index
                    for v1, v2, v3 in (f,)
                ]
              )

            has_normal = triset.normal_index is not None
            has_uv = len(triset.texcoord_indexset) > 0
            if has_normal :
                # TODO import normals
                for i, f in enumerate(b_mesh.polygons) :
                    f.use_smooth = not _is_flat_face(triset.normal[triset.normal_index[i]])
                #end for
            #end if
            if has_uv :
                for j in range(len(triset.texcoord_indexset)) :
                    self.texcoord_layer \
                      (
                        triset,
                        triset.texcoordset[j],
                        triset.texcoord_indexset[j],
                        b_mesh
                      )
                #end for
            #end if
            b_mesh.update()
            return b_mesh
        #end if
    #end geometry_triangleset

    def texcoord_layer(self, triset, texcoord, index, b_mesh) :
        uv = b_mesh.uv_layers.new()
        for i, f in enumerate(b_mesh.polygons) :
            t1, t2, t3 = index[i]
            # eekadoodle -- is this really necessary?
            if triset.vertex_index[i][2] == 0 :
                t1, t2, t3 = t3, t1, t2
            #end if
            j = f.loop_start
            uv.data[j].uv = texcoord[t1]
            uv.data[j + 1].uv = texcoord[t2]
            uv.data[j + 2].uv = texcoord[t3]
        #end for
    #end texcoord_layer

    def light(self, light, i) :

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

        def position_direction_matrix(position, direction) :
            return \
                Matrix.Translation(self._units * position) @ direction_matrix(direction)
        #end position_direction_matrix

    #begin light
        if isinstance(light.original, AmbientLight) :
            return
        b_name = self.name(light.original, i)
        if b_name not in bpy.data.lights :
            light_type = tuple \
              (
                elt
                for elt in
                    (
                        (DirectionalLight, "SUN", "direction", direction_matrix),
                        (PointLight, "POINT", "position", Matrix.Translation),
                          # note Collada common profile doesn’t support
                          # direction-dependent light intensity
                        (SpotLight, "SPOT", ("position", "direction"), position_direction_matrix),
                    )
                if isinstance(light.original, elt[0])
              )
            if len(light_type) != 0 :
                light_type = light_type[0]
                b_lamp = bpy.data.lights.new(b_name, type = light_type[1])
                b_obj = bpy.data.objects.new(b_name, b_lamp)
                if isinstance(light_type[2], tuple) :
                    args = tuple(getattr(light, a) for a in light_type[2])
                else :
                    args = (getattr(light, light_type[2]),)
                #end if
                b_obj.matrix_world = light_type[3](*args)
                self._ctx.scene.collection.objects.link(b_obj)
            #end if
        #end if
    #end light

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
            transparent_shadows = self.parent._kwargs.get("transparent_shadows", False)
            b_mat.shadow_method = ("OPAQUE", "HASHED")[transparent_shadows]
              # best I can do for non-Cycles
            b_mat.cycles.use_transparent_shadow = transparent_shadows
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
                b_shader.inputs["Alpha"].default_value = alpha
                if alpha < 1.0 :
                    b_mat.blend_method = "BLEND"
                    b_mat.diffuse_color[3] = alpha
                #end if
            #end if
            b_shader.inputs["Transmission"].default_value = \
                (0.0, 1.0)[self.parent._kwargs.get("raytrace_transparency", False)]
            if isinstance(effect.index_of_refraction, Real) :
                b_shader.inputs["IOR"].default_value = effect.index_of_refraction
            #end if
        #end rendering_transparency

        def rendering_emission(self) :
            self.color_or_texture(self.effect.emission, "emission", "Emission")
        #end rendering_emission

        def color_or_texture(self, color_or_texture, tex_name, shader_input_name, set_mat_color = False) :

            def try_texture(c_image) :
                basename = os.path.split(c_image.path)[1]
                imgfile_name = os.path.join(self.create_tempdir(), basename)
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
                    image = None
                #end try
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

    def node(self, node, parent) :
        if isinstance(node, (Node, NodeNode)) :
            b_obj = bpy.data.objects.new(self.name(node), None)
            b_obj.matrix_world = self._convert_units_matrix(Matrix(node.matrix))
            self._ctx.scene.collection.objects.link(b_obj)
            if parent != None :
                b_obj.parent = parent
            #end if
            parent = b_obj
        elif isinstance(node, GeometryNode) :
            for bgeom in node.objects("geometry") :
                b_geoms = self.geometry(bgeom)
                for b_obj in b_geoms :
                    b_obj.parent = parent
                #end for
            #end for
        #end if
        return parent
    #end node

    def name(self, obj, index = 0) :
        """ Trying to get efficient and human readable name, workarounds
        Blender's object name limitations.
        """
        if hasattr(obj, "id") :
            uid = obj.id.replace("material", "m")
        else :
            self._namecount += 1
            uid = "Untitled." + str(self._namecount)
        #end if
        base = "%s-%d" % (uid, index)
        if base not in self._names :
            self._namecount += 1
            suffix = "-%.4d" % self._namecount
            self._names[base] = base[:MAX_NAME_LENGTH - len(suffix)] + suffix
        #end if
        return self._names[base]
    #end name

    def _transform(self, t) :
        return self._kwargs["transformation"] == t
    #end _transform

#end ColladaImport

class SketchUpImport(ColladaImport) :
    "SketchUp specific COLLADA import."

    class Material(ColladaImport.Material) :
        "SketchUp-specific material handling."

        def rendering_phong(self) :
            super().rendering_lambert()
        #end rendering_phong

        def rendering_transparency(self) :
            effect = self.effect
            # fudge for some disappearing SketchUp models
            if (
                    effect.opaque_mode == "A_ONE"
                      # actually problem SketchUp files leave this unspecified
                and
                    effect.transparent != None
                and
                    effect.transparency != None
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

    @staticmethod
    def match_test2(xml) :
        return \
            any \
              (
                t.get("profile") == "GOOGLEEARTH"
                for t in xml.findall(".//dae:extra/dae:technique", namespaces = DAE_NS)
              )
    #end match_test2

    @classmethod
    def match(celf, collada) :
        "Does this look like a Collada file from SketchUp."

        def test1(xml) :
            return \
                any \
                  (
                    "SketchUp" in s
                    for s in
                        (
                            xml.find(".//dae:instance_visual_scene", namespaces = DAE_NS).get("url"),
                            xml.find(".//dae:authoring_tool", namespaces = DAE_NS),
                        )
                    if s != None
                  )
        #end test1

    #begin match
        xml = collada.xmlnode
        return test1(xml) or celf.match_test2(xml)
    #end match

#end SketchUpImport

VENDOR_SPECIFIC.append(SketchUpImport)

def _is_flat_face(normal) :
    a = Vector(normal[0])
    for n in normal[1:] :
        dp = a.dot(Vector(n))
        if dp < 0.99999 or dp > 1.00001 :
            return False
        #end if
    #end for
    return True
#end _is_flat_face

def _children(node) :
    if isinstance(node, Scene) :
        return node.nodes
    elif isinstance(node, Node) :
        return node.children
    elif isinstance(node, NodeNode) :
        return node.node.children
    else :
        return []
    #end if
#end _children

def _dfs(node, cb, parent = None) :
    """ Depth first search taking a callback function.
    Its return value will be passed recursively as a parent argument.

    :param node: COLLADA node
    :param callable cb:
     """
    parent = cb(node, parent)
    for child in _children(node) :
        _dfs(child, cb, parent)
    #end for
#end _dfs
