import os
import math
from numbers import \
    Real
from tempfile import NamedTemporaryFile
from contextlib import contextmanager

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


__all__ = ['load']

VENDOR_SPECIFIC = []
COLLADA_NS      = 'http://www.collada.org/2005/11/COLLADASchema'
DAE_NS          = {'dae': COLLADA_NS}
TRANSPARENCY_RAY_DEPTH = 8
MAX_NAME_LENGTH        = 27


def load(op, ctx, filepath=None, **kwargs):
    c = Collada(filepath, ignore=[DaeBrokenRefError])
    impclass = get_import(c)
    imp = impclass(ctx, c, os.path.dirname(filepath), **kwargs)

    tf = kwargs['transformation']

    if tf in ('MUL', 'APPLY'):
        for i, obj in enumerate(c.scene.objects('geometry')):
            b_geoms = imp.geometry(obj)
            if tf == 'MUL':
                tf_mat = Matrix(obj.matrix)
                for b_obj in b_geoms:
                    b_obj.matrix_world = tf_mat
    elif tf == 'PARENT':
        _dfs(c.scene, imp.node)

    for i, obj in enumerate(c.scene.objects('light')):
        imp.light(obj, i)

    for obj in c.scene.objects('camera'):
        imp.camera(obj)

    return {'FINISHED'}

def get_import(collada):
    for i in VENDOR_SPECIFIC:
        if i.match(collada):
            return i
    return ColladaImport


class ColladaImport:
    """ Standard COLLADA importer. """

    def __init__(self, ctx, collada, basedir, **kwargs):
        self._ctx = ctx
        self._collada = collada
        self._kwargs = kwargs
        self._namecount = 0
        self._names = {}
    #end __init__

    def camera(self, bcam):
        bpy.ops.object.add(type='CAMERA')
        b_obj = self._ctx.object
        b_obj.name = self.name(bcam.original, id(bcam))
        b_obj.matrix_world = Matrix(bcam.matrix)
        b_cam = b_obj.data
        if isinstance(bcam.original, PerspectiveCamera):
            b_cam.type = 'PERSP'
            prop = b_cam.bl_rna.properties.get('lens_unit')
            if 'DEGREES' in prop.enum_items:
                b_cam.lens_unit = 'DEGREES'
            elif 'FOV' in prop.enum_items:
                b_cam.lens_unit = 'FOV'
            else:
                b_cam.lens_unit = prop.default
            b_cam.angle = math.radians(max(
                    bcam.xfov or bcam.yfov,
                    bcam.yfov or bcam.xfov))
        elif isinstance(bcam.original, OrthographicCamera):
            b_cam.type = 'ORTHO'
            b_cam.ortho_scale = max(
                    bcam.xmag or bcam.ymag,
                    bcam.ymag or bcam.xmag)
        #end if
        if bcam.znear:
            b_cam.clip_start = bcam.znear
        #end if
        if bcam.zfar:
            b_cam.clip_end = bcam.zfar
        #end if
    #end camera

    def geometry(self, bgeom):
        b_materials = {}
        for sym, matnode in bgeom.materialnodebysymbol.items():
            mat = matnode.target
            b_matname = self.name(mat)
            if b_matname not in bpy.data.materials:
                b_matname = self.material(mat, b_matname)
            #end if
            b_materials[sym] = bpy.data.materials[b_matname]
        #end for

        primitives = bgeom.original.primitives
        if self._transform('APPLY'):
            primitives = bgeom.primitives()
        #end if

        b_geoms = []
        for i, p in enumerate(primitives):
            if isinstance(p, BoundPrimitive):
                b_mat_key = p.original.material
            else:
                b_mat_key = p.material
            #end if
            b_mat = b_materials.get(b_mat_key, None)
            b_meshname = self.name(bgeom.original, i)

            if isinstance(p, (TriangleSet, BoundTriangleSet)):
                b_mesh = self.geometry_triangleset(
                        p, b_meshname, b_mat)
            elif isinstance(p, (Polylist, BoundPolylist)):
                b_mesh = self.geometry_triangleset(
                        p.triangleset(), b_meshname, b_mat)
            else:
                continue
            #end if
            if not b_mesh:
                continue

            b_obj = bpy.data.objects.new(b_meshname, b_mesh)
            b_obj.data = b_mesh

            self._ctx.scene.collection.objects.link(b_obj)
            self._ctx.view_layer.objects.active = b_obj

            if len(b_obj.material_slots) == 0:
                bpy.ops.object.material_slot_add()
            #end if
            b_obj.material_slots[0].link = 'OBJECT'
            b_obj.material_slots[0].material = b_mat
            b_obj.active_material = b_mat

            if self._transform('APPLY'):
                # TODO import normals
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.normals_make_consistent()
                bpy.ops.object.mode_set(mode='OBJECT')
            #end if

            b_geoms.append(b_obj)
        #end for

        return b_geoms
    #end geometry

    def geometry_triangleset(self, triset, b_name, b_mat):
        if not self._transform('APPLY') and b_name in bpy.data.meshes:
            # with applied transformation, mesh reuse is not possible
            return bpy.data.meshes[b_name]
        else:
            if triset.vertex_index is None or not len(triset.vertex_index):
                return

            b_mesh = bpy.data.meshes.new(b_name)
            b_mesh.from_pydata \
              (
                triset.vertex,
                [],
                [((v3, v1, v2), (v1, v2, v3))[v3 != 0]
                    for f in triset.vertex_index
                    for v1, v2, v3 in (f,)
                ]
              )

            has_normal = triset.normal_index is not None
            has_uv = len(triset.texcoord_indexset) > 0
            if has_normal:
                # TODO import normals
                for i, f in enumerate(b_mesh.polygons):
                    f.use_smooth = not _is_flat_face(triset.normal[triset.normal_index[i]])
                #end for
            #end if
            if has_uv:
                for j in range(len(triset.texcoord_indexset)):
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

    def texcoord_layer(self, triset, texcoord, index, b_mesh):
        uv = b_mesh.uv_layers.new()
        for i, f in enumerate(b_mesh.polygons):
            t1, t2, t3 = index[i]
            # eekadoodle
            if triset.vertex_index[i][2] == 0:
                t1, t2, t3 = t3, t1, t2
            #end if
            j = f.loop_start
            uv.data[j].uv = texcoord[t1]
            uv.data[j + 1].uv = texcoord[t2]
            uv.data[j + 2].uv = texcoord[t3]
        #end for
    #end texcoord_layer

    def light(self, light, i):
        if isinstance(light.original, AmbientLight):
            return
        b_name = self.name(light.original, i)
        if b_name not in bpy.data.lamps:
            if isinstance(light.original, DirectionalLight):
                b_lamp = bpy.data.lamps.new(b_name, type='SUN')
            elif isinstance(light.original, PointLight):
                b_lamp = bpy.data.lamps.new(b_name, type='POINT')
                b_obj = bpy.data.objects.new(b_name, b_lamp)
                self._ctx.scene.collection.objects.link(b_obj)
                b_obj.matrix_world = Matrix.Translation(light.position)
            elif isinstance(light.original, SpotLight):
                b_lamp = bpy.data.lamps.new(b_name, type='SPOT')
            #end if
        #end if
    #end light

    class Material :
        "interpretation of Collada material settings."

        def __init__(self, parent, mat, b_name) :
            self.parent = parent
            rendering = \
                { # initialize at instantiation time to allow overriding by subclasses
                    "blinn" : self.rendering_blinn,
                    "constant" : self.rendering_constant,
                    "lambert" : self.rendering_lambert,
                    "phong" : self.rendering_phong,
                    "diffuse" : self.rendering_diffuse,
                    "specular" : self.rendering_specular,
                    "reflectivity" : self.rendering_reflectivity,
                    "transparency" : self.rendering_transparency,
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
            transparent_shadows = self.parent._kwargs.get('transparent_shadows', False)
            b_mat.shadow_method = ("OPAQUE", "HASHED")[transparent_shadows]
              # best I can do for non-Cycles
            b_mat.cycles.use_transparent_shadow = transparent_shadows
            if isinstance(effect.emission, tuple) :
                b_shader.inputs["Emission"].default_value = effect.emission
            # Map option NYI for now
            #end if
            self.rendering_transparency()
            self.rendering_reflectivity()
        #end __init__

        def rendering_blinn(self):
            # for the difference between Blinn (actually Blinn-Phong) and Phong shaders,
            # see <https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model>
            self.rendering_diffuse()
            self.rendering_specular(True)
        #end rendering_blinn

        def rendering_constant(self):
            pass # no real option for shadeless materials in current Blender renderers.
        #end rendering_constant

        def rendering_lambert(self):
            self.rendering_diffuse()
            self.b_shader.inputs["Specular"].default_value = 0
        #end rendering_lambert

        def rendering_phong(self):
            self.rendering_diffuse()
            self.rendering_specular(False)
        #end rendering_phong

        def rendering_diffuse(self):
            self.color_or_texture(self.effect.diffuse, "diffuse", "Base Color")
        #end rendering_diffuse

        def rendering_specular(self, blinn = False):
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

        def rendering_reflectivity(self):
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

        def rendering_transparency(self):
            effect = self.effect
            if effect.transparency == None :
                return
            b_mat = self.b_mat
            b_shader = self.b_shader
            if isinstance(effect.transparency, Real):
                b_shader.inputs["Alpha"].default_value = effect.transparency
                if effect.transparency < 1.0 :
                    b_mat.blend_method = "BLEND"
                    b_mat.diffuse_color[3] = effect.transparency
                #end if
            #end if
            b_shader.inputs["Transmission"].default_value = \
                (0.0, 1.0)[self.parent._kwargs.get('raytrace_transparency', False)]
            if isinstance(effect.index_of_refraction, Real):
                b_shader.inputs["IOR"].default_value = effect.index_of_refraction
            #end if
        #end rendering_transparency

        @contextmanager
        def _tmpwrite(self, relpath, data):
            with NamedTemporaryFile(suffix='.' + relpath.split('.')[-1]) as out:
                out.write(data)
                out.flush()
                yield out.name
            #end with
        #end _tmpwrite

        def color_or_texture(self, color_or_texture, tex_name, shader_input_name):

            def try_texture(c_image):
                mtex = None
                with self._tmpwrite(c_image.path, c_image.data) as tmpname:
                    image = load_image(tmpname)
                    if image != None:
                        node_graph = self.b_mat.node_tree
                        image.pack()
                        # wipe all traces of original file path
                        image.filepath = "//textures/%s" % os.path.split(c_image.path)[1]
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
                    #end if
                return mtex
            #end try_texture

        #begin color_or_texture
            if isinstance(color_or_texture, Map):
                image = color_or_texture.sampler.surface.image
                mtex = try_texture(image)
                if mtex == None :
                    mtex = (1, 0, 1, 1) # same hideous colour Blender uses
                #end if
            elif isinstance(color_or_texture, tuple):
                mtex = color_or_texture
            else :
                mtex = None
            #end if
            shader_input = self.b_shader.inputs[shader_input_name]
            if isinstance(mtex, tuple):
                shader_input.default_value = mtex
            elif isinstance(mtex, bpy.types.NodeSocket) :
                self.b_mat.node_tree.links.new(mtex, shader_input)
            #end if
        #end color_or_texture

    #end Material

    def material(self, mat, b_name):
        matctx = type(self).Material(self, mat, b_name)
          # all material setup happens here
        return matctx.name
    #end material

    def node(self, node, parent):
        if isinstance(node, (Node, NodeNode)):
            b_obj = bpy.data.objects.new(self.name(node), None)
            b_obj.matrix_world = Matrix(node.matrix)
            self._ctx.scene.collection.objects.link(b_obj)
            if parent:
                b_obj.parent = parent
            parent = b_obj
        elif isinstance(node, GeometryNode):
            for bgeom in node.objects('geometry'):
                b_geoms = self.geometry(bgeom)
                for b_obj in b_geoms:
                    b_obj.parent = parent
                #end for
            #end for
        return parent
    #end node

    def name(self, obj, index=0):
        """ Trying to get efficient and human readable name, workarounds
        Blender's object name limitations.
        """
        if hasattr(obj, 'id'):
            uid = obj.id.replace('material', 'm')
        else:
            self._namecount += 1
            uid = 'Untitled.' + str(self._namecount)
        base = '%s-%d' % (uid, index)
        if base not in self._names:
            self._namecount += 1
            self._names[base] = '%s-%.4d' % (base[:MAX_NAME_LENGTH], self._namecount)
        return self._names[base]
    #end name

    def _transform(self, t):
        return self._kwargs['transformation'] == t
    #end _transform

#end ColladaImport

class SketchUpImport(ColladaImport):
    """ SketchUp specific COLLADA import. """

    class Material(ColladaImport.Material) :

        def rendering_phong(self):
            super().rendering_lambert()
        #end rendering_phong

        def rendering_reflectivity(self):
            """ There are no reflectivity controls in SketchUp """
            if not self.parent.match_test2(self.effect.xmlnode) :
                super().rendering_reflectivity()
            #end if
        #end rendering_reflectivity

    #end Material

    @staticmethod
    def match_test2(xml):
        return \
            any \
              (
                t.get('profile') == 'GOOGLEEARTH'
                for t in xml.findall('.//dae:extra/dae:technique', namespaces = DAE_NS)
              )
    #end match_test2

    @classmethod
    def match(celf, collada):
        "Does this look like a Collada file from SketchUp."

        def test1(xml):
            return \
                any \
                  (
                    'SketchUp' in s
                    for s in
                        (
                            xml.find('.//dae:instance_visual_scene', namespaces = DAE_NS).get('url'),
                            xml.find('.//dae:authoring_tool', namespaces = DAE_NS),
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

def _is_flat_face(normal):
    a = Vector(normal[0])
    for n in normal[1:]:
        dp = a.dot(Vector(n))
        if dp < 0.99999 or dp > 1.00001:
            return False
    return True

def _children(node):
    if isinstance(node, Scene):
        return node.nodes
    elif isinstance(node, Node):
        return node.children
    elif isinstance(node, NodeNode):
        return node.node.children
    else:
        return []


def _dfs(node, cb, parent=None):
    """ Depth first search taking a callback function.
    Its return value will be passed recursively as a parent argument.

    :param node: COLLADA node
    :param callable cb:
     """
    parent = cb(node, parent)
    for child in _children(node):
        _dfs(child, cb, parent)
