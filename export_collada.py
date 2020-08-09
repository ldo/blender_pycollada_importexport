import math
import bpy
import numpy as np
from mathutils import Matrix, Vector

from collada import Collada
from collada.camera import PerspectiveCamera, OrthographicCamera
from collada.geometry import Geometry
from collada.light import DirectionalLight, PointLight, SpotLight
from collada.material import Effect, Material
from collada.scene import Node, Scene
from collada.scene import CameraNode, GeometryNode, LightNode, MaterialNode
from collada.scene import MatrixTransform
from collada.source import FloatSource, InputList

DEG = math.pi / 180

def save(op, context, filepath = None, directory = None, export_as = None, **kwargs) :
    exporter = ColladaExport(directory, export_as)
    for o in context.scene.objects :
        exporter.object(o)
    #end for
    # Note that, in Collada, lights and cameras are not part of
    # the object-parenting hierarchy, the way they are in Blender.
    for action, objtype in \
        (
            (exporter.obj_camera, "CAMERA"),
            (exporter.obj_light, "LIGHT"),
        ) \
    :
        for o in context.scene.objects :
            if o.type == objtype :
                action(o)
            #end if
        #end for
    #end for
    exporter.save(filepath)
    return {"FINISHED"}
#end save

class ColladaExport :

    def __init__(self, directory, export_as = "dae_only") :
        self._dir = directory
        self._export_as = export_as # TODO: NYI
        self._geometries = {}
        self._materials = {}
        self._collada = Collada()
        self._collada.assetInfo.unitmeter = 1
        self._collada.assetInfo.unitname = "metre"
        self._collada.assetInfo.upaxis = "Z_UP"
        self._collada.assetInfo.save()

        self._scene = Scene("main", [])
        self._collada.scenes.append(self._scene)
        self._collada.scene = self._scene

    #end __init__

    def save(self, fp) :
        self._collada.write(fp)
    #end save

    def object(self, b_obj, parent = None, do_children = True) :
        inode_meth = self.obj_type_handlers.get(b_obj.type)
        if inode_meth != None :
            b_matrix = b_obj.matrix_world
            if parent != None :
                if do_children :
                    b_matrix = b_obj.matrix_local
                else :
                    b_matrix = Matrix()
                #end if
            #end if

            node = self.node(b_obj.name, b_matrix)
            # todo: docs say computing b_obj.children takes O(N) time. Perhaps
            # build my own parent-child mapping table for all objects in scene
            # to speed this up?
            if do_children and any(b_obj.children) :
                self.object(b_obj, parent = node, do_children = False)
                for child in b_obj.children :
                    self.object(child, parent = node)
                #end for
            #end if

            if parent != None :
                parent.children.append(node)
            else :
                self._scene.nodes.append(node)
            #end if

            node.children.extend(inode_meth(self, b_obj))
        #end if
    #end object

    def node(self, b_name, b_matrix = None) :
        tf = []
        if b_matrix != None :
            tf.append(self.matrix(b_matrix))
        #end if
        node = Node(id = b_name, transforms = tf)
        node.save()
        return node
    #end node

    def obj_camera(self, b_obj) :
        b_cam = b_obj.data
        if b_cam.type == "PERSP" :
            cam_class = PerspectiveCamera
            args = \
                {
                    "xfov" : b_cam.angle_x / DEG,
                    "yfov" : b_cam.angle_y / DEG,
                }
        elif b_cam.type == "ORTHO" :
            cam_class = OrthographicCamera
            args = \
                {
                    "xmag" : b_cam.ortho_scale,
                    "ymag" : b_cam.ortho_scale,
                }
        else :
            cam_class = None
        #end if
        if cam_class != None :
            # todo: shared datablock
            cam = cam_class(id = b_obj.name, znear = b_cam.clip_start, zfar = b_cam.clip_end, **args)
            camnode = self.node("CA-%s" % b_obj.name, b_matrix = b_obj.matrix_world)
            camnode.children.append(CameraNode(cam))
            self._collada.cameras.append(cam)
            self._scene.nodes.append(camnode)
        #end if
    #end obj_camera

    def obj_light(self, b_obj) :
        b_light = b_obj.data
        v_pos, q_rot, v_scale = b_obj.matrix_world.decompose()
        if b_light.type == "POINT" :
            light_class, use_pos, use_dirn = PointLight, True, False
        elif b_light.type == "SPOT" :
            light_class, use_pos, use_dirn = SpotLight, True, True
        elif b_light.type == "SUN" :
            light_class, use_pos, use_dirn = DirectionalLight, False, True
        else :
            light_class = None
        #end if
        if light_class != None :
            # todo: colour, falloff, shared datablock
            light = light_class(id = b_obj.name, color = (1, 1, 1, 1))
            lightnode = self.node("LA-%s" % b_obj.name)
            lightnode.children.append(LightNode(light))
            if use_pos :
                lightnode.transforms.append(self.matrix(Matrix.Translation((v_pos))))
            #end if
            if use_dirn :
                lightnode.transforms.append(self.matrix(q_rot.to_matrix().to_4x4()))
            #end if
            self._collada.lights.append(light)
            self._scene.nodes.append(lightnode)
        #end if
    #end obj_light

    def obj_mesh(self, b_obj) :
        geom = self._geometries.get(b_obj.data.name, None)
        if not geom :
            geom = self.mesh(b_obj.data)
            self._geometries[b_obj.data.name] = geom
        #end if
        matnodes = []
        for slot in b_obj.material_slots :
            sname = slot.material.name
            if sname not in self._materials :
                self._materials[sname] = self.material(slot.material)
            #end if
            matnodes.append(MaterialNode("none", self._materials[sname], inputs = []))
        #end for
        return [GeometryNode(geom, matnodes)]
    #end obj_mesh

    obj_type_handlers = \
        {
            "MESH" : obj_mesh,
        }

    def mesh(self, b_mesh) :
        vert_srcid = b_mesh.name + "-vertary"
        vert_f = [c for v in b_mesh.vertices for c in v.co]
        vert_src = FloatSource(vert_srcid, np.array(vert_f), ("X", "Y", "Z"))

        sources = [vert_src]

        smooth = list(filter(lambda f : f.use_smooth, b_mesh.polygons))
        if any(smooth) :
            vnorm_srcid = b_mesh.name + "-vnormary"
            norm_f = [c for v in b_mesh.vertices for c in v.normal]
            norm_src = FloatSource(vnorm_srcid, np.array(norm_f), ("X", "Y", "Z"))
            sources.append(norm_src)
        #end if
        flat = list(filter(lambda f : not f.use_smooth, b_mesh.polygons))
        if any(flat) :
            fnorm_srcid = b_mesh.name + "-fnormary"
            norm_f = [c for f in flat for c in f.normal]
            norm_src = FloatSource(fnorm_srcid, np.array(norm_f), ("X", "Y", "Z"))
            sources.append(norm_src)
        #end if

        name = b_mesh.name + "-geom"
        geom = Geometry(self._collada, name, name, sources)

        if any(smooth) :
            ilist = InputList()
            ilist.addInput(0, "VERTEX", _url(vert_srcid))
            ilist.addInput(1, "NORMAL", _url(vnorm_srcid))
            # per vertex normals
            indices = np.array([
                i for v in [
                    (v, v) for f in smooth for v in f.vertices
                ] for i in v])
            if _is_trimesh(smooth) :
                p = geom.createTriangleSet(indices, ilist, "none")
            else :
                vcount = [len(f.vertices) for f in smooth]
                p = geom.createPolylist(indices, vcount, ilist, "none")
            #end if
            geom.primitives.append(p)
        #end if
        if any(flat) :
            ilist = InputList()
            ilist.addInput(0, "VERTEX", _url(vert_srcid))
            ilist.addInput(1, "NORMAL", _url(fnorm_srcid))
            indices = []
            # per face normals
            for i, f in enumerate(flat) :
                for v in f.vertices :
                    indices.extend([v, i])
                #end for
            #end for
            indices = np.array(indices)
            if _is_trimesh(flat) :
                p = geom.createTriangleSet(indices, ilist, "none")
            else :
                vcount = [len(f.vertices) for f in flat]
                p = geom.createPolylist(indices, vcount, ilist, "none")
            #end if
            geom.primitives.append(p)
        #end if

        self._collada.geometries.append(geom)
        return geom
    #end mesh

    def material(self, b_mat) :
        shader = "lambert"
        effect_kwargs = \
            {
                "diffuse" : tuple(b_mat.diffuse_color[:3]),
                "double_sided" : not b_mat.use_backface_culling,
            }
        if b_mat.diffuse_color[3] != 1.0 :
            effect_kwargs["transparency"] = b_mat.diffuse_color[3]
        #end if
        if b_mat.use_nodes :
            b_shader = list(n for n in b_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            if len(b_shader) == 1 :
                # assume node setup somewhat resembles what importer creates
                b_shader = b_shader[0]
            else :
                b_shader = None
            #end if
            if b_shader != None :
                def get_input(name) :
                    input = b_shader.inputs[name]
                    if not input.is_linked :
                        value = input.default_value
                    else :
                        # todo: try to extract Map definition
                        value = None
                    #end if
                    return value
                #end get_input
                value = get_input("Base Color")
                if value != None :
                    effect_kwargs["diffuse"] = value[:3]
                #end if
                value = get_input("Metallic")
                metallic = True
                if value == None or value == 0 :
                    value = get_input("Specular")
                    metallic = False
                #end if
                if value != None and value != 0 :
                    shader = "phong" # do I care about “blinn”?
                    if metallic :
                        effect_kwargs["reflective"] = effect_kwargs["diffuse"]
                    else :
                        effect_kwargs["reflective"] = (1, 1, 1)
                    #end if
                    effect_kwargs["reflectivity"] = value
                #end if
                value = get_input("Alpha")
                if value != None and value != 1.0 :
                    effect_kwargs["transparency"] = value
                      # overridden by Transmission (below) if any
                #end if
                value = get_input("Transmission")
                if value != None and value != 0 :
                    effect_kwargs["transparency"] = value
                    effect_kwargs["transparent"] = effect_kwargs["diffuse"]
                    value = get_input("IOR")
                    if value != None :
                        effect_kwargs["index_of_refraction"] = value
                    #end if
                #end if
            else :
                pass # give up for now
            #end if
        else :
            # quick fudge based only on Viewport Display settings
            if b_mat.metallic > 0 or b_mat.roughness < 1 :
                shader = "phong" # do I care about “blinn”?
                try :
                    shininess = 1 / b_mat.roughness - 1
                      # inverse of formula used in importer
                except ZeroDivisionError :
                    shininess = math.inf
                #end try
                shininess = min(shininess, 10000) # just some arbitrary finite upper limit
                effect_kwargs["reflectivity"] = b_mat.specular_intensity
                effect_kwargs["shininess"] = shininess
                if b_mat.metallic > 0 :
                    # not paying attention to actual value of b_mat.metallic!
                    effect_kwargs["reflective"] = b_mat.specular_color[:3]
                else :
                    effect_kwargs["reflective"] = (1, 1, 1)
                #end if
            #end if
        #end if
        effect = Effect(b_mat.name + "-fx", [], shader, **effect_kwargs)
        mat = Material(b_mat.name, b_mat.name, effect)
        self._collada.effects.append(effect)
        self._collada.materials.append(mat)
        return mat
    #end material

    @staticmethod
    def matrix(b_matrix) :
        return \
            MatrixTransform \
              (
                np.array
                  (
                    [e for r in tuple(map(tuple, b_matrix)) for e in r],
                    dtype = np.float32
                  )
              )
    #end matrix

#end ColladaExport

def _is_trimesh(faces) :
    return all([len(f.vertices) == 3 for f in faces])
#end _is_trimesh

def _url(uid) :
    return "#" + uid
#end _url

