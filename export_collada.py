import math
import enum
import io
import time
import zipfile
import numpy as np
import bpy
from mathutils import Matrix, Vector

from collada import Collada
from collada.camera import PerspectiveCamera, OrthographicCamera
from collada.common import E
from collada.geometry import Geometry
from collada.light import DirectionalLight, PointLight, SpotLight
from collada.material import Effect, Material
from collada.scene import Node, Scene
from collada.scene import CameraNode, GeometryNode, LightNode, MaterialNode
from collada.scene import MatrixTransform
from collada.source import FloatSource, InputList
from collada.xmlutil import etree as ElementTree

DEG = math.pi / 180 # angle unit conversion factor

class DATABLOCK(enum.Enum) :
    # Note on uniqueness of IDs: Blender’s datablock names have to be
    # unique among datablocks of the same type, so these can be used as
    # IDs in XML, with a different prefix for each datablock type.
    CAMERA = "CA"
    LAMP = "LA"
    MATERIAL = "MA"
    MATERIAL_FX = "MA-FX"
    MESH = "ME"

    def nameid(self, name) :
        return \
            "%s-%s" % (self.value, name)
    #end nameid

    def node_nameid(self, name) :
        return \
            "N%s-%s" % (self.value, name)
    #end node_nameid

#end DATABLOCK

def idurl(uid) :
    return "#" + uid
#end idurl

class ColladaExport :

    def __init__(self, is_zae, objects, directory, kwargs) :
        self._is_zae = is_zae
        self._add_blender_extensions = kwargs["add_blender_extensions"]
        self._dir = directory
        self._up_axis = kwargs["up_axis"]
        if self._up_axis == "Z_UP" :
            self._orient = Matrix.Identity(4)
        elif self._up_axis == "X_UP" :
            self._orient = Matrix.Rotation(- 120 * DEG, 4, Vector(1, -1, 1))
        else : # "Y_UP" or unspecified
            self._orient = Matrix.Rotation(- 90 * DEG, 4, "X")
        #end if
        obj_children = {}
        for obj in objects :
            parent = obj.parent
            if parent == None :
                parentname = None
            else :
                parentname = parent.name
            #end if
            if parentname not in obj_children :
                obj_children[parentname] = set()
            #end if
            obj_children[parentname].add(obj.name)
        #end for
        self._obj_children = obj_children
        self._export_as = kwargs["export_as"] # TODO: NYI
        self._selected_only = kwargs["use_selection"]
        self._geometries = {}
        self._materials = {}
        self._collada = Collada()
        self._collada.xmlnode.getroot().set("version", kwargs["collada_version"])
        self._collada.assetInfo.unitmeter = 1
        self._collada.assetInfo.unitname = "metre"
        self._collada.assetInfo.upaxis = self._up_axis
        self._collada.assetInfo.save()

        self._scene = Scene("main", [])
        self._collada.scenes.append(self._scene)
        self._collada.scene = self._scene

    #end __init__

    def save(self, filepath) :
        if self._is_zae :
            timestamp = time.gmtime()[:6]
            scene_name = "scene.dae"
            out = zipfile.ZipFile(filepath, "w")
            manifest = ElementTree.Element("dae_root")
            manifest.text = scene_name
            item = zipfile.ZipInfo()
            item.filename = "manifest.xml"
            item.compress_type = zipfile.ZIP_DEFLATED
            item.date_time = timestamp
            out.writestr(item, ElementTree.tostring(manifest))
            item = zipfile.ZipInfo()
            item.filename = scene_name
            item.compress_type = zipfile.ZIP_DEFLATED
            item.date_time = timestamp
            dae = io.BytesIO()
            self._collada.write(dae)
            out.writestr(item, dae.getvalue())
            out.close()
        else :
            self._collada.write(filepath)
        #end if
    #end save

    def blender_technique(self, as_extra, obj, b_data, attribs) :
        # experimental: add Blender-specific attributes via a custom <technique>.
        if self._add_blender_extensions :
            blendstuff = E.technique(profile = "BLENDER028")
            for tagname, format, attrname in attribs :
                subtag = getattr(E, tagname)(format(getattr(b_data, attrname)))
                blendstuff.append(subtag)
            #end for
            if as_extra :
                parent = E.extra()
            else :
                parent = obj.xmlnode
            #end if
            parent.append(blendstuff)
            if as_extra :
                obj.xmlnode.append(parent)
            #end if
        #end if
    #end blender_technique

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
        result = []
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
            cam = cam_class \
              (
                id = DATABLOCK.CAMERA.nameid(b_obj.name),
                znear = b_cam.clip_start,
                zfar = b_cam.clip_end,
                **args
              )
            result.append(self.matrix(b_obj.matrix_local))
            self._collada.cameras.append(cam)
            result.append(CameraNode(cam))
        #end if
        return result
    #end obj_camera

    def obj_light(self, b_obj) :
        result = []
        b_light = b_obj.data
        v_pos, q_rot, v_scale = b_obj.matrix_local.decompose()
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
            # todo: falloff, shared datablock
            light = light_class \
              (
                DATABLOCK.LAMP.nameid(b_obj.name),
                color = tuple(b_light.color) + (1,)
              )
            self.blender_technique \
              (
                True,
                light,
                b_light,
                [
                    ("energy", lambda f : "%.3f" % f, "energy"),
                    # more TBD
                ]
              )
            if use_pos :
                result.append(self.matrix(Matrix.Translation((v_pos))))
            #end if
            if use_dirn :
                result.append(self.matrix(q_rot.to_matrix().to_4x4()))
            #end if
            self._collada.lights.append(light)
            result.append(LightNode(light))
        #end if
        return result
    #end obj_light

    def obj_mesh(self, b_obj) :

        b_mesh = b_obj.data

        def make_slotname(slotindex) :
            return "slot%.3d" % slotindex
        #end make_slotname

        def encode_mesh() :

            def is_trimesh(faces) :
                return all([len(f.vertices) == 3 for f in faces])
            #end is_trimesh

        #begin encode_mesh
            mesh_name = DATABLOCK.MESH.nameid(b_mesh.name)
            vert_srcid = mesh_name + "-vertcoords"
            vert_src = FloatSource \
              (
                id = vert_srcid,
                data = np.array([c for v in b_mesh.vertices for c in v.co]),
                components = ("X", "Y", "Z")
              )

            sources = [vert_src]

            if any(f for f in b_mesh.polygons if f.use_smooth) :
                vnorm_srcid = mesh_name + "-vnormals"
                sources.append \
                  (
                    FloatSource
                      (
                        id = vnorm_srcid,
                        data = np.array([c for v in b_mesh.vertices for c in v.normal]),
                        components = ("X", "Y", "Z")
                      )
                  )
            #end if
            flat = [f for f in b_mesh.polygons if not f.use_smooth]
            if any(flat) :
                fnorm_srcid = mesh_name + "-fnormals"
                sources.append \
                  (
                    FloatSource
                      (
                        id = fnorm_srcid,
                        data = np.array([c for f in flat for c in f.normal]),
                        components = ("X", "Y", "Z")
                      )
                  )
            #end if

            name = mesh_name + "-geom"
            geom = Geometry(self._collada, name, name, sources)

            for slotindex in range(max(len(b_obj.material_slots), 1)) :
                slotname = make_slotname(slotindex)
                smooth = \
                    [
                        f
                        for f in b_mesh.polygons
                        if f.material_index == slotindex and f.use_smooth
                    ]
                flat = \
                    [
                        f
                        for f in b_mesh.polygons
                        if f.material_index == slotindex and not f.use_smooth
                    ]
                if any(smooth) :
                    ilist = InputList()
                    ilist.addInput(0, "VERTEX", idurl(vert_srcid))
                    ilist.addInput(1, "NORMAL", idurl(vnorm_srcid))
                    # per vertex normals
                    indices = np.array \
                      (
                        [
                            i
                            for v in [(v, v) for f in smooth for v in f.vertices]
                            for i in v
                        ]
                      )
                    if is_trimesh(smooth) :
                        p = geom.createTriangleSet(indices, ilist, slotname)
                    else :
                        vcount = [len(f.vertices) for f in smooth]
                        p = geom.createPolylist(indices, vcount, ilist, slotname)
                    #end if
                    geom.primitives.append(p)
                #end if
                if any(flat) :
                    ilist = InputList()
                    ilist.addInput(0, "VERTEX", idurl(vert_srcid))
                    ilist.addInput(1, "NORMAL", idurl(fnorm_srcid))
                    indices = []
                    # per face normals
                    for i, f in enumerate(flat) :
                        for v in f.vertices :
                            indices.extend([v, i])
                        #end for
                    #end for
                    indices = np.array(indices)
                    if is_trimesh(flat) :
                        p = geom.createTriangleSet(indices, ilist, slotname)
                    else :
                        vcount = [len(f.vertices) for f in flat]
                        p = geom.createPolylist(indices, vcount, ilist, slotname)
                    #end if
                    geom.primitives.append(p)
                #end if
            #end for

            self._collada.geometries.append(geom)
            return geom
        #end encode_mesh

    #begin obj_mesh
        geom = self._geometries.get(b_mesh.name, None)
        if not geom :
            geom = encode_mesh()
            self._geometries[b_mesh.name] = geom
        #end if
        matnodes = []
        for slotindex, slot in enumerate(b_obj.material_slots) :
            sname = slot.material.name
            if sname not in self._materials :
                self._materials[sname] = self.material(slot.material)
            #end if
            matnodes.append \
              (
                MaterialNode(make_slotname(slotindex), self._materials[sname], inputs = [])
              )
        #end for
        return [GeometryNode(geom, matnodes)]
    #end obj_mesh

    obj_type_handlers = \
        {
            "CAMERA" : (obj_camera, DATABLOCK.CAMERA, True),
            "LIGHT" : (obj_light, DATABLOCK.LAMP, True),
            "MESH" : (obj_mesh, DATABLOCK.MESH, False),
        }

    def object(self, b_obj, parent = None) :
        handle_type = self.obj_type_handlers.get(b_obj.type)
        if handle_type != None :
            if handle_type[2] :
                if parent != None :
                    b_matrix = None
                else :
                    b_matrix = self._orient.copy()
                #end if
            else :
                if parent != None :
                    b_matrix = b_obj.matrix_local
                else :
                    b_matrix = self._orient @ b_obj.matrix_world
                #end if
            #end if

            node = self.node(handle_type[1].node_nameid(b_obj.name), b_matrix)
            children = self._obj_children.get(b_obj.name)
            if children != None :
                for childname in children :
                    self.object(bpy.data.objects[childname], parent = node)
                #end for
            #end if

            if parent != None :
                parent.children.append(node)
            else :
                self._scene.nodes.append(node)
            #end if

            node.children.extend(handle_type[0](self, b_obj))
        #end if
    #end object

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
        effect = Effect(DATABLOCK.MATERIAL_FX.nameid(b_mat.name), [], shader, **effect_kwargs)
        mat = Material(DATABLOCK.MATERIAL.nameid(b_mat.name), b_mat.name, effect)
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

def save(op, context, is_zae, filepath, directory, **kwargs) :
    objects = context.scene.objects
    exporter = ColladaExport(is_zae, objects, directory, kwargs)
    for o in objects :
        if o.parent == None and (not exporter._selected_only or o.select_get()) :
            exporter.object(o)
        #end if
    #end for
    exporter.save(filepath)
    return {"FINISHED"}
#end save
