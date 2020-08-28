import os
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
from collada.common import DaeObject, E
from collada.geometry import Geometry
from collada.light import DirectionalLight, PointLight, SpotLight
from collada.material import CImage, Effect, Map, Material, Sampler2D, Surface
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
    EMPTY = "EM" # actually there is no type-specific datablock for these
    LAMP = "LA"
    MATERIAL = "MA"
    MESH = "ME"
    SCENE = "SCE"
    INTERNAL_ID = "IID"

    def nameid(self, name) :
        celf = type(self)
        if self not in celf._name_maps :
            celf._name_maps[self] = {}
            celf._name_revmaps[self] = {}
        #end if
        name_map = celf._name_maps[self]
        name_revmap = celf._name_revmaps[self]
        if name in name_map :
            clean_name = name_map[name]
        else :
            base_clean_name = name.replace(" ", "_")
              # are spaces the only illegal characters in XML IDs?
            clean_name = base_clean_name
            seq = 0
            while clean_name in name_revmap :
                seq += 1
                clean_name = "%s-%0.3d" % (base_clean_name, seq)
            #end while
            name_map[name] = clean_name
            name_revmap[clean_name] = name
        #end if
        return \
            "%s-%s" % (self.value, clean_name)
    #end nameid

    @property
    def internal_only(self) :
        "indicates IDs of this type are not used to name Blender objects on (re)import."
        return self == DATABLOCK.INTERNAL_ID
    #end internal_only

#end DATABLOCK
DATABLOCK._name_maps = {}
DATABLOCK._name_revmaps = {}

class EXT_FILE(enum.Enum) :

    TEXTURE = "textures"

    @property
    def subdir(self) :
        return self.value
    #end subdir

#end EXT_FILE

def idurl(uid) :
    return "#" + uid
#end idurl

class ColladaExport :

    def __init__(self, objects, filepath, directory, kwargs) :
        self._is_zae = kwargs["export_as"] == "zae"
        self._export_textures = kwargs["export_textures"]
        self._add_blender_extensions = kwargs["add_blender_extensions"]
        self._filepath = filepath
        self._dir = directory
        self._ext_files_map = {}
        self._ext_files_revmap = {}
        if self._is_zae :
            self._zip = zipfile.ZipFile(self._filepath, "w")
            class ZipAttr :
                compress_type = zipfile.ZIP_DEFLATED
                file_attr = 0o100644 << 16
                date_time = time.gmtime()[:6]
                scene_name = "scene.dae"

                @classmethod
                def new_item(celf, filename) :
                    item = zipfile.ZipInfo()
                    item.compress_type = celf.compress_type
                    item.external_attr = celf.file_attr
                    item.date_time = celf.date_time
                    item.filename = filename
                    return item
                #end new_item

            #end ZipAttr
            self._zipattr = ZipAttr
            # First item in archive is uncompressed and named “mimetype”, and
            # contents is MIME type for archive. This way it ends up at a fixed
            # offset (filename at 30 bytes from start, contents at 38 bytes) for
            # easy detection by format-sniffing tools. This convention is used
            # by ODF and other formats similarly based on Zip archives.
            mimetype = zipfile.ZipInfo()
            mimetype.filename = "mimetype"
            mimetype.compress_type = zipfile.ZIP_STORED
            mimetype.external_attr = ZipAttr.file_attr
            mimetype.date_time = (2020, 8, 23, 1, 33, 52)
              # about when I started getting .zae export working
            self._zip.writestr(mimetype, b"model/vnd.collada+xml+zip")
              # extrapolating from the fact that the official type for .dae files
              # is “model/vnd.collada+xml”
        else :
            self._zip = None
            self._zipattr = None
        #end if
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
        self._selected_only = kwargs["use_selection"]
        self._geometries = {}
        self._materials = {}
        self._collada = Collada()
        self._collada.xmlnode.getroot().set("version", kwargs["collada_version"])
        self._collada.assetInfo.unitmeter = 1
        self._collada.assetInfo.unitname = "metre"
        self._collada.assetInfo.upaxis = self._up_axis
        self._collada.assetInfo.save()
        asset_technique = self.blender_technique(True, self._collada.xmlnode.getroot())
          # I wanted to attach this under <asset>, but pycollada loses it there
        if asset_technique != None :
            prefixes = E.id_prefixes()
            for k in sorted(DATABLOCK.__members__.keys()) :
                v = DATABLOCK[k]
                if not v.internal_only :
                    prefix = E.prefix(name = k, value = v.nameid(""))
                    prefixes.append(prefix)
                #end if
            #end for
            asset_technique.append(prefixes)
        #end if

        self._scene = Scene(DATABLOCK.SCENE.nameid("main"), [])
        self._collada.scenes.append(self._scene)
        self._collada.scene = self._scene
        self._id_seq = 0

    #end __init__

    def write_ext_file(self, category, obj_name, filename, contents) :
        if category not in self._ext_files_map :
            self._ext_files_map[category] = {}
            self._ext_files_revmap[category] = {}
        #end if
        ext_files_map = self._ext_files_map[category]
        ext_files_revmap = self._ext_files_revmap[category]
        if obj_name in ext_files_map :
            # already encountered this external file
            out_filename = ext_files_map[obj_name]
        else :
            if not self._is_zae :
                outdir = os.path.join(self._dir, category.subdir)
                os.makedirs(outdir, exist_ok = True)
            #end if
            base_out_filename = os.path.join(category.subdir, filename)
            out_filename = base_out_filename
            seq = 0
            while out_filename in ext_files_revmap :
                if seq == 0 :
                    base_parts = os.path.splitext(base_out_filename)
                #end if
                seq += 1
                assert seq < 1000000 # impose some ridiculous but finite upper limit
                out_filename = "%s-%0.3d%s" % (base_parts[0], seq, base_parts[1])
            #end while
            ext_files_map[obj_name] = out_filename
            ext_files_revmap[out_filename] = obj_name
            if self._is_zae :
                item = self._zipattr.new_item(out_filename)
                self._zip.writestr(item, contents)
            else :
                out = open(os.path.join(self._dir, out_filename), "wb")
                out.write(contents)
                out.close()
            #end if
        #end if
        return out_filename
    #end write_ext_file

    def save(self) :
        if self._is_zae :
            item = self._zipattr.new_item(self._zipattr.scene_name)
            dae = io.BytesIO()
            self._collada.write(dae)
            self._zip.writestr(item, dae.getvalue())
            manifest = ElementTree.Element("dae_root")
            manifest.text = self._zipattr.scene_name
            item = self._zipattr.new_item("manifest.xml")
            self._zip.writestr(item, ElementTree.tostring(manifest))
            # all done
            self._zip.close()
        else :
            self._collada.write(self._filepath)
        #end if
    #end save

    def blender_technique(self, as_extra, obj) :
        # experimental: add Blender-specific attributes via a custom <technique>.
        if self._add_blender_extensions :
            if isinstance(obj, DaeObject) :
                obj = obj.xmlnode
            #end if
            blendstuff = E.technique(profile = "BLENDER028")
            if as_extra :
                parent = E.extra()
            else :
                parent = obj
            #end if
            parent.append(blendstuff)
            if as_extra :
                obj.append(parent)
            #end if
        else :
            blendstuff = None
        #end if
        return blendstuff
    #end blender_technique

    def obj_blender_technique(self, as_extra, obj, b_data, attribs) :
        # save any custom technique settings for this object.
        blendstuff = self.blender_technique(as_extra, obj)
        if blendstuff != None :
            for tagname, attrname in attribs :
                if hasattr(b_data, attrname) :
                    subtag = getattr(E, tagname)(str(getattr(b_data, attrname)))
                    blendstuff.append(subtag)
                #end if
            #end for
        #end if
    #end obj_blender_technique

    def next_internal_id(self) :
        self._id_seq += 1
        return DATABLOCK.INTERNAL_ID.nameid("%0.5d" % self._id_seq)
    #end next_internal_id

    def node(self, b_matrix = None) :
        node = Node(id = None, xmlnode = E.node())
          # construct my own xmlnode to avoid setting an id or name
          # (should be optional according to Collada spec)
        if b_matrix != None :
            node.transforms.append(self.matrix(b_matrix))
        #end if
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
            self._collada.cameras.append(cam)
            result.append(CameraNode(cam))
        #end if
        return result
    #end obj_camera

    def obj_light(self, b_obj) :
        result = []
        b_light = b_obj.data
        light_type = tuple \
          (
            t for t in
                (
                    ("POINT", PointLight),
                    ("SPOT", SpotLight),
                    ("SUN", DirectionalLight),
                )
            if b_light.type == t[0]
          )
        if len(light_type) != 0 :
            light_type = light_type[0][1]
        else :
            light_type = None
        #end if
        if light_type != None :
            # todo: falloff, shared datablock
            light = light_type \
              (
                DATABLOCK.LAMP.nameid(b_obj.name),
                color = tuple(b_light.color) + (1,)
              )
            self.obj_blender_technique \
              (
                True,
                light,
                b_light,
                [
                    ("angle", "angle"),
                    ("power", "energy"),
                    ("shadow_soft_size", "shadow_soft_size"),
                    ("spot_blend", "spot_blend"),
                    ("spot_size", "spot_size"),
                ]
              )
            self._collada.lights.append(light)
            result.append(LightNode(light))
        #end if
        return result
    #end obj_light

    def obj_empty(self, b_obj) :
        result = Node(id = DATABLOCK.EMPTY.nameid(b_obj.name))
        return [result]
    #end obj_empty

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
            vert_srcid = self.next_internal_id()
            vert_src = FloatSource \
              (
                id = vert_srcid,
                data = np.array([c for v in b_mesh.vertices for c in v.co]),
                components = ("X", "Y", "Z")
              )

            sources = [vert_src]

            if any(f for f in b_mesh.polygons if f.use_smooth) :
                vnorm_srcid = self.next_internal_id()
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
                fnorm_srcid = self.next_internal_id()
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

            geom = Geometry(self._collada, mesh_name, mesh_name, sources)

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
            "CAMERA" : (obj_camera, DATABLOCK.CAMERA),
            "EMPTY" : (obj_empty, DATABLOCK.EMPTY),
            "LIGHT" : (obj_light, DATABLOCK.LAMP),
            "MESH" : (obj_mesh, DATABLOCK.MESH),
        }

    def object(self, b_obj, parent = None) :
        handle_type = self.obj_type_handlers.get(b_obj.type)
        if handle_type != None :
            if parent != None :
                b_matrix = b_obj.matrix_local
            else :
                b_matrix = self._orient @ b_obj.matrix_world
            #end if

            obj = handle_type[0](self, b_obj)
            is_node = len(obj) == 1 and isinstance(obj[0], Node)
            if is_node :
                obj = obj[0]
                assert b_matrix != None
                obj.transforms.append(self.matrix(b_matrix))
                node = obj
            else :
                node = self.node(b_matrix)
            #end if
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
            if not is_node :
                node.children.extend(obj)
            #end if
        #end if
    #end object

    def material(self, b_mat) :
        shader = "lambert"
        effect_kwargs = \
            {
                "diffuse" : tuple(b_mat.diffuse_color[:3]),
                "double_sided" : not b_mat.use_backface_culling,
            }
        effect_params = []
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
                        value = None
                    #end if
                    return value
                #end get_input

                def get_input_map(name) :
                    input = b_shader.inputs[name]
                    map = None # to begin with
                    if input.is_linked :
                        links = input.links
                          # note docs say this takes O(N) in total nr links in node graph to compute
                        teximage = list \
                          (
                            l.from_node for l in links
                            if isinstance(l.from_node, bpy.types.ShaderNodeTexImage) and l.from_socket.name == "Color"
                          )
                        if len(teximage) != 0 :
                            teximage = teximage[0].image
                            if teximage.packed_file != None :
                                contents = teximage.packed_file.data
                            else :
                                contents = open(bpy.path.abspath(teximage.filepath), "rb").read()
                            #end if
                            out_filepath = self.write_ext_file \
                              (
                                category = EXT_FILE.TEXTURE,
                                obj_name = teximage.name,
                                filename = os.path.basename(teximage.filepath),
                                contents = contents
                              )
                            image = CImage(id = self.next_internal_id(), path = out_filepath)
                            surface = Surface(id = self.next_internal_id(), img = image)
                            sampler = Sampler2D(id = self.next_internal_id(), surface = surface)
                            map = Map(sampler = sampler, texcoord = "UVMap")
                              # TBD match up texcoord with material binding somehow
                            effect_params.extend([image, surface, sampler])
                        #end if
                    #end if
                    return map
                #end get_input_map

                value = get_input("Base Color")
                if value != None :
                    effect_kwargs["diffuse"] = value[:3]
                elif self._export_textures :
                    map = get_input_map("Base Color")
                    if map != None :
                        effect_kwargs["diffuse"] = map
                    #end if
                #end if
                # todo: support maps for more inputs
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
        effect = Effect(self.next_internal_id(), effect_params, shader, **effect_kwargs)
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

def save(op, context, filepath, directory, **kwargs) :
    objects = context.scene.objects
    exporter = ColladaExport(objects, filepath, directory, kwargs)
    for o in objects :
        if o.parent == None and (not exporter._selected_only or o.select_get()) :
            exporter.object(o)
        #end if
    #end for
    exporter.save()
    return {"FINISHED"}
#end save
