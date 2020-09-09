import os
import math
import enum
import re
import io
import time
import zipfile
import numpy as np
import bpy
import bmesh
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

XML_NCNAMESTART_CHARS = \
    r"\u0041-\u005A\u005F\u0061-\u007A\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF" \
    r"\u0370-\u037D\u037F-\u1FFF\u200C\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF" \
    r"\uF900-\uFDCF\uFDF0-\uFFFD\U00010000-\U000EFFFF"
      # valid chars that can start an NCName, as per the xml:id spec;
      # these are all of NameStartChar (see section 2.3 of the XML 1.0
      # or 1.1 specs) except “:”.
XML_NCNAMEREST_CHARS = r"\u002D\u002E\u0030-\u0039\u00B7\u0300-\u036F\u203F\u2040"
      # valid chars that, together with XML_NCNAMESTART_CHARS, can make up the rest of an NCName.

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
    INTERNAL_ID = "IID" # for things which have no names in Blender

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
        elif len(name) == 0 :
            # allowed for generating id-prefix-mapping table
            clean_name = ""
        else :
            clean_char = "_" # permissible anywhere in an XML ID
            base_clean_name = \
                (
                    re.sub("^[^" + XML_NCNAMESTART_CHARS + "]$", clean_char, name[0])
                +
                    re.sub("[^" + XML_NCNAMESTART_CHARS + XML_NCNAMEREST_CHARS + "]", clean_char, name[1:])
                )
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
        root_technique = self.blender_technique(True, self._collada.xmlnode.getroot())
        if root_technique != None :
            prefixes = E.id_prefixes()
            for k in sorted(DATABLOCK.__members__.keys()) :
                v = DATABLOCK[k]
                if not v.internal_only :
                    prefix = E.prefix(name = k, value = v.nameid(""))
                    prefixes.append(prefix)
                #end if
            #end for
            root_technique.append(prefixes)
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
            # todo: shared datablock
            light = light_type \
              (
                DATABLOCK.LAMP.nameid(b_obj.name),
                color = tuple(b_light.color) + (1,)
              )
            for attr, battr, conv in \
                (
                  # conversions are inverses of those done in importer
                    ("falloff_ang", "spot_size", lambda ang : ang / DEG),
                    ("falloff_exp", "spot_blend", lambda blend : 1 / max(blend, 0.00001) - 1),
                      # some very small-magnitude positive value to avoid division by zero
                ) \
            :
                if hasattr(b_light, battr) and hasattr(light, attr) :
                    setattr(light, attr, conv(getattr(b_light, battr)))
                #end if
            #end for
            if b_light.use_nodes :
                node_graph = b_light.node_tree
                the_node = list(n for n in node_graph.nodes if n.type == "OUTPUT_LIGHT")[0]
                trace_path = iter \
                  (
                    (
                        ("Surface", "EMISSION"),
                        ("Strength", "LIGHT_FALLOFF"),
                    )
                  )
                found = False
                while True :
                    trace = next(trace_path, None)
                    if trace == None :
                        if not the_node.inputs["Strength"].is_linked :
                            found = True
                        #end if
                        break
                    #end if
                    input_name, want_shader_type = trace
                    input = the_node.inputs[input_name]
                    if not input.is_linked :
                        break
                    links = input.links
                      # note docs say this takes O(N) in total nr links in node graph to compute
                    if len(links) == 0 :
                        break
                    the_node = links[0].from_node
                    if the_node.type != want_shader_type :
                        break
                    output_name = links[0].from_socket.name
                #end while
                if found :
                    strength = the_node.inputs["Strength"].default_value
                    if strength != 0 :
                        atten = \
                            {
                                "Constant" : "constant_att",
                                "Linear" : "linear_att",
                                "Quadratic" : "quad_att",
                            }.get(output_name)
                        if atten != None and hasattr(light, atten) :
                            setattr(light, atten, 1 / strength)
                        #end if
                    #end if
                #end if
            #end if
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

        def make_slotname(slotindex) :
            # Blender doesn’t name material slots, but Collada does
            return "slot%.3d" % slotindex
        #end make_slotname

        def encode_mesh(b_mesh, b_mesh_name, b_material_slots) :

            def is_trimesh(faces) :
                return all([len(f.verts) == 3 for f in faces])
            #end is_trimesh

        #begin encode_mesh
            mesh_name = DATABLOCK.MESH.nameid(b_mesh_name)
            sources = []
            vert_srcid = self.next_internal_id()
            sources.append \
              (
                FloatSource
                  (
                    id = vert_srcid,
                    data = np.array([c for v in b_mesh.verts for c in v.co]),
                    components = ("X", "Y", "Z")
                  )
              )
            vnorm_srcid = self.next_internal_id()
            sources.append \
              (
                FloatSource
                  (
                    id = vnorm_srcid,
                    data = np.array([c for v in b_mesh.verts for c in v.normal]),
                    components = ("X", "Y", "Z")
                  )
              ) # todo: face normal might be different for flat shading
            uv_ids = []
            if b_mesh.loops.layers.uv.active != None :
                active_uv_name = b_mesh.loops.layers.uv.active.name
            else :
                active_uv_name = None
            #end if
            for i, (b_uvname, uvlayer) in enumerate(b_mesh.loops.layers.uv.items()) :
                uv_name = self.next_internal_id()
                uv_ids.append((uv_name, b_uvname))
                sources.append \
                  (
                    FloatSource
                      (
                        id = uv_name,
                        data = np.array
                          (
                            [
                                x
                                for f in b_mesh.faces
                                for l in f.loops
                                for x in l[uvlayer].uv
                            ]
                          ),
                        components = ("S", "T")
                      )
                  )
            #end for
            geom = Geometry(self._collada, mesh_name, mesh_name, sources)
            blendstuff = self.blender_technique(True, geom)
            if blendstuff != None :
                names = E.layer_names()
                for u in uv_ids :
                    names.append(E.name(name = u[1], refid = u[0], type = "UV"))
                #end for
                blendstuff.append(names)
            #end if

            for slotindex in range(max(len(b_material_slots), 1)) :
                slotname = make_slotname(slotindex)
                assigned = \
                    [
                        f
                        for f in b_mesh.faces
                        if f.material_index == slotindex
                    ]
                if any(assigned) :
                    ilist = InputList()
                    ilist.addInput(0, "VERTEX", idurl(vert_srcid))
                    ilist.addInput(0, "NORMAL", idurl(vnorm_srcid))
                    setnr = 0
                    for u in uv_ids :
                        setnr += 1
                        ilist.addInput(1, "TEXCOORD", idurl(u[0]), (setnr, 0)[u[1] == active_uv_name])
                        # always assign set 0 to active UV layer
                    #end for
                    indices = []
                    for face in b_mesh.faces :
                        for face_loop in face.loops :
                            this_face = [face_loop.vert.index, face_loop.index]
                            indices.extend(this_face)
                        #end for
                    #end for
                    indices = np.array(indices)
                    if is_trimesh(assigned) :
                        p = geom.createTriangleSet(indices, ilist, slotname)
                    else :
                        vcounts = [len(f.verts) for f in assigned]
                        p = geom.createPolylist(indices, vcounts, ilist, slotname)
                    #end if
                    geom.primitives.append(p)
                #end if
            #end for

            self._collada.geometries.append(geom)
            return geom
        #end encode_mesh

    #begin obj_mesh
        b_mesh_name = b_obj.data.name
        b_material_slots = b_obj.material_slots
        b_mesh = bmesh.new()
        geom = self._geometries.get(b_mesh_name, None)
        if not geom :
            b_mesh.from_mesh(b_obj.data)
            geom = encode_mesh(b_mesh, b_mesh_name, b_material_slots)
            self._geometries[b_mesh_name] = geom
        #end if
        matnodes = []
        for slotindex, slot in enumerate(b_material_slots) :
            sname = slot.material.name
            if sname not in self._materials :
                self._materials[sname] = self.material(slot.material)
            #end if
            matnodes.append \
              (
                MaterialNode
                  (
                    make_slotname(slotindex),
                    self._materials[sname],
                    inputs = [("ACTIVE_UV", "TEXCOORD", "0")]
                      # always assign set 0 to active UV layer
                  )
              )
        #end for
        b_mesh.free()
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
                            map = Map(sampler = sampler, texcoord = "ACTIVE_UV")
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
