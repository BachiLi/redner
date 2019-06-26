import torch
import re
import pyredner
import os

class WavefrontMaterial:
    def __init__(self):
        self.name = ""
        self.Kd = (0.0, 0.0, 0.0)
        self.Ks = (0.0, 0.0, 0.0)
        self.Ns = 0.0
        self.Ke = (0.0, 0.0, 0.0)
        self.map_Kd = None
        self.map_Ks = None
        self.map_Ns = None

class TriangleMesh:
    def __init__(self, vertices, indices, uvs, normals):
        self.vertices = vertices
        self.indices = indices
        self.uvs = uvs
        self.normals = normals

def load_mtl(filename):
    mtllib = {}
    current_mtl = WavefrontMaterial()
    for line in open(filename, 'r'):
        line = line.strip()
        splitted = re.split('\ +', line)
        if splitted[0] == 'newmtl':
            if current_mtl.name != "":
                mtllib[current_mtl.name] = current_mtl
            current_mtl = WavefrontMaterial()
            current_mtl.name = splitted[1]
        elif splitted[0] == 'Kd':
            current_mtl.Kd = (float(splitted[1]), float(splitted[2]), float(splitted[3]))
        elif splitted[0] == 'Ks':
            current_mtl.Ks = (float(splitted[1]), float(splitted[2]), float(splitted[3]))
        elif splitted[0] == 'Ns':
            current_mtl.Ns = float(splitted[1])
        elif splitted[0] == 'Ke':
            current_mtl.Ke = (float(splitted[1]), float(splitted[2]), float(splitted[3]))
        elif splitted[0] == 'map_Kd':
            current_mtl.map_Kd = splitted[1]
        elif splitted[0] == 'map_Ks':
            current_mtl.map_Ks = splitted[1]
        elif splitted[0] == 'map_Ns':
            current_mtl.map_Ns = splitted[1]
    if current_mtl.name != "":
        mtllib[current_mtl.name] = current_mtl
    return mtllib

def load_obj(filename, obj_group = True, flip_tex_coords = True):
    """
        Load from a Wavefront obj file as PyTorch tensors.
        XXX: this is slow, maybe move to C++?
    """
    vertices_pool = []
    uvs_pool = []
    normals_pool = []
    indices = []
    vertices = []
    normals = []
    uvs = []
    vertices_map = {}
    material_map = {}
    current_mtllib = {}
    current_material_name = None

    def create_mesh(indices, vertices, normals, uvs):
        indices = torch.tensor(indices, dtype = torch.int32, device = pyredner.get_device())
        vertices = torch.tensor(vertices, device = pyredner.get_device())
        if len(uvs) == 0:
            uvs = None
        else:
            uvs = torch.tensor(uvs, device = pyredner.get_device())
        if len(normals) == 0:
            normals = None
        else:
            normals = torch.tensor(normals, device = pyredner.get_device())
        return TriangleMesh(vertices, indices, uvs, normals)

    mesh_list = []
    light_map = {}

    f = open(filename, 'r')
    d = os.path.dirname(filename)
    cwd = os.getcwd()
    if d != '':
        os.chdir(d)
    for line in f:
        line = line.strip()
        splitted = re.split('\ +', line)
        if splitted[0] == 'mtllib':
            current_mtllib = load_mtl(splitted[1])
        elif splitted[0] == 'usemtl':
            if len(indices) > 0 and obj_group is True:
                # Flush
                mesh_list.append((current_material_name, create_mesh(indices, vertices, normals, uvs)))
                indices = []
                vertices = []
                normals = []
                uvs = []
                vertices_map = {}
            mtl_name = splitted[1]
            current_material_name = mtl_name
            if mtl_name not in material_map:
                m = current_mtllib[mtl_name]
                if m.map_Kd is None:
                    diffuse_reflectance = torch.tensor(m.Kd,
                        dtype = torch.float32, device = pyredner.get_device())
                else:
                    diffuse_reflectance = pyredner.imread(m.map_Kd)
                    if pyredner.get_use_gpu():
                        diffuse_reflectance = diffuse_reflectance.cuda(device = pyredner.get_device())
                if m.map_Ks is None:
                    specular_reflectance = torch.tensor(m.Ks,
                        dtype = torch.float32, device = pyredner.get_device())
                else:
                    specular_reflectance = pyredner.imread(m.map_Ks)
                    if pyredner.get_use_gpu():
                        specular_reflectance = specular_reflectance.cuda(device = pyredner.get_device())
                if m.map_Ns is None:
                    roughness = torch.tensor([2.0 / (m.Ns + 2.0)],
                        dtype = torch.float32, device = pyredner.get_device())
                else:
                    roughness = 2.0 / (pyredner.imread(m.map_Ks) + 2.0)
                    if pyredner.get_use_gpu():
                        roughness = roughness.cuda(device = pyredner.get_device())
                if m.Ke != (0.0, 0.0, 0.0):
                    light_map[mtl_name] = torch.tensor(m.Ke, dtype = torch.float32)
                material_map[mtl_name] = pyredner.Material(\
                    diffuse_reflectance, specular_reflectance, roughness)
        elif splitted[0] == 'v':
            vertices_pool.append([float(splitted[1]), float(splitted[2]), float(splitted[3])])
        elif splitted[0] == 'vt':
            u = float(splitted[1])
            v = float(splitted[2])
            if flip_tex_coords:
                v = 1 - v
            uvs_pool.append([u, v])
        elif splitted[0] == 'vn':
            normals_pool.append([float(splitted[1]), float(splitted[2]), float(splitted[3])])
        elif splitted[0] == 'f':
            def num_indices(x):
                return len(re.split('/', x))
            def get_index(x, i):
                return int(re.split('/', x)[i])
            def parse_face_index(x, i):
                f = get_index(x, i)
                if f < 0:
                    if (i == 0):
                        f += len(vertices)
                    if (i == 1):
                        f += len(uvs)
                else:
                    f -= 1
                return f
            assert(len(splitted) <= 5)
            def get_vertex_id(indices):
                pi = parse_face_index(indices, 0)
                uvi = None
                if (num_indices(indices) > 1 and re.split('/', indices)[1] != ''):
                    uvi = parse_face_index(indices, 1)
                ni = None
                if (num_indices(indices) > 2 and re.split('/', indices)[2] != ''):
                    ni = parse_face_index(indices, 2)
                key = (pi, uvi, ni)
                if key in vertices_map:
                    return vertices_map[key]

                vertex_id = len(vertices)
                vertices_map[key] = vertex_id
                vertices.append(vertices_pool[pi])
                if uvi is not None:
                    uvs.append(uvs_pool[uvi])
                if ni is not None:
                    normals.append(normals_pool[ni])
                return vertex_id
            vid0 = get_vertex_id(splitted[1])
            vid1 = get_vertex_id(splitted[2])
            vid2 = get_vertex_id(splitted[3])

            indices.append([vid0, vid1, vid2])
            if (len(splitted) == 5):
                vid3 = get_vertex_id(splitted[4])
                indices.append([vid0, vid2, vid3])
    
    mesh_list.append((current_material_name,
        create_mesh(indices, vertices, normals, uvs)))
    if d != '':
        os.chdir(cwd)
    return material_map, mesh_list, light_map
