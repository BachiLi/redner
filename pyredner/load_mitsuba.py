import torch
import xml.etree.ElementTree as etree
import numpy as np
import redner
import os
import pyredner
import pyredner.transform as transform

def parse_transform(node):
    ret = torch.eye(4)
    for child in node:
        if child.tag == 'matrix':
            value = torch.from_numpy(\
                np.reshape(\
                    np.fromstring(child.attrib['value'], dtype=np.float32, sep=' '),
                    (4, 4)))
            ret = value @ ret
        elif child.tag == 'translate':
            x = float(child.attrib['x'])
            y = float(child.attrib['y'])
            z = float(child.attrib['z'])
            value = transform.gen_translate_matrix(torch.tensor([x, y, z]))
            ret = value @ ret
        elif child.tag == 'scale':
            x = float(child.attrib['x'])
            y = float(child.attrib['y'])
            z = float(child.attrib['z'])
            value = transform.gen_scale_matrix(torch.tensor([x, y, z]))
            ret = value @ ret
    return ret

def parse_vector(str):
    v = np.fromstring(str, dtype=np.float32, sep=',')
    if v.shape[0] != 3:
        v = np.fromstring(str, dtype=np.float32, sep=' ')
    assert(v.ndim == 1)
    return torch.from_numpy(v)

def parse_camera(node):
    fov = torch.tensor([45.0])
    position = None
    look_at = None
    up = None
    clip_near = 1e-2
    resolution = [256, 256]
    for child in node:
        if 'name' in child.attrib:
            if child.attrib['name'] == 'fov':
                fov = torch.tensor([float(child.attrib['value'])])
            elif child.attrib['name'] == 'toWorld':
                has_lookat = False
                for grandchild in child:
                    if grandchild.tag == 'lookat':
                        has_lookat = True
                        position = parse_vector(grandchild.attrib['origin'])
                        look_at = parse_vector(grandchild.attrib['target'])
                        up = parse_vector(grandchild.attrib['up'])
                if not has_lookat:
                    print('Unsupported Mitsuba scene format: please use a look at transform')
                    assert(False)
        if child.tag == 'film':
            for grandchild in child:
                if 'name' in grandchild.attrib:
                    if grandchild.attrib['name'] == 'width':
                        resolution[0] = int(grandchild.attrib['value'])
                    elif grandchild.attrib['name'] == 'height':
                        resolution[1] = int(grandchild.attrib['value'])

    return pyredner.Camera(position     = position,
                           look_at      = look_at,
                           up           = up,
                           fov          = fov,
                           clip_near    = clip_near,
                           resolution   = resolution)

def parse_material(node, two_sided = False):
    node_id = None
    if 'id' in node.attrib:
        node_id = node.attrib['id']
    if node.attrib['type'] == 'diffuse':
        diffuse_reflectance = torch.tensor([0.5, 0.5, 0.5])
        diffuse_uv_scale = torch.tensor([1.0, 1.0])
        specular_reflectance = torch.tensor([0.0, 0.0, 0.0])
        specular_uv_scale = torch.tensor([1.0, 1.0])
        roughness = torch.tensor([1.0])
        for child in node:
            if child.attrib['name'] == 'reflectance':
                if child.tag == 'texture':
                    for grandchild in child:
                        if grandchild.attrib['name'] == 'filename':
                            diffuse_reflectance = pyredner.imread(grandchild.attrib['value'])
                        elif grandchild.attrib['name'] == 'uscale':
                            diffuse_uv_scale[0] = float(grandchild.attrib['value'])
                        elif grandchild.attrib['name'] == 'vscale':
                            diffuse_uv_scale[1] = float(grandchild.attrib['value'])
                elif child.tag == 'rgb':
                    diffuse_reflectance = parse_vector(child.attrib['value'])
            elif child.attrib['name'] == 'specular':
                if child.tag == 'texture':
                    for grandchild in child:
                        if grandchild.attrib['name'] == 'filename':
                            specular_reflectance = image.imread(grandchild.attrib['value'])
                        elif grandchild.attrib['name'] == 'uscale':
                            specular_uv_scale[0] = float(grandchild.attrib['value'])
                        elif grandchild.attrib['name'] == 'vscale':
                            specular_uv_scale[1] = float(grandchild.attrib['value'])
                elif child.tag == 'rgb':
                    specular_reflectance = parse_vector(child.attrib['value'])
            elif child.attrib['name'] == 'roughness':
                roughness = torch.tensor(float(child.attrib['value']))
        if pyredner.get_use_gpu():
            # Copy to GPU
            diffuse_reflectance = diffuse_reflectance.cuda()
            specular_reflectance = specular_reflectance.cuda()
            roughness = roughness.cuda()
        return (node_id, pyredner.Material(diffuse_reflectance,
                diffuse_uv_scale = diffuse_uv_scale,
                specular_reflectance = specular_reflectance,
                specular_uv_scale = specular_uv_scale,
                roughness = roughness,
                two_sided = two_sided))
    elif node.attrib['type'] == 'twosided':
        ret = parse_material(node[0], True)
        return (node_id, ret[1])

def parse_shape(node, material_dict, shape_id):
    if node.attrib['type'] == 'obj' or node.attrib['type'] == 'serialized':
        to_world = torch.eye(4)
        serialized_shape_id = 0
        mat_id = -1
        light_intensity = None
        filename = ''
        for child in node:
            if 'name' in child.attrib:
                if child.attrib['name'] == 'filename':
                    filename = child.attrib['value']
                elif child.attrib['name'] == 'toWorld':
                    to_world = parse_transform(child)
                elif child.attrib['name'] == 'shapeIndex':
                    serialized_shape_id = int(child.attrib['value'])
            if child.tag == 'ref':
                mat_id = material_dict[child.attrib['id']]
            elif child.tag == 'emitter':
                for grandchild in child:
                    if grandchild.attrib['name'] == 'radiance':
                        light_intensity = parse_vector(grandchild.attrib['value'])
                        if light_intensity.shape[0] == 1:
                            light_intensity = torch.tensor(\
                                         [light_intensity[0],
                                          light_intensity[0],
                                          light_intensity[0]])

        if node.attrib['type'] == 'obj':
            vertices, indices, uvs, normals = pyredner.load_obj(filename)
        else:
            assert(node.attrib['type'] == 'serialized')
            mitsuba_tri_mesh = redner.load_serialized(filename, serialized_shape_id)
            vertices = torch.from_numpy(mitsuba_tri_mesh.vertices)
            indices = torch.from_numpy(mitsuba_tri_mesh.indices)
            uvs = torch.from_numpy(mitsuba_tri_mesh.uvs)
            normals = torch.from_numpy(mitsuba_tri_mesh.normals)
            if uvs.shape[0] == 0:
                uvs = None
            if normals.shape[0] == 0:
                normals = None

        # Transform the vertices and normals
        vertices = torch.cat((vertices, torch.ones(vertices.shape[0], 1)), dim = 1)
        vertices = vertices @ torch.transpose(to_world, 0, 1)
        vertices = vertices / vertices[:, 3:4]
        vertices = vertices[:, 0:3].contiguous()
        if normals is not None:
            normals = normals @ (torch.inverse(torch.transpose(to_world, 0, 1))[:3, :3])
            normals = normals.contiguous()
        assert(vertices is not None)
        assert(indices is not None)
        lgt = None
        if light_intensity is not None:
            lgt = pyredner.Light(shape_id, light_intensity)

        if pyredner.get_use_gpu():
            # Copy to GPU
            vertices = vertices.cuda()
            indices = indices.cuda()
            if uvs is not None:
                uvs = uvs.cuda()
            if normals is not None:
                normals = normals.cuda()
        return pyredner.Shape(vertices, indices, uvs, normals, mat_id), lgt
    elif node.attrib['type'] == 'rectangle':
        indices = torch.tensor([[0, 2, 1], [1, 2, 3]],
                               dtype = torch.int32)
        vertices = torch.tensor([[-1.0, -1.0, 0.0],
                                 [-1.0,  1.0, 0.0],
                                 [ 1.0, -1.0, 0.0],
                                 [ 1.0,  1.0, 0.0]])
        uvs = None
        normals = None
        to_world = torch.eye(4)
        mat_id = -1
        light_intensity = None
        for child in node:
            if 'name' in child.attrib:
                if child.attrib['name'] == 'toWorld':
                    to_world = parse_transform(child)
            if child.tag == 'ref':
                mat_id = material_dict[child.attrib['id']]
            elif child.tag == 'emitter':
                for grandchild in child:
                    if grandchild.attrib['name'] == 'radiance':
                        light_intensity = parse_vector(grandchild.attrib['value'])
                        if light_intensity.shape[0] == 1:
                            light_intensity = torch.tensor(\
                                         [light_intensity[0],
                                          light_intensity[0],
                                          light_intensity[0]])
        # Transform the vertices
        # Transform the vertices and normals
        vertices = torch.cat((vertices, torch.ones(vertices.shape[0], 1)), dim = 1)
        vertices = vertices @ torch.transpose(to_world, 0, 1)
        vertices = vertices / vertices[:, 3:4]
        vertices = vertices[:, 0:3].contiguous()
        if normals is not None:
            normals = normals @ (torch.inverse(torch.transpose(to_world, 0, 1))[:3, :3])
            normals = normals.contiguous()
        assert(vertices is not None)
        assert(indices is not None)
        lgt = None
        if light_intensity is not None:
            lgt = pyrender.Light(shape_id, light_intensity)

        if pyredner.get_use_gpu():
            # Copy to GPU
            vertices = vertices.cuda()
            indices = indices.cuda()
            if uvs is not None:
                uvs = uvs.cuda()
            if normals is not None:
                normals = normals.cuda()
        return shape.Shape(vertices, indices, uvs, normals, mat_id), lgt
    else:
        assert(False)

def parse_scene(node):
    cam = None
    resolution = None
    materials = []
    material_dict = {}
    shapes = []
    lights = []
    for child in node:
        if child.tag == 'sensor':
            cam = parse_camera(child)
        elif child.tag == 'bsdf':
            node_id, material = parse_material(child)
            if node_id is not None:
                material_dict[node_id] = len(materials)
                materials.append(material)
        elif child.tag == 'shape':
            shape, light = parse_shape(child, material_dict, len(shapes))
            shapes.append(shape)
            if light is not None:
                lights.append(light)
    return pyredner.Scene(cam, shapes, materials, lights)

def load_mitsuba(filename):
    """
        Load from a Mitsuba scene file as PyTorch tensors.
    """

    tree = etree.parse(filename)
    root = tree.getroot()
    cwd = os.getcwd()
    os.chdir(os.path.dirname(filename))
    ret = parse_scene(root)
    os.chdir(cwd)
    return ret
