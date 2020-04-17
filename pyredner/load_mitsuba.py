import torch
import xml.etree.ElementTree as etree
import numpy as np
import redner
import os
import pyredner
import pyredner.transform as transform
from typing import Optional
import math

def parse_transform(node):
    ret = torch.eye(4)
    for child in node:
        if child.tag == 'matrix':
            value = torch.from_numpy(\
                np.reshape(\
                    # support both ',' and ' ' seperator
                    np.fromstring(child.attrib['value'], dtype=np.float32, sep=',' if ',' in child.attrib['value'] else ' '),
                    (4, 4)))
            ret = value @ ret
        elif child.tag == 'translate':
            x = float(child.attrib['x'])
            y = float(child.attrib['y'])
            z = float(child.attrib['z'])
            value = transform.gen_translate_matrix(torch.tensor([x, y, z]))
            ret = value @ ret
        elif child.tag == 'scale':
            # single scale value
            if 'value' in child.attrib:
                x = y = z = float(child.attrib['value'])
            else:
                x = float(child.attrib['x'])
                y = float(child.attrib['y'])
                z = float(child.attrib['z'])
            value = transform.gen_scale_matrix(torch.tensor([x, y, z]))
            ret = value @ ret
        elif child.tag == 'rotate':
            x = float(child.attrib['x']) if 'x' in child.attrib else 0.0
            y = float(child.attrib['y']) if 'y' in child.attrib else 0.0
            z = float(child.attrib['z']) if 'z' in child.attrib else 0.0
            angle = transform.radians(float(child.attrib['angle']))
            axis = np.array([x, y, z])
            axis = axis / np.linalg.norm(axis)
            cos_theta = math.cos(angle)
            sin_theta = math.sin(angle)
            mat = torch.zeros(4, 4)
            mat[0, 0] = axis[0] * axis[0] + (1.0 - axis[0] * axis[0]) * cos_theta
            mat[0, 1] = axis[0] * axis[1] * (1.0 - cos_theta) - axis[2] * sin_theta
            mat[0, 2] = axis[0] * axis[2] * (1.0 - cos_theta) + axis[1] * sin_theta

            mat[1, 0] = axis[0] * axis[1] * (1.0 - cos_theta) + axis[2] * sin_theta
            mat[1, 1] = axis[1] * axis[1] + (1.0 - axis[1] * axis[1]) * cos_theta
            mat[1, 2] = axis[1] * axis[2] * (1.0 - cos_theta) - axis[0] * sin_theta

            mat[2, 0] = axis[0] * axis[2] * (1.0 - cos_theta) - axis[1] * sin_theta
            mat[2, 1] = axis[1] * axis[2] * (1.0 - cos_theta) + axis[0] * sin_theta
            mat[2, 2] = axis[2] * axis[2] + (1.0 - axis[2] * axis[2]) * cos_theta

            mat[3, 3] = 1.0
            ret = mat @ ret
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
                    if grandchild.tag.lower() == 'lookat':
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
                        resolution[1] = int(grandchild.attrib['value'])
                    elif grandchild.attrib['name'] == 'height':
                        resolution[0] = int(grandchild.attrib['value'])

    return pyredner.Camera(position     = position,
                           look_at      = look_at,
                           up           = up,
                           fov          = fov,
                           clip_near    = clip_near,
                           resolution   = resolution)

def parse_material(node, device, two_sided = False):

    def parse_material_bitmap(node, scale = None):
        reflectance_texture = None
        uv_scale = torch.tensor([1.0, 1.0])
        for grandchild in node:
            if grandchild.attrib['name'] == 'filename':
                reflectance_texture = pyredner.imread(grandchild.attrib['value'])
                if scale:
                    reflectance_texture = reflectance_texture * scale
            elif grandchild.attrib['name'] == 'uscale':
                uv_scale[0] = float(grandchild.attrib['value'])
            elif grandchild.attrib['name'] == 'vscale':
                uv_scale[1] = float(grandchild.attrib['value'])
        assert reflectance_texture is not None
        uv_scale = uv_scale.to(device = device)
        return reflectance_texture, uv_scale
    
    # support mitsuba pulgin 'scale' for texture
    def parse_texture(node):
        if node.attrib['type'] == 'scale':
            scale_value = None
            for grandchild in node:
                if grandchild.attrib['name'] == 'scale' and grandchild.tag == 'float':
                    scale_value = float(grandchild.attrib['value'])
                elif grandchild.attrib['type'] == 'bitmap' and grandchild.tag == 'texture':
                    assert scale_value is not None # avoid 'scale' element is declared below the 'bitmap'
                    return parse_material_bitmap(grandchild, scale_value)
                else:
                    raise NotImplementedError('Unsupported scale param type {}'.format(grandchild.child['type']))
        elif node.attrib['type'] == 'bitmap':
            return parse_material_bitmap(node)
        else:
            raise NotImplementedError('Unsupported Texture type {}'.format(node.attrib['type']))
    
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
                    diffuse_reflectance, diffuse_uv_scale = parse_texture(child)
                elif child.tag == 'rgb' or child.tag == 'spectrum' or child.tag == 'srgb':
                    diffuse_reflectance = parse_vector(child.attrib['value'])
                    if child.tag == 'srgb':
                        diffuse_reflectance = pyredner.srgb_to_linear(diffuse_reflectance)
            elif child.attrib['name'] == 'specular':
                if child.tag == 'texture':
                    specular_reflectance, specular_uv_scale = parse_texture(child)
                elif child.tag == 'rgb' or child.tag == 'spectrum' or child.tag == 'srgb':
                    specular_reflectance = parse_vector(child.attrib['value'])
                    if child.tag == 'srgb':
                        specular_reflectance = pyredner.srgb_to_linear(specular_reflectance)
            elif child.attrib['name'] == 'roughness':
                roughness = torch.tensor([float(child.attrib['value'])])
        diffuse_reflectance = diffuse_reflectance.to(device = device)
        specular_reflectance = specular_reflectance.to(device = device)
        roughness = roughness.to(device = device)
        diffuse_uv_scale = diffuse_uv_scale.to(device = device)
        specular_uv_scale = specular_uv_scale.to(device = device)
        return (node_id, pyredner.Material(\
                diffuse_reflectance = pyredner.Texture(diffuse_reflectance, diffuse_uv_scale),
                specular_reflectance = pyredner.Texture(specular_reflectance, specular_uv_scale),
                roughness = pyredner.Texture(roughness),
                two_sided = two_sided))
    elif node.attrib['type'] == 'roughplastic':
        diffuse_reflectance = torch.tensor([0.5, 0.5, 0.5])
        diffuse_uv_scale = torch.tensor([1.0, 1.0])
        specular_reflectance = torch.tensor([1.0, 1.0, 1.0])
        specular_uv_scale = torch.tensor([1.0, 1.0])
        roughness = torch.tensor([0.01])
        roughness_uv_scale = torch.tensor([1.0, 1.0])

        for child in node:
            if child.attrib['name'] == 'diffuseReflectance' or child.attrib['name'] == 'diffuse_reflectance':
                if child.tag == 'texture':
                    diffuse_reflectance, diffuse_uv_scale = parse_texture(child)
                elif child.tag == 'rgb' or child.tag == 'spectrum' or child.tag == 'srgb':
                    diffuse_reflectance = parse_vector(child.attrib['value'])
                    if child.tag == 'srgb':
                        diffuse_reflectance = pyredner.srgb_to_linear(diffuse_reflectance)
            elif child.attrib['name'] == 'specularReflectance' or child.attrib['name'] == 'specular_reflectance':
                if child.tag == 'texture':
                    specular_reflectance, specular_uv_scale = parse_texture(child)
                elif child.tag == 'rgb' or child.tag == 'spectrum' or child.tag == 'srgb':
                    specular_reflectance = parse_vector(child.attrib['value'])
                    if child.tag == 'srgb':
                        specular_reflectance = pyredner.srgb_to_linear(specular_reflectance)
            elif child.attrib['name'] == 'alpha':
                if child.tag == 'texture':
                    roughness, roughness_uv_scale = parse_texture(child)
                    roughness = roughness * roughness
                elif child.tag == 'float':
                    alpha = float(child.attrib['value'])
                    roughness = torch.tensor([alpha * alpha])
        diffuse_reflectance = diffuse_reflectance.to(device = device)
        specular_reflectance = specular_reflectance.to(device = device)
        roughness = roughness.to(device = device)
        diffuse_uv_scale = diffuse_uv_scale.to(device = device)
        specular_uv_scale = specular_uv_scale.to(device = device)
        roughness_uv_scale = roughness_uv_scale.to(device = device)
        return (node_id, pyredner.Material(\
                diffuse_reflectance = pyredner.Texture(diffuse_reflectance, diffuse_uv_scale),
                specular_reflectance = pyredner.Texture(specular_reflectance, specular_uv_scale),
                roughness = pyredner.Texture(roughness, roughness_uv_scale),
                two_sided = two_sided))
    elif node.attrib['type'] == 'twosided':
        ret = parse_material(node[0], device, True)
        return (node_id, ret[1])
    # Simply bypass mask's opacity
    elif node.attrib['type'] == 'mask': #TODO add opacity!!!
        ret = parse_material(node[0], device)
        return (node_id, ret[1])
    else:
        print('Unsupported material type:', node.attrib['type'])
        assert(False)

def parse_shape(node, material_dict, shape_id, device, shape_group_dict = None):
    if node.attrib['type'] == 'obj' or node.attrib['type'] == 'serialized':
        to_world = torch.eye(4)
        serialized_shape_id = 0
        mat_id = -1
        light_intensity = None
        filename = ''
        max_smooth_angle = -1
        for child in node:
            if 'name' in child.attrib:
                if child.attrib['name'] == 'filename':
                    filename = child.attrib['value']
                elif child.attrib['name'] == 'toWorld':
                    to_world = parse_transform(child)
                elif child.attrib['name'] == 'shapeIndex':
                    serialized_shape_id = int(child.attrib['value'])
                elif child.attrib['name'] == 'maxSmoothAngle':
                    max_smooth_angle = float(child.attrib['value'])
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
            # Load in CPU for rebuild_topology
            _, mesh_list, _ = pyredner.load_obj(filename, obj_group = False, device = torch.device('cpu'))
            vertices = mesh_list[0][1].vertices
            indices = mesh_list[0][1].indices
            uvs = mesh_list[0][1].uvs
            normals = mesh_list[0][1].normals
            uv_indices = mesh_list[0][1].uv_indices
            normal_indices = mesh_list[0][1].normal_indices
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
            uv_indices = None # Serialized doesn't use different indices for UV & normal
            normal_indices = None

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
        if max_smooth_angle >= 0:
            if normals is None:
                normals = torch.zeros_like(vertices)
            new_num_vertices = redner.rebuild_topology(\
                redner.float_ptr(vertices.data_ptr()),
                redner.int_ptr(indices.data_ptr()),
                redner.float_ptr(uvs.data_ptr() if uvs is not None else 0),
                redner.float_ptr(normals.data_ptr() if normals is not None else 0),
                redner.int_ptr(uv_indices.data_ptr() if uv_indices is not None else 0),
                int(vertices.shape[0]),
                int(indices.shape[0]),
                max_smooth_angle)
            print('Rebuilt topology, original vertices size: {}, new vertices size: {}'.format(\
                int(vertices.shape[0]), new_num_vertices))
            vertices.resize_(new_num_vertices, 3)
            if uvs is not None:
                uvs.resize_(new_num_vertices, 2)
            if normals is not None:
                normals.resize_(new_num_vertices, 3)

        lgt = None
        if light_intensity is not None:
            lgt = pyredner.AreaLight(shape_id, light_intensity)

        vertices = vertices.to(device)
        indices = indices.to(device)
        if uvs is not None:
            uvs = uvs.to(device)
        if normals is not None:
            normals = normals.to(device)
        if uv_indices is not None:
            uv_indices = uv_indices.to(device)
        if normal_indices is not None:
            normal_indices = normal_indices.to(device)
        return pyredner.Shape(vertices,
                              indices,
                              uvs=uvs,
                              normals=normals,
                              uv_indices=uv_indices,
                              normal_indices=normal_indices,
                              material_id=mat_id), lgt
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
            lgt = pyredner.AreaLight(shape_id, light_intensity)

        vertices = vertices.to(device)
        indices = indices.to(device)
        if uvs is not None:
            uvs = uvs.to(device)
        if normals is not None:
            normals = normals.to(device)
        return pyredner.Shape(vertices, indices, uvs=uvs, normals=normals, material_id=mat_id), lgt
    # Add instance support 
    # TODO (simply transform & create a new shape now)
    elif node.attrib['type'] == 'instance':
        shape = None
        for child in node:
            if 'name' in child.attrib:
                if child.attrib['name'] == 'toWorld':
                    to_world = parse_transform(child)
            if child.tag == 'ref':
                shape = shape_group_dict[child.attrib['id']]
        # transform instance
        vertices = shape.vertices
        normals = shape.normals
        vector1 = torch.ones(vertices.shape[0], 1, device = vertices.device)
        to_world = to_world.to(vertices.device)
        vertices = torch.cat((vertices, vector1), dim = 1)
        vertices = vertices @ torch.transpose(to_world, 0, 1)
        vertices = vertices / vertices[:, 3:4]
        vertices = vertices[:, 0:3].contiguous()
        if normals is not None:
            normals = normals @ (torch.inverse(torch.transpose(to_world, 0, 1))[:3, :3])
            normals = normals.contiguous()
        # assert(vertices is not None)
        # assert(indices is not None)
        # lgt = None
        # if light_intensity is not None:
        #     lgt = pyredner.AreaLight(shape_id, light_intensity)

        return pyredner.Shape(vertices, shape.indices, uvs=shape.uvs, normals=normals, material_ids=shape.material_id), None
    else:
        print('Shape type {} is not supported!'.format(node.attrib['type']))
        assert(False)

def parse_scene(node, device):
    cam = None
    resolution = None
    materials = []
    material_dict = {}
    shapes = []
    lights = []
    shape_group_dict = {}
    envmap = None

    for child in node:
        if child.tag == 'sensor':
            cam = parse_camera(child)
        elif child.tag == 'bsdf':
            node_id, material = parse_material(child, device)
            if node_id is not None:
                material_dict[node_id] = len(materials)
                materials.append(material)
        # shapegroup for instancing
        elif child.tag == 'shape' and child.attrib['type'] == 'shapegroup':
            for child_s in child:
                if child_s.tag == 'shape':
                    shape_group_dict[child.attrib['id']] = parse_shape(child_s, material_dict, None)[0]
        elif child.tag == 'shape':
            shape, light = parse_shape(child, material_dict, len(shapes), device, shape_group_dict if child.attrib['type'] == 'instance' else None)
            shapes.append(shape)
            if light is not None:
                lights.append(light)
        # Add envmap loading support
        elif child.tag == 'emitter' and child.attrib['type'] == 'envmap':
            # read envmap params from xml
            scale = 1.0
            envmap_filename = None
            to_world = torch.eye(4)
            for child_s in child:
                if child_s.attrib['name'] == 'scale':
                    assert child_s.tag == 'float'
                    scale = float(child_s.attrib['value'])
                if child_s.attrib['name'] == 'filename':
                    assert child_s.tag == 'string'
                    envmap_filename = child_s.attrib['value']
                if child_s.attrib['name'] == 'toWorld':
                    to_world = parse_transform(child_s)
            # load envmap
            envmap = scale * pyredner.imread(envmap_filename).to(device)
            envmap = pyredner.EnvironmentMap(envmap, env_to_world=to_world)
    return pyredner.Scene(cam, shapes, materials, lights, envmap)

def load_mitsuba(filename: str,
                 device: Optional[torch.device] = None):
    """
        Load from a Mitsuba scene file as PyTorch tensors.

        Args
        ====
        filename: str
            Path to the Mitsuba scene file.
        device: Optional[torch.device]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device().

        Returns
        =======
        pyredner.Scene
    """
    if device is None:
        device = pyredner.get_device()

    tree = etree.parse(filename)
    root = tree.getroot()
    cwd = os.getcwd()
    os.chdir(os.path.dirname(filename))
    ret = parse_scene(root, device)
    os.chdir(cwd)
    return ret
