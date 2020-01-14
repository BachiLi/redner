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
        if pyredner.get_use_gpu():
            uv_scale = uv_scale.cuda(device = pyredner.get_device())
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
                elif child.tag == 'rgb' or child.tag == 'spectrum':
                    diffuse_reflectance = parse_vector(child.attrib['value'])
            elif child.attrib['name'] == 'specular':
                if child.tag == 'texture':
                    specular_reflectance, specular_uv_scale = parse_texture(child)
                elif child.tag == 'rgb' or child.tag == 'spectrum':
                    specular_reflectance = parse_vector(child.attrib['value'])
            elif child.attrib['name'] == 'roughness':
                roughness = torch.tensor([float(child.attrib['value'])])
        if pyredner.get_use_gpu():
            # Copy to GPU
            diffuse_reflectance = diffuse_reflectance.cuda(device=pyredner.get_device())
            specular_reflectance = specular_reflectance.cuda(device=pyredner.get_device())
            roughness = roughness.cuda(device=pyredner.get_device())
            diffuse_uv_scale = diffuse_uv_scale.cuda(device = pyredner.get_device())
            specular_uv_scale = specular_uv_scale.cuda(device = pyredner.get_device())
        return (node_id, pyredner.Material(\
                diffuse_reflectance = pyredner.Texture(diffuse_reflectance, diffuse_uv_scale),
                specular_reflectance = pyredner.Texture(specular_reflectance, specular_uv_scale),
                roughness = pyredner.Texture(roughness),
                two_sided = two_sided))
    elif node.attrib['type'] == 'roughplastic':
        diffuse_reflectance = torch.tensor([0.5, 0.5, 0.5])
        diffuse_uv_scale = torch.tensor([1.0, 1.0])
        specular_reflectance = torch.tensor([0.0, 0.0, 0.0])
        specular_uv_scale = torch.tensor([1.0, 1.0])
        roughness = torch.tensor([1.0])
        roughness_uv_scale = torch.tensor([1.0, 1.0])

        for child in node:
            if child.attrib['name'] == 'diffuseReflectance':
                if child.tag == 'texture':
                    diffuse_reflectance, diffuse_uv_scale = parse_texture(child)
                elif child.tag == 'rgb' or child.tag == 'spectrum':
                    diffuse_reflectance = parse_vector(child.attrib['value'])
            elif child.attrib['name'] == 'specularReflectance':
                if child.tag == 'texture':
                    specular_reflectance, specular_uv_scale = parse_texture(child)
                elif child.tag == 'rgb' or child.tag == 'spectrum':
                    specular_reflectance = parse_vector(child.attrib['value'])
            elif child.attrib['name'] == 'alpha':
                # Add 'alpha texture' support
                if child.tag == 'texture':
                    roughness, roughness_uv_scale = parse_texture(child) #? not sure to do square here
                elif child.tag == 'float':
                    alpha = float(child.attrib['value'])
                    roughness = torch.tensor([alpha * alpha])
        if pyredner.get_use_gpu():
            # Copy to GPU
            diffuse_reflectance = diffuse_reflectance.cuda(device=pyredner.get_device())
            specular_reflectance = specular_reflectance.cuda(device=pyredner.get_device())
            roughness = roughness.cuda(device=pyredner.get_device())
            diffuse_uv_scale = diffuse_uv_scale.cuda(device = pyredner.get_device())
            specular_uv_scale = specular_uv_scale.cuda(device = pyredner.get_device())
            roughness_uv_scale = roughness_uv_scale.cuda(device = pyredner.get_device())
        return (node_id, pyredner.Material(\
                diffuse_reflectance = pyredner.Texture(diffuse_reflectance, diffuse_uv_scale),
                specular_reflectance = pyredner.Texture(specular_reflectance, specular_uv_scale),
                roughness = pyredner.Texture(roughness, roughness_uv_scale),
                two_sided = two_sided))
    elif node.attrib['type'] == 'twosided':
        ret = parse_material(node[0], True)
        return (node_id, ret[1])
    # Simply bypass mask's opacity
    elif node.attrib['type'] == 'mask': #TODO add opacity!!!
        ret = parse_material(node[0])
        return (node_id, ret[1])
    else:
        print('Unsupported material type:', node.attrib['type'])
        assert(False)

def parse_shape(node, material_dict, shape_id, shape_group_dict = None):
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
            _, mesh_list, _ = pyredner.load_obj(filename)
            # Convert to CPU for rebuild_topology
            vertices = mesh_list[0][1].vertices.cpu()
            indices = mesh_list[0][1].indices.cpu()
            uvs = mesh_list[0][1].uvs
            normals = mesh_list[0][1].normals
            uv_indices = mesh_list[0][1].uv_indices
            normal_indices = mesh_list[0][1].normal_indices
            if uvs is not None:
                uvs = uvs.cpu()
            if normals is not None:
                normals = normals.cpu()
            if uv_indices is not None:
                uv_indices = uv_indices.cpu()
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

        if pyredner.get_use_gpu():
            # Copy to GPU
            vertices = vertices.cuda(device=pyredner.get_device())
            indices = indices.cuda(device=pyredner.get_device())
            if uvs is not None:
                uvs = uvs.cuda(device=pyredner.get_device())
            if normals is not None:
                normals = normals.cuda(device=pyredner.get_device())
            if uv_indices is not None:
                uv_indices = uv_indices.cuda(device=pyredner.get_device())
            if normal_indices is not None:
                normal_indices = normal_indices.cuda(device=pyredner.get_device())
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

        if pyredner.get_use_gpu():
            # Copy to GPU
            vertices = vertices.cuda(device=pyredner.get_device())
            indices = indices.cuda(device=pyredner.get_device())
            if uvs is not None:
                uvs = uvs.cuda(device=pyredner.get_device())
            if normals is not None:
                normals = normals.cuda(device=pyredner.get_device())
        return pyredner.Shape(vertices, indices, uvs=uvs, normals=normals, material_id=mat_id), lgt
    # Add instance support 
    # TODO (simply transform & create a new shape now)
    elif node.attrib['type'] == 'instance':
        shape = None
        for child in node:
            if 'name' in child.attrib:
                if child.attrib['name'] == 'toWorld':
                    to_world = parse_transform(child)
                    if pyredner.get_use_gpu():
                        to_world = to_world.cuda()
            if child.tag == 'ref':
                shape = shape_group_dict[child.attrib['id']]
        # transform instance
        vertices = shape.vertices
        normals = shape.normals
        vector1 = torch.ones(vertices.shape[0], 1)
        vertices = torch.cat((vertices, vector1.cuda() if pyredner.get_use_gpu() else vector1), dim = 1)
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

def parse_scene(node):
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
            node_id, material = parse_material(child)
            if node_id is not None:
                material_dict[node_id] = len(materials)
                materials.append(material)
        # shapegroup for instancing
        elif child.tag == 'shape' and child.attrib['type'] == 'shapegroup':
            for child_s in child:
                if child_s.tag == 'shape':
                    shape_group_dict[child.attrib['id']] = parse_shape(child_s, material_dict, None)[0]
        elif child.tag == 'shape':
            shape, light = parse_shape(child, material_dict, len(shapes), shape_group_dict if child.attrib['type'] == 'instance' else None)
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
            envmap = scale * pyredner.imread(envmap_filename)
            if pyredner.get_use_gpu():
                envmap = envmap.cuda()
            envmap = pyredner.EnvironmentMap(envmap, env_to_world=to_world)
    return pyredner.Scene(cam, shapes, materials, lights, envmap)

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
