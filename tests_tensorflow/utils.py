from __future__ import absolute_import, division, print_function
from typing import List, Set, Dict, Tuple, Optional, Callable, Union

import tensorflow as tf
import torch

import pyrednertensorflow as pyrednertorch
import pyrednertensorflow as pyredner
import numpy as np

import pdb

def is_same_tensor(a:tf.Tensor, b:torch.Tensor, atol=0.0001) -> bool:
    is_same = False
    try:
        is_same = np.alltrue(a.numpy() == b.numpy()) or np.allclose(a.numpy(), b.numpy(), atol=atol)
    except RuntimeError:
        print("Detach")
        is_same = np.alltrue(a.numpy() == b.detach().numpy()) \
            or np.allclose(a.numpy(), b.detach().numpy(), atol=atol)
    finally:
        return is_same

def is_same_grads(a:pyredner.SceneGrads, b:List) -> bool:
    if not is_same_tensor(a.d_position, b[4]): 
        print(4); return False
    if not is_same_tensor(a.d_look_at, b[5]): 
        print(5); return False
    if not is_same_tensor(a.d_up, b[6]): 
        print(6); return False
    if not is_same_tensor(a.d_ndc_to_cam, b[7]): 
        print(7); return False
    if not is_same_tensor(a.d_cam_to_ndc, b[8]): 
        print(8); return False
    if not is_same_container(a.d_vertices_list, [b[12]]): 
        print(12); return False
    if not is_same_container(a.d_uvs_list, [b[14]]): 
        print(14); return False
    if not is_same_container(a.d_normals_list, [b[15]]): 
        print(15); return False
    if not is_same_container(a.d_diffuse_list, [b[18]]): 
        print(18); return False
    if not is_same_container(a.d_specular_list, [b[20]]): 
        print(20); return False
    if not is_same_container(a.d_roughness_list, [b[22]]): 
        print(22); return False
    if not is_same_container(a.d_intensity_list, [b[26]]): 
        print(26); return False
    if not is_same_tensor(a.d_envmap_values, b[28]): 
        print(28); return False
    if not is_same_tensor(a.d_world_to_env, b[31]): 
        print(31); return False
    return True


def is_same_optional(
    a: Union[pyredner.EnvironmentMap, tf.Tensor, None], 
    b: Union[pyrednertorch.EnvironmentMap, torch.Tensor, None], 
    func:Callable, 
    atol=0.0001) -> bool:
    '''Test for optional tensor parameters. 

    Args:
        a(tf.Tensor or None)
        b(torch.Tensor or None)
        
    '''
    if is_only_one_element_none_in_pair(a, b):
        return False
    elif a is None and b is None:
        return True
    elif func == is_same_tensor:
        return func(a,b, atol)
    else:
        return func(a,b)

def is_same_camera(a:pyredner.Camera, b:pyrednertorch.Camera):
    if not is_same_tensor(a.position, b.position):
        pdb.set_trace()
    if not is_same_tensor(a.look_at, b.look_at):
        pdb.set_trace()
    if not is_same_tensor(a.up, b.up):
        pdb.set_trace()
    if not is_same_tensor(a.fov, b.fov):
        pdb.set_trace()
    if not is_same_tensor(a.cam_to_ndc, b.cam_to_ndc):
        pdb.set_trace()
    if not is_same_tensor(a.ndc_to_cam, b.ndc_to_cam):
        pdb.set_trace()
    return True 

def is_same_pdf_norm(a:Union[tf.Tensor, float], b:float) -> bool:
    if isinstance(a, tf.Tensor):
        return  np.isclose(a.numpy(), b)
    else:
        return  np.isclose(a, b)

def is_same_envmap(a:pyredner.EnvironmentMap, b:pyredner.EnvironmentMap):
    return is_same_texture(a.values, b.values) \
        and is_same_tensor(a.env_to_world, b.env_to_world) \
        and is_same_tensor(a.world_to_env, b.world_to_env) \
        and is_same_tensor(a.sample_cdf_xs, b.sample_cdf_xs) \
        and is_same_tensor(a.sample_cdf_ys, b.sample_cdf_ys) \
        and is_same_pdf_norm(a.pdf_norm, b.pdf_norm)
        

def is_same_texture(a:pyredner.Texture, b:pyrednertorch.Texture) -> bool:
    return is_same_tensor(a.texels, b.texels) \
        and is_same_tensor(a.mipmap, b.mipmap) \
        and is_same_tensor(a.uv_scale, b.uv_scale) 


def is_same_material(a:pyredner.Material, b:pyrednertorch.Material):
    if not is_same_texture(a.diffuse_reflectance, b.diffuse_reflectance):
        return False

    if not is_same_texture(a.specular_reflectance, b.specular_reflectance):
        return False

    if not is_same_texture(a.roughness, b.roughness):
        return False

    return a.two_sided == b.two_sided

def is_only_one_element_none_in_pair(a, b):
    is_a_none_b_not_none = a is None and b is not None
    is_a_not_none_b_none = a is not None and b is None

    return is_a_none_b_not_none or is_a_not_none_b_none


def is_same_shape(a:pyredner.Shape, b:pyrednertorch.Shape):
    if not is_same_tensor(a.vertices, b.vertices):
        pdb.set_trace()
    if not is_same_tensor(a.indices, b.indices):
        pdb.set_trace()
    
    if not is_same_optional(a.normals, b.normals, is_same_tensor):
        pdb.set_trace()

    if not is_same_optional(a.uvs, b.uvs, is_same_tensor):
        pdb.set_trace()

    return True

def is_same_area_light(a:pyredner.AreaLight, b:pyrednertorch.AreaLight) -> bool:
    return a.shape_id == b.shape_id \
        and is_same_tensor(a.intensity, b.intensity) \
        and a.two_sided == b.two_sided

def is_same_container(container1:List, container2:List) -> bool:
    assert len(container1) == len(container2)

    if len(container1) == 0:
        return True

    compare_func = None

    if isinstance(container1[0], pyredner.Material):
        compare_func = is_same_material
    elif isinstance(container1[0], pyredner.AreaLight):
        compare_func = is_same_area_light
    elif isinstance(container1[0], pyredner.Shape):
        compare_func = is_same_shape
    elif isinstance(container1[0], tf.Tensor) or isinstance(container1[0], tf.Variable):
        compare_func = is_same_tensor
    else:
        return False

    for c1, c2 in zip(container1, container2):
        if not compare_func(c1, c2):
            return False

    return True

def is_same_image(a: tf.Tensor, b:torch.Tensor) -> bool:
    diff_channels = [
        i for i in range(a.shape[2]) if not is_same_tensor(a[:,:,i], b[:,:,i])
        ]
    if len(diff_channels) == 0:
        return True


    diff_channels = [
        'RGB'[i] for i in diff_channels
    ]
    
    for c in diff_channels:
        print(f'{c} - channel is different')

    return False


def is_same_scene(scene1:pyredner.Scene, scene2:pyrednertorch.Scene) -> bool:        

    if not is_same_optional(scene1.envmap, scene2.envmap, is_same_envmap):
        return False

    return is_same_camera(scene1.camera, scene2.camera) \
        and is_same_container(scene1.shapes, scene2.shapes) \
        and is_same_container(scene1.materials, scene2.materials) \
        and is_same_container(scene1.area_lights, scene2.area_lights)

def is_same_scene_args(args1:pyredner.SceneArgs, args2:List) -> bool:
    i = 0
    assert args1.num_shapes == args2[i]
    i += 1
    assert args1.num_materials == args2[i]
    i += 1
    assert args1.num_lights == args2[i]
    i += 1
    assert is_same_tensor(args1.position, args2[i])
    i += 1
    assert is_same_tensor(args1.look_at, args2[i])
    i += 1
    assert is_same_tensor(args1.up, args2[i])
    i += 1
    assert is_same_tensor(args1.ndc_to_cam, args2[i])
    i += 1
    assert is_same_tensor(args1.cam_to_ndc, args2[i])
    i += 1
    
    assert args1.clip_near == args2[i]
    i += 1
    assert args1.resolution == args2[i]
    i += 1
    assert args1.fisheye == args2[i]
    i += 1

    for j in range(len(args1.shapes)):
        
        assert is_same_tensor(args1.shapes[j].vertices, args2[i])
        i += 1
        assert is_same_tensor(args1.shapes[j].indices, args2[i])
        i += 1
        assert is_same_optional(args1.shapes[j].uvs, args2[i], is_same_tensor)
        i += 1
        assert is_same_optional(args1.shapes[j].normals, args2[i], is_same_tensor)
        i += 1
        assert args1.shapes[j].material_id == args2[i]
        i += 1
        assert args1.shapes[j].light_id == args2[i]
        i += 1
        
    for j in range(len(args1.materials)):
        assert is_same_tensor(args1.materials[j].diffuse_reflectance.mipmap, args2[i])
        i += 1
        assert is_same_tensor(args1.materials[j].diffuse_reflectance.uv_scale, args2[i])
        i += 1
        assert is_same_tensor(args1.materials[j].specular_reflectance.mipmap, args2[i])
        i += 1
        assert is_same_tensor(args1.materials[j].specular_reflectance.uv_scale, args2[i])
        i += 1
        assert is_same_tensor(args1.materials[j].roughness.mipmap, args2[i])
        i += 1
        assert is_same_tensor(args1.materials[j].roughness.uv_scale, args2[i])
        i += 1
        assert args1.materials[j].two_sided == args2[i]
        i += 1
        
    for j in range(len(args1.lights)):
        assert args1.lights[j].shape_id == args2[i]
        i += 1
        assert is_same_tensor(args1.lights[j].intensity, args2[i])
        i += 1
        assert args1.lights[j].two_sided == args2[i]
        i += 1
        
    if args2[i] is not None:
        assert is_same_tensor(args1.envmap_mipmap, args2[i])
        i += 1
        assert is_same_tensor(args1.envmap_uv_scale, args2[i])
        i += 1
        assert is_same_tensor(args1.envmap_env_to_world, args2[i])
        i += 1
        assert is_same_tensor(args1.envmap_world_to_env, args2[i])
        i += 1
        assert is_same_tensor(args1.envmap_sample_cdf_ys, args2[i])
        i += 1
        assert is_same_tensor(args1.envmap_sample_cdf_xs, args2[i])
        i += 1
        assert is_same_pdf_norm(args1.envmap_pdf_norm, args2[i])
        i += 1
        
    else:
        i += 7



    

    assert args1.num_samples == args2[i]
    i += 1
    assert args1.max_bounces == args2[i]
    i += 1
    assert args1.channels == args2[i]
    i += 1
    assert args1.sampler_type == args2[i]
    i += 1
    assert args1.use_primary_edge_sampling == args2[i]
    i += 1
    assert args1.use_secondary_edge_sampling == args2[i]
    i += 1
    return True