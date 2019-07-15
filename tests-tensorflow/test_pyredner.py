
from __future__ import absolute_import, division, print_function
from typing import List, Set, Dict, Tuple, Optional

import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager

import torch

import redner
import pyrednertensorflow as pyrednertorch
import pyrednertensorflow as pyredner
import numpy as np
import pdb
import pickle
from utils import *


class TestRednerFunction(tf.test.TestCase):

    def testSerializeScene(self):
        cam = pyredner.Camera(position = tfe.Variable([0.0, 0.0, -5.0], dtype=tf.float32),
                      look_at = tfe.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
                      up = tfe.Variable([0.0, 1.0, 0.0], dtype=tf.float32),
                      fov = tfe.Variable([45.0], dtype=tf.float32), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)

        mat_grey = pyredner.Material(
            diffuse_reflectance = \
                tfe.Variable([0.4, 0.4, 0.4], ),
            specular_reflectance = \
                tfe.Variable([0.5, 0.5, 0.5], ),
            roughness = \
                tfe.Variable([0.05], ))

        materials = [mat_grey]

        vertices, indices, uvs, normals = pyredner.generate_sphere(128, 64)
        shape_sphere = pyredner.Shape(
            vertices = vertices,
            indices = indices,
            uvs = uvs,
            normals = normals,
            material_id = 0)
        shapes = [shape_sphere]

        envmap = pyredner.imread('sunsky.exr')
        envmap = pyredner.EnvironmentMap(envmap)
        scene = pyredner.Scene(cam, shapes, materials, [], envmap)
        scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 256,
            max_bounces = 1)

        for e in scene_args:
            print(e)
            self.assertTrue(isinstance(e, tf.Tensor) or isinstance(e, tf.Variable), f'{e}')


    def testSerializeSceneWithoutUVS(self):
        cam = pyredner.Camera(position = tfe.Variable([0.0, 0.0, -5.0], dtype=tf.float32),
                      look_at = tfe.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
                      up = tfe.Variable([0.0, 1.0, 0.0], dtype=tf.float32),
                      fov = tfe.Variable([45.0], dtype=tf.float32), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)

        mat_grey = pyredner.Material(
            diffuse_reflectance = \
                tfe.Variable([0.4, 0.4, 0.4], ),
            specular_reflectance = \
                tfe.Variable([0.5, 0.5, 0.5], ),
            roughness = \
                tfe.Variable([0.05], ))

        materials = [mat_grey]

        vertices, indices, uvs, normals = pyredner.generate_sphere(128, 64)
        shape_sphere = pyredner.Shape(
            vertices = vertices,
            indices = indices,
            material_id = 0)
        shapes = [shape_sphere]

        envmap = pyredner.imread('sunsky.exr')
        envmap = pyredner.EnvironmentMap(envmap)
        scene = pyredner.Scene(cam, shapes, materials, [], envmap)
        scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 256,
            max_bounces = 1)

        for e in scene_args:
            print(e)
            self.assertTrue(isinstance(e, tf.Tensor) or isinstance(e, tf.Variable), f'{e}')

    def testSerializeSceneWithoutUVSAndEnvmap(self):
        cam = pyredner.Camera(position = tfe.Variable([0.0, 0.0, -5.0], dtype=tf.float32),
                      look_at = tfe.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
                      up = tfe.Variable([0.0, 1.0, 0.0], dtype=tf.float32),
                      fov = tfe.Variable([45.0], dtype=tf.float32), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)

        mat_grey = pyredner.Material(
            diffuse_reflectance = \
                tfe.Variable([0.4, 0.4, 0.4], ),
            specular_reflectance = \
                tfe.Variable([0.5, 0.5, 0.5], ),
            roughness = \
                tfe.Variable([0.05], ))

        materials = [mat_grey]

        vertices, indices, uvs, normals = pyredner.generate_sphere(128, 64)
        shape_sphere = pyredner.Shape(
            vertices = vertices,
            indices = indices,
            material_id = 0)
        shapes = [shape_sphere]

        scene = pyredner.Scene(cam, shapes, materials, [])
        scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 256,
            max_bounces = 1)

        for e in scene_args:
            print(e)
            self.assertTrue(isinstance(e, tf.Tensor) or isinstance(e, tf.Variable), f'{e}')



# class TestCamera(tf.test.TestCase):

#     def setUp(self):
#         position = np.array([0.0, 0.0, -5.0], dtype=np.float32)
#         look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
#         up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
#         fov = np.array([45.0], dtype=np.float32)
#         clip_near = 1e-2

#         resolution = (256, 256)
#         self.cam = pyredner.Camera(
#             position = tfe.Variable(position),
#             look_at = tfe.Variable(look_at),
#             up = tfe.Variable(up),
#             fov = tfe.Variable(fov),
#             clip_near = clip_near,
#             resolution = resolution,
#             fisheye = True)

#         self.cam_torch = pyrednertorch.Camera(
#             position = torch.tensor(position),
#             look_at = torch.tensor(look_at),
#             up = torch.tensor(up),
#             fov = torch.tensor(fov), # in degree
#             clip_near = clip_near, # needs to > 0
#             resolution = resolution,
#             fisheye = False)


#     def testCameraPropertiesCoherenceToTorch(self):
#         """Test Camera property values between TF version and Torch.
#         """
#         # with open('cam_state_dict.pickle', 'rb') as handle:
#         #     state_dict_torch = pickle.load(handle)
#         # state_dict_tf = self.cam.state_numpy_dict()
#         # for k in state_dict_torch.keys():
#         #     pdb.set_trace()
#         #     self.assertAllEqual(
#         #         state_dict_torch[k], state_dict_tf[k]
#         #     )
#         self.assertAllEqual(self.cam.position.numpy(), self.cam_torch.position.numpy())
#         self.assertAllEqual(self.cam.look_at.numpy(), self.cam_torch.look_at.numpy())
#         self.assertAllEqual(self.cam.up.numpy(), self.cam_torch.up.numpy())
#         self.assertAllEqual(self.cam.fov.numpy(), self.cam_torch.fov.numpy())
#         self.assertAllEqual(self.cam.cam_to_ndc.numpy(), self.cam_torch.cam_to_ndc.numpy())
#         self.assertAllEqual(self.cam.ndc_to_cam.numpy(), self.cam_torch.ndc_to_cam.numpy())

# class TestTexture(tf.test.TestCase):
#     '''Test is meaningful only for tensors with dimension >= 2
#     '''

#     def setUp(self):
#         pyredner.DEBUG = True
#         pyrednertorch.DEBUG = True
#         texels = pyredner.imread('sunsky.exr')
#         self.texture_tf = pyredner.Texture(texels)

#         texels = pyrednertorch.imread('sunsky.exr')
#         self.texture_torch = pyrednertorch.Texture(texels)

#     def tearDown(self):
#         pyredner.DEBUG = False
#         pyrednertorch.DEBUG = False

#     def testWidth(self):
#         self.assertEqual(self.texture_tf.width, self.texture_torch.width)

#     def testConvolution(self):
#         import os
#         for fname in os.listdir('results/texture-tf'):
#             if not fname.endswith('png'):
#                 continue
#             img_tf = pyredner.imread(f'results/texture-tf/{fname}')
#             img_torch = pyredner.imread(f'results/texture-torch/{fname}')

#             self.assertTrue(np.allclose(img_tf.numpy(), img_torch.numpy()))

        
#     def testNumLevels(self):
#         self.assertEqual(self.texture_tf.num_levels, self.texture_torch.num_levels)

#     def testMipMapAfterBroadcast(self):
#         self.assertAllEqual(
#             self.texture_tf.mipmap_after_broadcast.numpy(), self.texture_torch.mipmap_after_broadcast.numpy()
#             )

#     def testBoxFilter(self):
#         self.assertAllEqual(
#             tf.transpose(
#                 tf.expand_dims(
#                     self.texture_tf.box_filter[:,:,:,0], axis=-1), 
#                     perm=[2,3,0,1]
#                 ).numpy(), 
#             self.texture_torch.box_filter.numpy())

#     def testBaseLevel(self):
#         self.assertAllEqual(
#             self.texture_tf.base_level.numpy(), 
#             self.texture_torch.base_level.numpy())

#     def testLevel(self):
#         self.assertAllEqual(
#             self.texture_tf.level.numpy(), 
#             self.texture_torch.level.numpy())

#     def testMipMap(self):
#         self.assertAllEqual(
#             self.texture_tf.mipmap.numpy(), self.texture_torch.mipmap.numpy())

#     def testUV(self):
#         self.assertAllEqual(
#             self.texture_tf.uv_scale.numpy(), self.texture_torch.uv_scale.numpy())
                




# class TestMaterial(tf.test.TestCase):

#     def setUp(self):
#         self.mat_tf = pyredner.Material(
#             diffuse_reflectance=tfe.Variable([0.4, 0.4, 0.4]),
#             specular_reflectance=tfe.Variable([0.5, 0.5, 0.5]),
#             roughness=tfe.Variable([0.05]))
#         self.mat_torch = pyrednertorch.Material(
#             diffuse_reflectance=torch.tensor([0.4, 0.4, 0.4]),
#             specular_reflectance=torch.tensor([0.5, 0.5, 0.5]),
#             roughness=torch.tensor([0.05]))

#     def testDiffuseReflectance(self):
#         self.assertTrue(is_same_texture(
#             self.mat_tf.diffuse_reflectance, 
#             self.mat_torch.diffuse_reflectance))
    
#     def testSpecularReflectance(self):
#         self.assertTrue(is_same_texture(
#             self.mat_tf.specular_reflectance, 
#             self.mat_torch.specular_reflectance))
    
#     def testRoughness(self):
#         self.assertTrue(is_same_texture(
#             self.mat_tf.roughness, 
#             self.mat_torch.roughness))

#     def testTwoSided(self):
#         self.assertEqual(self.mat_tf.two_sided, self.mat_torch.two_sided)
    


# class TestEnvironmentMap(tf.test.TestCase):

#     def setUp(self):
#         pyredner.DEBUG = False
#         pyrednertorch.DEBUG = False
#         envmap = pyredner.imread('sunsky.exr')
#         self.envmap_tf = pyredner.EnvironmentMap(envmap)
#         envmap = pyrednertorch.imread('sunsky.exr')
#         self.envmap_torch = pyrednertorch.EnvironmentMap(envmap)

#     def testTexture(self):
#         self.assertTrue(is_same_texture(self.envmap_tf.values, self.envmap_torch.values))
        
#     def testEnvToWorld(self):
#         self.assertTrue(is_same_tensor(
#             self.envmap_tf.env_to_world, self.envmap_torch.env_to_world
#         ))

#     def testWorldToEnv(self):
#         self.assertTrue(is_same_tensor(
#             self.envmap_tf.world_to_env, self.envmap_torch.world_to_env
#         ))

#     def testSampleCDFYs(self):
#         # pdb.set_trace()
#         self.assertTrue(is_same_tensor(
#             self.envmap_tf.sample_cdf_ys, self.envmap_torch.sample_cdf_ys
#         ))

#     def testSampleCDFXs(self):
#         self.assertTrue(is_same_tensor(
#             self.envmap_tf.sample_cdf_xs, self.envmap_torch.sample_cdf_xs
#         ))

#     def testPDFNorm(self):
#         self.assertAlmostEqual(
#             self.envmap_tf.pdf_norm.numpy(), self.envmap_torch.pdf_norm
#         )


# class TestShape(tf.test.TestCase):

#     def setUp(self):
#         floor_vertices = tfe.Variable([[-20.0,0.0,-20.0],[-20.0,0.0,20.0],[20.0,0.0,-20.0],[20.0,0.0,20.0]],
#     )
#         floor_indices = tfe.Variable([[0,1,2], [1,3,2]], dtype=tf.int32)
#         self.shape_tf = pyredner.Shape(floor_vertices, floor_indices, None, None, 0)

#         floor_vertices = torch.tensor([[-20.0,0.0,-20.0],[-20.0,0.0,20.0],[20.0,0.0,-20.0],[20.0,0.0,20.0]])
#         floor_indices = torch.tensor([[0,1,2], [1,3,2]], dtype = torch.int32)
#         self.shape_torch = pyrednertorch.Shape(floor_vertices, floor_indices, None, None, 0)

#     def testShapes(self):
#         self.assertTrue(is_same_shape(self.shape_tf, self.shape_torch))

# class TestTransform(tf.test.TestCase):

#     def testNormalize(self):
#         v = np.random.rand(3,10,5,7)
#         self.assertTrue(is_same_tensor(
#             pyredner.transform.normalize(v),
#             pyrednertorch.transform.normalize(torch.tensor(v))
#         ))

#     def testGenLookAtMatrix(self):
#         self.assertTrue(is_same_tensor(
#             pyredner.transform.gen_scale_matrix(tf.constant([1, 2, 3], dtype=tf.float32)),
#             pyrednertorch.transform.gen_scale_matrix(torch.tensor([1, 2, 3], dtype=torch.float32))
#         ))

#     def testGenTranslateMatrix(self):
#         self.assertTrue(is_same_tensor(
#             pyredner.transform.gen_translate_matrix(
#                 tf.constant([1, 2, 3], dtype=tf.float32)),
#             pyrednertorch.transform.gen_translate_matrix(
#                 torch.tensor([1, 2, 3], dtype=torch.float32))
#         ))

#     # def testGenPerspectiveMatrix(self):
#     #     self.assertTrue(is_same_tensor(
#     #         pyredner.transform.gen_perspective_matrix(
#     #             tf.constant([1, 2, 3], dtype=tf.float32)),
#     #         pyrednertorch.transform.gen_perspective_matrix(
#     #             torch.tensor([1, 2, 3], dtype=torch.float32))
#     #     ))
        
#     def testGenRotateMatrix(self):
#         self.assertTrue(is_same_tensor(
#             pyredner.transform.gen_rotate_matrix(
#                 tf.constant([0.1, -0.1, 0.1], dtype=tf.float32)),
#             pyrednertorch.transform.gen_rotate_matrix(
#                 torch.tensor([0.1, -0.1, 0.1], dtype=torch.float32))
#         ))


# class TestUtils(tf.test.TestCase):

#     def testGenerateSphere(self):
#         vertices_tf, indices_tf, uvs_tf, normals_tf = pyredner.generate_sphere(128, 64)
#         vertices_torch, indices_torch, uvs_torch, normals_torch = pyredner.generate_sphere(128, 64)

#         self.assertTrue(is_same_tensor(
#             vertices_tf, vertices_tf
#         ))
#         self.assertTrue(is_same_tensor(
#             indices_tf, indices_torch
#         ))
#         self.assertTrue(is_same_tensor(
#             uvs_tf, uvs_torch
#         ))
#         self.assertTrue(is_same_tensor(
#             normals_tf, normals_torch
#         ))
        




# class TestComputeVertexNormal(tf.test.TestCase):

#     def setUp(self):
#         pyredner.IS_UNIT_TEST = True
#         pyrednertorch.IS_UNIT_TEST = True

#     def tearDown(self):
#         pyredner.IS_UNIT_TEST = False
#         pyrednertorch.IS_UNIT_TEST = False
    

#     def testComputeVertexNormal(self):
#         material_map, mesh_list_tf, light_map = pyredner.load_obj('scenes/teapot.obj')
#         material_maptorch, mesh_list_torch, light_map = pyrednertorch.load_obj('scenes/teapot.obj')

#         assert len(mesh_list_tf) == len(mesh_list_torch)

        
#         for (_, mesh_tf), (_, mesh_torch) in zip(mesh_list_tf, mesh_list_torch):

#             mesh_tf.normals, contribs_tf, v_tf = pyredner.compute_vertex_normal(
#                 mesh_tf.vertices, 
#                 mesh_tf.indices
#             )
#             # pdb.set_trace()
#             mesh_torch.normals, contribs_torch, v_torch = pyrednertorch.compute_vertex_normal(
#                 mesh_torch.vertices, 
#                 mesh_torch.indices
#             )

#             self.assertTrue(is_same_container(contribs_tf, contribs_torch))
#             self.assertTrue(is_same_container(v_tf, v_torch))
#             self.assertTrue(is_same_tensor(mesh_tf.indices, mesh_torch.indices))
#             self.assertTrue(is_same_tensor(mesh_tf.normals, mesh_torch.normals))


# class TestTexture(tf.test.TestCase):

#     def setUp(self):
#         tex = tfe.Variable(
#             np.ones((256, 256, 3), dtype=np.float32) * 0.5
#         )
#         self.tex_tf = pyredner.Texture(tex)

#         tex = torch.tensor(
#             np.ones((256, 256, 3), dtype=np.float32) * 0.5,
#         )
#         self.tex_torch = pyrednertorch.Texture(tex)

#     def testTextures(self):
#         self.assertTrue(is_same_texture(self.tex_tf, self.tex_torch))



        

# class TestLoadMitsuba(tf.test.TestCase):

#     def setUp(self):
#         pass

#     def testParseScene(self):
#         scene_tf = pyredner.load_mitsuba('scenes/teapot_specular.xml')
#         scene_torch = pyrednertorch.load_mitsuba('scenes/teapot_specular.xml')

#         self.assertEqual(len(scene_tf.shapes), len(scene_torch.shapes))

#         self.assertTrue(
#             is_same_scene(scene_tf, scene_torch)
#         )








    

        

if __name__ == "__main__":
    tf.test.main()
