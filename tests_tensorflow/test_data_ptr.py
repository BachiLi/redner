from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()

from pyredner import read_tensor
import redner
import pyrednertensorflow as pyredner
import numpy as np
import pdb

class TestFloat32(tf.test.TestCase):
    def setUp(self):
        self.shape_ref = [10, 12, 7, 31]
        self.array_ref = np.array(np.random.rand(*self.shape_ref), dtype=np.float32)
        self.length_ref = np.prod(self.shape_ref)
        self.tensor = tfe.Variable(np.copy(self.array_ref), dtype=tf.float32)
        if tf.executing_eagerly():
            # print("Eager mode!")
            self.addr = pyredner.data_ptr(self.tensor)
        else:
            # print("Non-Eager mode!")
            with tf.Session(''):
                self.addr = pyredner.data_ptr(np.array([1.0, 2.2], dtype=np.float32)).eval()

    def testAddressIsInt(self):
        self.assertIsInstance(self.addr, int, "Memory should be integer")

    def testTensorEquality(self):        
        # print("Address: ", self.addr)
        redner.write_float32_data_ptr(redner.float_ptr(self.addr), self.length_ref)

        self.assertAllClose(
            read_tensor("temp/test_float32_data_ptr_reference.txt", self.array_ref.shape),
            self.array_ref
        ) 
        
    def testModifiedTensorEquality(self):
        # print("Address: ", self.addr)
        redner.multiply_float32_data_ptr(redner.float_ptr(self.addr), self.length_ref, 1000)
        self.assertAllEqual(
            self.array_ref * 1000,
            self.tensor.numpy()
        )

class TestInt32(tf.test.TestCase):
    def setUp(self):
        self.shape_ref = [256, 256, 3]
        self.array_ref = np.array(np.random.rand(*self.shape_ref), dtype=np.int32)
        self.length_ref = np.prod(self.shape_ref)
        self.tensor = tfe.Variable(np.copy(self.array_ref), dtype=tf.int32)
        if tf.executing_eagerly():
            # print("Eager mode!")
            self.addr = pyredner.data_ptr(self.tensor)
        else:
            # print("Non-Eager mode!")
            with tf.Session(''):
                self.addr = pyredner.data_ptr(np.array([1.0, 2.2], dtype=np.int32)).eval()

    def testTensorEquality(self):        
        # print("Address: ", self.addr)
        redner.write_int32_data_ptr(redner.int_ptr(self.addr), self.length_ref)

        self.assertAllClose(
            read_tensor("temp/test_int32_data_ptr_reference.txt", self.array_ref.shape),
            self.array_ref
        ) 
        
    def testModifiedTensorEquality(self):
        # print("Address: ", self.addr)
        redner.multiply_int32_data_ptr(redner.int_ptr(self.addr), self.length_ref, 1000)
        self.assertAllEqual(
            self.array_ref * 1000,
            self.tensor.numpy()
        )

class TestOnes(tf.test.TestCase):
    def setUp(self):
        self.shape_ref = [256,256,3]
        self.array_ref = np.ones(self.shape_ref, dtype=np.float32)
        self.tensor = tf.ones(self.shape_ref, dtype=tf.float32)
        self.length_ref = np.prod(self.shape_ref)
        if tf.executing_eagerly():
            # print("Eager mode!")
            self.addr = pyredner.data_ptr(self.tensor)
        else:
            # print("Non-Eager mode!")
            with tf.Session(''):
                self.addr = pyredner.data_ptr(np.array([1.0, 2.2], dtype=np.int32)).eval()


    def testTensorEquality(self):        
        # print("Address: ", self.addr)
        redner.write_float32_data_ptr(redner.float_ptr(self.addr), self.length_ref)

        self.assertAllClose(
            read_tensor("temp/test_float32_data_ptr_reference.txt", self.array_ref.shape),
            self.array_ref
        ) 
        
    def testModifiedTensorEquality(self):
        # print("Address: ", self.addr)
        redner.multiply_float32_data_ptr(redner.float_ptr(self.addr), self.length_ref, 1000)
        self.assertAllEqual(
            self.array_ref * 1000,
            self.tensor
        )

class TestConstant(tf.test.TestCase):
    """Even `tf.constant` can be manipulated in C++
    """
    def setUp(self):
        self.array_ref = np.array([1,2,3,4], dtype=np.float32)
        self.shape_ref = [4]
        self.tensor = tf.constant([1,2,3,4], dtype=tf.float32)
        self.length_ref = np.prod(self.shape_ref)
        if tf.executing_eagerly():
            # print("Eager mode!")
            self.addr = pyredner.data_ptr(self.tensor)
        else:
            # print("Non-Eager mode!")
            with tf.Session(''):
                self.addr = pyredner.data_ptr(np.array([1.0, 2.2], dtype=np.int32)).eval()


    def testTensorEquality(self):        
        # print("Address: ", self.addr)
        redner.write_float32_data_ptr(redner.float_ptr(self.addr), self.length_ref)

        self.assertAllClose(
            read_tensor("temp/test_float32_data_ptr_reference.txt", self.array_ref.shape),
            self.array_ref
        ) 
        
    def testModifiedTensorEquality(self):
        # print("Address: ", self.addr)
        redner.multiply_float32_data_ptr(redner.float_ptr(self.addr), self.length_ref, 1000)
        self.assertAllEqual(
            self.array_ref * 1000,
            self.tensor
        )

class TestCamera(tf.test.TestCase):
    def setUp(self):
        self.cam = pyredner.Camera(position = tfe.Variable([0.0, 0.0, -5.0],
                dtype=tf.float32),
                look_at = tfe.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
                up = tfe.Variable([0.0, 1.0, 0.0], dtype=tf.float32),
                fov = tfe.Variable([45.0], dtype=tf.float32), # in degree
                clip_near = 1e-2, # needs to > 0
                resolution = (256, 256),
                fisheye = False
                )


    def testCamToWorld(self):
        redner.write_float32_data_ptr(
            redner.float_ptr(pyredner.data_ptr(self.cam.cam_to_world)), 
            16
        )
        self.assertAllEqual(
            self.cam.cam_to_world,
            read_tensor("temp/test_float32_data_ptr_reference.txt", self.cam.cam_to_world.shape)
        )

    def testWorldToCam(self):
        redner.write_float32_data_ptr(
            redner.float_ptr(pyredner.data_ptr(self.cam.world_to_cam)), 
            16
        )
        self.assertAllEqual(
            self.cam.world_to_cam,
            read_tensor("temp/test_float32_data_ptr_reference.txt", self.cam.world_to_cam.shape)
        )

    def testNDCToCam(self):
        length = np.prod(self.cam.ndc_to_cam.shape).value
        redner.write_float32_data_ptr(
            redner.float_ptr(pyredner.data_ptr(self.cam.ndc_to_cam)), 
            length
        )
        self.assertAllEqual(
            self.cam.ndc_to_cam,
            read_tensor("temp/test_float32_data_ptr_reference.txt", self.cam.ndc_to_cam.shape)
        )

    def testCamToNDC(self):
        length = np.prod(self.cam.cam_to_ndc.shape).value
        redner.write_float32_data_ptr(
            redner.float_ptr(pyredner.data_ptr(self.cam.cam_to_ndc)), 
            length
        )
        self.assertAllEqual(
            self.cam.cam_to_ndc,
            read_tensor("temp/test_float32_data_ptr_reference.txt", self.cam.cam_to_ndc.shape)
        )
    
    
class TestShape(tf.test.TestCase):
    def setUp(self):
        self.shape = pyredner.Shape(
            vertices = tfe.Variable([[-1.7, 1.0, 0.0], [1.0, 1.0, 0.0], 
                                    [-0.5, -1.0, 0.0]],
                            trainable=True, 
                            name='shape_triangle_vertices', 
                            dtype=tf.float32),
                        indices = tfe.Variable([[0, 1, 2]], dtype=tf.int32),
                        uvs = None,
                        normals = None,
                        material_id = 0)


    def testVertices(self):
        length = np.prod(self.shape.vertices.shape).value
        redner.write_float32_data_ptr(
            redner.float_ptr(pyredner.data_ptr(self.shape.vertices)), 
            length
        )
        # pdb.set_trace()
        self.assertAllEqual(
            self.shape.vertices.numpy(),
            read_tensor("temp/test_float32_data_ptr_reference.txt", self.shape.vertices.shape)
        )

    def testIndices(self):
        length = np.prod(self.shape.indices.shape).value
        redner.write_int32_data_ptr(
            redner.int_ptr(pyredner.data_ptr(self.shape.indices)), 
            length
        )
        # pdb.set_trace()
        self.assertAllEqual(
            self.shape.indices.numpy(),
            read_tensor("temp/test_int32_data_ptr_reference.txt", self.shape.indices.shape)
        )

class TestScene(tf.test.TestCase):
    def setUp(self):
        cam = pyredner.Camera(position = tfe.Variable([0.0, 0.0, -5.0], dtype=tf.float32),
                      look_at = tfe.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
                      up = tfe.Variable([0.0, 1.0, 0.0], dtype=tf.float32),
                      fov = tfe.Variable([45.0], dtype=tf.float32), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)

        mat_grey = pyredner.Material(
            diffuse_reflectance=tfe.Variable([0.5, 0.5, 0.5], dtype=tf.float32)
        )
        materials = [mat_grey]

        shape_triangle = pyredner.Shape(
            vertices = tfe.Variable([[-1.7, 1.0, 0.0], [1.0, 1.0, 0.0], [-0.5, -1.0, 0.0]],
                                    trainable=True, 
                                    name='shape_triangle_vertices', 
                                    dtype=tf.float32),
            indices = tfe.Variable([[0, 1, 2]], dtype=tf.int32),
            uvs = None,
            normals = None,
            material_id = 0)

        shape_light = pyredner.Shape(
            vertices = tfe.Variable([[-1.0, -1.0, -7.0],
                                    [ 1.0, -1.0, -7.0],
                                    [-1.0,  1.0, -7.0],
                                    [ 1.0,  1.0, -7.0]],
                                    dtype=tf.float32),
            indices = tfe.Variable([[0, 1, 2],[1, 3, 2]],
                                    dtype = tf.int32),
            uvs = None,
            normals = None,
            material_id = 0)
        shapes = [shape_triangle, shape_light]

        light = pyredner.AreaLight(shape_id = 1, 
                                intensity = tfe.Variable([20.0,20.0,20.0], dtype=tf.float32))
        area_lights = [light]

        self.scene = pyredner.Scene(cam, shapes, materials, area_lights)

if __name__ == "__main__":
    tf.test.main()

    # read_tensor("temp/test_float32_data_ptr_reference.txt", self.shape.vertices.shape)