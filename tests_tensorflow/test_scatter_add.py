from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()
import pyrednertensorflow as pyrednertf
import pyredner as pyrednertorch

import numpy as np

class TestScatterAdd(tf.test.TestCase):

    def testComputeVertexNormal(self):
        _, mesh_list_torch, _ = pyrednertorch.load_obj('scenes/teapot.obj')
        _, mesh_list_tf, _ = pyrednertf.load_obj('scenes/teapot.obj')

        for (_, mesh1), (_ ,mesh2) in zip(mesh_list_tf, mesh_list_torch):
            normalstf = pyrednertf.compute_vertex_normal(mesh1.vertices, mesh1.indices)
            normalstorch = pyrednertorch.compute_vertex_normal(mesh2.vertices, mesh2.indices)
            self.assertTrue(
                np.allclose(normalstf.numpy(), normalstorch.detach().numpy(), atol=1e-07)
            )

if __name__ == "__main__":
    tf.test.main()
