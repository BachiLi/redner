from typing import Optional
import pyredner_tensorflow as pyredner
import tensorflow as tf
import math
import numpy as np

def compute_vertex_normal(vertices, indices):
    def dot(v1, v2):
        return tf.math.reduce_sum(v1 * v2, axis=1)
    def squared_length(v):
        return tf.math.reduce_sum(v * v, axis=1)
    def length(v):
        return tf.sqrt(squared_length(v))
    def safe_asin(v):
        # Hack: asin(1)' is infinite, so we want to clamp the contribution
        return tf.asin(tf.clip_by_value(v, 0, 1-1e-6))

    # Nelson Max, "Weights for Computing Vertex Normals from Facet Vectors", 1999
    normals = tf.zeros(vertices.shape, dtype = tf.float32)

    # NOTE: Try tf.TensorArray()
    v = [tf.gather(vertices, indices[:,0]),
         tf.gather(vertices, indices[:,1]),
         tf.gather(vertices, indices[:,2])]

    for i in range(3):
        v0 = v[i]
        v1 = v[(i + 1) % 3]
        v2 = v[(i + 2) % 3]
        e1 = v1 - v0
        e2 = v2 - v0
        e1_len = length(e1)
        e2_len = length(e2)
        side_a = e1 / tf.reshape(e1_len, [-1, 1])
        side_b = e2 / tf.reshape(e2_len, [-1, 1])
        if i == 0:
            n = tf.linalg.cross(side_a, side_b)
            n = tf.where(\
                tf.broadcast_to(tf.reshape(length(n) > 0, (-1, 1)), tf.shape(n)),
                n / tf.reshape(length(n), (-1, 1)),
                tf.zeros(tf.shape(n), dtype=n.dtype))

        angle = tf.where(dot(side_a, side_b) < 0,
            math.pi - 2.0 * safe_asin(0.5 * length(side_a + side_b)),
            2.0 * safe_asin(0.5 * length(side_b - side_a)))
        sin_angle = tf.sin(angle)

        e1e2 = e1_len * e2_len
        # contrib is 0 when e1e2 is 0
        contrib = tf.reshape(\
            tf.where(e1e2 > 0, sin_angle / e1e2, tf.zeros(tf.shape(e1e2), dtype=e1e2.dtype)), (-1, 1))
        contrib = n * tf.broadcast_to(contrib, [tf.shape(contrib)[0],3]) # In torch, `expand(-1, 3)`
        normals += tf.scatter_nd(tf.reshape(indices[:, i], [-1, 1]), contrib, shape = tf.shape(normals))

    degenerate_normals = tf.constant((0.0, 0.0, 1.0))
    degenerate_normals = tf.broadcast_to(tf.reshape(degenerate_normals, (1, 3)), tf.shape(normals))
    normals = tf.where(tf.broadcast_to(tf.reshape(length(normals) > 0, (-1, 1)), tf.shape(normals)),
        normals / tf.reshape(length(normals), (-1, 1)),
        degenerate_normals)
    return normals

class Shape:
    def __init__(self,
                 vertices: tf.Tensor,
                 indices: tf.Tensor,
                 uvs: Optional[tf.Tensor] = None,
                 normals: Optional[tf.Tensor] = None,
                 material_id: int = 0):
        assert(tf.executing_eagerly())
        assert(vertices.dtype == tf.float32)
        assert(indices.dtype == tf.int32)
        if uvs is not None:
            assert(uvs.dtype == tf.float32)
        if normals is not None:
            assert(normals.dtype == tf.float32)
        if pyredner.get_use_gpu():
            # Automatically copy all tensors to GPU
            # tf.Variable doesn't support .gpu(), so we'll wrap it with an identity().
            vertices = tf.identity(vertices).gpu(pyredner.get_gpu_device_id())
            indices = tf.identity(indices).gpu(pyredner.get_gpu_device_id())
            if uvs is not None:
                uvs = tf.identity(uvs).gpu(pyredner.get_gpu_device_id())
            if normals is not None:
                normals = tf.identity(normals).gpu(pyredner.get_gpu_device_id())
        else:
            # Automatically copy to CPU
            vertices = tf.identity(vertices).cpu()
            indices = tf.identity(indices).cpu()
            if uvs is not None:
                uvs = tf.identity(uvs).cpu()
            if normals is not None:
                normals = tf.identity(normals).cpu()

        self.vertices = vertices
        self.indices = indices
        self.uvs = uvs
        self.normals = normals
        self.material_id = material_id
        self.light_id = -1

    def state_dict(self):
        return {
            'vertices': self.vertices,
            'indices': self.indices,
            'uvs': self.uvs,
            'normals': self.normals,
            'material_id': self.material_id,
            'light_id': self.light_id,
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        out = cls(
            state_dict['vertices'],
            state_dict['indices'],
            state_dict['uvs'],
            state_dict['normals'],
            state_dict['material_id'])
        return out
