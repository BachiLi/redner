import numpy as np
import tensorflow as tf
import math
import pyredner_tensorflow as pyredner

def generate_geometry_image(size: int):
    """
        Generate an spherical geometry image [Gu et al. 2002 and Praun and Hoppe 2003]
        of size [2 * size + 1, 2 * size + 1]. This can be used for encoding a genus-0
        surface into a regular image, so that it is more convienent for a CNN to process.
        The topology is given by a tesselated octahedron. UV is given by the spherical mapping.
        Duplicated vertex are mapped to the one with smaller index (so some vertices on the
        geometry image is unused by the indices).

        Args
        ====
        size: int
            size of the geometry image
        device_name: Optional[str]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device_name().

        Returns
        =======
        tf.Tensor
            vertices of size [(2 * size + 1 * 2 * size + 1), 3]
        tf.Tensor
            indices of size [2 * (2 * size + 1 * 2 * size + 1), 3]
        tf.Tensor
            uvs of size [(2 * size + 1 * 2 * size + 1), 2]
    """
    size *= 2

    # Generate vertices and uv by going through each vertex.
    left_top = np.array([0.0, 0.0, 1.0])
    top = np.array([0.0, 1.0, 0.0])
    right_top = np.array([0.0, 0.0, 1.0])
    left = np.array([-1.0, 0.0, 0.0])
    middle = np.array([0.0, 0.0, -1.0])
    right = np.array([1.0, 0.0, 0.0])
    left_bottom = np.array([0.0, 0.0, 1.0])
    bottom = np.array([0.0, -1.0, 0.0])
    right_bottom = np.array([0.0, 0.0, 1.0])
    vertices = np.zeros([(size+1) * (size+1), 3])
    uvs = np.zeros([(size+1) * (size + 1), 2])
    vertex_id = 0
    half_size = size / 2.0
    for i in range(size+1): # height
        for j in range(size+1): # width
            # Left Top
            if i  + j <= half_size:
                org = left_top
                i_axis = left - left_top
                j_axis = top - left_top
                i_ = float(i) / half_size
                j_ = float(j) / half_size
            elif (i + j >= half_size and i <= half_size and j <= half_size):
                org = middle
                i_axis = top - middle
                j_axis = left - middle
                i_ = 1.0 - float(i) / half_size
                j_ = 1.0 - float(j) / half_size
            # Right Top
            elif ((half_size - i + j - half_size) <= half_size and i <= half_size and j >= half_size):
                org = middle
                i_axis = top - middle
                j_axis = right - middle
                i_ = 1.0 - float(i) / half_size
                j_ = float(j) / half_size - 1.0
            elif ((i + size - j) <= half_size):
                org = right_top
                i_axis = right - right_top
                j_axis = top - right_top
                i_ = float(i) / half_size
                j_ = 2.0 - float(j) / half_size
            # Left Bottom
            elif ((i - half_size + half_size - j) <= half_size and i >= half_size and j <= half_size):
                org = middle
                i_axis = bottom - middle
                j_axis = left - middle
                i_ = float(i) / half_size - 1.0
                j_ = 1.0 - float(j) / half_size
            elif ((size - i + j) <= half_size):
                org = left_bottom
                i_axis = left - left_bottom
                j_axis = bottom - left_bottom
                i_ = 2.0 - float(i) / half_size
                j_ = float(j) / half_size
            # Right Bottom
            elif ((i - half_size + j - half_size) <= half_size and i >= half_size and j >= half_size):
                org = middle
                i_axis = bottom - middle
                j_axis = right - middle
                i_ = float(i) / half_size - 1.0
                j_ = float(j) / half_size - 1.0
            else:
                org = right_bottom
                i_axis = right - right_bottom
                j_axis = bottom - right_bottom
                i_ = 2.0 - float(i) / half_size
                j_ = 2.0 - float(j) / half_size
            p = org + i_ * i_axis + j_ * j_axis
            vertices[vertex_id, :] = p / np.linalg.norm(p)
            # Spherical UV mapping
            u = 0.5 + math.atan2(float(p[2]), float(p[0])) / (2 * math.pi)
            v = 0.5 - math.asin(float(p[1])) / math.pi
            uvs[vertex_id, :] = np.array([u, v])
            vertex_id += 1

    # Generate indices by going through each triangle.
    # Duplicated vertex are mapped to the one with smaller index.
    indices = []
    for i in range(size): # height
        for j in range(size): # width
            left_top = i * (size + 1) + j
            right_top = i * (size + 1) + j + 1
            left_bottom = (i + 1) * (size + 1) + j
            right_bottom = (i + 1) * (size + 1) + j + 1
            # Wrap rule for octahedron topology
            if i == 0 and j >= half_size:
                if j > half_size:
                    left_top = i * (size + 1) + size - j
                right_top = i * (size + 1) + (size - (j + 1))
            elif i == size - 1 and j >= half_size:
                if j > half_size:
                    left_bottom = (i + 1) * (size + 1) + size - j
                right_bottom = (i + 1) * (size + 1) + (size - (j + 1))
                if j == size - 1:
                    right_bottom = 0
            elif j == 0 and i >= half_size:
                if i > half_size:
                    left_top = (size - i) * (size + 1) + j
                left_bottom = (size - (i + 1)) * (size + 1) + j
            elif j == size - 1 and i >= half_size:
                if i > half_size:
                    right_top = (size - i) * (size + 1) + j + 1
                right_bottom = (size - (i + 1)) * (size + 1) + j + 1

            # Left Top
            if i < half_size and j < half_size:
                indices.append((left_top, left_bottom, right_top))
                indices.append((right_top, left_bottom, right_bottom))
            # Right Top
            elif i < half_size and j >= half_size:
                indices.append((left_top, left_bottom, right_bottom))
                indices.append((left_top, right_bottom, right_top))
            # Left Bottom
            elif i >= half_size and j < half_size:
                indices.append((left_top, right_bottom, right_top))
                indices.append((left_top, left_bottom, right_bottom))
            # Right Bottom
            else:
                indices.append((left_top, left_bottom, right_top))
                indices.append((right_top, left_bottom, right_bottom))

    vertices = tf.constant(vertices, dtype = tf.float32)
    uvs = tf.constant(uvs, dtype = tf.float32)
    indices = tf.constant(indices, dtype = tf.int32)
    return vertices, indices, uvs
