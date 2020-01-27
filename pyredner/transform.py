import math
import numpy as np
import torch

def radians(deg):
    return (math.pi / 180.0) * deg

def normalize(v):
    return v / torch.norm(v)

def gen_look_at_matrix(pos, look, up):
    d = normalize(look - pos)
    right = normalize(torch.cross(d, normalize(up)))
    new_up = normalize(torch.cross(right, d))
    z = torch.zeros([1], dtype=torch.float32, device = d.device)
    o = torch.ones([1], dtype=torch.float32, device = d.device)
    return torch.transpose(torch.stack([torch.cat([right , z], 0),
                                        torch.cat([new_up, z], 0),
                                        torch.cat([d     , z], 0),
                                        torch.cat([pos   , o], 0)]), 0, 1).contiguous()

def gen_scale_matrix(scale):
    o = torch.ones([1], dtype=torch.float32, device = scale.device)
    return torch.diag(torch.cat([scale, o], 0))

def gen_translate_matrix(translate):
    z = torch.zeros([1], dtype=torch.float32, device = translate.device)
    o = torch.ones([1], dtype=torch.float32, device = translate.device)
    return torch.stack([torch.cat([o, z, z, translate[0:1]], 0),
                        torch.cat([z, o, z, translate[1:2]], 0),
                        torch.cat([z, z, o, translate[2:3]], 0),
                        torch.cat([z, z, z, o], 0)])

def gen_perspective_matrix(fov, clip_near, clip_far):
    clip_dist = clip_far - clip_near
    cot = 1 / torch.tan(radians(fov / 2.0))
    z = torch.zeros([1], dtype=torch.float32, device = clip_near.device)
    o = torch.ones([1], dtype=torch.float32, device = clip_near.device)
    return torch.stack([torch.cat([cot,   z,             z,                       z], 0),
                        torch.cat([  z, cot,             z,                       z], 0),
                        torch.cat([  z,   z, 1 / clip_dist, - clip_near / clip_dist], 0),
                        torch.cat([  z,   z,             o,                       z], 0)])

def gen_rotate_matrix(angles: torch.Tensor):
    """
        Given a 3D Euler angle vector, outputs a rotation matrix.

        Args
        ====
            angles: torch.Tensor
                3D Euler angle

        Returns
        =======
            torch.Tensor
                3x3 rotation matrix
    """

    theta = angles[0]
    phi = angles[1]
    psi = angles[2]
    rot_x = torch.zeros((3, 3), device=angles.device, dtype=torch.float32)
    rot_y = torch.zeros((3, 3), device=angles.device, dtype=torch.float32)
    rot_z = torch.zeros((3, 3), device=angles.device, dtype=torch.float32)
    rot_x[0, 0] = 1
    rot_x[0, 1] = 0
    rot_x[0, 2] = 0
    rot_x[1, 0] = 0
    rot_x[1, 1] = theta.cos()
    rot_x[1, 2] = theta.sin()
    rot_x[2, 0] = 0
    rot_x[2, 1] = -theta.sin()
    rot_x[2, 2] = theta.cos()
    
    rot_y[0, 0] = phi.cos()
    rot_y[0, 1] = 0
    rot_y[0, 2] = -phi.sin()
    rot_y[1, 0] = 0
    rot_y[1, 1] = 1
    rot_y[1, 2] = 0
    rot_y[2, 0] = phi.sin()
    rot_y[2, 1] = 0
    rot_y[2, 2] = phi.cos()
    
    rot_z[0, 0] = psi.cos()
    rot_z[0, 1] = -psi.sin()
    rot_z[0, 2] = 0
    rot_z[1, 0] = psi.sin()
    rot_z[1, 1] = psi.cos()
    rot_z[1, 2] = 0
    rot_z[2, 0] = 0
    rot_z[2, 1] = 0
    rot_z[2, 2] = 1
    return rot_z @ (rot_y @ rot_x)
