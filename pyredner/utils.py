import torch
import math
import numpy as np
import pyredner
from typing import Optional

####################### Spherical Harmonics utilities ########################
# Code adapted from "Spherical Harmonic Lighting: The Gritty Details", Robin Green
# http://silviojemma.com/public/papers/lighting/spherical-harmonic-lighting.pdf
def associated_legendre_polynomial(l, m, x):
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll

def SH_renormalization(l, m):
    return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m) / \
        (4 * math.pi * math.factorial(l + m)))

def SH(l, m, theta, phi):
    if m == 0:
        return SH_renormalization(l, m) * associated_legendre_polynomial(l, m, torch.cos(theta))
    elif m > 0:
        return math.sqrt(2.0) * SH_renormalization(l, m) * \
            torch.cos(m * phi) * associated_legendre_polynomial(l, m, torch.cos(theta))
    else:
        return math.sqrt(2.0) * SH_renormalization(l, -m) * \
            torch.sin(-m * phi) * associated_legendre_polynomial(l, -m, torch.cos(theta))

def SH_reconstruct(coeffs, res):
    uv = np.mgrid[0:res[1], 0:res[0]].astype(np.float32)
    theta = torch.from_numpy((math.pi / res[1]) * (uv[1, :, :] + 0.5))
    phi = torch.from_numpy((2 * math.pi / res[0]) * (uv[0, :, :] + 0.5))
    theta = theta.to(coeffs.device)
    phi = phi.to(coeffs.device)
    result = torch.zeros(res[1], res[0], coeffs.shape[0], device = coeffs.device)
    num_order = int(math.sqrt(coeffs.shape[1]))
    i = 0
    for l in range(num_order):
        for m in range(-l, l + 1):
            sh_factor = SH(l, m, theta, phi)
            result = result + sh_factor.view(sh_factor.shape[0], sh_factor.shape[1], 1) * coeffs[:, i]
            i += 1
    result = torch.max(result,
        torch.zeros(res[1], res[0], coeffs.shape[0], device = coeffs.device))
    return result
#######################################################################################

def generate_sphere(theta_steps: int,
                    phi_steps: int,
                    device: Optional[torch.device] = None):
    """
        Generate a triangle mesh representing a UV sphere,
        center at (0, 0, 0) with radius 1.

        Args
        ====
        theta_steps: int
            zenith subdivision
        phi_steps: int
            azimuth subdivision
        device: Optional[torch.device]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device().

        Returns
        =======
        torch.Tensor
            vertices
        torch.Tensor
            indices
        torch.Tensor
            uvs
        torch.Tensor
            normals
    """
    if device is None:
        device = pyredner.get_device()

    d_theta = math.pi / (theta_steps - 1)
    d_phi = (2 * math.pi) / (phi_steps - 1)

    num_vertices = theta_steps * phi_steps - 2 * (phi_steps - 1)
    vertices = torch.zeros(num_vertices, 3, device = device)
    uvs = torch.zeros(num_vertices, 2, device = device)
    vertices_index = 0
    for theta_index in range(theta_steps):
        sin_theta = math.sin(theta_index * d_theta)
        cos_theta = math.cos(theta_index * d_theta)
        if theta_index == 0:
            # For the two polars of the sphere, only generate one vertex
            vertices[vertices_index, :] = \
                torch.tensor([0.0, 1.0, 0.0])
            uvs[vertices_index, 0] = 0.0
            uvs[vertices_index, 1] = 0.0
            vertices_index += 1
        elif theta_index == theta_steps - 1:
            # For the two polars of the sphere, only generate one vertex
            vertices[vertices_index, :] = \
                torch.tensor([0.0, -1.0, 0.0])
            uvs[vertices_index, 0] = 0.0
            uvs[vertices_index, 1] = 1.0
            vertices_index += 1
        else:
            for phi_index in range(phi_steps):
                sin_phi = math.sin(phi_index * d_phi)
                cos_phi = math.cos(phi_index * d_phi)
                vertices[vertices_index, :] = \
                    torch.tensor([sin_theta * cos_phi, cos_theta, sin_theta * sin_phi],
                        device = device)
                uvs[vertices_index, 0] = phi_index * d_phi / (2 * math.pi)
                uvs[vertices_index, 1] = theta_index * d_theta / math.pi
                vertices_index += 1

    indices = []
    for theta_index in range(1, theta_steps):
        for phi_index in range(phi_steps - 1):
            if theta_index < theta_steps - 1:
                id0 = phi_steps * theta_index + phi_index - (phi_steps - 1)
                id1 = phi_steps * theta_index + phi_index + 1 - (phi_steps - 1)
            else:
                # There is only one vertex at the pole
                assert(theta_index == theta_steps - 1)
                id0 = num_vertices - 1
                id1 = num_vertices - 1
            if theta_index > 1:
                id2 = phi_steps * (theta_index - 1) + phi_index - (phi_steps - 1)
                id3 = phi_steps * (theta_index - 1) + phi_index + 1 - (phi_steps - 1)
            else:
                # There is only one vertex at the pole
                assert(theta_index == 1)
                id2 = 0
                id3 = 0

            if (theta_index < theta_steps - 1):
                indices.append([id0, id2, id1])
            if (theta_index > 1):
                indices.append([id1, id2, id3])

    indices = torch.tensor(indices, dtype = torch.int32, device = device)

    normals = vertices.clone()
    return (vertices, indices, uvs, normals)

def generate_quad_light(position: torch.Tensor,
                        look_at: torch.Tensor,
                        size: torch.Tensor,
                        intensity: torch.Tensor):
    """
        Generate a pyredner.Object that is a quad light source.

        Args
        ====
        position: torch.Tensor
            1-d tensor of size 3
        look_at: torch.Tensor
            1-d tensor of size 3
        size: torch.Tensor
            1-d tensor of size 2
        intensity: torch.Tensor
            1-d tensor of size 3

        Returns
        =======
        pyredner.Object
            quad light source
    """
    d = look_at - position
    d = d / torch.norm(d)
    # ONB -- generate two axes that are orthogonal to d
    a = 1 / (1 + d[2])
    b = -d[0] * d[1] * a
    x = torch.where(d[2] < (-1 + 1e-6),
                    torch.tensor([0.0, -1.0, 0.0], device = d.device),
                    torch.stack([1 - d[0] * d[0] * a, b, -d[0]]))
    y = torch.where(d[2] < (-1 + 1e-6),
                    torch.tensor([-1.0, 0.0, 0.0], device = d.device),
                    torch.stack([b, 1 - d[1] * d[1] * a, -d[1]]))
    v0 = position - x * size[0] * 0.5 - y * size[1] * 0.5
    v1 = position + x * size[0] * 0.5 - y * size[1] * 0.5
    v2 = position - x * size[0] * 0.5 + y * size[1] * 0.5
    v3 = position + x * size[0] * 0.5 + y * size[1] * 0.5

    vertices = torch.stack((v0, v1, v2, v3), dim = 0).to(d.device)
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = d.device)
    m = pyredner.Material(diffuse_reflectance = torch.tensor([0.0, 0.0, 0.0], device = d.device))
    return pyredner.Object(vertices = vertices,
                           indices = indices,
                           material = m,
                           light_intensity = intensity)

def linear_to_srgb(x):
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * torch.pow(x, 1.0 / 2.4) - 0.055)

def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, torch.pow((x + 0.055) / 1.055, 2.4))
