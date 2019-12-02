import pyredner
import random
import redner

def render_albedo(scene,
                  num_samples=(16, 4),
                  seed=None):
    """
        render the diffuse albedo color of the scene.
    """
    if seed==None:
        seed = random.randint(0, 16777216)
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = num_samples,
        max_bounces = 0,
        channels = [redner.channels.diffuse_reflectance])
    return pyredner.RenderFunction.apply(seed, *scene_args)
