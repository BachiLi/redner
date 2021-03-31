"""
    Defines the render command and builds the render options for the
    edge sampling integrator.
"""
import redner
import time

from .base import Integrator

class EdgeSamplingIntegrator(Integrator):
    def __init__( self,
                  num_samples = 64,
                  max_bounces = 2,
                  channels = [redner.channels.radiance],
                  sampler_type = redner.SamplerType.independent,
                  sample_pixel_center = True,
                  use_primary_edge_sampling = True,
                  use_secondary_edge_sampling = True,
                  ):

        if isinstance(num_samples, int):
            num_samples = (num_samples, num_samples)

        self.num_samples = num_samples
        self.max_bounces = max_bounces
        self.channels = channels
        self.sampler_type = sampler_type
        self.sample_pixel_center = sample_pixel_center
        self.use_primary_edge_sampling = use_primary_edge_sampling
        self.use_secondary_edge_sampling = use_secondary_edge_sampling

        self.timing = False

    def disable_exterior_derivative(self):
        self.use_primary_edge_sampling = False
        self.use_secondary_edge_sampling = False
    
    def enable_exterior_derivative(self):
        self.use_primary_edge_sampling = True
        self.use_secondary_edge_sampling = True

    def enable_timing(self):
        self.timing = True

    def disable_timing(self):
        self.timing = False

    def render_image(self, seed, scene, img):
        num_samples = self.num_samples[0]
        return self.render(seed=seed[0] if hasattr(seed, '__iter__') else seed,
                           scene=scene,
                           img=img,
                           num_samples=num_samples)

    def render_derivs(self, seed, scene, d_img, d_scene):
        num_samples = self.num_samples[1]
        return self.render(seed=seed[1] if hasattr(seed, '__iter__') else seed,
                           scene=scene,
                           d_img=d_img,
                           d_scene=d_scene,
                           num_samples=num_samples
                           )

    def render_screen_gradient(self, seed, scene, d_img, d_scene, screen_grad):
        num_samples = self.num_samples[1]
        return self.render(seed=seed[1] if hasattr(seed, '__iter__') else seed,
                           scene=scene,
                           d_img=d_img,
                           d_scene=d_scene,
                           screen_gradient_img=screen_grad,
                           num_samples=num_samples)

    def render_debug_image(self, seed, scene, d_img, d_scene, debug_img):
        num_samples = self.num_samples[1]
        return self.render(seed=seed[1] if hasattr(seed, '__iter__') else seed,
                           scene=scene,
                           d_img=d_img,
                           d_scene=d_scene,
                           debug_img=debug_img,
                           num_samples=num_samples)

    def render(self,
               seed, scene,
               img=redner.float_ptr(0),
               d_img=redner.float_ptr(0),
               d_scene=None,
               screen_gradient_img=redner.float_ptr(0),
               debug_img=redner.float_ptr(0),
               num_samples=None):

        use_secondary_edge_sampling = self.use_secondary_edge_sampling
        if self.max_bounces == 0:
            use_secondary_edge_sampling = False
        options = redner.RenderOptions(
                        seed, 
                        self.num_samples if not num_samples else num_samples,
                        self.max_bounces, 
                        self.channels,
                        self.sampler_type,
                        self.sample_pixel_center,
                        self.use_primary_edge_sampling,
                        use_secondary_edge_sampling)

        if self.timing:
            start = time.perf_counter()

        redner.render(scene,
                      options,
                      img,
                      d_img,
                      d_scene,
                      screen_gradient_img,
                      debug_img)

        if self.timing:
            print("Time elapsed: ", time.perf_counter() - start, "s")