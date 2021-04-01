"""
    Defines the render command and builds the render options for the
    warp field integrator.
"""
import redner
import time

from .base import Integrator
from pyredner import get_use_gpu


class KernelParameters:
    def __init__(self,
        vMFConcentration = 1e5,
        auxPrimaryGaussianStddev = 0.2,
        auxPdfEpsilonRegularizer = 1e-10,
        asymptoteInvGaussSigma = 0.2,
        asymptoteBoundaryTemp = 1.0,
        asymptoteGamma = 2,
        pixelBoundaryMultiplier = 0,
        numAuxillaryRays = 8,
        rrEnable = False,
        rrGeometricP = 0.4,
        batchSize = 4,
        isBasicNormal = False):

        self.vMFConcentration = vMFConcentration
        self.auxPrimaryGaussianStddev = auxPrimaryGaussianStddev
        self.auxPdfEpsilonRegularizer = auxPdfEpsilonRegularizer
        self.asymptoteInvGaussSigma = asymptoteInvGaussSigma
        self.asymptoteBoundaryTemp = asymptoteBoundaryTemp
        self.asymptoteGamma = asymptoteGamma
        self.pixelBoundaryMultiplier = pixelBoundaryMultiplier
        self.numAuxillaryRays = numAuxillaryRays
        self.rrEnable = rrEnable
        self.rrGeometricP = rrGeometricP
        self.batchSize = batchSize
        self.isBasicNormal = isBasicNormal
    
    def as_redner_object(self):
        return redner.KernelParameters(
            self.vMFConcentration,
            self.auxPrimaryGaussianStddev,
            self.auxPdfEpsilonRegularizer,
            self.asymptoteInvGaussSigma,
            self.asymptoteBoundaryTemp,
            self.asymptoteGamma,
            self.pixelBoundaryMultiplier,
            self.numAuxillaryRays,
            self.rrEnable,
            self.rrGeometricP,
            self.batchSize,
            self.isBasicNormal)


class VarianceReductionSettings:
    def __init__(self,
        primary_antithetic_variates = True,
        aux_antithetic_variates = True,
        primary_control_variates = False,
        aux_control_variates = False,
        secondary_antithetic_variates = False,
        num_control_rays = 1):

        self.primary_antithetic_variates = primary_antithetic_variates
        self.aux_antithetic_variates = aux_antithetic_variates
        self.primary_control_variates = primary_control_variates
        self.aux_control_variates = aux_control_variates
        self.secondary_antithetic_variates = secondary_antithetic_variates
        self.num_control_rays = num_control_rays

    def as_redner_object(self):
        return redner.VarianceReductionSettings(
                            self.primary_antithetic_variates, 
                            self.aux_antithetic_variates,
                            self.primary_control_variates,
                            self.aux_control_variates,
                            self.secondary_antithetic_variates,
                            self.num_control_rays)


class WarpFieldIntegrator(Integrator):
    """
        Interface for the Warped-area sampling method for differentiable rendering based
        on the SIGGRAPH Asia 2020 paper of the same name: https://www.saipraveenb.com/projects/was-2020/
    """
    def __init__( self,
                  num_samples = 64,
                  max_bounces = 2,
                  channels = [redner.channels.radiance],
                  variance_reduction = VarianceReductionSettings(True, True, False, False, False),
                  importance_sampling = redner.ImportanceSamplingVField.cosine_hemisphere,
                  sampler_type = redner.SamplerType.independent,
                  aux_sampler_type = redner.SamplerType.independent,
                  kernel_parameters = KernelParameters(),
                  use_primary_warp_field = True,
                  use_secondary_warp_field = True,
                  clear_loss = False,
                  sample_pixel_center = False
                ):

        if isinstance(num_samples, int):
            num_samples = (num_samples, num_samples)

        self.num_samples = num_samples
        self.sampler_type = sampler_type
        self.max_bounces = max_bounces
        self.channels = channels

        self.kernel_parameters = kernel_parameters
        self.variance_reduction = variance_reduction
        self.importance_sampling = importance_sampling
        self.aux_sampler_type = aux_sampler_type

        self.enable_primary_warp_field = use_primary_warp_field
        self.enable_secondary_warp_field = use_secondary_warp_field

        self.clear_loss = clear_loss
        self.sample_pixel_center = sample_pixel_center
        self.timing = False

    def disable_primary_warp_field(self):
        self.enable_primary_warp_field = False

    def enable_primary_warp_field(self):
        self.enable_primary_warp_field = True

    def disable_secondary_warp_field(self):
        self.enable_secondary_warp_field = False

    def enable_secondary_warp_field(self):
        self.enable_secondary_warp_field = True

    def disable_warp_fields(self):
        self.disable_primary_warp_field()
        self.disable_secondary_warp_field()

    def enable_warp_fields(self):
        self.enable_primary_warp_field()
        self.enable_secondary_warp_field()

    def enable_antithetic_variates(self):
        self.variance_reduction.primary_antithetic_variates = True
        self.variance_reduction.aux_antithetic_variates = True
    
    def disable_antithetic_variates(self):
        self.variance_reduction.primary_antithetic_variates = False
        self.variance_reduction.aux_antithetic_variates = False

    def enable_secondary_variates(self):
        self.variance_reduction.secondary_antithetic_variates = True

    def disable_secondary_variates(self):
        self.variance_reduction.secondary_antithetic_variates = False

    def enable_control_variates(self):
        self.variance_reduction.primary_control_variates = True
        self.variance_reduction.aux_control_variates = True

    def disable_control_variates(self):
        self.variance_reduction.primary_control_variates = False
        self.variance_reduction.aux_control_variates = False

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

        assert not get_use_gpu(), f'Scene.use_gpu is True. WarpFieldIntegrator cannot use the gpu'

        options = redner.RenderOptionsVField(
                        seed,
                        self.num_samples if not num_samples else num_samples,
                        self.max_bounces,
                        self.channels,
                        self.sampler_type, 
                        self.aux_sampler_type,
                        self.variance_reduction.as_redner_object(),
                        self.importance_sampling,
                        self.kernel_parameters.as_redner_object(),
                        self.enable_primary_warp_field,
                        self.enable_secondary_warp_field,
                        self.sample_pixel_center)

        if self.timing:
            start = time.perf_counter()

        redner.render_warped(
                            scene,
                            options,
                            img,
                            d_img,
                            d_scene,
                            screen_gradient_img,
                            debug_img)

        if self.timing:
            print("Time elapsed: ", time.perf_counter() - start, "s")
