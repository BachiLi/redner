import redner
import torch

def unit_tests():
    redner.test_sample_primary_rays(False)
    redner.test_scene_intersect(False)
    redner.test_sample_point_on_light(False)
    redner.test_active_pixels(False)
    redner.test_camera_derivatives()
    redner.test_camera_distortion()
    redner.test_d_bsdf()
    redner.test_d_bsdf_sample()
    redner.test_d_bsdf_pdf()
    redner.test_d_intersect()
    redner.test_d_sample_shape()
    redner.test_atomic()

    if torch.cuda.is_available():
        redner.test_sample_primary_rays(True)
        redner.test_scene_intersect(True)
        redner.test_sample_point_on_light(True)
        redner.test_active_pixels(True)

unit_tests()

