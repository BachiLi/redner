
class Integrator:
    def render_image(self, seed, scene, img):
        yield NotImplementedError()

    def render_derivs(self, seed, scene, d_img, d_scene):
        yield NotImplementedError()

    def render_sceen_gradient(self, seed, scene, d_img, d_scene, screen_grad):
        yield NotImplementedError()

    def render_debug_image(self, seed, scene, d_img, d_scene, debug_img):
        yield NotImplementedError()

    def render(self, seed, scene, img, d_img, d_scene,
               screen_gradient_img, debug_img, num_samples):
        yield NotImplementedError()