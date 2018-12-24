class Scene:
    def __init__(self, camera, shapes, materials, area_lights, envmap = None):
        self.camera = camera
        self.shapes = shapes
        self.materials = materials
        self.area_lights = area_lights
        self.envmap = envmap
