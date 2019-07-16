class Scene:
    def __init__(self, camera, shapes, materials, area_lights, envmap = None):
        self.camera = camera
        self.shapes = shapes
        self.materials = materials
        self.area_lights = area_lights
        self.envmap = envmap

    def state_dict(self):
        return {
            'camera': self.camera.state_dict(),
            'shapes': [s.state_dict() for s in self.shapes],
            'materials': [m.state_dict() for m in self.materials],
            'area_lights': [l.state_dict() for l in self.area_lights],
            'envmap': self.envmap.state_dict() if self.envmap is not None else None
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        envmap_dict = state_dict['envmap']
        return cls(
            pyredner.Camera.load_state_dict(state_dict['camera']),
            [pyredner.Shape.load_state_dict(s) for s in state_dict['shapes']],
            [pyredner.Material.load_state_dict(m) for m in state_dict['materials']],
            [pyredner.AreaLight.load_state_dict(l) for l in state_dict['area_lights']],
            pyredner.EnvironmentMap.load_state_dict(envmap_dict) if envmap_dict is not None else None)
