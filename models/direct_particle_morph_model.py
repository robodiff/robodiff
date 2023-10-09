import torch
from .base_models import BaseMorphModel

class DirectParticleMorphModel(BaseMorphModel):
    def __init__(self, scene=None,
                        mode="position",
                        genome=None,
                        optimizer=None,
                        gui=None,
                        device=None,
                        **kwargs):
        super().__init__(genome=genome, optimizer=optimizer, gui=gui, device=device, **kwargs)
        assert mode in ["position", "mass"]
        self.mode = mode
        self.is_in_position_mode = mode == "position"
        self._scene = scene
        self._final_tensor = None

    def get_summary_string(self):
        return super().get_summary_string(name=f"Morph Direct Particle: {self.mode}", truncate=10)

    def generate(self):
        mass_t = self._genome["mass"]

        return_tensors = {
            'mass':mass_t.detach().cpu().numpy()
        }

        if self.is_in_position_mode:
            x_t =  self._genome["x"]
            self._final_tensor = x_t
            return_tensors['x'] = x_t.detach().cpu().numpy()
        else:
            self._final_tensor = mass_t

        return return_tensors


    def backward(self, **kwargs):
        super().backward(**kwargs)

        if self.is_in_position_mode:
            self._final_tensor.backward(self.validate_grads('x', kwargs['x']))
        else:
            self._final_tensor.backward(self.validate_grads('mass', kwargs['mass']))

def DefaultParticleMorphModel(scene=None, device=None, **kwargs):
    return DirectParticleMorphModel(scene=scene,
                                    mode="mass",
                                    genome={"mass":[1.0]*(scene.n_particles - scene.n_terrain_particles)},
                                    optimizer=None,
                                    device=device,
                                    )