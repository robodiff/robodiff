#%%
import torch
from .base_models import BaseActuationModel
import math
import numpy as np
#%%

class MixedSineActuatorModel(BaseActuationModel):
    """
    Genome should be a dict containing:
    particle_actuator_map
    weights
    bias
    actuation_omega
    """

    def __init__(self, genome=None, optimizer=None, gui=None, device=None, **kwargs):
        super().__init__(genome=genome, optimizer=optimizer, gui=gui, device=device, **kwargs)
        self._particle_map = self._genome["particle_actuator_map"]
        self._n_actuators = self._genome["weights"].shape[0]
        self._n_sine_waves = self._genome["weights"].shape[1]

        phases = [2 * math.pi / self._n_sine_waves * j for j in range(self._n_sine_waves)]

        self._phases = torch.tensor(phases, requires_grad=True, device=self.device)

    def get_summary_string(self):
        return super().get_summary_string(truncate=10, name="Act Mixed Sine")

    def generate(self):
        weights_t =  self._genome["weights"]
        bias_t =  self._genome["bias"]

        frequencies = torch.ones(self._n_actuators, requires_grad=True, device=self.device) * self._genome["actuation_omega"]

        phases = []
        amplitudes = []
        for idx in range(self._n_actuators):
            amplitude_sq = torch.tensor([0.0], device=self.device)
            for i in range(self._n_sine_waves):
                for j in range(self._n_sine_waves):
                    amplitude_sq += weights_t[idx, i] * weights_t[idx, j] * torch.cos(self._phases[i] - self._phases[j])

            phase_numerator = torch.sum(weights_t[idx, :] * torch.sin(self._phases))[None]
            phase_denominator = torch.sum(weights_t[idx, :] * torch.cos(self._phases))[None]
            phases.append(torch.atan2(phase_numerator, phase_denominator))
            amplitudes.append(torch.sqrt(amplitude_sq))
        
        phases_t = torch.cat(phases)
        amplitudes_t = torch.cat(amplitudes)

        self.output_tensors['phase'] = torch.index_select(phases_t, 0, self._particle_map)
        self.output_tensors['bias'] = torch.index_select(bias_t, 0, self._particle_map)
        self.output_tensors['frequency'] = torch.index_select(frequencies, 0, self._particle_map)
        self.output_tensors['amplitude'] = torch.index_select(amplitudes_t, 0, self._particle_map)
        
        return super().generate()

    def backward(self, **kwargs):
        super().backward(**kwargs)

def DefaultMixedSineActuatorModel(scene=None, device=None, weights_seed=0, **kwargs):
    # if we are loading from an exported optimized bot, we should ignore the request to use the default model
    if ("genome" in kwargs):
        if (kwargs["genome"] is not None):
            return MixedSineActuatorModel(scene=scene, device=device, **kwargs)
        else:
            del kwargs["genome"]    

    n_actuators = 1
    n_sine_waves = 4
    actuation_omega = 40.0 / (np.pi * 2)

    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(weights_seed)))
    np.random.set_state(rs.get_state())

    weights_np = np.random.randn(n_actuators, n_sine_waves) * 0.1
    weights = torch.zeros(weights_np.shape, requires_grad=True, device=device) 
    with torch.no_grad():
        weights[...] = torch.from_numpy(weights_np)

    bias = torch.zeros(n_actuators, requires_grad=True, device=device)
    omega = torch.tensor([actuation_omega], requires_grad=True, device=device)


    return MixedSineActuatorModel(
                                    genome={
                                        "particle_actuator_map":torch.tensor(scene.actuator_id[scene.n_terrain_particles:], device=device),
                                        "weights":weights,
                                        "bias":bias,
                                        "actuation_omega": omega
                                    },
                                    optimizerGenome = [weights, bias],
                                    device=device,
                                    **kwargs)
