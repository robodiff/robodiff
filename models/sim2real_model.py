# Hole patches (hard fringe)
# sine Actuators 
# No actuator within wall_thickness of robot exterior
#    * within hole_radius + wall_thickness of any hole
#    * within wall_thickness of bounds of rectangle.
#%%
from copy import deepcopy
import torch
import numpy as np
import IPython
import matplotlib.pyplot as plt
import math

import sys
sys.path.insert(0, "..")

from models.base_models import BaseUnifiedModel
from models.direct_circle_actuator_model import DirectCircleActuatorModel
from utils import draw

def validate_hole_tensor(x):
    return torch.where(torch.isfinite(x), x, torch.ones_like(x)*100)
    

def get_particle_pos(ms):
    x_t = torch.linspace(0, 1, ms.w_count, requires_grad=True)
    xx_t, yy_t = torch.meshgrid(x_t, x_t)
    stacked_particles = torch.stack([xx_t[:, :ms.h_count].flatten(), yy_t[:, :ms.h_count].flatten()], axis=1)
    return stacked_particles

def get_summed_dists(ms, h_loc, patch_size, sum=True):
    x_t = get_particle_pos(ms)
    x_t_expand = x_t[:, :, None].repeat((1, 1, h_loc.shape[2]))

    # compute distance to each hole, normalized so that 1 == radius of hole.
    dist_to_each_hole = torch.pow(x_t_expand - h_loc, 2).sum(axis=1).sqrt() / (patch_size+1e-9) # hole size

    # threshold such that no hole has a distance greater than 1
    dist_to_each_hole_thresholded = torch.where(dist_to_each_hole < 1.0,
                                                        dist_to_each_hole,
                                                        torch.ones_like(dist_to_each_hole))
    if sum:
        return dist_to_each_hole_thresholded.sum(axis=1)
    else:
        return dist_to_each_hole_thresholded

def get_actuation_soft_fringe(x, softness=2.0):
    return torch.where(x < 1.0,
                        torch.pow(1-x * np.sqrt(0.1), softness),
                        torch.zeros_like(x) + 1e-6)

def get_soft_fringe(x, softness=2.0):
        return torch.where(x < 1.0,
                            torch.pow(torch.abs(x), softness),
                            torch.ones_like(x))

def get_hard_fringe(summed_thresholded_dists, n_holes):
    # Set the inner 50% of each hole to be zero. This occurs when any hole is close to a given point thus the sum is less than 0.5 from the max possible sum of n_holes
    inner_hole_omitted = torch.where(summed_thresholded_dists <n_holes - 0.5,
                                        summed_thresholded_dists * 0.0,
                                        summed_thresholded_dists)

    # IPython.embed()
    # set other points to have a mass of 1
    return torch.where(inner_hole_omitted < n_holes,
                                inner_hole_omitted / n_holes,
                                torch.ones_like(inner_hole_omitted))

def get_wall_mask(ms,wall_thickness):

    x_t = get_particle_pos(ms)
    
    (x_max, y_max), _ = x_t.max(axis=0)
    active_mask = ((x_t > torch.tensor([wall_thickness, wall_thickness])) & 
                    (x_t < torch.tensor([x_max-wall_thickness, y_max - wall_thickness]))).all(axis=1)
    return active_mask
   
def get_actuation_permitted_mask(ms, h_loc, patch_size=0.2, wall_thickness=0.05):
    
    # dist_to_each_hole_padded = get_summed_dists(ms, h_loc, patch_size=patch_size+wall_thickness*2)
    active_patch_mask = patch_size > 0.0
    dist_to_each_hole_padded = get_summed_dists(ms, h_loc[:,:, active_patch_mask], patch_size=patch_size[active_patch_mask] * np.sqrt(0.1) * 1.3 + wall_thickness, sum=False).min(axis=1).values 
    masses = (dist_to_each_hole_padded >= 1.0) * 1.0
    # masses = get_hard_fringe(dist_to_each_hole_padded, h_loc.shape[2])
    
    wall_mask = get_wall_mask(ms, wall_thickness=wall_thickness)

    act_mask = (wall_mask + masses)  > 1.1

    return act_mask

#%%


class UnifiedSim2RealModel(BaseUnifiedModel):
    def __init__(self, scene=None, actPatchSize=0.2 * math.sqrt(0.1), wallThickness=0.01, genome=None, optimizer=None, device=None, **kwargs):
        
        super().__init__(genome=genome, optimizer=optimizer, device=device, **kwargs)

        self._scene = scene
        assert self._scene.w_count >= 128, "sim2real model requires a higher resolution robot. Please run with --high_resolution_robot"

        self._actPatchSize = actPatchSize
        self._wallThickness = wallThickness
        self.dim = 2
        self._omega = 40.0 / (np.pi * 2)

    def step(self):
        # self._actModel.step()

        if self._optimizer is not None:
            self._optimizer.step()
        

    def generate_mass(self):
        h_t = validate_hole_tensor(self._genome["hole_loc"])
        __hole_size_t = torch.max(self._genome["hole_size"], torch.zeros_like(self._genome["hole_size"]))
        _hole_size_t = torch.where(__hole_size_t < 1, __hole_size_t, torch.zeros_like(__hole_size_t))
        hole_size_t = torch.where(torch.isfinite(_hole_size_t), _hole_size_t, torch.zeros_like(_hole_size_t))

        n_holes = h_t.shape[2]

        min_dists = get_summed_dists(self._scene, h_t, patch_size=hole_size_t, sum=False).min(axis=1).values 
        self._final_masses = get_soft_fringe(min_dists)

        return self._final_masses

    def generate_actuation(self):
        h_t = validate_hole_tensor(self._genome["hole_loc"])
        __hole_size_t = torch.max(self._genome["hole_size"], torch.zeros_like(self._genome["hole_size"]))
        _hole_size_t = torch.where(__hole_size_t < 1, __hole_size_t, torch.zeros_like(__hole_size_t))
        hole_size_t = torch.where(torch.isfinite(_hole_size_t), _hole_size_t, torch.zeros_like(_hole_size_t))

        a_t = validate_hole_tensor(self._genome["act_loc"])

        act_mask = get_actuation_permitted_mask(self._scene, h_t, patch_size=hole_size_t, wall_thickness=self._wallThickness)

        dists = get_summed_dists(self._scene, a_t, self._actPatchSize, sum=False)
        dists_mined = dists.min(axis=1)[0]

        actuation_amplitude = get_actuation_soft_fringe(dists_mined)
        self._actuation_amplitude =  torch.where(act_mask, actuation_amplitude, torch.zeros_like(actuation_amplitude))


        return self._actuation_amplitude

    def generate(self):
        self._output_tensors =  {
                'mass':self.generate_mass(),
                'amplitude':self.generate_actuation(),
                'phase':  torch.zeros(self._actuation_amplitude.shape, requires_grad=True, device=self.device),
                'bias':  torch.zeros(self._actuation_amplitude.shape, requires_grad=True, device=self.device),
                'frequency':torch.ones(self._actuation_amplitude.shape, requires_grad=True, device=self.device) * self._omega
            }
        output_numpy = dict((k, v.detach().cpu().numpy()) if isinstance(v, torch.Tensor) else (k, v) for (k, v) in self._output_tensors.items() )

        return output_numpy

    def backward(self, **kwargs):
        """ 
        implements both
        """
        self._grads_dict = kwargs

        act_keys_of_interest = ["frequency", "amplitude", "phase", "bias"]
        keys_of_interest = ["mass"] + act_keys_of_interest
        for k in keys_of_interest:
            assert k in kwargs, f"UnifiedSim2Real must be passed '{k}' as a keyword argument to the backward method"
            assert kwargs[k] is not None

        cat_outputs = torch.cat([self._final_masses, self._actuation_amplitude])
        cat_grads   = torch.cat([self.validate_grads('mass', kwargs['mass']),
                                    self.validate_grads('amplitude', kwargs['amplitude'])])

        cat_outputs.backward(cat_grads)


#%%
if __name__ == "__main__":
    from models.utils import MockScene

    ms = MockScene()
    a_t = torch.rand(1, 2, 10) # hole locations
    dists = get_summed_dists(ms, a_t, 0.2, sum=False)
    dists_mined = dists.min(axis=1)[0]

    soft_fringe = get_soft_fringe(dists_mined)
    draw(dists_mined)
    draw(soft_fringe)


# %%
# %%
