import torch
from .base_models import BaseActuationModel

import math
import pprint
import IPython

class DirectCircleActuatorModel(BaseActuationModel):
    def __init__(self, scene=None, actuator_size=0.2, actuator_sizes=None, genome=None, optimizer=None, gui=None, device=None, reset_patches=False, act_morph_fringe_softness="quadratic", **kwargs):
        super().__init__(genome=genome, optimizer=optimizer, gui=gui, device=device, **kwargs)
        if "act_size" not in self._genome:
            if actuator_sizes is not None:
                self._genome["act_size"] = torch.tensor(actuator_sizes, device=device)
            else:
                self._genome["act_size"] = torch.tensor([actuator_size]*self._genome["hole_x_sine"].shape[0], device=device)

        self._actuator_size = actuator_size
        self._scene = scene
        self.orig_act_omega = self._genome["actuation_omega"].clone()
        self._reset_patches = reset_patches

        self.act_morph_fringe_softness = act_morph_fringe_softness

    def get_summary_string(self):
        return super().get_summary_string(name="Act direct circle")

    def generate(self):
        with torch.no_grad():
            self._active_patch_log = []

        _c_x_a = self._genome["hole_x_sine"].to(self.device)
        _c_y_a = self._genome["hole_y_sine"].to(self.device)
        _c_x_b = self._genome["hole_x_cosine"].to(self.device)
        _c_y_b = self._genome["hole_y_cosine"].to(self.device)
        
        c_x_a = torch.where(torch.isfinite(_c_x_a), _c_x_a, torch.ones_like(_c_x_a)*100)
        c_y_a = torch.where(torch.isfinite(_c_y_a), _c_y_a, torch.ones_like(_c_y_a)*100)
        c_x_b = torch.where(torch.isfinite(_c_x_b), _c_x_b, torch.ones_like(_c_x_b)*100)
        c_y_b = torch.where(torch.isfinite(_c_y_b), _c_y_b, torch.ones_like(_c_y_b)*100)

        act_size_t = torch.max(self._genome["act_size"], torch.zeros_like(self._genome["act_size"]))

        x_t = torch.linspace(0, 1, self._scene.w_count, requires_grad=True, device=self.device)
        xx_t, yy_t = torch.meshgrid(x_t, x_t)
        xx_t = xx_t[:, :, None].repeat((1, 1, c_x_a.shape[0]))
        yy_t = yy_t[:, :, None].repeat((1, 1, c_x_a.shape[0]))

        brightnesses = [] # sine at index 0, cosine at index 1
        for n, (c_x_t, c_y_t) in enumerate([(c_x_a, c_y_a), (c_x_b, c_y_b)]):
            dx_t = torch.pow(torch.abs(xx_t - c_x_t)+1e-9, 2) # dx_t == 0 => we get NaNs during backprop... :(
            dy_t = torch.pow(torch.abs(yy_t - c_y_t)+1e-9, 2) # dy_t == 0 => we get NaNs during backprop... :(

            dist_t = torch.sqrt(dx_t + dy_t) / (act_size_t+1e-9)
            frac_towards_center_t, active_patches = (dist_t).min(axis=2)
            with torch.no_grad():
                self._active_patch_log.append( active_patches.detach().unique())
            if self.act_morph_fringe_softness == "none":
                 brightness_t = torch.where(frac_towards_center_t < 1,
                                           torch.ones_like(frac_towards_center_t),
                                           torch.zeros_like(frac_towards_center_t)+1e-6)
            else:
                interpolation_power = None
                if self.act_morph_fringe_softness == "quadratic":
                    interpolation_power = 2
                elif self.act_morph_fringe_softness == "linear":
                    interpolation_power = 1
                elif self.act_morph_fringe_softness == "cubic":
                    interpolation_power = 3
                elif self.act_morph_fringe_softness == "quartic":
                    interpolation_power = 4
                else:
                    raise NotImplementedError()
                sign = 1 if n == 0 else -1
                brightness_t = torch.where(frac_towards_center_t < 1,
                                            sign * torch.pow((1-frac_towards_center_t * math.sqrt(0.1)), interpolation_power),
                                            torch.zeros_like(frac_towards_center_t) + 1e-6
                                            )
                
            brightnesses.append(brightness_t)

        amplitude = torch.sqrt(
                                torch.pow(brightnesses[0], 2)
                                +
                                torch.pow(brightnesses[1], 2)
                            )

        amplitude_normalized = amplitude / torch.sqrt(torch.tensor(2.0))
        phase = torch.atan2(-brightnesses[0], brightnesses[1]) + math.pi/2 # https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Sine_and_cosine cos(t + phi) = arctan(-sum sin / sum cos)
        
        out_amplitude = torch.nn.functional.interpolate(amplitude_normalized[None, None, :, :], (self._scene.w_count, self._scene.w_count))
        out_phase = torch.nn.functional.interpolate(phase[None, None, :, :], (self._scene.w_count, self._scene.w_count))

        out_amplitude_flat = out_amplitude[0,0][:, :self._scene.h_count].flatten()
        out_phase_flat = out_phase[0,0][:, :self._scene.h_count].flatten()

        out_frequencies = torch.ones(out_phase_flat.shape[0], requires_grad=True, device=self.device) * self.orig_act_omega.to(self.device)
        out_bias = torch.zeros(out_phase_flat.shape[0], requires_grad=True, device=self.device)

        self.output_tensors['phase'] = out_phase_flat
        self.output_tensors['bias'] = out_bias
        self.output_tensors['frequency'] = out_frequencies
        self.output_tensors['amplitude'] = out_amplitude_flat

        return super().generate()

    def step(self):
        super().step()

        if self._optimizer is not None and self._reset_patches:
            with torch.no_grad():
                hx = self._genome["hole_x_sine"]

                num_holes = hx.shape[0]
                uniques, counts = torch.cat((torch.arange(num_holes), self._active_patch_log[0])).unique(return_counts=True)
                inactive_patches = uniques[counts==1]
                num_inactive = inactive_patches.shape[0]

                # reset the holes
                self._genome['hole_x_sine'][inactive_patches]  = torch.rand(num_inactive)
                self._genome['hole_y_sine'][inactive_patches]  = torch.rand(num_inactive) * 0.7

                # reset the adam momentum.
                for paramGroupKey in ['hole_x_sine', 'hole_y_sine']:
                    paramGroup =self._genome[paramGroupKey]
                    paramGroupState = self._optimizer.state[paramGroup]
                
                    paramGroupState['exp_avg'][inactive_patches] = 0
                    paramGroupState['exp_avg_sq'][inactive_patches] = 0
                
                print(f"resetting {num_inactive} act patches\n\n")
                # print(f"resetting {num_inactive} holes ({inactive_holes_size.sum()}s, {inactive_holes_location.sum()}loc)\n\n")


    def backward(self, **kwargs):
        super().backward(**kwargs)
