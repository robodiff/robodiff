import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt

import numpy as np

try:
    from visualization.visualizer_gl_taichi_free import visualize
except ImportError:
    visualize = None

import pickle

def run_visualization(positions, masses, realized_actuation, title=None, gui_auto_close=False, **kwargs):
    
    step_count = positions.shape[0]
    particle_count = positions.shape[1]
    colors = np.ones(shape=(step_count, particle_count, 3))

    act_range = realized_actuation.max() - realized_actuation.min()
    realized_actuation /= act_range
    realized_actuation -= realized_actuation.min() # 0 to 1. midpoint is 0.5
    realized_actuation = realized_actuation 
    colors[:,:,0] = np.clip(realized_actuation*1.5, 0,1)
    colors[:,:,1] = (0.5 - np.abs(realized_actuation - 0.5))*1.5
    colors[:,:,2] = np.clip((1 - realized_actuation)*1.5,0,1)

    if visualize is not None:
        visualize(pos=positions,
                radius=masses*10,
                colors=colors, 
                title=title,
                initial_stepsize=5,
                gui_auto_close=gui_auto_close,  **kwargs)

x_avgs = []
v_avgs = []

def get_grad_per_step(simModel):
    mass_grads = simModel.morphModel._grads_dict["mass"].copy()
    mass_grads_raw = simModel.morphModel._grads_dict["mass_raw"].copy()

    grads_buffer = np.zeros((simModel.steps, 1, 2))

    dcmm = simModel.morphModel
    _ = dcmm.generate()
    dcmm.backward(mass=mass_grads_raw[0]) # can be anything. Just priming the genome.grad tensors to be valid. 

    for step_idx in range(simModel.steps):
        dcmm._genome["hole_x"].grad[:] = 0
        dcmm._genome["hole_y"].grad[:] = 0

        morphModel = dcmm.generate()
        dcmm.backward(mass=mass_grads_raw[step_idx])
        grads_buffer[step_idx, 0, 0] =dcmm._genome["hole_x"].grad.detach().cpu().numpy().copy()[0] * -1
        grads_buffer[step_idx, 0, 1] =dcmm._genome["hole_y"].grad.detach().cpu().numpy().copy()[0] * -1

    dcmm._genome["hole_x"].grad[:] = 0
    dcmm._genome["hole_y"].grad[:] = 0
    morphModel = dcmm.generate()
    dcmm.backward(mass=mass_grads) # need to return to prior value or gradient descent optimization will not work.
    dcmm._grads_dict["mass_raw"] = mass_grads_raw # need to return to prior value or visualization will fail.

    return grads_buffer



def visualization_callback(losses, positions, masses, realized_actuation, title="", gui_auto_close=False, x_avg=None, v_avg=None, simModel=None):
    if simModel is not None:
        hole_particle_indicies = np.arange(64*44)[(simModel.morphModel.generate()['mass'] < 1.0) & (simModel.morphModel.generate()['mass'] > 0.1)] # ROBUSTNESS_STUDY_CONSIDER_EXTENDING
        
        real_hole_loc_x = simModel.morphModel._genome['hole_x'].detach().cpu().numpy()
        real_hole_loc_xgd = simModel.morphModel._genome['hole_x'].grad.detach().cpu().numpy() *-1 # point in the direction of movement rather than the direction of gradient.
        real_hole_loc_y = simModel.morphModel._genome['hole_y'].detach().cpu().numpy()
        real_hole_loc_ygd = simModel.morphModel._genome['hole_y'].grad.detach().cpu().numpy() * -1


        initial_hole_particle_mean = positions[0, hole_particle_indicies].mean(axis=0)[:, None]
        real_hole_loc = (np.array([real_hole_loc_x, real_hole_loc_y])*0.2 + [[0.3], [0.03]])
        initial_offsets =  real_hole_loc - initial_hole_particle_mean

        arrow_centers = positions[:, hole_particle_indicies].mean(axis=1)[:,None,:] + initial_offsets.T
        arrow_dts = np.dstack([real_hole_loc_xgd, real_hole_loc_ygd]).repeat(positions.shape[0], axis=0)
        
        per_step_grads = get_grad_per_step(simModel) * simModel.steps

        arrow_dts = np.hstack([per_step_grads, arrow_dts])
        arrow_centers = arrow_centers.repeat(2, axis=1)

        
        realized_actuation_per_step = simModel.morphModel._grads_dict["mass_raw"].copy()
        realized_actuation = simModel.morphModel._grads_dict["mass"].copy()[None, :].repeat(realized_actuation_per_step.shape[0],axis=0)
        magnitude = np.nanmax(np.abs(realized_actuation))
        realized_actuation /= (magnitude )
        realized_actuation *= 3

        run_visualization(positions,
                    masses,
                    realized_actuation[:simModel.steps],
                    title=title,
                    gui_auto_close=gui_auto_close)
                    # arrow_centers=arrow_centers[:, 1:, :],
                    # arrow_dts=arrow_dts[:, 1:, :],
                    # arrow_colors=[[0,1,0,0.5]], #,[1,0,0,0.7]],
                    # arrow_widths=[10,])# 10]) # 

    else:
        run_visualization(positions, masses, realized_actuation, title=title, gui_auto_close=gui_auto_close)
    return
