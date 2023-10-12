import argparse
import datetime
import os

import taichi as ti
from models.model_loaders import SimulationModel

from model_tests.test_models import run_all_tests

from http_utils.utils import handle_cluster_env
from utils import is_in_cluster

from optimizers.gd_optimizer import GDOptimizer
from scene import Scene

from visualization.utils import visualization_callback
from config import FIGURE_DATA_DIR

import IPython
import pickle

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--verbose', '-v', action='count', default=0)

parser.add_argument("--gui", action="store_true")
parser.add_argument("--gui_auto_close", action="store_true", help="auto closes GUI window after showing all time steps of simulation")
parser.add_argument("--client", action="store_true", help="Force to run in HTTP server client mode")
parser.add_argument("--ignore_exit_request", action="store_true")
parser.add_argument("--ip", default="10.243.16.52")
parser.add_argument("--port", default="24478")
parser.add_argument("--resim", action="store_true", help="resimulate each robot for a longer period of time")
parser.add_argument("--wide_world", action="store_true", help="sets the simulation world to be twice as wide.")
parser.add_argument("--figure_export", action="store_true", help="Save simulation data to pickle for figure creation")
parser.add_argument("--playback", default="", type=str, help="replay simulations especially for plotting and visualizing.")
parser.add_argument("--playback_old_actuator_size", action="store_true", help="if using old json files where actuator patch sizes were converted at runtime, use this flag to trigger a conversion")
parser.add_argument("--force_local", action="store_true", help="Force to run in local mode. Used when working on a SLURM cluster and not wanting to startup in HTTP client mode")
parser.add_argument("--square_body", action="store_true", help="Switch from rectangle to square base body")
parser.add_argument("--local_selections", nargs="+", default=["base"], help="Select which local demo to run", choices=["base",
                                                                                                                     "design_starting_mask",
                                                                                                                    "void_interpolation",
                                                                                                                    "constrained_actuation",
                                                                                                                    "object_manipulation",
                                                                                                                    "direct_particle",
                                                                                                                ])

parser.add_argument("--max_steps", default=2048, type=int)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--cpu_max_threads", default=0, type=int)
parser.add_argument("--clean_sim", action="store_true")
parser.add_argument("--terrain", action="store_true")
parser.add_argument("--terrain_mode", default=1, type=int)
parser.add_argument("--object", action="store_true")
parser.add_argument("--high_resolution_robot", action="store_true", help="simulates the robot at higher resolution. Used when imposing the actuator manufacturability constraints")


args = parser.parse_args()
if "constrained_actuation" in args.local_selections:
    args.high_resolution_robot = True

if "object_manipulation" in args.local_selections:
    args.object = True

if args.client and args.force_local:
    raise Exception("Can not force both http client mode and local mode")

print(args)
scene = None

def get_hill_gravity(slope):
    theta = np.arctan2(1, 1/slope)
    return np.array([np.sin(theta), -1 * np.cos(theta)])
locomote_loss_mode = 0
jump_loss_mode = 1

flat_gravity = [0.0,-1.0] / np.linalg.norm([0.0, -1.0])

hill_gravity = get_hill_gravity(1/4)


loss_mode_mapper = {
    "locomote_flat":(locomote_loss_mode, flat_gravity ),
    "locomote_inclined":(locomote_loss_mode, hill_gravity),
    "jump_flat":(jump_loss_mode, flat_gravity),
    "jump_inclined":(jump_loss_mode, hill_gravity)
    }
act_model_dict = {
    "x": np.array([[1,0],[0,0]]),
    "y": np.array([[0,0],[0,1]]),
    "iso": np.array([[1,0],[0,1]]),
}
def assign_with_terrain_padding(taichi_arr, generated_arr, default):
    tmp = np.zeros((scene.n_particles,), dtype=generated_arr.dtype)
    tmp[:scene.n_terrain_particles] = default
    tmp[scene.n_terrain_particles:] = generated_arr
    taichi_arr.from_numpy(tmp)

def assign_with_terrain_padding_per_timestep(taichi_arr, generated_arr, default):
    tmp = np.zeros((simulator.x.shape[0], scene.n_particles,), dtype=generated_arr.dtype)
    tmp[:, :scene.n_terrain_particles] = default
    tmp[:, scene.n_terrain_particles:] = generated_arr
    taichi_arr.from_numpy(tmp)

def simulate(simModel,  enable_gui=True, epoch_idx=None, object_weight=0.25,  **kwargs):
    assert isinstance(simModel, SimulationModel)
    
    unified_model = simModel.unifiedModel
    usingUnified = unified_model is not None
    morph_model = simModel.morphModel
    actuation_model = simModel.actuationModel

    if usingUnified:
        generated_actuation = unified_model.generate()
        morph_model = unified_model
        actuation_model = unified_model

    else:
        generated_actuation = actuation_model.generate()

    assign_with_terrain_padding(simulator.actuation_frequency, generated_actuation['frequency'], 0.)
    assign_with_terrain_padding(simulator.actuation_amplitude, generated_actuation['amplitude'], 0.)
    assign_with_terrain_padding(simulator.actuation_phase, generated_actuation['phase'], 0.)
    assign_with_terrain_padding(simulator.actuation_bias, generated_actuation['bias'], 0.)

    if usingUnified:
        generated_morph = generated_actuation
    else:
        generated_morph = morph_model.generate()


    # If we are exporting x for morphology generation, then we treat it as a perturbation from the initial rectangle shape.
    if ('x' in generated_morph):
        xnp = simulator.x.to_numpy()
        xnp[0, scene.n_terrain_particles:, :] += generated_morph['x']
        simulator.x.from_numpy(xnp)
    
    assign_with_terrain_padding_per_timestep(simulator.m, generated_morph['mass'][None, :].repeat(simulator.x.shape[0], axis=0), object_weight)

    simulator.simulation_inclusion_threshold[None] = simModel.simulation_inclusion_threshold
    simulator.internal_damping[None] = simModel.internalDamping
    simulator.global_damping[None] = simModel.globalDamping
    simulator.E_base[None] = simModel.baseE
    simulator.friction[None] = simModel.friction
    simulator.act_strength[None] = simModel.actuationStrength
    simulator.actuation_sharpness[None] = simModel.actuationSharpness
    simulator.actuation_max_signal[None] = simModel.actuationMaxSignal
    simulator.actuation_proportional_to_mass[None] = simModel.actuationProportionalToMass
    simulator.act_model.from_numpy(act_model_dict[simModel.act_model])
    

    (loss_mode, loss_gravity) = loss_mode_mapper[simModel.loss_mode] 
    
    simulator.gravity[None] = loss_gravity * simModel.gravityStrength
    particle_start_idx = scene.n_terrain_particles if "terrain" not in simModel.loss_mode_includes else 0 # if terrain in loss_mode_includes then add the particles into the loss
    particle_end_idx = scene.n_particles if "body" in simModel.loss_mode_includes else scene.n_terrain_particles # 
    simulator.simulate(pre_grads_steps=simModel.pre_grads_steps,
                        total_steps=simModel.steps,
                        loss_mode=loss_mode,
                        particle_start_idx=particle_start_idx,
                        particle_end_idx=particle_end_idx,
                        with_grads=simModel.requires_taichi_grads,
                        old_mode=not args.clean_sim)

    l = simulator.loss[None]
    if simModel.requires_taichi_grads:

        morph_backwards_grads = {
            "mass":simulator.m.grad.to_numpy().sum(axis=0)[scene.n_terrain_particles:],
        }
        
        if ('x' in generated_morph):
            morph_backwards_grads["x"] = simulator.x.grad.to_numpy()[0, scene.n_terrain_particles:, :]


        act_backwards_grads = {
                "frequency":simulator.actuation_frequency.grad.to_numpy()[scene.n_terrain_particles:],
                "amplitude":simulator.actuation_amplitude.grad.to_numpy()[scene.n_terrain_particles:],
                "phase":simulator.actuation_phase.grad.to_numpy()[scene.n_terrain_particles:],
                "bias":simulator.actuation_bias.grad.to_numpy()[scene.n_terrain_particles:]
        }
        if usingUnified:
            unified_model.backward(
                **morph_backwards_grads, **act_backwards_grads
            )
        else:
            morph_model.backward(
                **morph_backwards_grads
            )

            actuation_model.backward(
                **act_backwards_grads
            )
    
    x_np = simulator.x.to_numpy()
    m_np = simulator.m.to_numpy()
    m_np_vis = m_np.copy() * 2 * 0.85
    m_np_vis[:, :scene.n_terrain_particles] *= 2 
    actuation_np = simulator.realized_actuation.to_numpy()
    

    if epoch_idx is not None and args.verbose > 0: # The primary morph model utilizes holes, however not all models do (e.g. direct particle model).
        print_str = f"iter: {epoch_idx:5d} loss: {l:10.7f}"
       
        if "hole_x" in morph_model._genome:
            try:
                morph_hole_x = morph_model._genome["hole_x"].detach().cpu().numpy()
                morph_hole_y = morph_model._genome["hole_y"].detach().cpu().numpy()
                morph_hole_x_g = morph_model._genome["hole_x"].grad.detach().cpu().numpy()
                morph_hole_y_g = morph_model._genome["hole_y"].grad.detach().cpu().numpy()
                print_str += f"\nmorphGenome: {(morph_hole_x[0], morph_hole_y[0])} morphGenomeGrad: {(morph_hole_x_g[0], morph_hole_y_g[0])}"
            except:
                pass                
        if "mass" in morph_model._genome:
            try:
                m = morph_model._genome['mass'].detach().cpu().numpy()
                x = morph_model._genome['x'].detach().cpu().numpy()

                mg = morph_model._genome['mass'].grad
                if mg is not None:
                    mg = mg.detach().cpu().numpy()
                    m_min = m.min()
                    m_max = m.max()

                    print_str += f"\nmorphGenome:\n\tmass:{m[:2]} ({m_min:8.5f},{m_max:8.5f})\n\tmg  :{mg[:2]}"
                xg = morph_model._genome['x'].grad
                if xg is not None:
                    xg = xg.detach().cpu().numpy()
                    print_str += "\n\tx :{x[:8]}\n\txg:{xg[:8]}"
            except Exception as e:
                IPython.embed()
        if "hole_loc" in morph_model._genome:
            hole_loc = morph_model._genome["hole_loc"]
            try:
                print_str += f"\nmorphGenome: {hole_loc[:, :, 0].detach().cpu().numpy()} morphGenomeGrad: {hole_loc.grad[:, :, 0].detach().cpu().numpy()}"
            except:
                pass
        print(print_str)

    if args.figure_export:
        fname = simModel.get_kwarg('fig_export_name')
        if fname is None:
            fname_base = simModel.get_kwarg("fig_export_base_name")
            if fname_base is not None:
                fname = f"{fname_base}_idx_{epoch_idx}.pkl"
        if fname is None:
            t = datetime.datetime.now()
            fname = f"figure_export_{t.year}_{t.month}_{t.day}_idx_{epoch_idx}.pkl"
        if args.terrain:
            fname = f"terrain_mode_{args.terrain_mode}_{fname}"
            
        step_cnt = simModel.steps
        
        pickle.dump({
            "x": x_np[:step_cnt],
            "x_avg":simulator.x_avg.to_numpy()[:step_cnt],
            "m":  m_np.copy()[:step_cnt],
            "actuation": simulator.realized_actuation.to_numpy()[:step_cnt],
            "pressures": simulator.realized_pressure.to_numpy()[:step_cnt],
            'loss':l,
        }, open(f"{FIGURE_DATA_DIR}/{fname}", "wb"))

#   if args.gui and ( epoch_idx is None or epoch_idx in [0, 10,25, 50]) and not is_in_cluster(): # and enable_gui # epoch_idx < 10 or
    if args.gui and not is_in_cluster(): # and enable_gui # epoch_idx < 10 or

        vis_title = ""
        if "title" in kwargs:
            vis_title = kwargs["title"]

        os.system('mkdir frames')
        os.system('mkdir frames/epoch' + str(epoch_idx) )
        os.system('rm    frames/epoch' + str(epoch_idx) + '/*.npy') 
        np.save(        'frames/epoch' + str(epoch_idx) + '/positions' + str(epoch_idx) , x_np[        :simModel.steps] )
        np.save(        'frames/epoch' + str(epoch_idx) + '/actuation' + str(epoch_idx) , actuation_np[:simModel.steps] )
        print('saving frame ' + str(epoch_idx) )

        visualization_callback([l],
                                x_np[:simModel.steps],
                                m_np_vis, actuation_np[:simModel.steps],
                                x_avg = simulator.x_avg.to_numpy(),
                                v_avg = simulator.v_avg.to_numpy(),
                                title=vis_title, gui_auto_close=args.gui_auto_close)
                                # simModel=simModel)

        
    if args.verbose > 3: # if in debug mode, example usage: when simulation_inclusion_threshold is very low, actuation_proportional_to_mass is disabled 
        def validate_tensor(t, name):
            if (not np.isfinite(t).all()):
                print(f"{name} had NaNs!")
                return True
            else:
                print(f"{name} all finite")
                return False

        nans_exist =  np.array([ validate_tensor(simulator.x.to_numpy(), "x"),
                            validate_tensor(simulator.x.grad.to_numpy(), "x_grad"),
                            validate_tensor(simulator.m.to_numpy(), "m"),
                            validate_tensor(simulator.m.grad.to_numpy(), "m_grad"),
                            validate_tensor(simulator.v.to_numpy(), "v"),
                            validate_tensor(simulator.v.grad.to_numpy(), "v_grad")]).any()

        if nans_exist and args.verbose > 3: # if in trace mode
            IPython.embed()

    return {
        "loss":l,
        "x": x_np,
        "m": m_np,
        'actuation': actuation_np,
        "x_avg": simulator.x_avg.to_numpy()
    }


def run_optimization(simModel,
                    disable_optimization=False,
                    results_handler=None,
                    gui_handler=None):

    simulate_lambda = lambda **kwargs: simulate(simModel, **kwargs)

    optimizer = GDOptimizer(simModel,
                    simulate_lambda,
                    gui_handler = gui_handler,
                    results_handler = results_handler
                )

    return optimizer.optimize(disable_optimization)
        


def main(dim=2):
    global scene
    global simulator

    assert dim == 2, "3D simulations not yet supported"
    import simulator as simulator
    
    from base_robots.base_robots import robot

    # initialization
    scene = Scene(simulator.x, simulator.v, simulator.m, simulator.F, simulator.C, simulator.particle_type,  dx=simulator.dx, dim=dim, terrain=args.terrain, terrain_mode=args.terrain_mode)
    robot(scene, add_object=args.object)
    simulator.place()
    scene.reset()

    if not args.force_local and (is_in_cluster() or args.client):
        handle_cluster_env(scene, ip=args.ip, port=args.port, ignore_exit_request=args.ignore_exit_request, verbose=args.verbose)
        exit(0)

    run_all_tests()
    exit(0)

