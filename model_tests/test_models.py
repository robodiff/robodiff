import numpy as np
import math
import uuid
import pprint
from tqdm import tqdm

import sys
sys.path.insert(0, "..")
from config import FIGURE_DATA_DIR
from model_helpers import *

from model_tests import robustness_study

import IPython
import itertools

### UTILS: move to another file.
def cartesian_product_helper(**kwargs):
    k,v = zip(*[(k,v) for (k,v) in  kwargs.items()])
    for d in itertools.product(*v):
        yield dict(zip(k, d))


## GET individual optimization configuration starting points.
def get_direct_particle_sim(actuator_patch_count=64,  
                                actuation_omega=40.0/(2*np.pi),
                                optimize_actuators=True,
                                epochs=10):
    mass_with_act_patches = get_model_dict(
                        get_morph_direct_particle(mode='mass'),
                        get_act_rand_direct_circle_sine_only(hole_count=actuator_patch_count,
                                                    optimize=optimize_actuators,
                                                    actuation_omega=actuation_omega),
                        description_str=f"Hole count: {actuator_patch_count} with 'y' axis Actuation",
                        actuationProportionalToMass=True,
                        epochs=epochs,
                        steps=1024,
                        pre_grads_steps=0,
                        requires_taichi_grads=True,
                        gravityStrength=1.8*3,
                        act_model="y",
                        actuatorStrength=4.0, 
                        actuationSharpness=1.0,
                        friction=0.5,
                        fig_export_base_name="FigS2_DirectParticleMass")
    return mass_with_act_patches

def get_void_interpolation_sim(epochs=10, hole_count=64, fringe_softness=2.0):
    sim = get_base_experiments_sim(epochs, True, True, hole_count, fringe_softness=fringe_softness)
    sim["description_str"]  = f"{sim['description_str']} | Fringe Softness: {fringe_softness}"
    return sim

def get_constrained_actuation_sim(n_patches=64, epochs=10):
    act_patch_count = n_patches
    hole_patch_count = n_patches

    hole_patch_size = compute_mean_hole_size(0.6, 0.7, hole_patch_count)

    hole_loc = np.random.random((1, 2, hole_patch_count))
    hole_loc[:, 1, :] *= 0.7
    morph_hole_sizes = np.random.normal(hole_patch_size, hole_patch_size**2, size=(hole_patch_count)).tolist()

    act_loc = np.random.random((1, 2, act_patch_count))
    act_loc[:, 1, :] *= 0.7

    act_patch_size = 0.2 * np.sqrt(0.1) # compute_mean_hole_size(0.6, 0.7, act_patch_count)
    unified_model = {
                "name":"UnifiedSim2RealModel",
                "genome": {
                    "hole_loc":hole_loc.tolist(),
                    "act_loc":act_loc.tolist(),
                    "hole_size":morph_hole_sizes,
                    },
                "optimizer":get_adam_optimizer(),
                "actPatchSize":act_patch_size,
                "wallThickness":1/64,
                }

    return get_model_dict(
                None,
                None,
                unifiedModel=unified_model,
                description_str=f"unified sim2real model",
                actuationProportionalToMass=True,
                epochs=epochs,
                steps=1024,
                pre_grads_steps=0,
                requires_taichi_grads=True,
                gravityStrength=1.8*3,
                act_model="y",
                actuatorStrength=4.0, 
                actuationSharpness=1.0,
                # actuationMaxSignal = 0.0, # was default (100000)
                friction=0.5,
                fig_export_base_name="FigS4_ConstrainedsActuators")

def get_base_experiments_sim(epochs=10,
                optim_act=True,
                optim_morph=True,
                hole_count=64,
                fringe_softness=2.0,
                power_law_hole_sizes=False):

    morph_model = get_morph_rand_direct_circle_multi_sized_holes(hole_count=hole_count, optimize=optim_morph, fixed_hole_sizes=False, fringe_softness=fringe_softness, power_law_hole_sizes=power_law_hole_sizes)
    morph_model["morphMask"] = None
    act_model = get_act_rand_direct_circle_sine_only(hole_count=hole_count,
                                                    optimize=optim_act,
                                                    actuation_omega=40.0/(2*np.pi))
    return get_model_dict(
                morph_model,
                act_model,
                description_str=f"Hole count: {hole_count} with 'y' axis Actuation",
                actuationProportionalToMass=True,
                epochs=epochs,
                steps=1024,
                pre_grads_steps=0,
                requires_taichi_grads=True,
                gravityStrength=1.8*3,
                act_model="y",
                actuatorStrength=4.0, 
                actuationSharpness=1.0,
                # actuationMaxSignal = 0.0, # was default (100000)
                friction=0.5)

def get_diversity_sim(hole_count=64,
                    hole_size_constraints="independent",
                    fringe_softness=2.0,
                    actuator_count = 32,
                    actuation_omega = 40,
                    act_mode="y",
                    epochs=10,
                    steps=1024,
                    friction=0.5,
                    gravity=1.6*3,
                    actuatorStrength=4.0,
                    description_str='diversity',
                    morphMask=None,
                    loss_mode_includes='body',
                    robot_aspect_ratio=0.7,
                    ):
    morph_model = get_morph_rand_direct_circle_multi_sized_holes(hole_count=hole_count,
                                                                    fixed_hole_sizes= hole_size_constraints == "fixed",
                                                                    shared_hole_sizes= hole_size_constraints == "shared",
                                                                    fringe_softness=fringe_softness,
                                                                    robot_aspect_ratio=robot_aspect_ratio,
                                                                    )
    morph_model['morphMask'] = morphMask

    act_model = get_act_rand_direct_circle(hole_count=actuator_count,
                                                optimize=True,
                                                actuation_omega=actuation_omega/(2*np.pi),
                                                robot_aspect_ratio=robot_aspect_ratio,)
    model = get_model_dict(morph_model,
                            act_model,
                            description_str=description_str,
                            act_model=act_mode,
                            epochs=epochs,
                            steps=steps,
                            friction=friction,
                            gravityStrength=gravity,
                            actuatorStrength=actuatorStrength,
                            loss_mode_includes=loss_mode_includes,
                )
    return model


## GROUP together many starting points into a job.

def get_sims(sim_factory, *args, N=100,
                                K=10,
                                iterations_per_K=1,
                                job_title="default title",
                                **kwargs):

    # compute number of sims that we need to generate.
    restarts = N * K // iterations_per_K

    sims = [sim_factory(*args, **kwargs) for _ in range(restarts)]

    info =  {
        "job_title":job_title,
        "args": args,
        "kwargs": kwargs
    }
    return {
        "sims":sims,
        "info":info
    }

def direct_particle(N=25, K=50, seed=1, actuator_patch_count=64):
    np.random.seed(seed)
    
    return [get_sims(get_direct_particle_sim,
                        N=N,
                        K=K,
                        iterations_per_K=K,
                        job_title="direct_particle_sim",
                        actuator_patch_count=actuator_patch_count,
                        epochs=K
                        )] # a single job.


def void_interpolation(N=25, K=10, seed=1, hole_count=64):
    np.random.seed(seed)
    fringe_softnesses = [0] # just do a hard sim. other values (e.g. 1 would be softer, 2 is normal, and 4 is harder.)

    return [get_sims(get_void_interpolation_sim,
                        N=N,
                        K=K,
                        iterations_per_K=K,
                        job_title="fringe_softness",
                        hole_count=hole_count,
                        fringe_softness=fringe_softness,
                        epochs=K) for fringe_softness in fringe_softnesses]


def constrained_actuation(N=25, K=10, seed=1, n_patches=64):
    np.random.seed(seed) # ensure reproducibility of starting robot designs.

    return [get_sims(get_constrained_actuation_sim, 
                            N=N,
                            K=K,
                            iterations_per_K=K,
                            job_title="constrained actuation",
                            epochs=K,
                            n_patches=n_patches
                            )]


def base_experiments(N=100, K=10, seed=1, hole_count=64, fringe_softness=2.0,power_law_hole_sizes=False):
    """
    :param: N number of independent optimization run.
    :param: K number of design samples per run. If optimized, gradient descent is used, else random search is used.
    :param: seed, seed to use to generate the initial set of robot designs. 
    """
    np.random.seed(seed) # ensure reproducibility of starting robot designs.

    jobs = []
    # add the experiments
    for optim_act in [True, False]:
        for optim_morph in [True, False]:
            iterations_per_K = K if (optim_act or optim_morph) else 1
            jobs.append(
                get_sims(
                    get_base_experiments_sim,
                    N=N,
                    K=K,
                    iterations_per_K=iterations_per_K,
                    epochs=iterations_per_K,
                    optim_act=optim_act,
                    optim_morph=optim_morph,
                    hole_count=hole_count,
                    fringe_softness=fringe_softness,
                    power_law_hole_sizes=power_law_hole_sizes
                )
            )
    return jobs

def explore_diversity():
    jobs = []

    for kwargs in cartesian_product_helper(N = [25],
                                            restarts_per_K=[1],
                                            job_title=["diversity job"],
                                            steps=[1024],
                                            act_mode=["y"],
                                            fringe_softness=[2.0],
                                            hole_count=[16,32,64],
                                            actuator_count=[16, 32, 64],
                                            friction=[0.25, 0.5],
                                            gravity= [1.8, 1.8*2, 1.8*3],
                                            hole_size_constraints=["fixed", "shared", "independent"],
                                            epochs=[10]):
        jobs.append(get_sims(get_diversity_sim,
                                        **kwargs))
    return jobs
    
## SELECT the first design to visualize in local session.
def base_experiments_first(fringe_softness=2.0, power_law_hole_sizes=False):
    jobs = base_experiments(fringe_softness=fringe_softness, power_law_hole_sizes=power_law_hole_sizes)
    job = jobs[0]
    sims = job['sims']
    job_info = job['info']["kwargs"]
    assert job_info["optim_act"] == True and job_info["optim_morph"] == True
    return [sims[0]]

def direct_particle_first():
    return direct_particle()[0]["sims"][:1]

def void_interpolation_first():
    return void_interpolation()[0]["sims"][:1]

def constrained_actuation_first():
    return constrained_actuation()[0]["sims"][:1]


def insert_or_update(x, **kwargs):
    for k, v in kwargs.items():
        if k in x.keys() and isinstance(v, dict):
            insert_or_update(x[k], **v)
        else:
            x[k] = v

def f4_object_dict(secondary_losses=["erodeProportional", "rotationalMoment"],
                    loss_mode_includes="body",
                    erode_loss_amount=0.02,
                    erode_target_amount=0.5,
                    rotational_moment_amount=4e-4,
                    circle_mask_size=0.25,
                    mask_alignment_amount=1e-5
                    ):
    return {
        "morphModel":
            {
                "secondary_losses":secondary_losses,
                "erode_loss_amount": erode_loss_amount,
                "erode_target_amount":erode_target_amount,
                "rotational_moment_amount":rotational_moment_amount,
                "reset_patches":True,
                "mask_alignment_amount":mask_alignment_amount,
                "circle_mask_size":circle_mask_size,
            },
            "actuationModel": 
            {
                "reset_patches":True,
            },
            "loss_mode_includes":loss_mode_includes
    }


def object_manipulation(K=50, N=10, seed=1):
    np.random.seed(seed)

    jobs_modifications = [f4_object_dict(secondary_losses=["erodeProportional", "rotationalMoment"], loss_mode_includes="terrain"), # throw object
                            f4_object_dict(secondary_losses=["erodeProportional", "rotationalMoment"], loss_mode_includes="body"), # ignore object
                            f4_object_dict(secondary_losses=["erodeProportional", "rotationalMoment"], loss_mode_includes="bodyterrain"), # carry object
                            f4_object_dict(secondary_losses=["erodeProportional"], loss_mode_includes="body"), # ignore object
                            f4_object_dict(secondary_losses=["erodeProportional"], loss_mode_includes="bodyterrain"), # carry object
                            ]
    jobs = []
    for job_modification in jobs_modifications:

        job =  get_sims(
                        get_diversity_sim,
                        N=N,
                        K=K,
                        iterations_per_K=K,
                        epochs=K,
                        hole_count=64,
                        hole_size_constraints="independent",
                        fringe_softness=2.0,
                        actuator_count = 32,
                        actuation_omega = 40,
                        act_mode="y",
                        steps=1024,
                        friction=0.5,
                        gravity=1.6*3,
                        actuatorStrength=6,
                        description_str='diversity',
                        loss_mode_includes='bodyterrain', # body, terrrain, bodyterrain
                        morphMask='none' # triangleA, triangleB, circle, 
                    )
        insert_or_update(job['info'], **job_modification)

        for sim in job['sims']:
            insert_or_update(sim, **job_modification)
        jobs.append(job)

    return jobs
def object_manipulation_first(**kwargs):
    return object_manipulation(**kwargs)[0]["sims"][:1]

def f4_circle_mask(seed=1, N=10, circle_size=0.25):
    np.random.seed(seed)
    sims = [get_base_experiments_sim(epochs=100,
                optim_act=True,
                optim_morph=True,
                hole_count=64,
                fringe_softness=2.0,
                power_law_hole_sizes=False) for _ in range(N) ]
    object_dict = f4_object_dict(secondary_losses=["circleMask"],
                                                loss_mode_includes="body",
                                                mask_alignment_amount= np.linspace(0,  4 * 1e-5, 50).tolist(),
                                                circle_mask_size=circle_size,
                                                )
    for sim in sims:
        insert_or_update(sim, **object_dict )
    
    info =  {
        "job_title":"circle mask",
        "kwargs": object_dict
    }

    return {"sims":sims, "info":info}

def f4_rotational_moment(seed=1, N=10):
    np.random.seed(seed)
    sims = [get_base_experiments_sim(epochs=100,
                optim_act=True,
                optim_morph=True,
                hole_count=64,
                fringe_softness=2.0,
                power_law_hole_sizes=False) for _ in range(N)]

    object_dict = f4_object_dict(secondary_losses=["erodeProportional", "rotationalMoment"],
                                                loss_mode_includes="body",
                                                erode_loss_amount=0.01,
                                                erode_target_amount=0.5,
                                                rotational_moment_amount=4e-4,
                                                )
    for sim in sims:
        insert_or_update(sim, **object_dict )
    info =  {
        "job_title":"circle mask",
        "kwargs": object_dict
    }
    return {"sims":sims, "info":info}

def secondary_loss_functions():
    jobs = []
    jobs.append(f4_circle_mask(circle_size=0.25))
    jobs.append(f4_circle_mask(circle_size=0.35))
    jobs.append(f4_rotational_moment())
    return jobs

def design_masks(N=25, K=100, seed=1, hole_count=64, fringe_softness=2.0, power_law_hole_sizes=False):
    np.random.seed(seed) # ensure reproducibility of starting robot designs.

    jobs = []
    iterations_per_K = K # run optimization

    for circle_mask_size in [0.25, 0.3, 0.35]:
        sims = [get_base_experiments_sim(epochs=K,
                            optim_act=True,
                            optim_morph=True,
                            hole_count=64,
                            fringe_softness=2.0,
                            power_law_hole_sizes=False)]
        

        
    secondaryLosses = ["erodeProportional", "rotationalMoment", "circleMask"]
    secondaryLossesIdxs = [(0, 0), # erode only
                            (0, 1), # erode and rotational moment
                            (0, 2), # erode, rotational moment, and circle mask
                            (1, 1), # rotational moment only
                            (2, 2), # circle mask only
                            ]
                            
    for startIdx, stopIdx in secondaryLossesIdxs:
        for erode_target_amount in [0.25, 0.5, 0.75]:
            for erode_loss_amount in [0.01]:
                for rotational_moment_amount in [1e-4, 5e-4, 1e-3]:
                    for mask_alignment_amount in np.log10(np.logspace(1e-5, 1e-4, 15)):
                        sims = get_sims(
                                get_base_experiments_sim,
                                N=N,
                                K=K,
                                iterations_per_K=iterations_per_K,
                                epochs=iterations_per_K,
                                optim_act=True,
                                optim_morph=True,
                                hole_count=hole_count,
                                fringe_softness=fringe_softness,
                                power_law_hole_sizes=power_law_hole_sizes
                            )

                        for sim in sims['sims']:
                            sim["morphModel"]["secondary_losses"] = secondaryLosses[startIdx:stopIdx+1]

                            # secondary loss variables are only used if secondary losses are enabled.
                            sim['morphModel']['erode_loss_amount'] = erode_loss_amount
                            sim['morphModel']['erode_target_amount'] = erode_target_amount
                            sim['morphModel']['rotational_moment_amount'] = rotational_moment_amount
                            sim['morphModel']['mask_alignment_amount'] =  np.linspace(0, mask_alignment_amount, 100).tolist()

                            # always enable patch-resetting
                            sim['morphModel']['reset_patches'] = True
                            sim['actuationModel']['reset_patches'] = True
                        jobs.append(sims)
    return jobs

def design_masks_first(**kwargs):
    return design_masks(**kwargs)[0]["sims"][:1]

def f4_multi_phase(seed=1, hole_count=64,
                        optim_act=True, optim_morph=True, fringe_softness=2.0):
    np.random.seed(seed)
    morph_model = get_morph_rand_direct_circle_multi_sized_holes(hole_count=hole_count,
                                                                optimize=optim_morph, fixed_hole_sizes=False,
                                                                robot_aspect_ratio=0.7,
                                                                fringe_softness=fringe_softness)
    morph_model["morphMask"] = None
    morph_model['reset_patches'] = True
    act_model = get_act_rand_direct_circle(hole_count=hole_count,
                                                optimize=optim_act,
                                                actuation_omega=40.0/(2*np.pi))
    act_model['reset_patches'] = True
    return [get_model_dict(
                morph_model,
                act_model,
                description_str=f"Hole count: {hole_count} with 'y' axis Actuation",
                actuationProportionalToMass=True,
                epochs=10,
                steps=1024,
                pre_grads_steps=0,
                requires_taichi_grads=True,
                gravityStrength=1.8*3,
                act_model="y",
                actuatorStrength=4.0, 
                actuationSharpness=1.0,
                # actuationMaxSignal = 0.0, # was default (100000)
                friction=0.5)]

def masked_starting_points(K=10, N=10, seed=1, robot_aspect_ratio=0.7):
    np.random.seed(seed)

    jobs = []
    for morphMask in [ 'star5small', 'star6','star9b', 'star9a', 'star8', 'star7','star5', 'triangleC', 'triangleA', 'triangleB', 'circle']:

        job =  get_sims(
                        get_diversity_sim,
                        N=N,
                        K=K,
                        iterations_per_K=K,
                        epochs=K,
                        hole_count=64,
                        hole_size_constraints="independent",
                        fringe_softness=2.0,
                        actuator_count = 32,
                        actuation_omega = 40,
                        act_mode="y",
                        steps=1024,
                        friction=0.5,
                        gravity=1.6*3,
                        actuatorStrength=6,
                        description_str='masked starting point',
                        loss_mode_includes='body', # body, terrrain, bodyterrain
                        morphMask=morphMask, #'none'  triangleA, triangleB, circle, 
                        robot_aspect_ratio=robot_aspect_ratio,
                    )
        # insert_or_update(job['info'], fig_export_name=f"Masked_Optimization_{morphMask}.pkl")

        for sim in job['sims']:
            insert_or_update(sim, fig_export_name=f"Masked_Optimization_{morphMask}.pkl")
        jobs.append(job)

    return jobs
def masked_starting_points_first(**kwargs):
    return masked_starting_points(**kwargs)[0]["sims"][:1]

def validate_git_repo_and_get_sha():
    from subprocess import check_output
    if len(check_output(" git status --porcelain".split()).decode("utf-8")) != 0:
        raise RuntimeError("Aborting. Your Git Working Directory is not clean! Please commit or .gitignore all changes.")
    return check_output("git rev-parse HEAD".split()).decode("utf-8")

def export_model_simreal(simModel, seed=0, iteration=0,):
    import json
    import sys

    morph_model = simModel.morphModel
    actuation_model = simModel.actuationModel
    generated_actuation = actuation_model.generate()
    generated_morph = morph_model.generate()

    for k,v in generated_actuation.items():
        generated_actuation[k] = v.tolist()

    for k,v in generated_morph.items():
        generated_morph[k] = v.tolist()

    mass_morph = np.array(generated_morph["mass"]).reshape(64, -1)
    mass_mask = mass_morph > 0.1
    act_morph = np.array(generated_actuation["amplitude"]).reshape(64, -1) * mass_morph > 1e-6

    design = np.zeros((64, 44))
    design[mass_mask] = 1
    design[act_morph] = 2
    design = design.T[::-1,:].copy()
    xx, yy = np.meshgrid(np.arange(design.shape[1]), np.arange(design.shape[0]))
    # particle_type_locations = list(zip(design.flatten(), xx.flatten(), yy.flatten()))

    robot_dict = dict()
    robot_dict["genome"] = simModel.get_export_info()
    robot_dict["design_raw"] = dict(**generated_actuation, **generated_morph)
    robot_dict["design_points"] = {"particle_type":design.flatten().tolist(),
                                    "x":xx.flatten().tolist(),
                                    "y":yy.flatten().tolist(),
                                    "particle_type_map": {'no_material':0,
                                                            'passive':1,
                                                            'active':2}
                                    }

    robot_dict["design_image"] = design.flatten().tolist()
    robot_dict["git_hash"] = validate_git_repo_and_get_sha()
    robot_dict["cli_args"] = " ".join(sys.argv)
    json.dump(robot_dict, open(f"model_tests/data/sim2real_design_seed_{seed}_iteration_{iteration}.json", "w"))

def run_all_tests():
    import warnings
    warnings.filterwarnings("ignore")

    import sys
    # print(validate_git_repo_and_get_sha())
    print( " ".join(sys.argv))

    from utils import get_torch_device
    from models.model_loaders import SimulationModel
    import robodiff_startup as didv
    import torch 

    device =  torch.device("cpu") #get_torch_device()
    scene = didv.scene

    print(f"Particle Count: {scene.n_particles} (terrain: {scene.n_terrain_particles}, robot: {scene.n_particles - scene.n_terrain_particles})")

    def run_single_test(simDict):
        import pickle
        import base64
        results = []
        results_resim = []
        def handle(res, resim=False):
            if resim:
                results_resim.append(res)
            else:
                results.append(res)

        simDict["scene"] = scene
        simDict["device"] = device
        simModel = SimulationModel(**simDict)
        print(f"{simDict['description_str']}")
        # didv.args.gui = None
        final_results = didv.run_optimization(simModel,
                        results_handler=lambda x, **kwargs: handle(x, **kwargs), # lambda x: None
                        gui_handler=None)
        print(f"{simDict['description_str']} Time: {final_results['time']:8.5f}s Loss: {final_results['loss']:8.5f}")
        if didv.args.verbose > 2:
            pprint.pprint(simModel.get_genome())
        
        del simDict['scene']
        del simDict['device']
        simDict['requires_taichi_grads'] = False
        if didv.args.figure_export:
            fname = simModel.get_kwarg('fig_export_name')
            if fname is None:
                fname_base = simModel.get_kwarg("fig_export_base_name")
                if fname_base is not None:
                    fname = f"{fname_base}.pkl"

            if fname is None:
                from datetime import datetime
                t = datetime.now()
                fname = f"figure_export_{t.year}_{t.month}_{t.day}_results.pkl"
            else:
                fname = f"figure_export_results_{fname}"
            if didv.args.terrain:
                fname = f"terrain_mode_{didv.args.terrain_mode}_{fname}"
            pickle.dump((simDict, results, results_resim), open(f"{FIGURE_DATA_DIR}/{fname}", "wb"))

        if didv.args.gui:
            didv.args.gui = True    

            # grads are STILL needed for animation.
            simModel.requires_taichi_grads = True

            simModel.steps = didv.args.max_steps
            export_fname = f"20230918_carry_object.pkl"
            simModel.kwargs["fig_export_name"] = export_fname
            didv.simulate(simModel, enable_gui=True, title=simModel.kwargs['description_str'], epoch_idx=None)
    
    def run_playback():
        print("hello world")
        import json
        fname = didv.args.playback
        sims = json.load(open(fname, "r"))
        sims_of_interest_fig4 = [6,7, 39, 56, 84, 85]
        # sims_of_interest = sims_of_interest_fig4
        
        sims_of_interest = list(range(1, len(sims)+1))
        for simOfInterestIdx in sims_of_interest:
            simDict = sims[simOfInterestIdx - 1]
            simDict["scene"] = scene
            simDict["device"] = device
            simModel = SimulationModel(**simDict)
            if didv.args.playback_old_actuator_size:
                simModel.actuationModel._actuator_size *= math.sqrt(0.1)

            print(f"idx: {simOfInterestIdx}  {simDict['description_str']}")
            print(simModel.__dict__)

            simModel.requires_taichi_grads = False

            simModel.steps = didv.args.max_steps
            simModel.epochs = 1
            simModel.requires_taichi_grads = False
            didv.simulate(simModel, enable_gui=True, title=f"idx: {simOfInterestIdx} {simModel.kwargs['description_str']}")
    
 
    if didv.args.playback != "":
        run_playback()
    else:
        mapping_dict = dict(base=base_experiments_first,
                            direct_particle=direct_particle_first,
                            void_interpolation=void_interpolation_first,
                            constrained_actuation=constrained_actuation_first,
                            design_starting_mask=masked_starting_points_first,
                            object_manipulation=object_manipulation_first,
                            )
        for local_selection in didv.args.local_selections:
            for simModelDict in mapping_dict[local_selection]():
                run_single_test(simModelDict)

    
