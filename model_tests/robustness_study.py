#%%
import numpy as np
import math
import uuid
import pprint
from tqdm import tqdm

import sys
sys.path.insert(0, "..")
from config import FIGURE_DATA_DIR
from model_helpers import *

import IPython
import itertools
import random
#%%

def cartesian_product_helper(**kwargs):
    k,v = zip(*[(k,v) for (k,v) in  kwargs.items()])
    for d in itertools.product(*v):
        yield dict(zip(k, d))

def get_sims(sim_factory, *args, N=100,
                                K=10,
                                iterations_per_K=1,
                                job_title="default title",
                                **kwargs):
    # N is the number of trials (or seeds)
    # K // K => optimize
    # K // 1 => random search
    # => K // iterations_per_K indicates if random search is being used.
    # K => number of simulations per trial
    # iterations_per_K is the number of initial random robots to generate per trial.
    # iterations_per_K is the number of design samples to generate per trial of K simulations.
    # compute number of sims that we need to generate.

    restarts = N * K // iterations_per_K
    kwargs["epochs"] = iterations_per_K
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
# Above is duplicated from test_models.py don't copy back
#%%
def normal(mean, spread, size):
    return np.random.normal(mean, spread, size=size)

def uniform(mean, spread, size):
    return np.random.uniform(mean-spread, mean+spread, size=size)

def constant(mean, spread, size):
    return np.ones(size) * mean

def spread_lambda_mean(x):
    return x
def spread_lambda_mean_sq(x):
    return x**2

function_dictionary = {
    "normal": normal,
    "uniform": uniform,
    "spread_lambda_mean":spread_lambda_mean,
    "spread_lambda_mean_sq":spread_lambda_mean_sq
}

robot_aspect_ratio = 0.7

def get_robustness_morph_model(n_voids, void_patch_fraction, radius_distribution, spread_lambda, morph_fringe_softness_power):
    mean = compute_mean_hole_size(void_patch_fraction, robot_aspect_ratio, n_voids)
    spread = spread_lambda(mean)

    hole_x = np.random.random(n_voids).tolist()
    hole_y = (np.random.random(n_voids)*robot_aspect_ratio).tolist()
    hole_sizes = radius_distribution(mean, spread, size=(n_voids)).tolist()

    morphDict =  get_morph_direct_circle(
        hole_x=hole_x,
        hole_y=hole_y,
        hole_sizes=hole_sizes,
        optimize=True,
        target_hole_area=None,
        max_hole_size=None,
        fringe_softness=morph_fringe_softness_power
    )

    return morphDict

def get_robustness_act_model(n_acts, act_patch_fraction, radius_distribution, spread_lambda, optimize_act_radius, act_morph_fringe_softness):
    mean = compute_mean_hole_size(act_patch_fraction, robot_aspect_ratio, n_acts)
    spread = spread_lambda(mean)
    # np.sqrt(0.1) \approx 0.3.
    # was originally set so that the size of the actuator was measured in 


    actuation_omega=40.0/(2*np.pi)
    act_dict =  {
            "name":"DirectCircleActuatorModel",
            "genome": {
                "hole_x_sine":np.random.random(n_acts).tolist(),
                "hole_y_sine":(np.random.random(n_acts)*robot_aspect_ratio).tolist(),
                "hole_x_cosine":[1000 for _ in range(n_acts)],
                "hole_y_cosine":[1000 for _ in range(n_acts)],
                "actuation_omega":[actuation_omega],
            },
            "optimizer":get_adam_optimizer(),
            "act_morph_fringe_softness":act_morph_fringe_softness,
        }
    act_sizes = (radius_distribution(mean, spread, size=(n_acts))).tolist()
    
    if optimize_act_radius:
        act_dict["genome"]["act_size"] = act_sizes
    else:
        act_dict["actuator_sizes"] = act_sizes

    return act_dict

def get_sim(epochs=None,
            morph_fringe_softness_power=None,
            act_morph_fringe_softness=None,
            n_voids=None,
            n_acts=None,
            optimize_act_radius=None,
            simulation_inclusion_threshold=None,
            void_patch_fraction=None,
            act_patch_fraction=None,
            initial_morph_patch_radius_distribution=None,
            initial_act_patch_radius_distribution=None,
            spread_lambda=None):
    assert epochs is not None
    assert morph_fringe_softness_power is not None
    assert act_morph_fringe_softness is not None
    assert n_voids is not None
    assert n_acts is not None
    assert optimize_act_radius is not None
    assert simulation_inclusion_threshold is not None
    assert void_patch_fraction is not None
    assert act_patch_fraction is not None
    assert initial_morph_patch_radius_distribution is not None
    assert initial_act_patch_radius_distribution is not None
    assert spread_lambda is not None

    initial_act_patch_radius_distribution = function_dictionary[initial_act_patch_radius_distribution]
    initial_morph_patch_radius_distribution = function_dictionary[initial_morph_patch_radius_distribution]
    spread_lambda = function_dictionary[spread_lambda]

    morph_model = get_robustness_morph_model(n_voids, void_patch_fraction, initial_morph_patch_radius_distribution, spread_lambda, morph_fringe_softness_power)
    act_model = get_robustness_act_model(n_acts, act_patch_fraction, initial_act_patch_radius_distribution, spread_lambda, optimize_act_radius, act_morph_fringe_softness)
    return get_model_dict(morph_model, act_model,
                            description_str=f"robustness experiments",
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
                            simulation_inclusion_threshold=simulation_inclusion_threshold,
                          )


def hyper_params_of_experiments(n_trials=1, seed=1):
    sims_per_trial = 10
    void_counts = [ 64, 8, 16, 32, 128, 256]
    np.random.seed(seed)
    random.seed(seed)

    original_act_coverage = (np.pi * 64 * 0.2**2 * 0.1)/robot_aspect_ratio

    return cartesian_product_helper(N = [n_trials], # n seeds
                                           K = [sims_per_trial], # simulations per trial
                                            iterations_per_K = [sims_per_trial, 1], # 10 -> normal gradient descent, 1-> random search baseline. 
                                            morph_fringe_softness_power = [2, 1, 3], # quadratic, linear, cubic
                                            act_morph_fringe_softness=["quadratic", "cubic", "linear"],
                                            n_voids=void_counts, 
                                            n_acts=void_counts,
                                            optimize_act_radius=[False], # Currently, not optimizing actautor radii
                                            simulation_inclusion_threshold=[0.1, 0.2, 0.4],
                                            void_patch_fraction=[0.6, 0.3], 
                                            act_patch_fraction=[original_act_coverage, original_act_coverage/2],
                                            initial_morph_patch_radius_distribution=["normal", "uniform"], 
                                            initial_act_patch_radius_distribution=["uniform", "normal"], 
                                            spread_lambda=[  "spread_lambda_mean_sq", "spread_lambda_mean",]
                                            )
    
def experiments(n_trials=1, seed=1):
    jobs = []
    for n, kwargs in enumerate(hyper_params_of_experiments(n_trials=n_trials, seed=seed)):
        jobs.append(get_sims(get_sim, **kwargs))
    return jobs