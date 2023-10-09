import numpy as np

def get_model_dict(morphModel,
                    actModel,
                    loss_mode="locomote_flat", 
                    steps=2048,
                    internalDamping=30.0,
                    globalDamping=2.0,
                    baseE=20.0,
                    actuatorStrength=4.0,
                    epochs=1,
                    actuationProportionalToMass=1,
                    **kwargs):
    return {
            "steps":steps,
            "internalDamping":internalDamping,
            "globalDamping":globalDamping,
            "baseE":baseE,
            "actuationStrength":actuatorStrength,
            "loss_mode":loss_mode,
            "epochs":epochs,
            "morphModel": morphModel,
            "actuationModel": actModel,
            "actuationProportionalToMass":actuationProportionalToMass,
            **kwargs
        }

def get_adam_optimizer(lr=0.01):
    return {'name':"Adam", 'lr':lr}

def get_morph_rand_direct_circle_fixed_size(hole_count=1, optimize=True, holeSize=0.2,target_hole_area=None, max_hole_size=None):
    dict_to_return = get_morph_rand_direct_circle(hole_count=hole_count, optimize=optimize,target_hole_area=target_hole_area,max_hole_size=max_hole_size)
    del dict_to_return["genome"]["hole_size"]
    dict_to_return["holeSize"] = holeSize
    return dict_to_return

def get_morph_rand_direct_circle(hole_count=1, optimize=True, holeSize=0.2,target_hole_area=None, max_hole_size=None):
    return get_morph_direct_circle(
        hole_x=np.random.random(hole_count).tolist(),
        hole_y=(np.random.random(hole_count)*0.7).tolist(),
        optimize=optimize,
        hole_sizes=[holeSize],
        target_hole_area=target_hole_area,
        max_hole_size=max_hole_size)

def compute_mean_hole_size(hole_fraction, aspect_ratio, hole_count):
    return np.sqrt((hole_fraction * aspect_ratio)/(hole_count * np.pi))

def get_morph_rand_direct_circle_multi_sized_holes(hole_count=1,
                                optimize=True, hole_fraction=0.6, robot_aspect_ratio = 0.7,
                                fixed_hole_sizes=False,
                                shared_hole_sizes=False,
                                target_hole_area=None,
                                max_hole_size=None,
                                fringe_softness=2.0,
                                power_law_hole_sizes=False):
    # hole_fraction = 0.6, below giant connected conponent threshold on a square lattice. picked somewhat arbitrarly.
    mean_hole_size = compute_mean_hole_size(hole_fraction, robot_aspect_ratio, hole_count)
    hole_x = np.random.random(hole_count).tolist()
    hole_y = (np.random.random(hole_count)*robot_aspect_ratio).tolist()
    hole_sizes = [mean_hole_size] if shared_hole_sizes else np.random.normal(mean_hole_size, mean_hole_size**2, size=(hole_count)).tolist()
    if power_law_hole_sizes:
        sizes = 0.25 / np.power(2, (np.ceil(np.log2(np.arange(1,hole_count+1)))))
        total_area = np.power(sizes, 2).sum() * np.pi / (robot_aspect_ratio * hole_fraction)
        hole_sizes =  sizes/np.sqrt(total_area)


    morphDict =  get_morph_direct_circle(
        hole_x=hole_x,
        hole_y=hole_y,
        hole_sizes=hole_sizes,
        optimize=optimize,
        target_hole_area=target_hole_area,
        max_hole_size=max_hole_size,
        fringe_softness=fringe_softness
    )

    if fixed_hole_sizes:
        todel =  morphDict["genome"]
        del todel["hole_size"]
        morphDict["holeSize"] = mean_hole_size
    return morphDict


def get_morph_direct_circle(hole_x=[0.124],
                            hole_y=[0.4874],
                            offset_x=0.0,
                            offset_y=0.0,
                            width=1.0,
                            height=1.0,
                            hole_sizes=[0.2],
                            optimize=True,
                            target_hole_area=None,
                            max_hole_size=None,
                            **kwargs):
    morph_dict =  {
                "name":"DirectCircleMorphModel",
                "target_hole_area":target_hole_area,
                "max_hole_size":max_hole_size,
                "genome": {
                    "hole_x":[(d - offset_x)/width for d in hole_x],
                    "hole_y":[(d - offset_y)/height for d in hole_y],
                    "hole_size":hole_sizes
                },
                **kwargs}
    if optimize:
        morph_dict["optimizer"] = get_adam_optimizer()
    return morph_dict

def get_morph_direct_particle(mode="mass", mass=[1.0]*64*44, x=[[0.0,0.0]]*64*44):
    return {
        'name':'DirectParticleMorphModel',
        'mode':mode,
        'genome':{
            'mass':mass,
            'x':x
        },
        'optimizer': get_adam_optimizer()
    }

# standard rectangle robot without any holes. No shape optimization will occur.
def get_morph_default_particle():
    return {
        "name":"DefaultParticleMorphModel"
    }
    
def get_act_default_mixed_sine(optimize=True):
    act_dict =  { "name":"DefaultMixedSineActuatorModel"}
    if optimize:
        act_dict["optimizer"] = get_adam_optimizer()
    return act_dict

def get_act_rand_direct_circle_sine_only(hole_count=1, optimize=True, actuation_omega=40.0):
    return get_act_direct_circle(
                        hole_x_sine=np.random.random(hole_count).tolist(),
                        hole_y_sine=(np.random.random(hole_count)*0.7).tolist(),
                        hole_x_cosine=[1000 for _ in range(hole_count)],
                        hole_y_cosine=[1000 for _ in range(hole_count)],
                        optimize=optimize,
                        actuation_omega=[actuation_omega],
                        actuator_size=0.2 * np.sqrt(0.1))

def get_act_rand_direct_circle(hole_count=1, optimize=True, actuation_omega=40.0, robot_aspect_ratio=0.7):
    return get_act_direct_circle(
                        hole_x_sine=np.random.random(hole_count).tolist(),
                        hole_y_sine=(np.random.random(hole_count)*robot_aspect_ratio).tolist(),
                        hole_x_cosine=np.random.random(hole_count).tolist(),
                        hole_y_cosine=(np.random.random(hole_count)*robot_aspect_ratio).tolist(),
                        optimize=optimize,
                        actuation_omega=[actuation_omega])


def get_act_direct_circle(hole_x_sine=[0.1],
                          hole_y_sine=[0.1],
                          hole_x_cosine=[1.0],
                          hole_y_cosine=[1.0],
                          actuation_omega=[40.0],
                          optimize=True,
                          actuator_size=0.2 *  np.sqrt(0.1)):
    act_dict =  {
                "name":"DirectCircleActuatorModel",
                "actuator_size":actuator_size,
                "genome": {
                    "hole_x_sine":hole_x_sine,
                    "hole_y_sine":hole_y_sine,
                    "hole_x_cosine":hole_x_cosine,
                    "hole_y_cosine":hole_y_cosine,
                    "actuation_omega":actuation_omega
                }
            }
    if optimize:
        act_dict["optimizer"] = get_adam_optimizer()
    return act_dict


