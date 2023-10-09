import IPython
import numpy as np
from numpy.lib.arraysetops import isin
import pprint
import pathlib
import h5py

import matplotlib.colors
from PIL import Image
from io import BytesIO
import base64


from models import *
from models.base_models import BaseActuationModel, BaseMorphModel, BaseUnifiedModel

from models.direct_circle_actuator_model import DirectCircleActuatorModel
from models.direct_circle_morph_model import DirectCircleMorphModel
from models.mixed_sine_actuator_model import MixedSineActuatorModel, DefaultMixedSineActuatorModel
from models.direct_particle_morph_model import DirectParticleMorphModel, DefaultParticleMorphModel
from models.sim2real_model import UnifiedSim2RealModel

ModelMapper = {
    "MixedSineActuatorModel" : MixedSineActuatorModel,
    "DirectCircleActuatorModel" : DirectCircleActuatorModel,
    "DefaultMixedSineActuatorModel": DefaultMixedSineActuatorModel,
    "DirectCircleMorphModel" : DirectCircleMorphModel,
    "DirectParticleMorphModel" : DirectParticleMorphModel,
    "DefaultParticleMorphModel": DefaultParticleMorphModel,
    "UnifiedSim2RealModel":UnifiedSim2RealModel
}


class SimulationModel(object):
    def __init__(self, 
                    scene=None,
                    steps=1024,
                    pre_grads_steps=0,
                    internalDamping=30.0,
                    globalDamping=2.0,
                    enableDamping=1,
                    baseE = 20.0,
                    friction=0.5,
                    gravityStrength=3.8,
                    actuationStrength=4.0,
                    actuationSharpness=1.0,
                    actuationMaxSignal=100000.0,
                    act_model="y",
                    actuationProportionalToMass=1,
                    epochs=1,
                    unifiedModel=None,
                    morphModel=None,
                    actuationModel=None,
                    loss_mode="locomotion_flat",
                    loss_mode_includes="body",
                    work_guid = None,
                    save_sim_for_vis = None,
                    requires_taichi_grads = True,
                    simulation_inclusion_threshold=0.1,
                    **kwargs):
        
        self.scene = scene
        self.steps = steps
        self.pre_grads_steps = pre_grads_steps
        self.internalDamping = internalDamping
        self.globalDamping = globalDamping
        self.enableDamping = enableDamping
        self.baseE = baseE
        self.friction = friction
        self.gravityStrength = gravityStrength
        self.actuationStrength = actuationStrength
        self.actuationSharpness = actuationSharpness
        self.actuationMaxSignal = actuationMaxSignal
        self.act_model = act_model
        self.actuationProportionalToMass = actuationProportionalToMass
        self.requires_taichi_grads = requires_taichi_grads
        self.simulation_inclusion_threshold = simulation_inclusion_threshold
        self.loss_mode = loss_mode
        self.loss_mode_includes = loss_mode_includes
        self.epochs = epochs

        self.unifiedModel = unifiedModel
        self.morphModel = morphModel
        self.actuationModel = actuationModel

        self.work_guid = work_guid
        self.save_sim_for_vis = save_sim_for_vis

        self.kwargs = kwargs

        self.refreshModels()
  
    def get_kwarg(self, k):
        if k in self.__dict__:
            return self.__dict__[k]
        if k in self.kwargs:
            return self.kwargs[k]
        return None

    def get_title(self):
        if "description_str" in self.kwargs:
            return self.kwargs["description_str"]
        else:
            return ""
            
    def get_summary_string(self):
        self_desc = "" 
        if "description_str" in self.kwargs:
            self_desc = f"{self.kwargs['description_str']}\n"
            
        if self.unifiedModel is not None:
            return f"{self_desc}\nunified: {self.unifiedModel.get_summary_string()}"
        else:
            return f"{self_desc}\nmorph: {self.morphModel.get_summary_string()}\nact: {self.actuationModel.get_summary_string()}"

    def zero_grad(self):
        if self.unifiedModel is not None:
            self.unifiedModel.zero_grad()
        else:
            self.morphModel.zero_grad()
            self.actuationModel.zero_grad()


    def step(self):
        if self.unifiedModel is not None:
            self.unifiedModel.step()
        else:
            self.morphModel.step()
            self.actuationModel.step()

    def get_export_info(self, **kwargs):
        info = {**kwargs}
        if self.unifiedModel is not None:
            info["unifiedGenome"] = self.unifiedModel.export_genome_b64(grads=True)
        else:
            info["actuationGenome"] = self.actuationModel.export_genome_b64(grads=True)
            info["morphGenome"] = self.morphModel.export_genome_b64(grads=True)

        return info

    def get_genome(self, sanitize=False):
        if self.unifiedModel is not None:            
            return {
                "unifiedGenome":self.unifiedModel.get_genome(sanitize=sanitize),
            }
        else:
            return {
                "actuationGenome":self.actuationModel.get_genome(sanitize=sanitize),
                "morphGenome":self.morphModel.get_genome(sanitize=sanitize),
            }

        
    def get_scene(self):
        return self.scene

    def get_epoch_count(self):
        return self.epochs

    def try_save(self, simulation_details, title="default_group"):
        if self.save_sim_for_vis and self.work_guid is not None:
            guid = self.work_guid.replace("-", "")
            fname = f"sim_results/{guid[:2]}/{guid[2:4]}/{guid[4:6]}/{guid[6:]}.hdf5"
            path = pathlib.Path(fname)
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "a") as f:
                group = f.create_group(title)
                for key, value in simulation_details.items():
                    if not isinstance(value, np.ndarray):
                        value = np.array(value)
                    if value.shape != ():
                        group.create_dataset(key, data=value, compression="gzip")
                    else:
                        group.create_dataset(key, data=value)

        
    def refreshModels(self):
        """
        Checks if the morphModel  & or the actuationModel are fully setup or not, and parses them if needed.
        """
        # IPython.embed()
        unifiedModel = self.unifiedModel
        morphModel = self.morphModel
        actuationModel = self.actuationModel

        if (isinstance(unifiedModel, dict)):
            unifiedModelFactory = ModelMapper[unifiedModel["name"]]
            try:
                self.unifiedModel = unifiedModelFactory(**unifiedModel, **self.kwargs, scene=self.scene)
            except Exception as e:
                print(e)
                IPython.embed()

        if (isinstance(morphModel, dict)):
            morphModelFactory = ModelMapper[morphModel["name"]]
            try:
                self.morphModel = morphModelFactory(**morphModel, **self.kwargs, scene=self.scene)
            except Exception as e:
                print(e)
                IPython.embed()


        if (isinstance(actuationModel, dict)):
            actuationModelFactory = ModelMapper[actuationModel["name"]]
            try:
                self.actuationModel = actuationModelFactory(**actuationModel, **self.kwargs, scene=self.scene)
            except Exception as e:
                print(e)
                IPython.embed()

        unifiedModelExists = isinstance(self.unifiedModel, BaseUnifiedModel)
        morphModelExists = isinstance(self.actuationModel, BaseActuationModel)
        actModelExists = isinstance(self.morphModel, BaseMorphModel)

        # check that we have a valid model
        assert unifiedModelExists or (morphModelExists and actModelExists), "must use either unified model or both act and morph models."
        
        # check that we do not have any extra models.
        assert (self.morphModel is None) == (self.actuationModel is None), "if morph model is used, act model must also be used"
        assert (self.unifiedModel is None) != (self.morphModel is None), "if unified model is used, morphModel should not be used"
        