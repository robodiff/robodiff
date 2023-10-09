from optimizers.optimizer import Optimizer
from tqdm import tqdm
import numpy as np
import torch
from models.utils import sanitize_num, b64_encode
import time
import IPython
import robodiff_startup as didv
class GDOptimizer(Optimizer):
    
    def __init__(self,
                    simModel,
                    simulate,
                    gui_handler = None,
                    results_handler = None):
        super().__init__(simModel, simulate)
        self._scene = simModel.get_scene()
        self._gui_handler = gui_handler
        self._results_handler = results_handler
        self._ROBOT_ASPECT_RATIO = self._scene.robot_aspect_ratio

        self._epochs = self._simModel.get_epoch_count()
        self._results_dict = None
        self._t0 = time.time()
        
    def _opt_step(self, idx, disable_optimization):
        # save the genome corresponding to the loss that we are about to get...
        self._simModel.zero_grad()

        results_dict = self._simulate(title=self._simModel.get_title(),
                                        enable_gui=self._epochs == idx +1,
                                        epoch_idx = idx)
        self._results_dict = results_dict
        self._results_dict["time"] = time.time() - self._t0


        # handle sending results
        # Before taking an optimization step!   

        x_avg_b64 = b64_encode(results_dict["x_avg"])
        simModelExportInfo = self._simModel.get_export_info(epoch_idx=idx,
                loss=sanitize_num(results_dict['loss']),
                x_avg=x_avg_b64) 
        
        simModelGenomes = self._simModel.get_genome() 

        if self._results_handler:
            self._results_handler(simModelExportInfo)
            if didv.args.resim:

                origGradRequirements =  self._simModel.requires_taichi_grads
                origStepSize = self._simModel.steps
                self._simModel.requires_taichi_grads = False
                self._simModel.steps = didv.args.max_steps

                resim_results = self._simulate(title=self._simModel.get_title(), 
                                                    enable_gui=False,
                                                    epoch_idx = idx)
                self._simModel.requires_taichi_grads = origGradRequirements
                self._simModel.steps = origStepSize


                simModelExportInfo = self._simModel.get_export_info(epoch_idx=idx,
                    loss=sanitize_num(resim_results['loss']),
                    x_avg=resim_results["x_avg"]) 
                self._results_handler(simModelExportInfo, resim=True)

    
        self._simModel.try_save(self._results_dict, title=f"epoch_{idx}")
        
        msg = f"idx: {idx+1:4d}/{self._epochs:4d} l: { results_dict['loss'] :10.3E}" 
        if "unifiedGenome" in simModelGenomes:
            msg += f"\nunified: {simModelGenomes['unifiedGenome']}"
        else:
            msg += f"\nmorph: {simModelGenomes['morphGenome']}"
            msg += f"\nact: {simModelGenomes['actuationGenome']}"


        if not disable_optimization:
                self._simModel.step()

        return msg 
    
    def optimize(self, disable_optimization):
        self._scene.reset()

        tq = range(self._epochs)
        
        for idx in tq:

            # optim step will save results if applicable.
            msg = self._opt_step(idx, disable_optimization)
            
            if self._results_handler is None:
                print(msg, flush=True)

            if self._gui_handler and idx == self._epochs - 1:
                self._gui_handler()
        
        return self._results_dict