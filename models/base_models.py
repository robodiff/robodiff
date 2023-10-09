import torch
import numpy as np
import IPython
import pprint
import pickle
import base64

from .utils import sanitize_genome

class BaseModel(object):
    """
    Genome is a dictionary of all inputs needed to generate the model.
    Optimizer is None if no optimization will occur, otherwise of type torch.optim
    """
    def __init__(self,
                    genome=None,
                    optimizer=None,
                    optimizerGenome=None,
                    device=None,
                    gui=None,
                    always_include_genome=False,
                    **kwargs
                    ):
        assert optimizer is None or isinstance(optimizer, torch.optim.Optimizer) or isinstance(optimizer, dict), "Optimizer must be either None or a torch.optim.Optimizer or a dictionary which will be auto-expanded into an optimizer"
        self._genome = genome 
        self._optimizer = optimizer
        self._optimizerGenome = optimizerGenome

        self._always_include_genome = always_include_genome

        self.validateOptimizerGenome()


        self.device=device
        self.gui = gui

        self._grads_dict = None

        self.kwargs = kwargs
        self.refresh_genome()
        self.refresh_optimizer()

    def get_summary_string(self, grads=True, truncate=0, name="Base Model"):
        return  pprint.pformat(self.get_genome(grads=grads, truncate=truncate, force_include_genome=True))

    def validateOptimizerGenome(self):
        optimizerGenomeType = type(self._optimizerGenome) 
        assert optimizerGenomeType in [type(None), list, torch.Tensor], "optimizerGenome must be of type NoneType, list, or torch.Tensor"

        if optimizerGenomeType == list:
            for element in self._optimizerGenome:
                assert isinstance(element, torch.Tensor), "if optimizerGenome is a list, it must only contain objects of type torch.Tensor"

    def refresh_genome(self):
        arrays_in_genome = {}

        # convert non-torch objects to python lists of floats.
        for k, v in self._genome.items():
            if not isinstance(v, torch.Tensor):
                # check if genome object is likely to be encoded as base64 and attempt to decode it.
                if isinstance(v, str):
                    arrays_in_genome[k] = pickle.loads(base64.urlsafe_b64decode(v))
                # else: try to decode as a python list or an numpy array.
                else:
                    arrays_in_genome[k] = np.array(v, dtype=np.float32).tolist() # set string NaN and Inf to float values. Torch.tensor does not like strings.
 

        # some error checking
        if len(arrays_in_genome) != 0 and self._optimizerGenome is not None:
            raise ValueError("optimizerGenome can not be set if genome contains any elements that are not torch.Tensor objects")

        # convert python lists to torch objects.
        for k,v in arrays_in_genome.items():
            try:
                self._genome[k] = torch.tensor(v, device=self.device, requires_grad=True)
            except:
                self._genome[k] = torch.tensor(v, device=self.device)


    def refresh_optimizer(self):
        if isinstance(self._optimizer, dict):
            optimizerName = self._optimizer["name"]
            optimizerGenome = self._optimizerGenome if self._optimizerGenome is not None else list(self._genome.values())
            optimizerLR = self._optimizer["lr"]
            if (optimizerName == "Adam"):
                self._optimizer = torch.optim.Adam(optimizerGenome, lr=optimizerLR)
            else:
                raise NotImplementedError("Base Model does not support generating an optimizer of type other than Adam")


    def generate(self):
        raise NotImplementedError

    def backward(self, **kwargs):
        self._grads_dict = kwargs

    def validate_grads(self, key, value):

        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(self.device)

        elif isinstance(value, torch.Tensor):
            return value

        else:
            raise ValueError(f"{key} must be of type numpy.ndarray or torch.tensor. It is: {type(value)}")


    """
    Zero out optimizer grads if applicable.
    """
    def zero_grad(self):
        if self._optimizer is not None:
            self._optimizer.zero_grad()

    """
    Perform an optimization step if applicable.
    """
    def step(self):
        if self._optimizer is not None:
            self._optimizer.step()

    """
    Return dictionary of data needed to generate a robot with this model.
    """
    def get_genome(self, grads=True, truncate=0, force_include_genome=False, sanitize=False):
        
        dict_to_return = dict()
        if self._optimizer is None and not self._always_include_genome and not force_include_genome:
            dict_to_return["Msg"] = "No optimizer, omitting genome"
            return dict_to_return

        for k, v in self._genome.items():
                
            if isinstance(v, torch.Tensor):
                truncate = len(v) if truncate == 0 else truncate
                dict_to_return[k] = v.detach().cpu().numpy().tolist()[:truncate]
                if (v.grad is not None and grads):
                    dict_to_return[f"{k}_grad"] = v.grad.detach().cpu().numpy().tolist()[:truncate]
            elif isinstance(v, np.ndarray):
                truncate = len(v) if truncate == 0 else truncate
                dict_to_return[k] = v.tolist()[:truncate]
            else:
                dict_to_return[k] = v
        if sanitize:
            dict_to_return = sanitize_genome(dict_to_return)
        return dict_to_return
    
    def export_genome_b64(self, grads=True, sim_grads=True):
        dict_to_return = dict()

        items_to_save =  list(self._genome.items())
        if sim_grads:
            items_to_save += [(f"grad_from_sim_{k}",v) for k,v in self._grads_dict.items()]
        for k, v in items_to_save:
            val = None
            curr_grads = None
            if isinstance(v, torch.Tensor):
                val = v.detach().cpu().numpy()
                if (v.grad is not None and grads):
                    curr_grads = v.grad.detach().cpu().numpy()
            elif isinstance(v, np.ndarray):
                val = v
                
            dict_to_return[k] = base64.urlsafe_b64encode(pickle.dumps(val)).decode("utf-8")
            if curr_grads is not None:
                dict_to_return[f"{k}_grad"] = base64.urlsafe_b64encode(pickle.dumps(curr_grads)).decode("utf-8")

        return dict_to_return



class BaseUnifiedModel(BaseModel):
    def generate(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
    

class BaseMorphModel(BaseModel):
    """
    Return dictionary with key of 'mass' and value of mass of each particle.
    """
    def generate(self):
        raise NotImplementedError
    
    """
    Take gradients of mass and backprop gradients if appropiate
    """
    def backward(self, **kwargs):
        super().backward(**kwargs)
        assert "mass" in kwargs, "BaseMorphModel must be passed 'mass' as a keyword argument to the backward method"
        assert kwargs['mass'] is not None


class BaseActuationModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_tensors = {
            'frequency':None,
            'amplitude':None,
            'phase':None,
            'bias':None
        }
 
    """
    Return dictionary with keys:
    'frequency'
    'amplitude'
    'phase'
    'bias'
    for each particle.
    """
    def generate(self):
        for k, v in self.output_tensors.items():
            assert v != None, f"Derived class must set {k} to value other than None"
            
        return dict((k, v.detach().cpu().numpy()) for k,v in self.output_tensors.items())
    
    """
    Take gradients for each of:
    frequency,
    amplitude,
    phase,
    bias
    and perform back propagation if appropate.
    """
    def backward(self, **kwargs):
        super().backward(**kwargs)

        keys_of_interest = ["frequency", "amplitude", "phase", "bias"]
        for k in keys_of_interest:
            assert k in kwargs, f"BaseActuationModel must be passed '{k}' as a keyword argument to the backward method"
        
        output_tensors_filtered = dict((k,v) for k,v in self.output_tensors.items() if k in keys_of_interest)
        cat_output_tensors = torch.cat([v for k,v in sorted(output_tensors_filtered.items())])
        cat_grads = torch.cat([self.validate_grads(k, v) for k,v in sorted(kwargs.items())])
        # IPython.embed()
        cat_output_tensors.backward(cat_grads)

        # self.output_tensors[k].backward(self.validate_grads(k, kwargs[k]))

