#parent class for all optimizers

class Optimizer:
  def __init__(self,
               simModel,
               simulate): 
  
    self._simModel = simModel
    self._simulate = simulate

  def _opt_step(self, idx):
    raise NotImplementedError
    
   
  def optimize(self):
    raise NotImplementedError
    
