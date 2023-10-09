import numpy as np

from visualizer_gl_taichi_free import visualize

STEPS = 1000

N=8
pos = (np.random.random((STEPS, 1<<N, 2)) - 0.5)
pos[1:] *= 0.01
pos = np.cumsum(pos, axis=0)

gds_pos = (np.random.random((STEPS, 1<<N, 2)) - 0.5)
gds_pos[1:] *= 0.00
gds_pos = np.cumsum(gds_pos, axis=0)

gds_dt = (np.random.random((STEPS, 1<<N, 2)) - 0.5) *0.01
gds_dt[1:] *= 0.1
gds_dt = np.cumsum(gds_dt, axis=0)

visualize(pos = pos,
            enable_ticks_and_text=True,
            arrow_centers=pos,
            arrow_dts=gds_dt)

#%%
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "..")

from models.direct_circle_morph_model import DirectCircleMorphModel

from models.utils import MockScene
from model_helpers import get_morph_direct_circle

#%%

# %%
hole_x = 0.5
hole_y = 0.35
hole_size = 0.2
morph_model = get_morph_direct_circle([hole_x], [hole_y], holeSize=hole_size)
morph_model["morphMask"] = None
dcmm = DirectCircleMorphModel(scene=MockScene(), **morph_model)
# %%
plt.imshow(dcmm.generate()['mass'].reshape(64, 44).T)
plt.show()
# %%
np.arange(64*44)[dcmm.generate()['mass'] < 1]
# %%
