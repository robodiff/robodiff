from glob import glob

import numpy as np
import h5py
import argparse


from  visualization.utils import run_visualization

parser = argparse.ArgumentParser()
parser.add_argument('--guid', required=True)
args = parser.parse_args()


work_guid = args.guid

work_guid = work_guid.replace("-", "")

f_name = f"sim_results/{work_guid[:2]}/{work_guid[2:4]}/{work_guid[4:6]}/{work_guid[6:]}.hdf5"
with h5py.File(f_name, 'r') as f:
    m = f['epoch_0']['m'][()]
    x = f['epoch_0']['x'][()]
    act = f['epoch_0']['actuation'][()]
    run_visualization(x, m, act, title="")