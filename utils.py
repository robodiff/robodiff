from subprocess import check_output
import torch
import os
import matplotlib.pyplot as plt
import numpy as np


def is_in_cluster():
    return get_slurm_job_id() != None

def get_slurm_job_id():
    return os.getenv("SLURM_JOB_ID")

def get_slurm_time_remaining(job_id):
    try:
        timeleft_arr= check_output(['squeue', '-j', job_id, '-O', 'timeleft']).decode("utf-8").split()
        if len(timeleft_arr) == 1:
            raise Exception("SLURM JOB NOT FOUND")
    except:
        return -1 
        
    timeleft = timeleft_arr[1]
    days = 0
    hours = 0
    minutes = 0
    seconds = 0
    if "-" in timeleft:
        days, timeleft = timeleft.split("-")
        days = int(days)

    timeleft_tuple = timeleft.split(":")
    
    if len(timeleft_tuple) == 3:
        hours = int(timeleft_tuple[-3])
    if len(timeleft_tuple) >= 2: 
        minutes = int(timeleft_tuple[-2])
    if len(timeleft_tuple) >= 1:
        seconds = int(timeleft_tuple[-1])

    return seconds + 60 * (minutes + 60 * (hours + 24 * days))

def get_torch_device():
    return torch.device("cpu")
    # if (not is_in_cluster()):
    #     return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # else:
    #     return torch.device("cpu")

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def draw(x, width=64):
    plt.imshow(to_numpy(x).reshape(width, -1))
    plt.colorbar()
    plt.show()