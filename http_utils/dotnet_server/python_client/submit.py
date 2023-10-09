#%%
import requests
import uuid
import json
import tqdm
import time
import pprint
import random
import numpy as np
import io
import argparse
import datetime


#%%
parser = argparse.ArgumentParser()
parser.add_argument('--num_sims_per_job', "-n", type=int, default=100)
parser.add_argument('--n_jobs', "-m", type=int, default=10)
args = parser.parse_args()
print(args)

#%%
headers = {
    'Content-type':'application/json', 
    'Accept':'application/json'
}

url = 'http://localhost:24479' # Set destination URL here
#%%
session = requests.Session()
#%%
N_JOBS = args.n_jobs
N_SIMS_PER_JOB = args.num_sims_per_job

def get_sim():
    return {
        'steps': 1024,
        'internalDamping': 30.0,
        'globalDamping': 2.0,
        'baseE': 20.0,
        'actuatorStrength': 4.0,
        'loss_mode': 'locomote_flat',
        'epochs': 5,
        'morphModel': {
            'name': 'TestingModel',
            'model': {
                'modelFilename': 'TestingFileName'
            },
            'genome': {
                'z': np.random.random((16,1)).tolist()
                },
            'optimizer': {
                'name': 'Adam', 'lr': 0.01
            }
        },
        'actuationModel': {
            'name': 'DefaultMixedSineActuatorModel',
            'genome':{
                'z':np.random.random((4,1)).tolist()
            },
            'optimizer': {
                'name': 'Adam', 
                'lr': 0.01
            }
        }
    }
#%%
def submit_job(n_sims, job_title):
    inner_sims =  [get_sim() for _ in range(n_sims)]
    r = requests.post(f"{url}/submitjob",
                        headers=headers,
                        json= {
                            "innerSimulations":inner_sims,
                            "JobDetails":{
                                "title":job_title}
                        })
    print(r.status_code)
    print(r.text)


#%%
for job_idx in range(N_JOBS):
    submit_job(N_SIMS_PER_JOB, f"STRESS JOB {job_idx} of {N_JOBS} STARTED AT: {datetime.datetime.now().isoformat()}")
#%%
r = requests.get(f"{url}/stats", headers=headers)
r.status_code, json.loads(r.text)
#%%
# %%
