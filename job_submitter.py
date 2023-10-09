#%%
import requests
import numpy as np
import argparse

# local imports
from model_helpers import *
import model_tests.test_models as test_models
from model_tests import robustness_study
# import IPython
from subprocess import check_output


#%%
headers = {
    'Content-type':'application/json', 
    'Accept':'application/json'
    }
global url

session = requests.Session()


def validate_result(r, verbosity=0):
    assert r.status_code == 200
    if verbosity > 0:
        print(r.text)
    return r.text

def submit_job(inner_sims, job_title, submit_job_verbosity=0, **kwargs):
    r = requests.post(f"{url}/submitjob",
                            headers=headers,
                            json= {
                                "innerSimulations":inner_sims,
                                "JobDetails":{
                                    "title":job_title,
                                    **kwargs}
                            })
    return validate_result(r, verbosity=submit_job_verbosity)

def job_submission_helper(inner_sims, job_submission_helper_verbosity=0, **kwargs):
    partition_size = len(inner_sims)
    sims_to_submit = inner_sims
    joblet_idx = 0
    while len(sims_to_submit) != 0:
        if job_submission_helper_verbosity > 0:
            print(f"Idx: {joblet_idx} joblet size: {partition_size} to_submit: ({len(sims_to_submit)}/{len(inner_sims)})")
        current_joblet = sims_to_submit[:partition_size]
        job_submission_result = submit_job(current_joblet, joblet_idx=joblet_idx, joblet_size=partition_size, full_job_simulation_count=len(inner_sims), submit_job_verbosity=job_submission_helper_verbosity, **kwargs)
        if len(job_submission_result) != 0:
            joblet_idx+=1
            sims_to_submit = sims_to_submit[partition_size:]
        else:
            partition_size //= 2

def multi_job_submitter(jobs):
    for job in jobs:
        job_submission_helper(job['sims'], **job['info'])

def validate_git_repo_and_get_sha():
    if len(check_output(" git status --porcelain".split()).decode("utf-8")) != 0:
        raise RuntimeError("Aborting. Your Git Working Directory is not clean! Please commit or .gitignore all changes.")
    return check_output("git rev-parse HEAD".split()).decode("utf-8")


def submit_playback_job(fname):
    import json
    inner_sims = json.loads(open(fname, "r").read())

    for inner_sim in inner_sims:
        inner_sim["epochs"] = 1
        inner_sim["steps"] = 1024
        inner_sim["actuationProportionalToMass"] = 1
        inner_sim["requires_taichi_grads"] = True
        
    submit_job(inner_sims, job_title=f"Playback Job")

def submit_fig_job(job_factory):
    jobs = job_factory()
    for job in jobs:
        job_submission_helper(job["sims"], **(job["info"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="10.243.16.52")
    parser.add_argument("--port", default="24478")

    # Data used in many figures.
    parser.add_argument("--base_experiment", action="store_true")

    # Fig 4
    parser.add_argument("--secondary_loss_functions", action="store_true")
    parser.add_argument("--object_manipulation", action="store_true")
    parser.add_argument("--design_masks", action="store_true")
    parser.add_argument("--design_starting_mask", action="store_true")

    # Fig S4
    parser.add_argument("--constrained_actuation", action="store_true")

    # Fig S5
    parser.add_argument("--direct_particle", action="store_true")

    # Fig S6
    parser.add_argument("--void_interpolation", action="store_true")

    # Fig S7-9
    parser.add_argument("--robustness_study", action="store_true")

    parser.add_argument("--playback", default=None, help="json file to submit for playback")
    
    args = parser.parse_args()

    url = f"http://{args.ip}:{args.port}"

    if args.base_experiment:
        for fringe_softness in [0, 2]:
            jobs = test_models.base_experiments(fringe_softness=fringe_softness)
            for job in jobs:
                job_sims = job['sims']
                job_info = job['info']
                job_submission_helper(job_sims, **job_info)

    if args.secondary_loss_functions:
        multi_job_submitter(test_models.secondary_loss_functions())

    if args.object_manipulation:
        jobs = test_models.object_manipulation()
        multi_job_submitter(jobs)
        
    if args.design_starting_mask:
        jobs = test_models.masked_starting_points()
        multi_job_submitter(jobs)

    if args.design_masks:
        jobs = test_models.design_masks()
        multi_job_submitter(jobs)

    if args.constrained_actuation:
        print("Warning! This job requires simulation workers to be launched with --high_resolution_robot.")
        submit_fig_job(test_models.constrained_actuation)

    if args.direct_particle:
        submit_fig_job(test_models.direct_particle)
 
    if args.void_interpolation:
        submit_fig_job(test_models.void_interpolation)
 
    if args.robustness_study:
        jobs = robustness_study.experiments()
        for job in jobs:
            job_sims = job['sims']
            job_info = job['info']
            job_submission_helper(job_sims, **job_info)
   
    if args.playback is not None:
        submit_playback_job(args.playback)



# %%
