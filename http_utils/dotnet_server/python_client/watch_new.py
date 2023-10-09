#%%
import requests
import time
import pprint
from utils import *
import tqdm

#%%
session = requests.Session()
#%%
import sys
sleep_time = float(sys.argv[1])

stats = None
try:
    stats = fetch_stats(session)
except:
    stats = {'activeNodeCount': 0,
                'dispatchedJobCount': 0,
                'dispatchedSimulationCount': 0,
                'failedSimulationCount': 0,
                'outstandingSimulationCount': 0,
                'simulationsInRetryQueue': 0,
                'totalFinishedSimulationCount': 0,
                'totalJobCount': 0,
                'totalNodeCount': 0,
                'totalResultsReceivedCount': 0,
                'totalSimulationCount': 0,
                'totalSimulationErrorCount': 0}

nodes = tqdm.tqdm(desc="nodes", total=stats["totalNodeCount"], initial=stats["activeNodeCount"], position=0, ascii=True)
jobs =  tqdm.tqdm(desc="jobs", total=stats['totalJobCount'], initial=stats['dispatchedJobCount'], position=1, ascii=True)
fsims =  tqdm.tqdm(desc="finished sims", total=stats["totalSimulationCount"], initial=stats['totalFinishedSimulationCount'], position=2, ascii=True)
dsims =  tqdm.tqdm(desc="dispatched sims", total=stats["totalSimulationCount"], initial=stats['dispatchedSimulationCount'], position=3, ascii=True)
results = tqdm.tqdm(desc="results", initial=stats['totalResultsReceivedCount'], position=4, ascii=True)
errors = tqdm.tqdm(desc="errors", initial=stats["totalSimulationErrorCount"], position=5, ascii=True)
failures = tqdm.tqdm(desc="failures", initial=stats["failedSimulationCount"], position=6, ascii=True)

def get_delta(s, n):
    return s[n] - stats[n]

while True:
    try:
        new_stats = fetch_stats(session)
        nodes.total = new_stats["totalNodeCount"]
        nodes.update(get_delta(new_stats, "activeNodeCount"))

        jobs.total = new_stats["totalJobCount"]
        jobs.update(get_delta(new_stats, "dispatchedJobCount"))

        fsims.total = new_stats["totalSimulationCount"]
        fsims.update(get_delta(new_stats, "totalFinishedSimulationCount"))

        dsims.total = new_stats["totalSimulationCount"]
        dsims.update(get_delta(new_stats, "dispatchedSimulationCount"))


        results.update(get_delta(new_stats, "totalResultsReceivedCount"))
        errors.update(get_delta(new_stats, "totalSimulationErrorCount"))
        failures.update(get_delta(new_stats, "failedSimulationCount"))


        stats = new_stats


    except:
        pass
    time.sleep(sleep_time)

#%%
