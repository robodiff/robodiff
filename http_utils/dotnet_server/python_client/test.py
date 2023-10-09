#%%
from turtle import st
import requests
import uuid
import json
import tqdm
import time
import pprint
import random
import numpy as np
import io

#%%
headers = {
    'Content-type':'application/json', 
    'Accept':'application/json'    
}

url = 'http://localhost:24479' # Set destination URL here
#%%
session = requests.Session()
#%%
def print_results(result, stat_code=200, return_body=False):
    if stat_code is not None:
        assert result.status_code == stat_code
    result_body = json.loads(result.text)
    print(f"status_code: {result.status_code}")
    pprint.pprint(result_body)
    if return_body:
        return result_body


#%%
r = requests.get(f"{url}/stats", headers=headers)
print_results(r)

#%%
r = requests.post(f"{url}/register", headers=headers)
print_results(r)
#%%
r = requests.post(f"{url}/register", headers=headers, json={"nodeDetails":"test_details", "extra":"test_extra"})
node_guid = print_results(r, return_body=True)
#%%
# shutdown the node.
r = requests.post(f"{url}/shutdownnode/{node_guid}", headers=headers)
#%%

r = requests.post(f"{url}/register", headers=headers, json={"nodeDetails":"test_details", "extra":"test_extra"})
node_guid = print_results(r, return_body=True)
#%%
r = requests.get(f"{url}/exportnodes", headers=headers)
print_results(r)

#%%
r = requests.get(f"{url}/stats", headers=headers)
print_results(r)

#%%
# r = requests.get(f"{url}/exportresultsquick", headers=headers)
# print_results(r)
#%%
r = requests.post(f"{url}/submitjob",headers=headers, json={"innerSimulations":["a", "b"], "JobDetails":{"title":"jobDetailTile", "extra":"extra job details"}} )
print(r.status_code)
print(r.text)

#%%
r = requests.get(f"{url}/stats", headers=headers)
print_results(r)
#%%
# r = requests.get(f"{url}/exportresults", headers=headers)
# print_results(r)

#%%
# r = requests.get(f"{url}/exportresultsquick", headers=headers)
# print_results(r)
#%%
r = requests.get(f"{url}/simulation/{node_guid}", headers=headers)
simulation_instance = print_results(r, return_body=True)

#%%
r = requests.get(f"{url}/stats", headers=headers)
print_results(r)
#%%
# r = requests.get(f"{url}/exportresults", headers=headers)
# print_results(r)

# #%%
# r = requests.get(f"{url}/exportresultsquick", headers=headers)
# print_results(r)
#%%
r = requests.post(f"{url}/results/{simulation_instance['workGuid']}", json={"epoch":0, "loss":"incremental results one"} )
print_results(r)
#%%
# r = requests.get(f"{url}/exportresults", headers=headers)
# results = print_results(r, return_body=True)
#%%
r = requests.post(f"{url}/results/{simulation_instance['workGuid']}", json={"epoch":1, "loss":"incremental results two"} )
print_results(r)
#%%
# r = requests.get(f"{url}/exportresults", headers=headers)
# print_results(r)
#%%
# r = requests.get(f"{url}/exportresultsquick", headers=headers)
# print_results(r)
#%%
r = requests.post(f"{url}/finished/{simulation_instance['workGuid']}")
print_results(r)
#%%
# r = requests.get(f"{url}/exportresults", headers=headers)
# print_results(r)
# #%%
# r = requests.get(f"{url}/exportresultsquick", headers=headers)
# print_results(r)
#%%
r = requests.get(f"{url}/simulation/{node_guid}", headers=headers)
simulation_instance_will_fail = print_results(r, return_body=True)

#%%
r = requests.get(f"{url}/simulation/{node_guid}", headers=headers)
no_jobs = print_results(r, return_body=True)

#%%
r = requests.post(f"{url}/results/{simulation_instance_will_fail['workGuid']}", json={"epoch":0, "loss":"NaN"} )
print_results(r)
#%%
r = requests.post(f"{url}/errors/{simulation_instance_will_fail['workGuid']}", json={"details":"Test Error message!"} )
print_results(r)
#%%
# r = requests.get(f"{url}/exportresults", headers=headers)
# print_results(r)
# #%%
# r = requests.get(f"{url}/exportresultsquick", headers=headers)
# print_results(r)
#%%
r = requests.get(f"{url}/simulation/{node_guid}", headers=headers)
originally_failed_job = print_results(r, return_body=True)
r = requests.post(f"{url}/results/{originally_failed_job['workGuid']}", json={"epoch":0, "loss":0.1} )
print_results(r)
#%%
r = requests.post(f"{url}/finished/{originally_failed_job['workGuid']}")
print_results(r)
#%%
# r = requests.get(f"{url}/exportresults", headers=headers)
# print_results(r)
# #%%
# r = requests.get(f"{url}/exportresultsquick", headers=headers)
# print_results(r)
#%%
r = requests.get(f"{url}/stats", headers=headers)
print_results(r)
#%%
r = requests.post(f"{url}/exportcompletedfolder", headers=headers, json={"folderName":"/Users/David/Projects/minimalAPI/python_client/test_export_folder"})
print(r.status_code)
print(r.text)
# #%%
# r = requests.post(f"{url}/exportresultsfilequick", headers=headers, json={"fileName":"/Users/David/Projects/minimalAPI/python_test_export_quick.json"})
# print(r.status_code)
# print(r.text)

# %%

# %%

# %%

# %%
