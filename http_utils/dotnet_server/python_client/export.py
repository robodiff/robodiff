#%%
import requests
import time
import pprint
import argparse
import utils as ut

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--export_path', default=".", type=str)
parser.add_argument("--ip", default="10.243.16.52")
parser.add_argument("--port", default="24478")
args = parser.parse_args()

print(args)

ut.set_url(args)

#%%
compression = True
r = requests.post(f"{ut.url}/exportcompletedfolder", headers=ut.headers, json={"folderName":args.export_path, "compression":compression})
print(r.status_code)
print(r.text)


r = requests.post(f"{ut.url}/exportallfolder", headers=ut.headers, json={"folderName":args.export_path, "compression":compression})
print(r.status_code)
print(r.text)


r = requests.get(f"{ut.url}/exportnodes", headers=ut.headers)
open(f"{args.export_path}/nodeinfo.json", "w").write(r.text)
print(r.status_code)
print(r.text)

#%%
