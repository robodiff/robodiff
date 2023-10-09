import requests
import json
import time
import numpy as np

import sys
import traceback

from utils import get_slurm_job_id, get_slurm_time_remaining, get_torch_device
from http_utils.slurmutils import get_node_info
from models.model_loaders import SimulationModel
import robodiff_startup as didv

def get_server_url():
    # Server urls are usually assigned via cli. This provides some default ones if the cli options are left blank.
    server_urls = {
        "gpunode002":'http://10.243.16.52',
        "vacc-user1":"http://10.243.16.47",
        "vacc-user2":"http://10.243.16.48"
    }
    port = 24478
    return f"{server_urls['gpunode002']}:{port}"
    
# Infinite yielder. Returns None.
# Useful for tqdm.tqdm(infinite()) to get stats about a while true loop. 
def infinite():
    while True:
        yield   


class ServerManager(object):
    def __init__(self, scene=None,
                        torch_device=None,
                        retry_delay = None,
                        min_time_to_start_new_work=500,
                        shutdown_timer=60*10,
                        server_url=None,
                        exit_on_request=True,
                        verbose=0):

        self.verbose = verbose

        assert scene is not None

        # running sims variables.
        self.scene = scene
        self.torch_device = torch_device
        
        # reporting variables
        self.slurm_job_id = get_slurm_job_id()
        self.start_time = time.time()
        self.slurm_run_time = get_slurm_time_remaining(self.slurm_job_id)
        self.min_time_to_start_new_work = min_time_to_start_new_work
        self.slurm_expected_end_time = self.start_time + self.slurm_run_time - self.min_time_to_start_new_work


        self.headers = {
            'Content-type':'application/json', 
            'Accept':'application/json'
        }
        self.session = requests.Session()
        self.url = server_url if server_url is not None else get_server_url()
        self.init_with_url = server_url is not None
        
        self.retry_delay = retry_delay
        self.node_guid = None
        self.node_info = get_node_info()
        self.register()

        self.raw_work = None
        self.raw_sim = None
        self.work_guid = None
        
        self._shutdown_timer = shutdown_timer
        self._last_work_contact_time = time.time() # updated whenever current work is finished being processed. either due to an error, or due to successful finishing of the work.

        self.should_exit = False
        self._exit_on_request = exit_on_request

        self.info_msg(f"Running in cluster mode: JOB_ID: {self.slurm_job_id}. Remaining Time: {self.slurm_expected_end_time - time.time():.0f}s")

    def info_msg(self, msg):
        if self.verbose > 0:
            print(msg)

    def debug_msg(self, msg):
        if self.verbose > 1:
            print(msg)

    def log_or_throw(self, e, send_to_server=True):
        exc_info = sys.exc_info()
        msg = ''.join(traceback.format_exception(*exc_info))

        if send_to_server:
            try:
                self.send_error(details={"ExceptionString":msg}) 
            except Exception as ex:
                self.log_or_throw(ex, send_to_server=False)

        if self.verbose > 2:
            try:
                import IPython
                IPython.embed()
            except:
                raise e
        elif self.verbose > 1:
            raise e
        else:
            self.info_msg(msg)
            

    def retry_until_success(self, method, *args, **kwargs):
        success = False
        while(not success):
            try:
                method(*args, **kwargs)
                success = True
            except:
                self.info_msg(f"Retrying until success: {method}")
                if not self.init_with_url:
                    self.url = get_server_url()
                
                time.sleep(self.retry_delay if self.retry_delay is not None else 10)

    def validate_result(self, r, return_body=True):
        assert r.status_code == 200
        if return_body:
            return json.loads(r.text)


    def _register(self):
        r = self.session.post(f"{self.url}/register", headers=self.headers, json=self.node_info)
        self.node_guid =  self.validate_result(r)
        return True
    
    def register(self):
        self.info_msg("Registering node...")
        self.retry_until_success(self._register)
        self.info_msg(f"Node registered with GUID: {self.node_guid}")

    def fetch_sim(self):
        r = self.session.get(f"{self.url}/simulation/{self.node_guid}", headers=self.headers)
        return self.validate_result(r)

    def send_results(self, results=None):
        self.info_msg(f"{self.work_guid}: epoch_idx: {results['epoch_idx']} | loss: {results['loss']}")

        r = self.session.post(f"{self.url}/results/{self.work_guid}", json=results)
        results_response = self.validate_result(r)
        if not results_response:
            self.should_exit = True
            self.info_msg("send_results requested node shutdown.")
    
        return results_response

    def send_finished(self):
        r = self.session.post(f"{self.url}/finished/{self.work_guid}")
        response = self.validate_result(r)
        if not response:
            self.should_exit = True
            self.info_msg("send_finished requested node shutdown.")
        return response

    def shutdown(self) :
        r = self.session.post(f"{self.url}/shutdownnode/{self.node_guid}", headers=self.headers)
        return self.validate_result(r, return_body=False)

    def send_error(self, details=None):
        r = self.session.post(f"{self.url}/errors/{self.work_guid}", json=details)
        response = self.validate_result(r)
        if not response:
            self.should_exit = True
            self.info_msg("send_error requested node shutdown.")
                
        return response

    def process_simulation(self):
        try:
            self.raw_work = self.fetch_sim() # might throw!

            self.work_guid = self.raw_work["workGuid"]
            self.raw_sim = self.raw_work["innerSimulation"]

            if self.retry_delay is None or self.retry_delay == 0:
                self.retry_delay = self.raw_work["retryDelay"]

            if self.raw_sim is None:
                self.info_msg("Process Simulation: No jobs right now. Sleeping")
                time.sleep(self.retry_delay)
                return False

            return self._process_simulation() # might throw.
        except Exception as e:
            self._last_work_contact_time = time.time()

            self.info_msg(f"Process Simulation Exception: {str(e)}")
            self.log_or_throw(e)
            time.sleep(self.retry_delay)
            return False

    def _process_simulation(self):
        scene = self.scene
        device = self.torch_device
        # try catch simulation unpackaging + running to attempt to let server know about errors.
        # if error fails to send, server will automatically restart simulation.
        try:
            simModel = SimulationModel( scene=scene, device=device, work_guid = self.work_guid,  **self.raw_sim)
            self.info_msg("\n")
            if("description_str" in simModel.kwargs):
                self.info_msg(simModel.kwargs["description_str"])
                self.debug_msg(simModel.get_summary_string())
            else:
                self.info_msg("Got Simulation for Processing...")

            didv.run_optimization(simModel,
                        results_handler=lambda x: self.send_results(x),
                        gui_handler=None)

        except Exception as e:
            self._last_work_contact_time = time.time()

            # Note: sending error might throw. If so, process_simulation will catch.
            self.info_msg(f"_process_simulation raise Exception: {str(e)}")    
            self.log_or_throw(e)    
            return False

        self._last_work_contact_time = time.time()
        # Note: sending finished message might throw. If so, process_simulation will catch.
        return self.send_finished()

    def _has_time_remaining(self):
        return self.slurm_run_time == -1 or self.slurm_run_time - self.min_time_to_start_new_work > time.time() - self.start_time

    def should_continue(self):
        can_continue = self._has_time_remaining()
        if self._exit_on_request and self.should_exit:
            can_continue = False
        
        auto_shutdown_not_triggered = True
        if (self._shutdown_timer is not None and self._shutdown_timer > 0):
             auto_shutdown_not_triggered =  self._last_work_contact_time + self._shutdown_timer >= time.time()

        should_continue = can_continue and auto_shutdown_not_triggered
        return should_continue
            
        # return (not self.should_exit) and self._has_time_remaining()

def handle_cluster_env(scene, ip="10.243.16.52", port=24478, ignore_exit_request=False, verbose=0):
    # default to gpunode002 ip
    torch_device = get_torch_device()

    server_url = f"http://{ip}:{port}" if ip is not None and port is not None else None

    serverManager = ServerManager(torch_device = torch_device, scene=scene, server_url=server_url, verbose=verbose, exit_on_request=not ignore_exit_request)
    try:
        while serverManager.should_continue():
            serverManager.process_simulation()
    except KeyboardInterrupt:
        # catch the keyboard interrupt to bread out of the while loop.
        # allow the node to gracefully shutdown.
        pass

    try:
        serverManager.info_msg("Shutting down node")
        serverManager.shutdown() # It is polite to tell server that we are shutting down the node.
    except Exception as e:
        serverManager.info_msg("Error during shutdown of node")
        serverManager.log_or_throw(e, send_to_server=False)
