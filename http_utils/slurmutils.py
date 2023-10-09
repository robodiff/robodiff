# %%
import subprocess
import os


slurm_envs = ["SLURM_NODELIST",
            "SLURM_JOB_NAME",
            "SLURMD_NODENAME",
            "SLURM_TOPOLOGY_ADDR",
            "SLURM_PRIO_PROCESS",
            "SLURM_SRUN_COMM_PORT",
            "SLURM_JOB_QOS",
            "SLURM_PTY_WIN_ROW",
            "SLURM_TOPOLOGY_ADDR_PATTERN",
            "SLURM_NNODES",
            "SLURM_STEP_NUM_NODES",
            "SLURM_JOBID",
            "SLURM_NTASKS",
            "SLURM_LAUNCH_NODE_IPADDR",
            "SLURM_STEP_ID",
            "SLURM_STEP_LAUNCHER_PORT",
            "SLURM_TASKS_PER_NODE",
            "SLURM_WORKING_CLUSTER",
            "SLURM_CONF",
            "SLURM_JOB_ID",
            "SLURM_CPUS_PER_TASK",
            "SLURM_JOB_USER",
            "SLURM_STEPID",
            "SLURM_SRUN_COMM_HOST",
            "SLURM_PTY_WIN_COL",
            "SLURM_UMASK",
            "SLURM_JOB_UID",
            "SLURM_NODEID",
            "SLURM_SUBMIT_DIR",
            "SLURM_TASK_PID",
            "SLURM_NPROCS",
            "SLURM_CPUS_ON_NODE",
            "SLURM_PROCID",
            "SLURM_JOB_NODELIST",
            "SLURM_PTY_PORT",
            "SLURM_LOCALID",
            "SLURM_JOB_GID",
            "SLURM_JOB_CPUS_PER_NODE",
            "SLURM_CLUSTER_NAME",
            "SLURM_GTIDS",
            "SLURM_SUBMIT_HOST",
            "SLURM_JOB_PARTITION",
            "SLURM_STEP_NUM_TASKS",
            "SLURM_JOB_ACCOUNT",
            "SLURM_JOB_NUM_NODES",
            "SLURM_STEP_TASKS_PER_NODE",
            "SLURM_STEP_NODELIST",
            "SLURM_SCRIPT_CONTEXT"]

# %%

def get_slurm_vars():
    return dict([(k, os.getenv(k)) for k in slurm_envs])
# %%
def get_cpu_info():
    return dict([l for l in [[v.strip() for v in l.split(":")] for l in subprocess.check_output("lscpu").decode("utf-8").split("\n")] if len(l) > 1])

# %%
def get_node_info():
    return dict(**get_slurm_vars(), **get_cpu_info())