# Efficient automatic design of robots
This git repo contains the code used in `Efficient automatic design of robots`, published in [PNAS](https://www.pnas.org/doi/pdf/10.1073/pnas.2305180120) on Oct 10th, 2023.
## Bibtex
```Bibtex
@article{
doi:10.1073/pnas.2305180120,
author = {David Matthews  and Andrew Spielberg  and Daniela Rus  and Sam Kriegman  and Josh Bongard },
title = {Efficient automatic design of robots},
journal = {Proceedings of the National Academy of Sciences},
volume = {120},
number = {41},
pages = {e2305180120},
year = {2023},
doi = {10.1073/pnas.2305180120},
URL = {https://www.pnas.org/doi/abs/10.1073/pnas.2305180120},
eprint = {https://www.pnas.org/doi/pdf/10.1073/pnas.2305180120},
}



```

## Installation

```
conda create --name public-robodiff python=3.7.15
conda activate public-robodiff
conda install -c conda-forge libstdcxx-ng=12

pip install -r requirements.txt
```

## Running your first example...
```python
python main.py -vv --gui --cpu
```
This will load the first design attempt reported in the paper, and optimize it for 9 gradient descent steps (10 total design attempts).

The first & last design attempts will be visualized to the screen.

## Visualization.

If the design attempts are not visualized to the screen, there is a backup method for seeing your designs.

1. Make sure you have matplotlib installed: https://matplotlib.org/stable/users/installing/index.html

1. Open a command window, and navigate into ```robodiff/visualization```

## Running other examples.
To run other demos, add the `--local_selection <a,...>` flag.
Available options are:
* `base`: Runs optimization to produce the design manufactured with the paper.
* `design_starting_mask`: Constrains the initial robot shape to be a 5 pointed star. Also demonstrates aganoistic actuator groups.
* `void_interpolation`: Demonstrates a secondary void interpolation function.
* `direct_particle`: Demonstrates optimization morphology directly on each particle rather than using a circular void based encoding.
---
The following options should be run on their own. They trigger the robot to be simulated at higher resolution, or with an extra object added to the scene.
* `constrained_actuation`: Emposes the constraint that all actuators are fully enclosed by passive material. Mimicking the constraints of manufacturing actuators as enclosed volumes. 
* `object_manipulation`: Adds an object to the scene, and optimizes the robot to throw the object.

```python
python main.py -vv --gui --cpu --local_selection base design_starting_mask direct_particle
```
```python
python main.py -vv --gui --cpu --local_selection constrained_actuation
```
```python
python main.py -vv --gui --cpu --local_selection object_manipulation
```

## Running experiments.
When running multiple optimizations multiple independent times, it is helpful to parallelize across multiple computers. To do this, an ASP.NET HTTP server can be utilized.

### Launch the HTTP server
install C# dotnet with ASP.NET 6.0.

```
cd http_utils/dotnet_server
dotnet restore
dotnet build
dotnet run
```
Record IP address of the computer which the server was started on.

### Submit optimization work to the server.
Run `python job_submitter.py --ip <ip address of server> --<job to submit>`. Use `-h` flag to list available experiments.

### Start up worker nodes.
Typically this is done using a compute cluster job scheduler (e.g. SLURM). If you are using SLURM, `main.py` should automatically enter HTTP client worker mode.
Otherwise, you can trigger this with the `--client` flag. 

```
python main.py --client --IP <ip address of server>
```

If you are on a SLURM server, you can also trigger local mode, by adding the `--force_local` flag.

I typically use:
```
python main.py --ignore_exit_request -v --ip $IP --cpu_max_threads=$SLURM_JOB_CPUS_PER_NODE
```

### Exporting server data
The server has some limitations, 
(a) it does not automatically export worker result data to file.
(b) it stores everything in memory. For very large experiments, when running many jobs at once, this can consume quite a bit of memory.

Call
```
cd http_utils/dotnet_server/python_client
python export.py --ip <ip address of server> --export_path /path/to/results/directory
```
