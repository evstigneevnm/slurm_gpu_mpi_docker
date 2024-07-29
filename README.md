# Sample for Slurm NVIDIA GPU MPI Docker interaction.

This is a repository that contains a sample of how to make a Dockerfile and compile your program that uses MPI into slurm managed cluster with enroot and pyxis from NVIDIA.


## Problem we solve
One needs to run a custom program, possibly written on CUDA or OpenCL with MPI, on a cluster with only isolated container support. It requires a correct build of MPI inside a container and interaction of the internal MPI inside a container and external MPI which is managed by [Slurm](https://github.com/SchedMD/slurm) via **srun** command that is effectively, managed **mpiexec**. This container operation and interaction is managed by NVIDIA [enroot](https://github.com/NVIDIA/enroot) and [pyxis](https://github.com/NVIDIA/pyxis) installed on the cluster. But I was unable to find a good manual of how to build a custom C++/CUDA C++ program for said systems when multiple GPUs are used via MPI and contarization is mandatory. This manual and simple example should help anyone faced with the same complication. We solve it by using an NVIDIA sample Docker image and then use it to build our program with already built MPI.

### Approach
In this repository I provide a sample program that is written on C++/CUDA C++ which is build by a simple Makefile. It can be used to test a remote cluster or a local computer with multiple GPUs and MPI as well as serve as example of how to include your own code into the Docker image with to be executed by Slurm with pyxis and MPI. You can use the provided **Dockerfile** to configure the image and build your own program instead of the sample I provided.

### Definitions
**Cluster** is a remote system that has Slurm installed, it executes tasks by running **srun** and supports NVIDIA enroot and pyxis. It requires that only containers can be executed. 

**Local machine** is your local computer that has access to a Cluster. Assuming your local machine runs Docker and its OS is Linux. I haven't tested it on anything else.

### Instruction
**On the Local machine**
1. On your local machine install NVIDIA [enroot:](https://github.com/NVIDIA/enroot), instructions are found [here](https://github.com/NVIDIA/enroot/blob/master/doc/installation.md).

2. Clone this repository on your local machine.

3. From the root of the repository folder build Docker image as:

```
docker build -f docker_config/Dockerfile . -t slurmmpigpu/test:0.01

```
4. Import local docker image to enroot *sqsh* format:
```
enroot import dockerd://slurmmpigpu/test:0.01
```
This will create a file `slurmmpigpu+test+0.01.sqsh` which is generated in the repository directory. *Optionally* check the hash of your local image, e.g. use `sha512sum slurmmpigpu+test+0.01.sqsh`

5. Copy this file to your working directory on the Cluster.

**On the Cluster**

1. Navigate to your working directory on the Cluster and check that the file is copied correctly from your local machine. 
From now on we assume that your working directory on the Cluster is* **/user/workdir**.
*Optionally* check the hash of your remote image, e.g. use `sha512sum ~/workdir/slurmmpigpu+test+0.01.sqsh`, and compare it to the hash of your local `sqsh` file. 

2. Your current entrypoint is `/mpi_cuda_sample` inside the container, configured by the *Dockerfile*. It will be automatically accessed by the `srun`. To access other parts of the container provide a full path to `srun` as listed below.

3. *Optionally* To get inside your container you can run the command:
```
srun --container-image ~/workdir/slurmmpigpu+test+0.01.sqsh --pty bash
```
**WARNING!** This command *is not used* for launching applications, but only to get into the shell `bash` inside your container.

4. We execute our sample program on 4 nodes, each having 8 GPUs with MPI:
```
srun -N4 -n32 -G32 --container-image ~/workdir/slurmmpigpu+test+0.01.sqsh --container-entrypoint test_mpi_cuda.bin
```
It should return PASS for all reduce operations.

**WORNING!** One needs not execute `mpiexec` or `mpirun` inside the container. MPI call is configured automatically by pyxis.

### Additional options for pyxis srun are found [here](https://github.com/NVIDIA/pyxis).
Options that I find useful:
- `--container-mounts=SRC:DST`, where *SRC* is a FULL path to the directory on the cluster and *DST* is the FULL path inside the container.
- `--container-entrypoint` is the program inside the container to be executed in parallel my the managed MPI.
- `--container-env=NAME[,NAME...]` overrides system environment variables in the container, such as `PATH`, `LD_LD_LIBRARY_PATH` etc, e.g. the system variables form the Cluster override variables with the same name in the container.
- `-N number_of_nodes` is the number of nodes to be executed on.
- `-G number_of_gpus` is the number of GPUs to be passed to the container
- `-n number_of_procs` is the global number of MPI processes usually equals the number of GPUs.
