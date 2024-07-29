# Multigpu_sample for Slurm CUDA MPI Docker test program
# It allows one to test multigpu build for a custom assembled programm
# when one is using isolaiton by slurm with enroot and pyxis from Nvidia.
# Copyright (C) 2024 Evstigneev Nikolay Mikhaylovitch

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https:#www.gnu.org/licenses/>.

FROM nvcr.io/nvidia/hpc-benchmarks:24.06 as build
# update and install all you need to build your program
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install gfortran -y
# building a test case (anything you need goes here)
WORKDIR /mpi_cuda_sample
ADD ./cuda_mpi_prog/ /mpi_cuda_sample
#you can now run CMAKE or anything you need to compile and make your program. I here use a simple Makefile
RUN cd /mpi_cuda_sample
RUN make rel

#This will take the compiled program from your build stage but will keep your source code out of container.
FROM nvcr.io/nvidia/hpc-benchmarks:24.06
#copy executable to your container.
WORKDIR /mpi_cuda_sample
COPY --from=build /mpi_cuda_sample/*.bin /mpi_cuda_sample/

