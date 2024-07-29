INCLUDE_SRC = ./cuda_mpi_prog/src
INCLUDE_SCFD = ./cuda_mpi_prog/contrib/scfd/include
ROOT_CUDA = /usr/local/cuda
ROOT_MPI = /usr/local/mpi
INCLUDE_CUDA = $(ROOT_CUDA)/include
LIB_CUDA = $(ROOT_CUDA)/lib64
INCLUDE_MPI = $(ROOT_MPI)/include
LIB_MPI = $(ROOT_MPI)/lib
CC = $(ROOT_CUDA)/bin/nvcc
MPICC = $(ROOT_MPI)/bin/mpic++
OPENMP = -fopenmp -lpthread

rel:
	$(CC) -O3 -std=c++17 -I$(INCLUDE_SCFD) -I$(INCLUDE_SRC) -I$(INCLUDE_MPI) cuda_mpi_prog/src/test_mpi_cuda.cu -c -o test_mpi_cuda.obj -Xcompiler $(OPENMP)
	$(MPICC) test_mpi_cuda.obj -L$(LIB_CUDA) -L$(LIB_MPI) -lcudart $(OPENMP) -o test_mpi_cuda.bin
	rm test_mpi_cuda.obj



