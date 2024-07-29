// Multigpu_sample for Slurm CUDA MPI Docker test program
// It allows one to test multigpu build for a custom assembled programm
// when one is using isolaiton by slurm with enroot and pyxis from Nvidia.
// Copyright (C) 2024 Evstigneev Nikolay Mikhaylovitch

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef __SLURM_MPI_CUDA_ENROOT_PYXIS__VECTOR_OPERATIONS_H__
#define __SLURM_MPI_CUDA_ENROOT_PYXIS__VECTOR_OPERATIONS_H__


#include <array_utils.h>
#include <scfd/communication/mpi_comm_info.h>



template<class T, class Vec>
struct vector_operations
{
    using vector_type = Vec;
    
    vector_operations(scfd::communication::mpi_comm_info* mpi): mpi_(mpi)
    {}

    T all_reduce_sum(const Vec& loc_data)const
    {
        std::size_t size_loc = loc_data.size();
        T res_local = detail::sum_reduce<std::size_t, T, Vec>(size_loc, loc_data);
        // std::cout << "mpi_.myid = " << mpi_->myid << ", res_local = " << res_local << std::endl;
        return mpi_->all_reduce_sum<T>( res_local );
    }

    T all_reduce_max(const Vec& loc_data)const
    {
        std::size_t size_loc = loc_data.size();
        T res_local = detail::max_reduce<std::size_t, T, Vec>(size_loc, loc_data);
        return mpi_->all_reduce_max<T>( res_local );

    }

    T all_reduce_min(const Vec& loc_data)const
    {
        std::size_t size_loc = loc_data.size();
        T res_local = detail::min_reduce<std::size_t, T, Vec>(size_loc, loc_data);
        return mpi_->all_reduce_min<T>( res_local );
    }  


    scfd::communication::mpi_comm_info* mpi_;

};


#endif