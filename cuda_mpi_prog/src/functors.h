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

#ifndef __SLURM_MPI_CUDA_ENROOT_PYXIS__FUNCTORS_H__
#define __SLURM_MPI_CUDA_ENROOT_PYXIS__FUNCTORS_H__

#include <scfd/static_vec/vec.h>
#include <scfd/static_vec/rect.h>


namespace functors
{

namespace kernel
{

template<class Ord, int Dim, class Array>
struct fill_values
{
    fill_values(Ord shift, Array array):
    array_(array),
    shift_(shift)
    {}
    Array array_;
    Ord shift_;

    __DEVICE_TAG__ void operator()(const scfd::static_vec::vec<Ord, Dim> &idx)
    {
        array_(idx) = shift_;
    }
};


template<class Ord, int Dim, class Array>
struct make_zero
{
    make_zero(Array& output):
    output_(output)
    {}
    Array output_;
    __DEVICE_TAG__ void operator()(const scfd::static_vec::vec<Ord, Dim> &idx)
    {
        output_(idx) = 0;
    }
};
}


template<class ForEach, class Ord, int Dim, class Array>    
void fill_values( const ForEach &for_each, const scfd::static_vec::rect<Ord,Dim> &rect, Ord shift, Array array )
{   
    for_each( kernel::fill_values<Ord, Dim, Array>(shift, array), rect);
}

template<class ForEach, class Ord, int Dim, class Array>    
void make_zero( const ForEach &for_each, const scfd::static_vec::rect<Ord,Dim> &rect, Array output )
{   
    for_each( kernel::make_zero<Ord, Dim, Array>(output), rect);
}



}


#endif