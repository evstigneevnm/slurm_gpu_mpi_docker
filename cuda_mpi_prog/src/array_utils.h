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

#ifndef __SLURM_MPI_CUDA_ENROOT_PYXIS__ARRAY_UTILS_H__
#define __SLURM_MPI_CUDA_ENROOT_PYXIS__ARRAY_UTILS_H__

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/set_operations.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <limits>
#include <scfd/arrays/array_thrust_cast.h>

namespace detail{

template<class Ord, class T, class Keys, class Vals>
Ord size_by_key(Ord rows_n, const Keys keys_in, Keys keys_out, Vals vals_out)
{

    
    auto new_end = thrust::reduce_by_key(
        scfd::arrays::array_thrust_begin(keys_in),
        scfd::arrays::array_thrust_begin(keys_in ) + rows_n, 
        ::thrust::make_counting_iterator<T>(1),
        scfd::arrays::array_thrust_begin(keys_out ),
        scfd::arrays::array_thrust_begin(vals_out )
        ); 

    auto end_keys = new_end.first - scfd::arrays::array_thrust_begin(keys_out );
    auto end_vals = new_end.second - scfd::arrays::array_thrust_begin(vals_out );
    return end_keys;
}

template<class Ord, class Keys, class Vals>
Ord reduce_by_key(Ord rows_n, const Keys keys_in, const Vals vals_in, Keys keys_out, Vals vals_out)
{
    
    auto new_end = thrust::reduce_by_key(
        scfd::arrays::array_thrust_begin(keys_in ),
        scfd::arrays::array_thrust_begin(keys_in ) + rows_n, 
        scfd::arrays::array_thrust_begin(vals_in ),
        scfd::arrays::array_thrust_begin(keys_out ),
        scfd::arrays::array_thrust_begin(vals_out )
        ); 

    auto end_keys = new_end.first - scfd::arrays::array_thrust_begin(keys_out );
    auto end_vals = new_end.second - scfd::arrays::array_thrust_begin(vals_out );
    return end_keys;
}

template<class Ord, class Keys, class Vals>
Ord reduce_by_key_min(Ord rows_n, const Keys keys_in, const Vals vals_in, Keys keys_out, Vals vals_out)
{
    
    auto new_end = thrust::reduce_by_key(
        scfd::arrays::array_thrust_begin(keys_in ),
        scfd::arrays::array_thrust_begin(keys_in ) + rows_n, 
        scfd::arrays::array_thrust_begin(vals_in ),
        scfd::arrays::array_thrust_begin(keys_out ),
        scfd::arrays::array_thrust_begin(vals_out ),
        thrust::equal_to<typename Keys::value_type>(),
        thrust::minimum<typename Vals::value_type>()
        ); 

    auto end_keys = new_end.first - scfd::arrays::array_thrust_begin(keys_out );
    auto end_vals = new_end.second - scfd::arrays::array_thrust_begin(vals_out );
    return end_keys;
}


template<class Ord, class Keys, class Vals>
void sort_by_key(Ord rows_n, Keys keys, Vals vals)
{
    thrust::sort_by_key
    (
        scfd::arrays::array_thrust_begin(keys ),
        scfd::arrays::array_thrust_begin(keys )+ rows_n,
        scfd::arrays::array_thrust_begin(vals )
    );
}

template<class Ord, class Keys, class Vals>
void stable_sort_by_key(Ord rows_n, Keys keys, Vals vals)
{
    thrust::stable_sort_by_key
    (
        scfd::arrays::array_thrust_begin(keys ),
        scfd::arrays::array_thrust_begin(keys )+ rows_n,
        scfd::arrays::array_thrust_begin(vals )
    );
}


template<class Ord, class Vec>
void sort(Ord rows_n, Vec vec1)
{

    thrust::sort
    (
        scfd::arrays::array_thrust_begin(vec1 ), 
        scfd::arrays::array_thrust_begin(vec1 )+rows_n 
    );

}

template<class Ord, class Vec>
Ord unique(Ord rows_n, Vec vec1)
{

    auto ret_device = thrust::unique
    (
        scfd::arrays::array_thrust_begin(vec1 ), 
        scfd::arrays::array_thrust_begin(vec1 )+rows_n
    );
    
    return (ret_device - scfd::arrays::array_thrust_begin(vec1 ) );
}

template<class Ord, class Set>
Ord set_intersection(Set set1, Ord size_x0, Set set2, Ord size_xN, Set set_ret)
{

    auto ret_device = thrust::set_intersection
    (
        scfd::arrays::array_thrust_begin(set1 ), 
        scfd::arrays::array_thrust_begin(set1 )+size_x0,
        scfd::arrays::array_thrust_begin(set2 ), 
        scfd::arrays::array_thrust_begin(set2 )+size_xN,        
        scfd::arrays::array_thrust_begin(set_ret ) 
    );

    return (ret_device - scfd::arrays::array_thrust_begin(set_ret ) );

}

template<class Ord, class T, class Vec>
T inclusive_scan_inplace(Ord rows_n, Vec ptr)
{
    thrust::inclusive_scan
    (
        scfd::arrays::array_thrust_begin(ptr ),
        scfd::arrays::array_thrust_begin(ptr )+rows_n,
        scfd::arrays::array_thrust_begin(ptr )
    );

    T nnz_;
    thrust::copy
    (
        scfd::arrays::array_thrust_begin(ptr) + rows_n-1,
        scfd::arrays::array_thrust_begin(ptr) + rows_n,
        &nnz_
    );
    
    return nnz_;
}

template<class Ord, class T, class Vec>
T exclusive_scan_inplace(Ord rows_n, Vec ptr)
{
    T last_val;
    thrust::copy
    (
        scfd::arrays::array_thrust_begin(ptr) + rows_n-1,
        scfd::arrays::array_thrust_begin(ptr) + rows_n,
        &last_val
    );

    thrust::exclusive_scan
    (
        scfd::arrays::array_thrust_begin(ptr ),
        scfd::arrays::array_thrust_begin(ptr )+rows_n,
        scfd::arrays::array_thrust_begin(ptr )
    );

    T nnz_;
    thrust::copy
    (
        scfd::arrays::array_thrust_begin(ptr) + rows_n-1,
        scfd::arrays::array_thrust_begin(ptr) + rows_n,
        &nnz_
    );

    return nnz_ + last_val; //returned value is the sum of all aelement, since ptr[rows_n-1] is removed by exclusive_scan
}

template<class Ord, class T, class Vec>
T sum_reduce(Ord rows_n, Vec ptr)
{
    T res = thrust::reduce(
        scfd::arrays::array_thrust_begin(ptr ), 
        scfd::arrays::array_thrust_begin(ptr ) + rows_n, 
        T(0)
        );
    
    // T res = thrust::reduce(
    //     scfd::arrays::array_thrust_begin(ptr ), 
    //     scfd::arrays::array_thrust_begin(ptr ) + rows_n,
    //     T(0),
    //     thrust::plus<T>()
    //     );

    return res;
}

template<class Ord, class T, class Vec>
T max_reduce(Ord rows_n, Vec ptr)
{
    T res = thrust::reduce(
        scfd::arrays::array_thrust_begin(ptr ), 
        scfd::arrays::array_thrust_begin(ptr ) + rows_n, 
        std::numeric_limits<T>::min(),
        thrust::maximum<T>()
        );
    return res; 
}
template<class Ord, class T, class Vec>
T min_reduce(Ord rows_n, Vec ptr)
{
    T res = thrust::reduce(
        scfd::arrays::array_thrust_begin(ptr ), 
        scfd::arrays::array_thrust_begin(ptr ) + rows_n, 
        std::numeric_limits<T>::max(),
        thrust::minimum<T>()
        );
    return res; 
}

}//detail ends

#endif
