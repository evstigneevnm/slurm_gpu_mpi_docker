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

#ifndef __SLURM_MPI_CUDA_ENROOT_PYXIS__QUICK_TEST_CHECK_H__
#define __SLURM_MPI_CUDA_ENROOT_PYXIS__QUICK_TEST_CHECK_H__


#include <limits>
#include <utility>

namespace tests
{


template<class T>
std::pair<std::string, bool> check_test_to_eps(const T val)
{
    if( !std::isfinite(val) )
    {
        return {"\x1B[31mFAIL, NOT FINITE\033[0m", false};
    }
    else if(std::abs(val)>std::sqrt(std::numeric_limits<T>::epsilon()) )
    {
        return {"\x1B[31mFAIL\033[0m", false};
    }
    else
    {
        return {"\x1B[32mPASS\033[0m", true};
    }
}

template<class T>
std::pair<std::string, bool> check_test_to_zero(const T val)
{
    if( !std::isfinite(val) )
    {
        return {"\x1B[31mFAIL, NOT FINITE\033[0m", false};
    }
    else if(std::abs(val) != 0 )
    {
        return {"\x1B[31mFAIL\033[0m", false};
    }
    else
    {
        return {"\x1B[32mPASS\033[0m", true};
    }
}

std::pair<std::string, bool> check_to_bool(const bool val)
{
    if(!val)
    {
        return {"\x1B[31mFAIL\033[0m", false};
    }
    else
    {
        return {"\x1B[32mPASS\033[0m", true};
    }
}

}

#endif //__SLURM_MPI_CUDA_ENROOT_PYXIS__QUICK_TEST_CHECK_H__