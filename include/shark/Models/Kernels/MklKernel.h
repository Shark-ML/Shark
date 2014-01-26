//===========================================================================
/*!
 * 
 *
 * \brief       Weighted sum of base kernels, each acting on a subset of features only.
 * 
 * 
 *
 * \author      M. Tuma, O.Krause
 * \date        2012
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================

#ifndef SHARK_MODELS_KERNELS_MKL_KERNEL_H
#define SHARK_MODELS_KERNELS_MKL_KERNEL_H


#include <shark/Models/Kernels/WeightedSumKernel.h>
#include "Impl/MklKernelBase.h"
namespace shark {

/// \brief Weighted sum of kernel functions
///
/// For a set of positive definite kernels \f$ k_1, \dots, k_n \f$
/// with positive coeffitients \f$ w_1, \dots, w_n \f$ the sum
/// \f[ \tilde k(x_1, x_2) := \sum_{i=1}^{n} w_i \cdot k_i(x_1, x_2) \f]
/// is again a positive definite kernel function. This still holds when
/// the sub-kernels only operate of a subset of features, that is, when
/// we have a direct sum kernel ( see e.g. the UCSC Technical Report UCSC-CRL-99-10:
/// Convolution Kernels on Discrete Structures by David Haussler ).
///
/// This class is very similar to the #WeightedSumKernel , except that it assumes
/// its inputs to be tuples of values \f$ x=(x_1,\dots, x_n) \f$. It calculates
/// the direct sum of kernels
/// \f[ \tilde k(x, y) := \sum_{i=1}^{n} w_i \cdot k_i(x_i, y_i) \f]
///
/// Internally, the weights are represented as \f$ w_i = \exp(\xi_i) \f$
/// to allow for unconstrained optimization.
///
/// The result of the kernel evaluation is devided by the sum of the
/// kernel weights, so that in total, this amounts to fixing the sum
/// of the of the weights to one.
///
/// In the current implementation, we expect the InputType to be a
/// boost::fusion::vector. For example, boost::fusion::vector<RealVector,RealVector>
/// represents a tuple of two vectors.
///


template<class InputType>
class MklKernel
: private detail::MklKernelBase<InputType>//order is important!
, public WeightedSumKernel<InputType>
{
private:
    typedef detail::MklKernelBase<InputType> base_type1;
    typedef WeightedSumKernel<InputType> base_type2;
public:

    template<class KernelTuple>
    MklKernel(KernelTuple const& kernels):base_type1(kernels),base_type2(base_type1::makeKernelVector()){}

    /// \brief From INameable: return the class name.
    std::string name() const
    { return "MklKernel"; }
};

}
#endif
