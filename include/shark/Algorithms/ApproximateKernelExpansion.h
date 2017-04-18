//===========================================================================
/*!
 * 
 *
 * \brief       The k-means clustering algorithm.
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2011
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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


#ifndef SHARK_ALGORITHMS_APPROXIMATE_KERNEL_EXPANSION_H
#define SHARK_ALGORITHMS_APPROXIMATE_KERNEL_EXPANSION_H

#include <shark/Core/DLLSupport.h>
#include <shark/Core/Random.h>
#include <shark/Models/Kernels/KernelExpansion.h>
namespace shark{
/// \brief Approximates a kernel expansion by a smaller one using an optimized basis.
///
/// often, a kernel expansion can be represented much more compactly when the points defining the basis of the kernel
/// expansion are not fixed.  The resulting kernel expansion can be evaluated much quicker than the original
/// when k is small compared to the number of the nonzero elements in the weight vector of the supplied kernel expansion.
///
/// Given a kernel expansion with weight matrix alpha and Basis B of size m, finds a new weight matrix beta and Basis Z with k vectors
/// so that the difference of the resulting decision vectors is small in the RKHS defined by the supplied kernel.
///
/// The algorithm proceeds by first performing a kMeans clustering as a good initialization. This initial guess is then optimized
/// by finding the closest weight vector to the original vector representable by the basis. Using this estimate, the basis
/// can then be optimized.
///
/// The supplied kernel must be dereferentiable wrt its input parameters which excludes all kernels not defined on RealVector
///
/// The algorithms is O(k^3 + k m) in each iteration.
///
/// \param rng the Rng used for the kMeans clustering
/// \param model the kernel expansion to approximate
/// \param k the number of basis vectors to be used by the approximation
/// \param precision target precision of the gradient to be reached during optimization
SHARK_EXPORT_SYMBOL KernelExpansion<RealVector> approximateKernelExpansion(
	random::rng_type& rng,
	KernelExpansion<RealVector> const& model,
	std::size_t k,
	double precision = 1.e-8
);

} // namespace shark
#endif
