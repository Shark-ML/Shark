/*!
 * 
 *
 * \brief      -
 * \author    O.Krause
 * \date        2014
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_KERNELBASISDISTANCE_H
#define SHARK_OBJECTIVEFUNCTIONS_KERNELBASISDISTANCE_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/LinAlg/solveSystem.h>
#include <shark/Rng/GlobalRng.h>


namespace shark{

	
/// \brief Computes the distance between the optimal point in a basis to the point represented by a KernelExpansion.
///
/// Assume we are given a kernel expansion \f$ w= \sum_i \alpha_i k(x_i,\cdot) \f$. The KernelBasisDistance takes
/// a new set of basis vectors \f$ z_i \f$ and finds the linear combination in that space which is closest
/// to \f$ w \f$ . More formally the function measures the squared distance in kernel space:
/// \f[ f(z) = \min_{\beta} \frac 1 2 || \sum_j \beta_j k(z_j,\cdot)-w|| . \f]
/// In vector notation with \f$ (K_x)_{i,j} = k(x_i,x_j) \f$ , \f$ (K_z)_{i,j} = k(z_i,z_j) \f$ and \f$ (K_{zx})_{i,j} = k(z_i,x_j) \f$ it computes:
///\f[ f(z) = \min_{\beta} \frac 1 2  \beta^T K_z  \beta - \beta^T K_{zx} \alpha + \frac 1 2 \alpha^TK_x \alpha . \f]
/// as the last term is constant in \f$ z_i \f$ , it is omitted in the actual computation that is, the value is offsetted by some constant and the minimum is not 0.
/// The input of the function is a vector which continuously stores the set of points in the basis. That is, it stores
/// \f$ v=[z_1, z_2,\dots,z_k] \f$.
///
/// The target point \f$ w \f$ is set by the KernelExpansion in the constructor. When the kernel used by the expansion is differentiable
/// with respect to the input point, this function is differentiable as well.
///
/// The kernel expansion can represent more than one single point, in this point the error is the sum of approximation errors.
class KernelBasisDistance : public SingleObjectiveFunction
{
public:
	/// \brief Constructs the objective function.
	///
	/// This functions calls sparsify on the kernel expansion to save computation time in the case of sparse bases.
	///
	/// \param kernelExpansion a pointer to the kernel expansion to approximate
	/// \param approximatingVectors the number of vectors used to approximate the point - the basis size
	KernelBasisDistance(KernelExpansion<RealVector>* kernelExpansion,std::size_t approximatingVectors);

	/// \brief Returns the name of the class
	std::string name() const
	{ return "KernelBasisDistance"; }

	void configure( const PropertyTree & node ){}

	/// \brief Returns a starting point of the algorithm
	///
	/// Returns a random subset of the basis of the kernel expansion
	void proposeStartingPoint(SearchPointType& startingPoint) const;
	
	/// \brief Returns the number of variables of the function.
	std::size_t numberOfVariables()const;

	/// \brief Evaluate the (sum of) distance(s) between the closes point in the basis to the point(s) represented by the kernel expansion.
	///
	/// See the class description for more details on this computation.
	double eval(RealVector const& input) const;

	/// \brief computes the derivative of the function with respect to the supplied basis.
	///
	/// Assume \f$ \beta \f$ to be the optimal value. then the derivative with respect to the basis vectors is:
	/// \f[	\frac{ \partial f}{\partial z_l} = \beta_l \sum_i \beta_i \frac{ \partial f}{\partial z_l} k(z_l,z_i) - \beta_l \sum_i \alpha_i \frac{ \partial f}{\partial z_l} k(z_l, x_i) \f]
	ResultType evalDerivative( const SearchPointType & input, FirstOrderDerivative & derivative ) const;
private:
	KernelExpansion<RealVector>* mep_expansion;     ///< kernel expansion to approximate
	std::size_t m_approximatingVectors; ///< number of vectors in the basis
};


}
#endif
