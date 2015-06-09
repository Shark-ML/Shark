/*!
 * 
 *
 * \brief      -
 * \author    O.Krause
 * \date        2014
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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

#include <shark/Core/DLLSupport.h>
#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Models/Kernels/KernelExpansion.h>

namespace shark{

/// \brief Computes the squared distance between the optimal point in a basis to the point represented by a KernelExpansion.
///
/// Assume we are given a kernel expansion \f$ w = \sum_i \alpha_i k(x_i, \cdot) \f$. The KernelBasisDistance takes
/// a new set of basis vectors \f$ z_i \f$ and finds the linear combination in that space which is closest
/// to \f$ w \f$ . More formally the function measures the squared distance in the kernel-induced feature space:
/// \f[ f(z) = \min_{\beta} \frac 1 2 \| \sum_j \beta_j k(z_j, \cdot) - w \|^2 . \f]
/// In vector notation with \f$ (K_x)_{i,j} = k(x_i,x_j) \f$, \f$ (K_z)_{i,j} = k(z_i,z_j) \f$ and \f$ (K_{zx})_{i,j} = k(z_i,x_j) \f$ it computes:
///\f[ f(z) = \min_{\beta} \frac 1 2  \beta^T K_z  \beta - \beta^T K_{zx} \alpha + \frac 1 2 \alpha^TK_x \alpha . \f]
/// The last term is independent of \f$ z_i \f$. Thus it is omitted in the actual computation. That is, the value is offset by a constant and the minimum is not 0.
/// The input of the function consists of a vector which is the concatenation \f$ v=[z_1, z_2,\dots,z_k] \f$ of all basis vectors.
///
/// The target point \f$ w \f$ is set as a KernelExpansion in the constructor. If the kernel is differentiable
/// with respect to the input point then this objective function is differentiable as well.
///
/// The kernel expansion can represent more than one single point, in this case the error is the sum of approximation errors.
class KernelBasisDistance : public SingleObjectiveFunction
{
public:
	/// \brief Constructs the objective function.
	///
	/// This functions calls sparsify on the kernel expansion to save computation time in the case of sparse bases.
	///
	/// \param kernelExpansion a pointer to the kernel expansion to approximate
	/// \param numApproximatingVectors the number of vectors used to approximate the point - the basis size
	SHARK_EXPORT_SYMBOL KernelBasisDistance(KernelExpansion<RealVector>* kernelExpansion,std::size_t numApproximatingVectors = 1);

	/// \brief Returns the name of the class
	std::string name() const
	{ return "KernelBasisDistance"; }

	/// \brief Returns the number of vectors the uses to approximate the point - the basis size
	std::size_t numApproximatingVectors() const{
		return m_numApproximatingVectors;
	}
	
	/// \brief Returns a reference the number of vectors the uses to approximate the point - the basis size
	std::size_t& numApproximatingVectors(){
		return m_numApproximatingVectors;
	}
	/// \brief Returns a starting point of the algorithm
	///
	/// Returns a random subset of the basis of the kernel expansion
	SHARK_EXPORT_SYMBOL void proposeStartingPoint(SearchPointType& startingPoint) const;

	/// \brief Returns the number of variables of the function.
	SHARK_EXPORT_SYMBOL std::size_t numberOfVariables()const;
		
	/// \brief Given an input basis, returns the point with the minimum error.
	SHARK_EXPORT_SYMBOL RealMatrix findOptimalBeta(RealVector const& input)const;

	/// \brief Evaluate the (sum of) squared distance(s) between the closes point in the basis to the point(s) represented by the kernel expansion.
	///
	/// See the class description for more details on this computation.
	SHARK_EXPORT_SYMBOL double eval(RealVector const& input) const;

	/// \brief computes the derivative of the function with respect to the supplied basis.
	///
	/// Assume \f$ \beta \f$ to be the optimal value. Then the derivative with respect to the basis vectors is:
	/// \f[	\frac{ \partial f}{\partial z_l} = \beta_l \sum_i \beta_i \frac{ \partial f}{\partial z_l} k(z_l,z_i) - \beta_l \sum_i \alpha_i \frac{ \partial f}{\partial z_l} k(z_l, x_i) \f]
	SHARK_EXPORT_SYMBOL ResultType evalDerivative( const SearchPointType & input, FirstOrderDerivative & derivative ) const;

private:
	/// \brief Sets up and solves the regression problem for the base z.
	///
	/// calculates K_z, the linear part of the system of equations and solves for beta.
	SHARK_EXPORT_SYMBOL void setupAndSolve(RealMatrix& beta, RealVector const& input, RealMatrix& Kz, RealMatrix& linear)const;

	/// \brief Returns the error of the solution found
	SHARK_EXPORT_SYMBOL double errorOfSolution(RealMatrix const& beta, RealMatrix const& Kz, RealMatrix const& linear)const;

	KernelExpansion<RealVector>* mep_expansion;     ///< kernel expansion to approximate
	std::size_t m_numApproximatingVectors; ///< number of vectors in the basis
};


}
#endif
