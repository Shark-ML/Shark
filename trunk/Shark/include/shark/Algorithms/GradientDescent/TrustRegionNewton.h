/*!
 * 
 *
 * \brief       Trust-Region Newton-Step Method
 *
 * \author      O. Krause
 * \date        2015
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

#ifndef ALGORITHMS_GRADIENTDESCENT_TRUST_REGION_NEWTON_H
#define ALGORITHMS_GRADIENTDESCENT_TRUST_REGION_NEWTON_H

#include <shark/Core/DLLSupport.h>
#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>

namespace shark {

/// \brief Simple Trust-Region method based on the full Hessian matrix
///
/// While normal Newton methods compute the Newton steps and perform a line-search
/// In the Newton direction, trust region methods first choose a maximal step-length and
/// then try to find an approximate best point of the second order tailor expansion in that
/// region. more formally, we solve
/// \f[ \min_{p} m(p) = p^T B p +g^Tp, ||p||<\delta \f]
/// where \f$B\f$ is the Hessian and \f$g\f$ the gradient of the current point \f$x\f$.
/// Given this step, we compute how much the model agrees with the actual function, i.e.
/// \f[ \rho = \frac{ f(x+p)-f(p) }{m(p)} \f]
/// If this value is large, that is, the improvement in function value is approximately as
/// large or larger as the model predicted, we increase \f$\delta\f$ to make larger steps
/// possible, otherwise we decrease it, if the model predicted a much larger improvement
/// than observed - in the worst case, the new point is worse than the old one.
/// 
/// As a further check, to improve convergence, we do not accept every step, but those
/// with \f$ \rho > c > 0 \f$. This ensures that we do not overjump the optimum too much
/// and leads to a better (worst case) convergence rate.
///
/// The optimal step is computed by a conjugate gradient method that stops once a
/// target tolerance is reached, or the step approaches the boundary (which happens,
/// for example, when the Hessian is indefinite or rank-deficient). Thus, computation
/// time is not wasted for steps that are far away from the optimum. The tolerance
/// is set by a forcing-schedule so that accuracy increases in the vicinity of the
/// optimum, enabling solutions with arbitrary precision.
///
/// The algorithm is based on 
/// Jorge Nocedal, Stephen J. Wright
/// Numerical Optimization, 2nd Edition
/// Algorithm 4.1 with Algorithm 7.2 to solve the sub-problem
class TrustRegionNewton : public AbstractSingleObjectiveOptimizer<RealVector >
{
public:
	/// \brief Default constructor.
	SHARK_EXPORT_SYMBOL TrustRegionNewton();

	/// \brief Initialize the iterative optimizer with a problem (objective function) and a starting point.
	///
	/// The initial trust region radius is set to 0.1
	void init(ObjectiveFunctionType& objectiveFunction, SearchPointType const& startingPoint){
		init(objectiveFunction,startingPoint,0.1);
	}
	/// \brief Initialize the iterative optimizer with a problem (objective function), a starting point and an initial value for the trust-region
	SHARK_EXPORT_SYMBOL void init(ObjectiveFunctionType& objectiveFunction, SearchPointType const& startingPoint,double initialDelta);
	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	
	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "TrustRegionNewton"; }

	/// \brief Minimal improvement ratio (see the algorithm details in the class description).
	double minImprovementRatio()const{
		return m_minImprovementRatio;
	}
	
	/// \brief Minimal improvement ratio (see the algorithm details in the class description).
	double& minImprovementRatio(){
		return m_minImprovementRatio;
	}

	/// \brief Perform one trust region Newton step, update point and trust region radius.
	SHARK_EXPORT_SYMBOL void step(ObjectiveFunctionType const& objectiveFunction);

protected:
	double m_delta;                                               ///< Current trust region size
	double m_minImprovementRatio;                                 ///< Minimal improvement ratio (see the algorithm details in the class description).
	ObjectiveFunctionType::SecondOrderDerivative m_derivatives;   ///< First and second derivative of the objective function in the current point.
};
}
#endif
