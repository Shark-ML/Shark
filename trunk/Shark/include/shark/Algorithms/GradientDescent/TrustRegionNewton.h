/*!
 * 
 *
 * \brief       Trust-Region Newton-Step Method
 *
 * \author      O. Krause
 * \date        2010
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

/// \brief Implements a simple Trust-Region method based on the full hessian matrix
///
/// While normal Newton methods compute the newton steps and perform a line-search
/// In the Newton direction, trust region methods first choose a maximal step-length and
/// then try to find an approximate best point of the second order tailor expansion in that
/// region. more formally, we solve
/// \f[ \min_{p} m(p) = p^T B p +g^Tp, ||p||<\delta \f]
/// where \f$B\f$ is the hessian and \f$g\f$ the gradient of the current point \f$x\f$.
/// Given this step, we compute how much the model agrees with the actual function, i.e.
/// \f[ \rho = \frac{ f(x+p)-f(p) }{m(p)} \f]
/// If this value is large, that is, the improvement in function value is approximately as
/// large or larger as the model predicted, we increase \f$\delta\f$ to make larger steps
/// possible, otherwise we decrease it, if the model predicted a much larger improvement
/// then observed - in the worst case, the new point is worse then the old.
/// 
/// As a further check, to improve convergence, we do not accept every step, but those
/// with \f$ \rho > c > 0 \f$. This ensures that we do not overjump the optimum too much
/// and leads to better convergence rate.
///
/// The optimal step is computed by a conjugate gradient method which stops once a
/// target tolerance is reached, or the step approaches the boundary(which for example
/// happens, when the hessian is indefinite or rank-deficient). Thus computation time
/// is not wasted for steps that are far away from the optimum. The tolerance is set
///by a forcing-schedule so that accuracy increases the closer we are to the optimum, thus
/// arbitrary precision is possible.
///
/// The algorithm is based on 
/// Jorge Nocedal, Stephen J. Wright
/// Numerical Optimization, 2nd Edition
/// Algorithm 4.1 with Algorithm 7.2 to solve the sub-problem
class TrustRegionNewton : public AbstractSingleObjectiveOptimizer<RealVector >
{
public:
	SHARK_EXPORT_SYMBOL TrustRegionNewton();
	SHARK_EXPORT_SYMBOL void init(ObjectiveFunctionType& objectiveFunction, SearchPointType const& startingPoint);
	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	
	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "TrustRegionNewton"; }

	double minImprovementRatio()const{
		return m_minImprovementRatio;
	}
	
	double& minImprovementRatio(){
		return m_minImprovementRatio;
	}
	
	SHARK_EXPORT_SYMBOL void step(ObjectiveFunctionType const& objectiveFunction);
protected:
	double m_delta;
	double m_minImprovementRatio;
	ObjectiveFunctionType::SecondOrderDerivative m_derivatives;
};
}
#endif
