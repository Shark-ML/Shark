/*!
 * 
 *
 * \brief       Trust-Region Newton-Step Method
 * 
 * 
 * 
 * 
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
#define SHARK_COMPILE_DLL
#include <shark/Algorithms/GradientDescent/TrustRegionNewton.h>
#include <shark/LinAlg/solveSystem.h>

using namespace shark;

namespace{
	/// \brief Compute the maximal step size given point z, direction, and trust region radius delta.
	double borderDistance(
		RealVector const& z, 
		RealVector const& direction,
		double delta
	){
		double z2=norm_sqr(z);
		double d2=norm_sqr(direction);
		assert(z2 <= d2);   // otherwise z is outside the trust region and the square root will give a NaN

		//find tau such that ||z+tau*d||=delta
		//using p-q formula
		double p = 2*inner_prod(direction,z)/d2;
		double q=(z2-sqr(delta))/d2;

		double tau = p/2 + std::sqrt(sqr(p/2) - q);
		return tau;	
	}

	/// \brief Compute the function value improvement from step, residual, and gradient.
	double errorDifference(RealVector const& step, RealVector const& residual, RealVector const& gradient){
		double diff = inner_prod(residual,step)+inner_prod(gradient,step);
		return diff/2;
	}

	/// \brief CG-Variant that finds an approximate solution inside a trust region.
	///
	/// This variant of CG runs until either the solution is sufficiently accurate
	/// (norm of the gradient below the given tolerance) or the step hits the border
	/// of the trust region.
	///
	/// Returns the improvement in function value and the solution as a pair.
	///
	/// Algorithm 7.2 in Wright, Nocedal: Numerical Optimization
	std::pair<double,RealVector> trustRegionCG(
		RealMatrix const& hessian,
		RealVector gradient,
		double tolerance,   // bound on the norm of the gradient
		double delta        // trust region size (radius)
	){
		RealVector residual = gradient;
		RealVector direction = -gradient;
		std::pair<double,RealVector> solution(0.0,RealVector(residual.size(),0.0));
		RealVector& step = solution.second;
		
		double currentNormRes2 = norm_sqr(residual);
		if( currentNormRes2 <sqr(tolerance))
			return solution;

		for(std::size_t iter = 0; iter != 10*gradient.size(); ++iter ){//numerical safeguard(should never be called)
			RealVector Hdir=prod(hessian,direction);
			double normH=inner_prod(direction, Hdir);
			// if our Hessian is not positive definite then we just run to the boundary
			if(normH <= 0){
				double tau= borderDistance(step,direction,delta);
				noalias(step) += tau*direction;
				noalias(residual) += tau*Hdir;
				solution.first=errorDifference(step,residual,gradient);
				return solution;
			}
			double alpha = currentNormRes2/normH;   // compute step size
			// if the step brings us to the boundary then return the boundary point
			if(norm_sqr(step+alpha*direction) >= sqr(delta)){
				double tau= borderDistance(step,direction,delta);
				noalias(step) += tau*direction;
				noalias(residual) += tau*Hdir;
				solution.first=errorDifference(step,residual,gradient);
				return solution;
			}
			//update point, residual and direction
			noalias(step) +=alpha*direction;
			noalias(residual) +=alpha*Hdir;
			double normRes2 = norm_sqr(residual);

			// check accuracy-based stopping criterion
			if(normRes2 < sqr(tolerance)){
				solution.first=errorDifference(step,residual,gradient);
				return solution;
			}
			double beta = normRes2/currentNormRes2;
			noalias(direction) *= beta;
			noalias(direction) -= residual;
			currentNormRes2 = normRes2;
		}
		return solution; // added by CI
	}
}

TrustRegionNewton::TrustRegionNewton()
{
	m_features |= REQUIRES_VALUE;
	m_features |= REQUIRES_FIRST_DERIVATIVE;
	m_features |= REQUIRES_SECOND_DERIVATIVE;
}

void TrustRegionNewton::init(ObjectiveFunctionType& objectiveFunction, SearchPointType const& startingPoint, double initialDelta) {
	checkFeatures(objectiveFunction);

	m_delta = initialDelta;
	m_minImprovementRatio = 0.1;
	
	m_best.point = startingPoint;
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivatives);
}

void TrustRegionNewton::step(const ObjectiveFunctionType& objectiveFunction) {
	//compute step in approximated sub-system
	double gradNorm_2 = norm_2( m_derivatives.gradient);
	//Computing the forcing schedule for the solution
	//we set the tolerance in terms of the gradient norm, that is epsilon = gamma*||g||. If we set gamma<1 to a constant value
	//the solution found will approach the optimum linearly. more exactly in a quadratic model with exact hessian, the gradient
	//after k iterations will have length gamma^k*||g|| (assuming the trust-region bound is not a problem).
	//thus gamma in a quadratic model is exactly the convergence constant. To obtain superlinear convergence, we have to
	//find a schedule such that gamma ->0. This can be achieved by setting gamma =sqrt(||g||) once the gradient norm is smaller than sqrt(0.5).
	//we choose gamma such that we have linear convergence as long as we are far away from the optimum, (defined as ||g||>=1) 
	//and sqrt(||g||) when we are close. We could use a faster converging schedule, but it is not clear whether this is always good on a 
	//non-quadratic function, even though in the end quadratic convergence is obtained - however getting close enough might take a long time 
	//if we spend too much time on solving the newton steps before we enter this region(which might be extremely small!)
	//The initial guess of 0.5 might be too optimistic and we still spend a lot of time on finding the solution, but this is hugely problem dependent.
	double gamma =std::min(0.5,std::sqrt(gradNorm_2));
	double tolerance = gamma* gradNorm_2;
	std::pair<double,RealVector> solution =  trustRegionCG(m_derivatives.hessian, m_derivatives.gradient, tolerance, m_delta);
	if (solution.first == 0) return;//we are done

	//calculate the function value improvement of the point compared to the model prediction
	double newValue = objectiveFunction(m_best.point + solution.second);
	double functionImprovement = newValue - m_best.value;
	double rho = functionImprovement / solution.first;   // improvement relative to model forecast

	//update trust region size if the observed improvement is large or very small compared to the prediction
	if(rho < 0.25){
		m_delta /= 4;   // shrink trust region size
	}else if(rho > 0.75 && norm_sqr(solution.second) > 0.99*sqr(m_delta)){
		m_delta *= 2;   // grow trust region size
	}

	//accept the point only if the improvement is significant
	if(rho >= m_minImprovementRatio){
		noalias(m_best.point) +=solution.second;
		m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivatives);
	}
}
