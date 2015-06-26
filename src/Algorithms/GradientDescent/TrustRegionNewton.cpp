/*!
 * 
 *
 * \brief       simple Newton step method
 * 
 * 
 * 
 * 
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
 #define SHARK_COMPILE_DLL
#include <shark/Algorithms/GradientDescent/TrustRegionNewton.h>
#include <shark/LinAlg/solveSystem.h>

using namespace shark;

namespace{
	double borderDistance(
		RealVector const& z, 
		RealVector const& direction,
		double delta
	){
		double z2=norm_sqr(z);
		double d2=norm_sqr(direction);
		
		//find tau such that ||z+tau*d||=delta
		//using p-q formula
		double p = 2*inner_prod(direction,z)/d2;
		double q=(z2-sqr(delta))/d2;
		
		double tau = p/2+std::sqrt(sqr(p/2)-q);
		return tau;	
	}
	
	double errorDifference(RealVector const& step, RealVector const& residual, RealVector const& gradient){
		double diff = inner_prod(residual,step)+inner_prod(gradient,step);
		return diff/2;
	}
	
	/// \brief CG-Variant that finds an approximate solution inside a trust region.
	///
	/// This variant of CG runs until either the solution has the right tolerance or the
	/// step hits the border of the trust region.
	///
	/// returns the improvement in function value as well as the solution
	///
	/// Algorithm 7.1 in Wright, Nocedal: Numerical Optimization
	std::pair<double,RealVector> trustRegionCG(
		RealMatrix const& hessian,
		RealVector gradient,
		double tolerance,
		double delta
	){
		RealVector residual = gradient;
		RealVector direction = -gradient;
		std::pair<double,RealVector> solution(0.0,RealVector(residual.size(),0.0));
		RealVector& step = solution.second;
		
		double currentNormRes2 = norm_sqr(residual);
		if( currentNormRes2 <sqr(tolerance))
			return solution;
		
		while(true){
			RealVector Hdir= prod(hessian,direction);
			double normH=inner_prod(direction, Hdir);
			if(normH <= 0){//we are not positive definite, so we just run to the boundary
				double tau= borderDistance(step,direction,delta);
				noalias(step) += tau*direction;
				noalias(residual) += tau*Hdir;
				solution.first=errorDifference(step,residual,gradient);
				return solution;
			}
			double alpha = currentNormRes2/normH;
			//step brings us to the boundary, just return the boundary point
			if(norm_sqr(step+alpha*direction) >= sqr(delta)){
				double tau= borderDistance(step,direction,delta);
				noalias(step) += tau*direction;
				noalias(residual) += tau*Hdir;
				solution.first=errorDifference(step,residual,gradient);
				return solution;
			}
			//update point, residuals and direction
			noalias(step) +=alpha*direction;
			noalias(residual) +=alpha*Hdir;
			double normRes2 = norm_sqr(residual);
			
			if(normRes2 < sqr(tolerance)){
				solution.first=errorDifference(step,residual,gradient);
				return solution;
			}
			double beta = normRes2/currentNormRes2;
			noalias(direction) *= beta;
			noalias(direction) -= residual;
			currentNormRes2 = normRes2;
			
		}
	}
}

TrustRegionNewton::TrustRegionNewton()
{
	m_features |= REQUIRES_VALUE;
	m_features |= REQUIRES_FIRST_DERIVATIVE;
	m_features |= REQUIRES_SECOND_DERIVATIVE;
}
void TrustRegionNewton::init(ObjectiveFunctionType& objectiveFunction, SearchPointType const& startingPoint) {
	checkFeatures(objectiveFunction);
	objectiveFunction.init();
	
	m_delta = 0.1;
	m_minImprovementRatio = 0.1;
	
	m_best.point = startingPoint;
	m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivatives);
}


void TrustRegionNewton::step(const ObjectiveFunctionType& objectiveFunction) {
	//compute step in approximated sub-system
	double gradNorm=norm_2( m_derivatives.gradient);
	double tolerance = std::min(0.5,std::sqrt(gradNorm))*gradNorm;
	std::pair<double,RealVector> solution =  trustRegionCG(m_derivatives.hessian, m_derivatives.gradient, tolerance,m_delta);
	if(solution.first == 0)
		return;//we are done
	//calculate the function value improvement of the point compared to the model prediction
	double newValue = objectiveFunction(m_best.point + solution.second);
	double functionImprovement = newValue-m_best.value;
	double rho = functionImprovement/solution.first;
	//update step-size if the observed improvment is large or very small compared to the prediction
	if(rho < 0.25){
		m_delta /=4;
	}else if(rho > 0.75 && norm_sqr(solution.second) > 0.99*sqr(m_delta)){
		m_delta *= 2;
	}
	
	//accept the point only, if the improvement is better then some value
	if(rho >= m_minImprovementRatio){
		noalias(m_best.point) +=solution.second;
		m_best.value = objectiveFunction.evalDerivative(m_best.point,m_derivatives);
	}
	//~ std::cout<<m_best.value<<" "<< functionImprovement<<" "<<solution.first<<" "<<rho<<" "<<norm_2(solution.second)<<" "<<m_delta<<std::endl;

}