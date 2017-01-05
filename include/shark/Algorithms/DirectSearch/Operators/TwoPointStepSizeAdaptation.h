/*!
 * \brief       Implements a two point step size adaptation rule based on a line-search type of approach
 *
 * \author    O.Krause
 * \date        2014
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_TWO_POINT_STEP_SIZE_ADAPTATION_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_TWO_POINT_STEP_SIZE_ADAPTATION_H

#include <shark/LinAlg/Base.h>
#include <cmath>

namespace shark {

/// \brief Step size adaptation based on the success of the new population compared to the old
///
/// This is the step size adaptation algorithm as proposed in 
/// Ilya Loshchilov, "A Computationally Efficient Limited Memory CMA-ES for Large Scale Optimization"
///
/// It ranks the old and new population together and checks whether the mean rank of the new population
/// is lower than the old one in this combined population. If this is true, the step size is increased
/// in an exponential fashion. More formally, let \f$ r_t(i) \f$ be the rank of the i-th individual in the 
/// current population in the combined ranking and 	\f$ r_{t-1}(i) \f$ the rank of the i-th previous
/// individual. Then we have
/// \f[ z_t \leftarrow \frac 1 {\lamba^2} \sum_i^{\lambda} r_{t-1}(i) - r_t(i) - z*\f]
/// where \f$ z* \f$ is a target success value, which defaults to 0.25
/// this statistic is stabilised using an exponential average:
/// \f[ s_t \leftarrow (1-c)*s_{t-1} + c*z_t \f]
/// where the learning rate c defaults to 0.3
/// finally we adapt the step size sigma by
/// \f[ \sigma_t = \sigma_{t-1} exp(s_t/d) \f]
/// where the damping factor d defaults to 1
class TwoPointStepSizeAdaptation{
public:
	TwoPointStepSizeAdaptation():m_cAlpha(0.1), m_alphaStep(0.5){}
	
	double stepSize()const{
		return m_stepSize;
	}
	
	void setAlphaStep(double alphaStep){
		m_alphaStep = alphaStep;
	}
	
	void setLearningRate(double learningRate){
		m_cAlpha = learningRate;
	}
	
	///\brief Initializes a new trial by setting the initial step size and resetting the internal values.
	void init(double initialStepSize){
		m_stepSize = initialStepSize;
		m_alpha = 0;
	}
	
	void setStepSize(double stepSize){
		m_stepSize = stepSize;
	}
	
	/// \brief updates the step size using the newly sampled population
	///
	/// The offspring is assumed to be ordered in ascending order by their penalizedFitness
	/// (this is the same as ordering by the unpenalized fitness in an unconstrained setting)
	void update(SingleObjectiveFunction const& f, RealVector const& point,RealVector const& direction){
		double fminus = f.eval(point +m_alphaStep * m_stepSize * direction);
		double fplus = f.eval(point + (1+m_alphaStep) * m_stepSize * direction);
		
		//~ double alphaCurrent = fminus < fplus? -m_alphaStep : m_alphaStep; 
		double alphaCurrent = fminus < fplus? std::log(m_alphaStep) : std::log(1+m_alphaStep); 
		m_alpha= (1 - m_cAlpha) * m_alpha + m_cAlpha * alphaCurrent;
		m_stepSize *= std::exp(m_alpha);
	}
private:
	double m_stepSize;///< currnt value for the step size
	double m_alpha; ///< The time average of the successesful steps

	//hyper parameters
	double m_cAlpha;
	double m_alphaStep;
};

}

#endif
