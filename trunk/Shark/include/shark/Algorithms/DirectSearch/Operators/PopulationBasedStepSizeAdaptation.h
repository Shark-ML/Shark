/*!
 * \brief       Implements the tep size adaptation based on the success of the new population compared to the old
 *
 * \author    O.Krause
 * \date        2014
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_POPULATION_BASED_STEP_SIZE_ADAPTATION_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_OPERATORS_POPULATION_BASED_STEP_SIZE_ADAPTATION_H

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
class PopulationBasedStepSizeAdaptation{
public:
	PopulationBasedStepSizeAdaptation():m_targetSuccessRate(0.25), m_c(0.3), m_d(1.0){}
	
	/////Getter and Setter functions/////////
		
	double targetSuccessRate()const{
		return m_targetSuccessRate;
	}
	
	double& targetSuccessRate(){
		return m_targetSuccessRate;
	}
	
	double learningRate()const{
		return m_c;
	}
	
	double& learningRate(){
		return m_c;
	}
	
	double dampingFactor()const{
		return m_d;
	}
	
	double& dampingFactor(){
		return m_d;
	}
	
	double stepSize()const{
		return m_stepSize;
	}
	
	///\brief Initializes a new trial by setting the initial learning rate and resetting the internal values.
	void init(double initialStepSize){
		m_stepSize = initialStepSize;
		m_s = 0;
		m_prevFitness.resize(0);
	}
	
	/// \brief updates the step size using the newly sampled population
	///
	/// The offspring is assumed to be ordered in ascending order by their penalizedFitness
	/// (this is the same as ordering by the unpenalized fitness in an unconstrained setting)
	template<class Population>
	void update(Population const& offspring){
		std::size_t lambda = offspring.size();
		if (m_prevFitness.size() == lambda){
			//get estimate of z
			std::size_t indexOld = 0;
			std::size_t indexNew = 0;
			std::size_t rank = 1;
			double z =  0;
			while(indexOld < lambda && indexNew < lambda){
				if (offspring[indexNew].penalizedFitness() <= m_prevFitness[indexOld]){
					z-=rank;
					++indexNew;
				}
				else{
					z+=rank;
					++indexOld;
				}
				++rank;
			}
			//case 1: the worst elements in the old population are better than the worst in the new
			while(indexNew < lambda){
				z-=rank;
				++indexNew;
				++rank;
			}
			//case 2: the opposite
			while(indexOld< lambda){
				z += rank;
				++indexOld;
				++rank;
			}
			z /= lambda*lambda;
			z -= m_targetSuccessRate;
			m_s = (1-m_c)*m_s +m_c*z;
			m_stepSize *= std::exp(m_s/m_d);
		}
		
		//store fitness values of last iteration
		m_prevFitness.resize(lambda);
		for(std::size_t i = 0; i != lambda; ++i)
			m_prevFitness(i) = offspring[i].penalizedFitness();
	}
private:
	double m_stepSize;///< currnt value for the step size
	RealVector m_prevFitness;///< fitness values of the previous iteration for ranking
	double m_s; ///< The time average of the population success

	//hyper parameters
	double m_targetSuccessRate;
	double m_c;
	double m_d;
};

}

#endif