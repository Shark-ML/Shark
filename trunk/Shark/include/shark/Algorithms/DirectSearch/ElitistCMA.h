//===========================================================================
/*!
 * 
 *
 * \brief       Implements the most recent version of the elitist CMA-ES.
 * 
 * The algorithm is based on
 * 
 * C. Igel, T. Suttorp, and N. Hansen. A Computational Efficient
 * Covariance Matrix Update and a (1+1)-CMA for Evolution
 * Strategies. In Proceedings of the Genetic and Evolutionary
 * Computation Conference (GECCO 2006), pp. 453-460, ACM Press, 2006
 * 
 * D. V. Arnold and N. Hansen: Active covariance matrix adaptation for
 * the (1+1)-CMA-ES. In Proceedings of the Genetic and Evolutionary
 * Computation Conference (GECCO 2010): pp 385-392, ACM Press 2010
 * 
 *
 * \author      O. Krause T.Voss
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
//===========================================================================


#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_ELITIST_CMA_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_ELITIST_CMA_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Algorithms/DirectSearch/CMA/CMAIndividual.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>

namespace shark {

/**
* \brief Implements the elitist CMA-ES.
* 
* The algorithm is based on
* 
* C. Igel, T. Suttorp, and N. Hansen. A Computational Efficient
* Covariance Matrix Update and a (1+1)-CMA for Evolution
* Strategies. In Proceedings of the Genetic and Evolutionary
* Computation Conference (GECCO 2006), pp. 453-460, ACM Press, 2006
* 
* D. V. Arnold and N. Hansen: Active covariance matrix adaptation for
* the (1+1)-CMA-ES. In Proceedings of the Genetic and Evolutionary
*/
class ElitistCMA : public AbstractSingleObjectiveOptimizer<RealVector >{	    
public:

	ElitistCMA();

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ElitistCMA"; }

	void configure( const PropertyTree & node ) {}

	void read( InArchive & archive );

	void write( OutArchive & archive ) const;

	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	
	/// \brief Initializes the algorithm for the supplied objective function.
	void init( ObjectiveFunctionType const& function, SearchPointType const& p);

	///\brief Executes one iteration of the algorithm.
	void step(ObjectiveFunctionType const& function);
	
	/// \brief Returns true when the active update is used (default true).
	bool activeUpdate()const{
		return m_activeUpdate;
	}
	/// \brief Setter function to enable active update. Returns true when the active update is used (default true).
	bool& activeUpdate(){
		return m_activeUpdate;
	}
	
	/// \brief Returns the penalty factor for an individual that is outside the feasible area.
	///
	/// The value is multiplied with the distance to the nearest feasible point.
	double constrainedPenaltyFactor()const{
		return m_evaluator.m_penaltyFactor;
	}
	
	/// \brief Returns a reference to the penalty factor for an individual that is outside the feasible area.
	///
	/// The value is multiplied with the distance to the nearest feasible point.
	double& constrainedPenaltyFactor(){
		return m_evaluator.m_penaltyFactor;
	}
	
	/// \brief  Returns the current step length
	double sigma()const{
		return m_individual.chromosome().m_stepSize;
	}
	
	/// \brief  Returns the current step length
	double& sigma(){
		return m_individual.chromosome().m_stepSize;
	}

private:
	CMAIndividual<double> m_individual;///< Individual holding strategy parameter. usd as parent and offspring
	PenalizingEvaluator m_evaluator;///< evaluates the fitness of the individual and handles constraints
	std::vector<double> m_ancestralFitness; ///< stores the last k fitness values (by default 5).
	bool m_activeUpdate;///< Should bad individuals be actively purged from the strategy?
};
}

#endif
