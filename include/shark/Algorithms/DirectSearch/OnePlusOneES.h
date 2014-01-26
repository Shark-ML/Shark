/*!
 * 
 *
 * \brief       Implements the (1+1)-ES.
 * 
 * The algorithm is described in
 * 
 * http://www.scholarpedia.org/article/Evolution_strategies
 * 
 * 
 * 
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_EA_ONE_PLUS_ONE_ES_H
#define SHARK_EA_ONE_PLUS_ONE_ES_H

#include <shark/Core/Exception.h>
#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>
#include <shark/Algorithms/DirectSearch/TypedIndividual.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>

#include <boost/lambda/lambda.hpp>
#include <boost/math/special_functions.hpp>

namespace shark {

	namespace one_plus_one_es {
		/**
		* \brief Models the update strategy of the step size.
		*/
		enum SigmaUpdateStrategy {
			SELF_ADAPTATION_SIGMA_UPDATE,
			ONE_FIFTH_UPDATE,
			SYMMETRIC_ONE_FIFTH_UDPATE
		};
	}

/**
 * \brief Implements the (1+1)-ES.
 *
 * The algorithms is described in:
 *
 * http://www.scholarpedia.org/article/Evolution_strategies
 */
class OnePlusOneES : public AbstractSingleObjectiveOptimizer<VectorSpace<double> >{	    
	/**
	* \brief The individual type of the (1+1)-ES.
	*/
	typedef TypedIndividual< RealVector,double,double > Individual;

	/**
	* \brief Samples a random unit vector from the mutation distribution of the (1+1)-ES.
	*/
	RealVector random_unit_vector() {				
		RealVector v = m_mvn().first;
		return( v / blas::norm_2( v ) );
	}

public:
	/**
	* \brief Default c'tor.
	*/
	OnePlusOneES() : m_updateStrategy( one_plus_one_es::ONE_FIFTH_UPDATE ) {
		m_features |= REQUIRES_VALUE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "OnePlusOneES"; }
	
	void configure( const PropertyTree & node ) {
	}

	template<typename InArchive>
	void read( InArchive & archive ) {
		archive >> m_parent;
	}

	template<typename OutArchive>
	void write( OutArchive & archive ) const {
		archive << m_parent;
	}

	using AbstractSingleObjectiveOptimizer<VectorSpace<double> >::init;
	/**
	* \brief Initializes the algorithm for the supplied objective function.
	*/
	
	void init( ObjectiveFunctionType const& function, SearchPointType const& p){
		//~ if( !(function.features().test( ObjectiveFunctionType::CAN_PROPOSE_STARTING_POINT ) ) )
			//~ throw( shark::Exception( "Fitness Function does not propose starting point.", __FILE__, __LINE__ ) );
		//~ RealVector initialPoints[3];
		//~ function.proposeStartingPoint( initialPoints[0] );
		//~ function.proposeStartingPoint( initialPoints[1] );
		//~ function.proposeStartingPoint( initialPoints[2] );
		//~ double d[3];
		//~ d[0] = blas::norm_2( initialPoints[1] - initialPoints[0] );
		//~ d[1] = blas::norm_2( initialPoints[2] - initialPoints[0] );
		//~ d[2] = blas::norm_2( initialPoints[2] - initialPoints[1] );
		//~ std::sort( d, d+3 );

		m_mvn.resize( p.size() );

		//~ *m_parent = initialPoints[0];
		*m_parent = p;
		//~ m_parent.get<0>() = d[1];
		m_parent.get<0>() = 1.0;

		m_logStepSizeShift = 0.0;
		m_logStepSizeStdDev = 0.5;
		
		shark::soo::PenalizingEvaluator evaluator;
		boost::tuple<ResultType,ResultType > evalResult = evaluator( function, *m_parent );

		m_parent.fitness( shark::tag::UnpenalizedFitness() )[0] = boost::get< shark::soo::PenalizingEvaluator::UNPENALIZED_RESULT >( evalResult );
		m_parent.fitness( shark::tag::PenalizedFitness() )[0] = boost::get< shark::soo::PenalizingEvaluator::PENALIZED_RESULT >( evalResult );
	}

	/**
	* \brief Updates the step-size according to the selected update scheme.
	*/
	void updateStepSize( bool successful ) {
		switch( m_updateStrategy ) {
			
			case one_plus_one_es::SELF_ADAPTATION_SIGMA_UPDATE: {
				break;
			}
			
			case one_plus_one_es::ONE_FIFTH_UPDATE: {
				m_parent.get<0>() *= (successful ? ::exp( 0.8 ) : -0.2); 
				break;
			}
			case one_plus_one_es::SYMMETRIC_ONE_FIFTH_UDPATE: {
				if( successful )
					std::swap( m_parent.get<0>(), m_parent.get<1>() );
					
				m_logStepSizeShift = successful ? 0.0 : m_logStepSizeShift + 0.01;
				m_logStepSizeStdDev = successful ? 0.5 : m_logStepSizeStdDev + 0.01;
				break;
			}
			
		}
	}

	/**
	* \brief Executes one iteration of the algorithm.
	*/

	void step(const ObjectiveFunctionType& function) {

		Individual offspring = sampleOffspring( m_parent );
		 
		shark::soo::PenalizingEvaluator evaluator;
		boost::tuple<ResultType, ResultType > evalResult = evaluator( function, *offspring );

		offspring.fitness( shark::tag::UnpenalizedFitness() )[0] = boost::get< shark::soo::PenalizingEvaluator::UNPENALIZED_RESULT >( evalResult );
		offspring.fitness( shark::tag::PenalizedFitness() )[0] = boost::get< shark::soo::PenalizingEvaluator::PENALIZED_RESULT >( evalResult );

		if( offspring.fitness( shark::tag::PenalizedFitness() )[0] < m_parent.fitness( shark::tag::PenalizedFitness() )[0] )
			std::swap( m_parent, offspring );

		updateStepSize( offspring.fitness( shark::tag::PenalizedFitness() )[0] < m_parent.fitness( shark::tag::PenalizedFitness() )[0] );
		m_best.point = *m_parent;
		m_best.value = m_parent.fitness(shark::tag::UnpenalizedFitness())[0] ;
	}

	/** \cond */
	Individual sampleOffspring( const Individual & parent ) {
		Individual result( parent );
		
		switch( m_updateStrategy ) {
			
			case one_plus_one_es::SELF_ADAPTATION_SIGMA_UPDATE: {
					double tau = ::sqrt(0.5 / (*parent).size());
					double gauss = Rng::gauss();
					result.get<0>() *= ::exp(tau * gauss);
					*result = result.get<0>() * m_mvn().first;
				break;
			}
			
			case one_plus_one_es::ONE_FIFTH_UPDATE:
				*result = parent.get<0>() * m_mvn().first;
			break;
			
			case one_plus_one_es::SYMMETRIC_ONE_FIFTH_UDPATE: {
				double ls = ::log( parent.get<0>() ) + Rng::gauss( 0.0, m_logStepSizeStdDev );
				if (Rng::coinToss()) 
					ls += m_logStepSizeShift; 							
				else 
					ls -= m_logStepSizeShift;
				double s = ::exp(ls);
				*result = s * OnePlusOneES::random_unit_vector();
				result.get<1>() = s;
				break;
			}
			
		}
		return result;
	}

	shark::one_plus_one_es::SigmaUpdateStrategy updateStrategy() const {
		return( m_updateStrategy );
	}

	shark::one_plus_one_es::SigmaUpdateStrategy & updateStrategy() {
		return( m_updateStrategy );
	}
	/** \endcond */

protected:
	shark::one_plus_one_es::SigmaUpdateStrategy m_updateStrategy; ///< Stores the selected update strategy.	

	Individual m_parent; ///< The current parent individual.
	double m_logStepSizeShift;
	double m_logStepSizeStdDev;

	shark::MultiVariateNormalDistribution m_mvn; ///< Models the search distribution.
};

/** \brief Registers the (1+1)-ES with the factory. */
ANNOUNCE_SINGLE_OBJECTIVE_OPTIMIZER( OnePlusOneES, soo::RealValuedSingleObjectiveOptimizerFactory );
}

#endif
