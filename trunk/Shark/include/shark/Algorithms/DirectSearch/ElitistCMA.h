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
//===========================================================================


#ifndef SHARK_EA_ELITIST_CMA_H
#define SHARK_EA_ELITIST_CMA_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Core/Exception.h>
#include <shark/Core/Flags.h>

#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>
#include <shark/Algorithms/DirectSearch/TypedIndividual.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/Operators/Initializers/CovarianceMatrixInitializer.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/GlobalIntermediateRecombination.h>

#include <boost/math/special_functions.hpp>

namespace shark {
	namespace elitist_cma {

		/**
		* \brief Chromosome of the elitist CMA-ES.
		*/
		struct Chromosome {

			/**
			* \brief Default c'tor.
			*/
			Chromosome( unsigned int dimension = 0 ) : 
				m_fitnessUpdateFrequency( 5 ),
                                m_fitness( std::numeric_limits<double>::max() ),
				m_targetSuccessProbability( 0 ),
				m_successProbability( 0 ),
				m_cSuccessProb( 0 ),
				m_sigma( 0 ),
				m_cC( 0 ),
				m_cCov( 0 ),
				m_cCovMinus( 0 ),
				m_cSigma( 0 ),
				m_dSigma( 0 ) {
			}

			/**
			* \brief Adjusts the dimension of the chromosome.
			*/
			void setDimension( unsigned int dimension ) {
				m_mean.resize( dimension);
				m_evolutionPathC.resize( dimension);

				m_mutationDistribution.resize( dimension );
			}

			/**
			* \brief Serializes the chromosome to the supplied archive.
			* \tparam Archive Type of the archive.
			* \param [in, out] archive Object of type archive.
			* \param [in] version Currently unused.
			*/
			template<typename Archive>
			void serialize( Archive & archive, const unsigned int version ) {
				archive & m_fitness;
				archive & m_fitnessUpdateFrequency;
				archive & m_generationCounter;
				archive & m_targetSuccessProbability;
				archive & m_successProbability;
				archive & m_cSuccessProb;
				archive & m_sigma;

				archive & m_cC;
				archive & m_cCov;
				archive & m_cCovMinus;
				archive & m_cSigma;
				archive & m_dSigma;

				archive & m_mean;
				archive & m_evolutionPathC;

				archive & m_mutationDistribution;
			}

			/**
			* \brief Prints the chromosome for debugging purposes.
			*/
			template<typename Stream>
			void print( Stream & s ) const {
				s << "Fitness: " << m_fitness << std::endl;
				s << "Target Success Prob.: " << m_targetSuccessProbability << std::endl;
				s << "Success Prob.: " << m_successProbability << std::endl;
				s << "cSuccProb: " << m_cSuccessProb;
				s << "Sigma: " << m_sigma << std::endl;
				s << "cc: " << m_cC << std::endl;
				s << "ccov: " << m_cCov << std::endl;
				s << "ccov-: " << m_cCovMinus << std::endl;
				s << "csigma: " << m_cSigma << std::endl;
				s << "dsigma: " << m_dSigma << std::endl;
				s << "mean: " << m_mean << std::endl;
				s << "pc: " << m_evolutionPathC << std::endl;	
				m_mutationDistribution.print( s );
			}

			/** \cond */
			double m_lastFitness;
			unsigned int m_fitnessUpdateFrequency;
			unsigned int m_generationCounter;

			double m_fitness;
			double m_targetSuccessProbability;
			double m_successProbability;
			double m_cSuccessProb;
			double m_sigma;
			double m_cC;
			double m_cCov;
			double m_cCovMinus;
			double m_cSigma;
			double m_dSigma;

			RealVector m_mean;

			RealVector m_evolutionPathC;

			shark::MultiVariateNormalDistribution m_mutationDistribution;  
			/** \endcond */
		};

	}

	/**
	* \brief Implements the elitist CMA-ES.
	*
	*  The algorithm is described in
	*
	*  Hansen, N., S. Kern (2004). Evaluating the CMA Evolution Strategy
	*  on Multimodal Test Functions. In Proceedings of the Eighth
	*  International Conference on Parallel Problem Solving from Nature
	*  (PPSN VIII), pp. 282-291, LNCS, Springer-Verlag
	*/
	class ElitistCMA 
	: public AbstractSingleObjectiveOptimizer<VectorSpace<double> >{	    

		/** \brief Models the individual type of the elitist CMA-ES. */
		typedef TypedIndividual< RealVector > Individual;

		/** \brief Default success probability threshold */
		static const double SUCCESS_PROBABILITY_THRESHOLD;
		
		/** \brief Default target success probability */
		static const double TARGET_SUCCESS_PROBABILITY;

		/** \cond */
		struct FitnessComparator {
			template<typename Individual>
			bool operator()( const Individual & a, const Individual & b ) const {
				return( a.fitness( shark::tag::PenalizedFitness() )[0] < b.fitness( shark::tag::PenalizedFitness() )[0] );
			}
		};

		struct IdentityExtractor {
			template<typename T>
			const T & operator()( const T & t ) const {
				return( t );
			}
		};

	public:

		struct SigmaExtractor {
			double operator()( const ElitistCMA & cma ) const {
				return( cma.m_chromosome.m_sigma );
			}
		};
		/** \endcond */

		//~ /**
		//~ * \brief Models the features supported by the algorithms.
		//~ */
		//~ enum Feature {
			//~ ACTIVE_COVARIANCE_MATRIX_UPDATE = 0
		//~ };

		//~ typedef shark::TypedFlags< Feature > Features;

		/**
		* \brief Default c'tor.
		*/
		ElitistCMA() : m_lambda( 1 ),m_activeUpdate(false) {
			m_features |= REQUIRES_VALUE;
		}

		/// \brief From INameable: return the class name.
		std::string name() const
		{ return "ElitistCMA"; }

		/**
		* \brief Calculates the expected length of a vector of length n.
		* \param [in] n The length of the vector.
		*/ 
		static double chi( unsigned int n ) {
			return( std::sqrt( static_cast<double>( n ) )*(1. - 1./(4.*n) + 1./(21.*n*n)) );
		}

		void configure( const PropertyTree & node ) {
		}

		template<typename InArchive>
		void read( InArchive & archive ) {
			archive >> m_numberOfVariables;
			archive >> m_mu;
			archive >> m_lambda;
			archive >> m_chromosome;
		}

		template<typename OutArchive>
		void write( OutArchive & archive ) const {
			archive << m_numberOfVariables;
			archive << m_mu;
			archive << m_lambda;
			archive << m_chromosome;
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

			m_numberOfVariables = p.size();

			m_chromosome.m_generationCounter = 0;

			m_chromosome.m_evolutionPathC = blas::repeat( 0.0,m_numberOfVariables );
			m_chromosome.m_mean = blas::repeat( 0.0,m_numberOfVariables );
			m_chromosome.m_mutationDistribution.resize( m_numberOfVariables );

			
			//~ double d[3];
			//~ d[0] = blas::norm_2( initialPoints[1] - initialPoints[0] );
			//~ d[1] = blas::norm_2( initialPoints[2] - initialPoints[0] );
			//~ d[2] = blas::norm_2( initialPoints[2] - initialPoints[1] );
			//~ std::sort( d, d+3 );

			//~ m_chromosome.m_sigma = std::max( 1.0, d[1] );
			m_chromosome.m_sigma = 1.0;
			
			m_chromosome.m_dSigma = 1. + m_numberOfVariables/(2.*m_lambda);

			m_chromosome.m_targetSuccessProbability = 1./(5.+::sqrt( static_cast<double>( m_lambda ) )/2.);
			m_chromosome.m_cSuccessProb = (m_chromosome.m_targetSuccessProbability*m_lambda)/(2+m_chromosome.m_targetSuccessProbability*m_lambda);

			m_chromosome.m_cC = 2./(m_numberOfVariables + 2.);
			m_chromosome.m_cCov = 2./(boost::math::pow<2>(m_numberOfVariables) + 6);
			m_chromosome.m_cCovMinus = 0.4/( ::pow(m_numberOfVariables, 1.6 )+1. );

			m_chromosome.m_mean = p;
			//~ m_chromosome.m_mean = initialPoints[0];
			m_chromosome.m_fitness = function( m_chromosome.m_mean );
			m_chromosome.m_lastFitness = m_chromosome.m_fitness;

		}

		/**
		* \brief Updates the covariance matrix of the strategy.
		* \param [in] offspring Individual to update the covariance matrix for.
		*/
		void updateCovarianceMatrix( const Individual & offspring ) {
			if( m_chromosome.m_successProbability < ElitistCMA::SUCCESS_PROBABILITY_THRESHOLD ) {
				m_chromosome.m_evolutionPathC = (1-m_chromosome.m_cC)*m_chromosome.m_evolutionPathC + 
					::sqrt( m_chromosome.m_cC*(2-m_chromosome.m_cC) ) * 1./m_chromosome.m_sigma * (*offspring - m_chromosome.m_mean);
				RealMatrix & C = m_chromosome.m_mutationDistribution.covarianceMatrix();
				C = (1. - m_chromosome.m_cCov) * C + m_chromosome.m_cCov * blas::outer_prod( m_chromosome.m_evolutionPathC, m_chromosome.m_evolutionPathC );
			} else {
				m_chromosome.m_evolutionPathC = (1-m_chromosome.m_cC)*m_chromosome.m_evolutionPathC;		  
				RealMatrix & C = m_chromosome.m_mutationDistribution.covarianceMatrix();
				C = (1. - m_chromosome.m_cCov) * C + 
					m_chromosome.m_cCov * (
					blas::outer_prod( m_chromosome.m_evolutionPathC, m_chromosome.m_evolutionPathC ) + 
					m_chromosome.m_cC*(2.-m_chromosome.m_cC)*C);		    
			}
			m_chromosome.m_mutationDistribution.update();
		}

		/**
		* \brief Updates the step size based on the current success probability.
		*/
		void updateStepSize() {
			m_chromosome.m_sigma *= ::exp( 1./m_chromosome.m_dSigma * (m_chromosome.m_successProbability - m_chromosome.m_targetSuccessProbability)/(1-m_chromosome.m_targetSuccessProbability) );
		}


		/**
		* \brief Purges the information contributed by the individual from the strategy's covariance matrix.
		* \param [in] offspring The offspring to purge from the strategy's covariance matrix.
		*/
		void activeCovarianceMatrixUpdate( const Individual & offspring ) {
			RealVector z = (*offspring - m_chromosome.m_mean)/m_chromosome.m_sigma;
			if( 1 - boost::math::pow<2>( blas::norm_2( z ) ) * m_chromosome.m_cCovMinus/(1+m_chromosome.m_cCovMinus) <= 0 )
				return;
			RealMatrix & C = m_chromosome.m_mutationDistribution.covarianceMatrix();
			C = (1. - m_chromosome.m_cCovMinus) * C - m_chromosome.m_cCovMinus * blas::outer_prod( z, z );
			m_chromosome.m_mutationDistribution.update();

		}

		/**
		* \brief Updates the strategy parameters based on the supplied offspring population.
		*/
		void updateStrategyParameters( const Individual & offspring, bool successful ) {
			m_chromosome.m_successProbability = (1.-m_chromosome.m_cSuccessProb)*m_chromosome.m_successProbability +
			m_chromosome.m_cSuccessProb * (successful ? 1 : 0);

			// Covariance Matrix update
			if( successful )
				updateCovarianceMatrix( offspring );
			else if( offspring.fitness( tag::PenalizedFitness() )[0] > m_chromosome.m_lastFitness  
				&& m_activeUpdate )
				activeCovarianceMatrixUpdate( offspring );

			updateStepSize();

			if( successful ) {
				m_chromosome.m_mean = *offspring;
				m_chromosome.m_fitness = offspring.fitness( tag::PenalizedFitness() )[ 0 ];
			}

			if( m_chromosome.m_generationCounter % m_chromosome.m_fitnessUpdateFrequency == 0 )
				m_chromosome.m_lastFitness = m_chromosome.m_fitness;

			m_chromosome.m_generationCounter++;

		}

		/**
		* \brief Executes one iteration of the algorithm.
		*/
		void step(ObjectiveFunctionType const& function) {

			std::vector< Individual > offspring( m_lambda );

			shark::soo::PenalizingEvaluator evaluator;
			for( unsigned int i = 0; i < offspring.size(); i++ ) {
				MultiVariateNormalDistribution::ResultType sample = m_chromosome.m_mutationDistribution();
				*(offspring[i]) = m_chromosome.m_mean + m_chromosome.m_sigma * sample.first;
				boost::tuple< ObjectiveFunctionType::ResultType, ObjectiveFunctionType::ResultType > evalResult;
				evalResult = evaluator( function, *offspring[i] );

				offspring[i].fitness( shark::tag::UnpenalizedFitness() )[0] = boost::get< shark::soo::PenalizingEvaluator::UNPENALIZED_RESULT >( evalResult );
				offspring[i].fitness( shark::tag::PenalizedFitness() )[0] = boost::get< shark::soo::PenalizingEvaluator::PENALIZED_RESULT >( evalResult );		    

			}

			// Selection
			FitnessComparator comparator;
			std::sort( offspring.begin(), offspring.end(), comparator );

			updateStrategyParameters( offspring[ 0 ], offspring[0].fitness(shark::tag::PenalizedFitness())[0] < m_chromosome.m_fitness );

			m_best.point = *offspring.front();
			m_best.value = offspring.front().fitness(shark::tag::UnpenalizedFitness())[0] ;
		}

		const shark::elitist_cma::Chromosome & chromosome() const {
			return m_chromosome;
		}

		unsigned int lambda() const {
			return m_lambda;
		}

		unsigned int & lambda() {
			return m_lambda;
		}
		
		bool usesActiveUpdate(){
			return m_activeUpdate;
		}
		
		void setActiveUpdate(bool update){
			m_activeUpdate = update;
		}

		//~ const Features & features() const {
			//~ return m_features;
		//~ }

		//~ Features & features() {
			//~ return m_features;
		//~ }

	protected:
		unsigned int m_numberOfVariables; ///< Stores the dimensionality of the search space.
		unsigned int m_mu; ///< The size of the parent population.
		unsigned int m_lambda; ///< The size of the offspring population, defaults to one.
		Features m_features; ///< Flags that indicate the features activated for this instance of the algorithm.
		bool m_activeUpdate;///< Should the matrix be updated for non-successful offspring which is better than the previous?

		shark::elitist_cma::Chromosome m_chromosome;  ///< Stores the strategy parameters of the algorithm.
	};      

	const double ElitistCMA::SUCCESS_PROBABILITY_THRESHOLD = 0.44;
}

#endif
