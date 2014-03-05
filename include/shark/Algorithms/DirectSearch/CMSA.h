//===========================================================================
/*!
 * 
 *
 * \brief       Implements the CMSA.
 * 
 * The algorithm is described in
 * 
 * H. G. Beyer, B. Sendhoff (2008). 
 * Covariance Matrix Adaptation Revisited – The CMSA Evolution Strategy –
 * In Proceedings of the Tenth International Conference on Parallel Problem Solving from Nature
 * (PPSN X), pp. 123-132, LNCS, Springer-Verlag
 * 
 * \par Copyright (c) 1998-2008:
 * Institut f&uuml;r Neuroinformatik
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


#ifndef SHARK_EA_CMSA_H
#define SHARK_EA_CMSA_H

#include <shark/Algorithms/AbstractOptimizer.h>
#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Algorithms/DirectSearch/TypedIndividual.h>

#include <shark/Core/SearchSpaces/VectorSpace.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>


namespace shark {
	namespace cmsa {
		/**
		* \brief Chromosome of the CMSA-ES.
		*/
		struct Chromosome {

			/**
			* \brief Default c'tor.
			*/
			Chromosome( unsigned int dimension = 0 ) : m_sigma( 0 ),
				m_cSigma( 0 ),
				m_cC( 0 ) {
			}

			/**
			* \brief Adjusts the dimension of the chromosome.
			*/
			void setDimension( unsigned int dimension ) {
				m_mean.resize( dimension);
				m_mutationDistribution.resize( dimension );
			}

			/**
			* \brief Serializes the chromosome to the supplied archive.
			* \tparam Archive Type of the archive.y
			* \param [in, out] archive Object of type archive.
			* \param [in] version Currently unused.
			*/
			template<typename Archive>
			void serialize( Archive & archive, const unsigned int version ) {
				archive & m_sigma;

				archive & m_cC;
				archive & m_cSigma;

				archive & m_mean;
				archive & m_mutationDistribution;
			}

			double m_sigma; ///< The current step size.
			double m_cSigma; 
			double m_cC; ///< Constant for adapting the covariance matrix.

			RealVector m_mean; ///< The current cog of the population.

			shark::MultiVariateNormalDistribution m_mutationDistribution; ///< Multi-variate normal mutation distribution.      
		};

	}


	/**
	* \brief Implements the CMSA.
	*
	*  The algorithm is described in
	*
	*  H. G. Beyer, B. Sendhoff (2008). 
	*  Covariance Matrix Adaptation Revisited – The CMSA Evolution Strategy –
	*  In Proceedings of the Tenth International Conference on Parallel Problem Solving from Nature
	*  (PPSN X), pp. 123-132, LNCS, Springer-Verlag
	*/
	class CMSA : public AbstractSingleObjectiveOptimizer<VectorSpace<double> > {
		/** \cond */

		struct LightChromosome {
			RealVector m_step;
			double m_sigma;
		};
		/** \endcond */
		/**
		* \brief Individual type of the CMSA implementation.
		*/
		typedef TypedIndividual< RealVector, LightChromosome > Individual;

	public:

		/** \cond */
		struct SigmaExtractor {
			double operator()( const CMSA & cmsa ) const {
				return cmsa.m_chromosome.m_sigma;
			}
		};
		/** \endcond */

		/**
		* \brief Default c'tor.
		*/
		CMSA() : m_mu( 100 ), m_lambda( 200 ) {
			m_features |= REQUIRES_VALUE;
		}

		/// \brief From INameable: return the class name.
		std::string name() const
		{ return "CMSA"; }

		/**
		* \brief Calculates the center of gravity of the given population \f$ \in \mathbb{R}^d\f$.
		*
		* 
		*/
		template<typename Container, typename Extractor>
		RealVector cog( const Container & container, const Extractor & e ) {

			RealVector result( m_numberOfVariables, 0. );

			for( unsigned int j = 0; j < container.size(); j++ )
				result += 1./m_mu * e( container[j] );

			return result;
		}

		/**
		* \brief Configures the algorithm based on the supplied configuration.
		*/
		void configure( const PropertyTree & node ) {}

		void read( InArchive & archive );
		void write( OutArchive & archive ) const;

		using AbstractSingleObjectiveOptimizer<VectorSpace<double> >::init;
		/**
		* \brief Initializes the algorithm for the supplied objective function.
		*/
		void init( ObjectiveFunctionType const& function, SearchPointType const& p);

		/**
		* \brief Executes one iteration of the algorithm.
		*/
		void step(ObjectiveFunctionType const& function);

		/**
		* \brief Accesses the chromosome.
		*/
		const shark::cmsa::Chromosome & chromosome() const {
			return m_chromosome;
		}

		/**
		* \brief Accesses the size of the parent population.
		*/
		unsigned int mu() const {
			return m_mu;
		}

		/**
		* \brief Accesses the size of the parent population, allows for l-value semantics.
		*/
		unsigned int & mu() {
			return m_mu;
		}

		/**
		* \brief Accesses the size of the offspring population.
		*/
		unsigned int lambda() const {
			return m_lambda;
		}

		/**
		* \brief Accesses the size of the offspring population, allows for l-value semantics.
		*/
		unsigned int & lambda() {
			return m_lambda;
		}
	protected:
		
		unsigned int m_numberOfVariables; ///< Stores the dimensionality of the search space.
		unsigned int m_mu; ///< The size of the parent population.
		unsigned int m_lambda; ///< The size of the offspring population, needs to be larger than mu.

		cmsa::Chromosome m_chromosome; ///< Stores the strategy parameters of the algorithm.
	private:
		/**
		* \brief Updates the strategy parameters based on the supplied offspring population.
		*/
		void updateStrategyParameters( const std::vector< CMSA::Individual > & offspringNew ) ;
	};

	/** \brief Registers the CMSA with the factory. */
	ANNOUNCE_SINGLE_OBJECTIVE_OPTIMIZER( CMSA, soo::RealValuedSingleObjectiveOptimizerFactory );
}

#endif
