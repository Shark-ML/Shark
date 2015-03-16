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


#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_CMSA_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_CMSA_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>


namespace shark {
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
	class CMSA : public AbstractSingleObjectiveOptimizer<RealVector > {
		/** \cond */

		struct LightChromosome {
			RealVector step;
			double sigma;
		};
		/** \endcond */
		/**
		* \brief Individual type of the CMSA implementation.
		*/
		typedef Individual< RealVector, double, LightChromosome > IndividualType;

	public:

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

		void read( InArchive & archive );
		void write( OutArchive & archive ) const;

		using AbstractSingleObjectiveOptimizer<RealVector >::init;
		/**
		* \brief Initializes the algorithm for the supplied objective function.
		*/
		void init( ObjectiveFunctionType const& function, SearchPointType const& p);

		/**
		* \brief Executes one iteration of the algorithm.
		*/
		void step(ObjectiveFunctionType const& function);

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

		double m_sigma; ///< The current step size.
		double m_cSigma; 
		double m_cC; ///< Constant for adapting the covariance matrix.

		RealVector m_mean; ///< The current cog of the population.

		shark::MultiVariateNormalDistribution m_mutationDistribution; ///< Multi-variate normal mutation distribution.   
	private:
		/**
		* \brief Updates the strategy parameters based on the supplied offspring population.
		*/
		void updateStrategyParameters( const std::vector< IndividualType > & offspringNew ) ;
	};
}

#endif
