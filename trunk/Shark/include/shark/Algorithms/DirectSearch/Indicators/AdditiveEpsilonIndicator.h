/*!
 * 
 * \file        AdditiveEpsilonIndicator.h
 *
 * \brief       Calculates the additive approximation quality of a Pareto-front
 * approximation.
 * 
 * 
 *
 * \author      T.Voss
 * \date        2010
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_INDICATORS_ADDITIVE_EPSILON_INDICATOR_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_INDICATORS_ADDITIVE_EPSILON_INDICATOR_H

#include <shark/Algorithms/DirectSearch/Traits/QualityIndicatorTraits.h>

#include <shark/LinAlg/Base.h>

#include <algorithm>
#include <limits>
#include <vector>

namespace shark {

    /**
     * \brief Given a reference front R and an approximation F, calculates the
     * additive approximation quality of F.
     */
    struct AdditiveEpsilonIndicator {

	/**
	 * \brief Executes the algorithm for the given ranges of individuals and returns the additive approximation ratio.
	 * 
	 * \param [in] itPF Iterator pointing to the first valid individual of the front approximation.
	 * \param [in] itePF Iterator pointing behind the last valid individual of the front approximation.
	 * \param [in] itRF Iterator pointing to the first valid individual of the reference front.
	 * \param [in] iteRF Iterator pointing behind the last valid individual of the reference front.
	 * \param [in,out] e Extractor instance that maps elements of the set to \f$\mathbb{R}^d\f$.		
	 */
	template<
	typename IteratorTypeA,
	    typename IteratorTypeB,
	    typename Extractor
	    > double operator()( IteratorTypeA itPF, IteratorTypeA itePF, IteratorTypeB itRF, IteratorTypeB iteRF, Extractor & e ) {

	    double result = -std::numeric_limits<double>::max();

	    for( IteratorTypeA ita = itPF;
		 ita != itePF;
		 ++ita ) {

		double tmp = std::numeric_limits<double>::max();
		for( IteratorTypeB itb = itRF;
		     itb != iteRF;
		     ++itb ) {

		    double d = -std::numeric_limits<double>::max();
		    for( unsigned int i = 0; i < std::min( e(*ita).size(), e(*itb).size() ); i++ ) {
			d = std::max( d, e(*ita)[i]-e(*itb)[i] );
		    }
		    tmp = std::min( tmp, d );
		}
		result = std::max( result, tmp );
	    }

	    return( result );
	}

	/**
	 * \brief Executes the algorithm for the given sets of individuals and returns the additive approximation ratio.
	 * 
	 * \param [in] a Pareto-front approximation.
	 * \param [in] b Reference Pareto-front.
	 * \param [in,out] e Extractor instance that maps elements of the set to \f$\mathbb{R}^d\f$.		
	 */
	template<
	typename ParetoFrontTypeA,
	    typename ParetoFrontTypeB,
	    typename Extractor
	    > double operator()( const ParetoFrontTypeA & a, const ParetoFrontTypeB & b, Extractor & e ) {
	    return( (*this)( a.begin(), a.end(), b.begin(), b.end(), e ) );
	}

	/**
	 * \brief Adjusts the nadir fitness vector, implemented empty.
	 * \param [in] fitness The new nadir fitness vector.
	 */
	void setNadirFitness( const RealVector & fitness ) {
	    (void)fitness;
	}

	/**
	 * \brief Adjusts the utopian fitness vector, implemented empty.
	 * \param [in] fitness The new utopian fitness vector.
	 */
	void setUtopianFitness( const RealVector & fitness ) {
	    (void)fitness;
	}

	/**
	 * \brief Serializes/Deserializes the state of the indicator, implemented empty.
	 */
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {
	    (void) archive;
	    (void) version;
	}

    };

    /**
     * \brief Binary performance indicator inspired by AGE-I.
     *
     * See the following reference for further details:
     *	- Bringmann, Friedrich, Neumann, Wagner. Approximation-Guided Evolutionary Multi-Objective Optimization. IJCAI '11.
     */
    template<typename FitnessType>
	struct LocalitySensitiveAdditiveEpsilonIndicator {

	    typedef std::vector< double > ResultType;

	    template<typename IteratorTypeA, typename IteratorTypeB, typename Extractor>
		ResultType operator()( IteratorTypeA itPF, IteratorTypeA itePF, IteratorTypeB itRF, IteratorTypeB iteRF, Extractor & e ) {

		ResultType indicatorValues;

		for( IteratorTypeA ita = itRF; ita != iteRF; ++ita ) {
		    double tmp = std::numeric_limits<double>::max();
		    for( IteratorTypeB itb = itPF; itb != itePF; ++itb ) {
			double d = -std::numeric_limits<double>::max();

			// for( unsigned int i = 0; i < std::min( ita->fitness( FitnessType() ).size(), itb->fitness( FitnessType() ).size() ); i++ ) {
			for( unsigned int i = 0; i < std::min( e( *ita ).size(), e( *itb ).size() ); i++ ) {
			    // d = std::max( d, itb->fitness( FitnessType() )[i] - ita->fitness( FitnessType() )[i] );
			    d = std::max( d, e( *itb )[ i ] - e( *ita )[i] );
			}
			tmp = std::min( tmp, d );
		    }
		    indicatorValues.push_back( tmp );
		}

		std::sort( indicatorValues.begin(), indicatorValues.end() );
		std::reverse( indicatorValues.begin(), indicatorValues.end() );

		

		return( indicatorValues );
	    }

	    template<
	    typename ParetoFrontTypeA,
		typename ParetoFrontTypeB,
		typename Extractor>
		double operator()( const ParetoFrontTypeA & a, const ParetoFrontTypeB & b, Extractor & e ) {

		return( (*this)( a.begin(), a.end(), b.begin(), b.end(), e ) );

	    }

	    /**
	     * \brief Adjusts the nadir fitness vector, implemented empty.
	     * \param [in] fitness The new nadir fitness vector.
	     */
	    void setNadirFitness( const RealVector & fitness ) {
		(void)fitness;
	    }

	    /**
	     * \brief Adjusts the utopian fitness vector, implemented empty.
	     * \param [in] fitness The new utopian fitness vector.
	     */
	    void setUtopianFitness( const RealVector & fitness ) {
		(void)fitness;
	    }
	};
}

#include <shark/Algorithms/DirectSearch/EA.h>

/**
 * \brief Integrates the additive epsilon indicator with the shark library.
 */
DECLARE_BINARY_QUALITY_INDICATOR( shark::AdditiveEpsilonIndicator );

/**
 * \brief Integrates the locality sensitive additive epsilon indicator with the shark library.
 *
 * Template specialization for explicitly considering penalized fitness values.
 */
DECLARE_BINARY_QUALITY_INDICATOR( shark::LocalitySensitiveAdditiveEpsilonIndicator<tag::PenalizedFitness> );

/**
 * \brief Integrates the locality sensitive additive epsilon indicator with the shark library.
 *
 * Template specialization for explicitly considering unpenalized fitness values.
 */
DECLARE_BINARY_QUALITY_INDICATOR( shark::LocalitySensitiveAdditiveEpsilonIndicator<tag::UnpenalizedFitness> );

#endif // SHARK_EA_ADDITIVE_EPSILON_INDICATOR_H
