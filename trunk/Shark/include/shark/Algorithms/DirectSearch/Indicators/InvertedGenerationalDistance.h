/*!
 * 
 * \file        InvertedGenerationalDistance.h
 *
 * \brief       Inverted generational distance for comparing Pareto-front approximations.
 * 
 *
 * \author      -
 * \date        2010-2011
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_INVERTED_GENERATIONAL_DISTANCE_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_INDICATORS_INVERTED_GENERATIONAL_DISTANCE_H

#include <shark/Algorithms/DirectSearch/Traits/QualityIndicatorTraits.h>

namespace shark {

    /**
     * \brief Inverted generational distance for comparing Pareto-front approximations.
     */
	struct InvertedGenerationalDistance {

		template<typename IteratorTypeA,
			typename IteratorTypeB>
			double operator()( IteratorTypeA itPF, IteratorTypeA itePF, IteratorTypeB itRF, IteratorTypeB iteRF ) {
				typedef typename IteratorTypeA::value_type::const_iterator PointIteratorTypeA;
				typedef typename IteratorTypeB::value_type::const_iterator PointIteratorTypeB;

				double result = 0.;
				std::size_t noObjectives = std::numeric_limits<std::size_t>::max();
				for( IteratorTypeA ita = itPF;
					ita != itePF;
					++ita ) {

						double tmp = std::numeric_limits<double>::max();
						for( IteratorTypeB itb = itRF;
							itb != iteRF;
							++itb ) {

								
								PointIteratorTypeA itpa;
								PointIteratorTypeB itpb;
								
								double sum = 0.;

								for( unsigned int i = 0; i < std::min( (*ita).size(), (*itb).size() ); i++ ) {
									sum += boost::math::pow<2>( (*itb)[i] - (*ita)[i] );
									noObjectives = std::min( noObjectives, (*ita).size() );
								}
								sum = ::sqrt( sum );
								tmp = std::min( tmp, sum );
						}
						result += tmp * tmp;
				}

				result = ::sqrt( result ) / noObjectives;

				return( result );
			}

		template<typename ParetoFrontTypeA,typename ParetoFrontTypeB>
		double operator()( const ParetoFrontTypeA & a, const ParetoFrontTypeB & b ) {

			return( (*this)( a.begin(), a.end(), b.begin(), b.end() ) );

		}

	};
}

/**
* \brief Integrates the inverted generational distance indicator with the shark library.
*/
DECLARE_BINARY_QUALITY_INDICATOR( shark::AdditiveEpsilonIndicator );

#endif // SHARK_EA_INVERTED_GENERATIONAL_DISTANCE_H
