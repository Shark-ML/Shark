/*!
 * 
 * \file        MultiplicativeEpsilonIndicator.h
 *
 * \brief       Calculates the multiplicate approximation quality of a Pareto-front
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
#ifndef SHARK_EA_MULTIPLICATIVE_EPSILON_INDICATOR_H
#define SHARK_EA_MULTIPLICATIVE_EPSILON_INDICATOR_H

#include <shark/Algorithms/DirectSearch/Traits/QualityIndicatorTraits.h>

namespace shark {

    /**
     * \brief Given a reference front R and an approximation F, calculates the
     * multiplicative approximation quality of F.
     */
    struct MultiplicativeEpsilonIndicator {

	/**
	 * \brief Executes the algorithm for the given ranges of individuals and returns the multiplicative approximation ratio.
	 * 
	 * \param [in] itPF Iterator pointing to the first valid individual of the front approximation.
	 * \param [in] itePF Iterator pointing behind the last valid individual of the front approximation.
	 * \param [in] itRF Iterator pointing to the first valid individual of the reference front.
	 * \param [in] iteRF Iterator pointing behind the last valid individual of the reference front.
	 */
	template<
	    typename IteratorTypeA,
	    typename IteratorTypeB
	> double operator()( IteratorTypeA itPF, IteratorTypeA itePF, IteratorTypeB itRF, IteratorTypeB iteRF ) {
		typedef typename IteratorTypeA::value_type::const_iterator PointIteratorTypeA;
				typedef typename IteratorTypeB::value_type::const_iterator PointIteratorTypeB;

				double result = -std::numeric_limits<double>::max();

				for( IteratorTypeA ita = itPF;
					ita != itePF;
					++ita ) {

						double tmp = std::numeric_limits<double>::max();
						for( IteratorTypeB itb = itRF;
							itb != iteRF;
							++itb ) {

								double d = -std::numeric_limits<double>::max();
								PointIteratorTypeA itpa;
								PointIteratorTypeB itpb;
								/*for( itpa = ita->begin(), itpb = itb->begin();
									itpa != ita->end() && itpb != itb->end();
									++itpa, ++itpb
									) {*/
								for( unsigned int i = 0; i < std::min( (*ita).size(), (*itb).size() ); i++ ) {
									// d = std::max( d, (*itpb)/ ( (*itpa) > 0. ? (*itpa) : std::numeric_limits<double>::min()) );
									d = std::max( d, (*itb)[i]/(*ita)[i] );
								}
								tmp = std::min( tmp, d );
						}
						result = std::max( result, tmp );
				}

				return( result );
			}

		template<typename ParetoFrontTypeA,typename ParetoFrontTypeB>
		double operator()( const ParetoFrontTypeA & a, const ParetoFrontTypeB & b ) {

			return( (*this)( a.begin(), a.end(), b.begin(), b.end() ) );

		}

	};

}

/**
* \brief Integrates the additive epsilon indicator with the shark library.
*/
DECLARE_BINARY_QUALITY_INDICATOR( shark::MultiplicativeEpsilonIndicator );

#endif // SHARK_EA_MULTIPLICATIVE_EPSILON_INDICATOR_H
