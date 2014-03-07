/*!
 *
 *
 * \brief       Inverted generational distance for comparing Pareto-front approximations.
 *
 *
 * \author     T. Voss, O.Krause
 * \date        2010-2014
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_INDICATORS_INVERTED_GENERATIONAL_DISTANCE_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_INDICATORS_INVERTED_GENERATIONAL_DISTANCE_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/OpenMP.h>

#include <algorithm>
#include <limits>
#include <vector>

namespace shark {

/**
 * \brief Inverted generational distance for comparing Pareto-front approximations.
 */
struct InvertedGenerationalDistance {

	template<
		typename IteratorTypeA,
		typename IteratorTypeB
	>
	double operator()( IteratorTypeA itPF, IteratorTypeA itePF, IteratorTypeB itRF, IteratorTypeB iteRF )
	{
		double result = 0.;
		std::size_t noObjectives = e(*ita).size();
		for( IteratorTypeA ita = itPF; ita != itePF; ++ita ) {
			SIZE_CHECK(e(*ita).size() == noObjectives);
			
			double tmp = std::numeric_limits<double>::max();
			for( IteratorTypeB itb = itRF; itb != iteRF; ++itb ) {
				SIZE_CHECK(e(*itb).size() == noObjectives);
				double sum = 0.;				
				for( unsigned int i = 0; i < noObjectives; i++ ) {
					sum += sqr( e(*itb)[i] - e(*ita)[i] );
				}
				tmp = std::min( tmp, sum );
			}
			result += tmp;
		}

		return std::sqrt( result ) / noObjectives;
	}
	
	/// \brief Given a pareto front, returns the index of the points which is the least contributer
	template<typename Extractor, typename ParetofrontType>
	std::size_t leastContributor( Extractor & extractor, const ParetofrontType & front)
	{
		std::vector<double> relativeApproximation(front.size());
		SHARK_PARALLEL_FOR( int i = 0; i < static_cast< int >( front.size() ); i++ ) {
			relativeApproximation[i] = (*this)( front.begin()+i,front.begin()+(i+1), front.begin(), front.end(), extractor );
		}
		
		return std::min_element( relativeApproximation.begin(), relativeApproximation.end() ) - relativeApproximation.begin();
	}
	
	/**
	 * \brief Adjusts the nadir fitness vector, implemented empty.
	 * \param [in] fitness The new nadir fitness vector.
	 */
	void setNadirFitness( const RealVector & fitness )
	{
		(void)fitness;
	}

	/**
	 * \brief Adjusts the utopian fitness vector, implemented empty.
	 * \param [in] fitness The new utopian fitness vector.
	 */
	void setUtopianFitness( const RealVector & fitness )
	{
		(void)fitness;
	}

	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {
		(void)archive;
		(void)version;
	}

};
}
#endif
