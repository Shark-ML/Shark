/*!
 *
 *
 * \brief       Calculates the additive approximation quality of a Pareto-front
 * approximation.
 *
 *
 *
 * \author      T.Voss, O.Krause
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_INDICATORS_ADDITIVE_EPSILON_INDICATOR_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_INDICATORS_ADDITIVE_EPSILON_INDICATOR_H

#include <shark/LinAlg/Base.h>
#include <shark/LinAlg/Base.h>
#include <shark/Core/OpenMP.h>

#include <algorithm>
#include <limits>
#include <vector>


namespace shark {

/**
 * \brief Given a reference front R and an approximation F, calculates the
 * additive approximation quality of F.
 *
 * See the following reference for further details:
 *	- Bringmann, Friedrich, Neumann, Wagner. Approximation-Guided Evolutionary Multi-Objective Optimization. IJCAI '11.
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
	> 
	double operator()( IteratorTypeA itPF, IteratorTypeA itePF, IteratorTypeB itRF, IteratorTypeB iteRF, Extractor & e ){
		double result = -std::numeric_limits<double>::max();
		for( IteratorTypeB itb = itRF; itb != iteRF; ++itb ) {
			double delta = std::numeric_limits<double>::max();
			for( IteratorTypeA ita = itPF; ita != itePF;++ita ) {
				SIZE_CHECK(e( *ita ).size() == e( *itb ).size());
				delta = std::min( delta, max(e(*itb)-e(*ita)) );
			}
			result = std::max( result, delta );
		}

		return result;
	}
	
	/// \brief Given a pareto front, returns the index of the points which is the least contributer
	template<typename Extractor, typename ParetofrontType>
	unsigned int leastContributor( Extractor extractor, const ParetofrontType & front)
	{
		std::vector<double> relativeApproximation(front.size());
		SHARK_PARALLEL_FOR( int i = 0; i < static_cast< int >( front.size() ); i++ ) {
			//find the minimum distance the front with one point removed has to be moved to dominate the original front
			double result = -std::numeric_limits<double>::max();
			for(std::size_t j = 0; j != front.size(); ++j){
				if(j == i) continue; //this point is removed
				result = std::min<double>(result,max(extractor(front[i])-extractor(front[j]))); 
			}
			relativeApproximation[i] = result;
		}
		
		return std::min_element( relativeApproximation.begin(), relativeApproximation.end() ) - relativeApproximation.begin();
	}
	
	/// \brief Updates the internal variables of the indicator using a whole population.
	///
	/// Empty for this Indicator
	/// \param extractor Functor returning the fitness values
	/// \param set The set of points.
	template<typename Extractor, typename PointSet>
	void updateInternals(Extractor extractor, PointSet const& set){
		(void)extractor;
		(void)set;
	}

	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {
		(void)archive;
		(void)version;
	}
};

}

#endif
