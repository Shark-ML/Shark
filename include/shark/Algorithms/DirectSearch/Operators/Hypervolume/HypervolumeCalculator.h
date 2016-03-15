/*!
 * 
 *
 * \brief       Implementation of the exact hypervolume calculation in m dimensions.
 * 
 * The algorithm is described in
 * 
 * Nicola Beume und Günter Rudolph. 
 * Faster S-Metric Calculation by Considering Dominated Hypervolume as Klee's Measure Problem.
 * In: B. Kovalerchuk (ed.): Proceedings of the Second IASTED Conference on Computational Intelligence (CI 2006), 
 * pp. 231-236. ACTA Press: Anaheim, 2006. 
 * 
 * A specialized algorithm is used for the three-objective case:
 *
 * M. T. M. Emmerich and C. M. Fonseca.
 * Computing hypervolume contributions in low dimensions: Asymptotically optimal algorithm and complexity results.
 * In: Evolutionary Multi-Criterion Optimization (EMO) 2011.
 * Vol. 6576 of Lecture Notes in Computer Science, pp. 121--135, Berlin: Springer, 2011.
 *
 *
 * \author      T.Voss, O.Krause, T. Glasmachers
 * \date        2014-2016
 *
 *
 * \par Copyright 1995-2016 Shark Development Team
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMECALCULATOR_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMECALCULATOR_H

#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator2D.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator3D.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculatorND.h>

namespace shark {
/// \brief Implementation of the exact hypervolume calculation in n dimensions.
///
///  Depending on the dimensionality of the problem, one of the specialized algorithms is called.
struct HypervolumeCalculator {

	/// \brief Default c'tor.
	HypervolumeCalculator() : m_useLogHyp( false ) {}

	///\brief True if the logarithmic volume is to be used, e.g. taking the logarithm of all values.
	void useLogHyp(bool useLogHyp){m_useLogHyp = useLogHyp;}
	
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {}
	
	/// \brief Executes the algorithm.
	/// \param [in] extractor Function object \f$f\f$to "project" elements of the points to \f$\mathbb{R}^n\f$.
	/// \param [in] points The set \f$S\f$ of points for which the following assumption needs to hold: \f$\forall s \in S: \lnot \exists s' \in S: f( s' ) \preceq f( s ) \f$
	/// \param [in] refPoint The reference point \f$\vec{r} \in \mathbb{R}^n\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$. .
	template<typename Points,typename Extractor, typename VectorType>
	double operator()( Extractor const& extractor, Points const& points, VectorType const& refPoint){
		SIZE_CHECK( extractor(*points.begin()).size() == refPoint.size() );
		typedef typename Points::const_reference Point;
		std::size_t numObjectives = refPoint.size();
		if(numObjectives == 2){
			HypervolumeCalculator2D algorithm;
			if(m_useLogHyp){
				return algorithm([&](Point x){return log(extractor(x));}, points, refPoint);
			}else{
				return algorithm(extractor, points, refPoint);
			}
		}else if(numObjectives == 3){
			HypervolumeCalculator3D algorithm;
			if(m_useLogHyp){
				return algorithm([&](Point x){return log(extractor(x));}, points, refPoint);
			}else{
				return algorithm(extractor, points, refPoint);
			}
		}else{
			HypervolumeCalculatorND algorithm;
			if(m_useLogHyp){
				return algorithm([&](Point x){return log(extractor(x));}, points, refPoint);
			}else{
				return algorithm(extractor, points, refPoint);
			}
		}
	}

private:
	bool m_useLogHyp;
};

}
#endif