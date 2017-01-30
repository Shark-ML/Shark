/*!
 * 
 *
 * \brief       Implementation of the exact hypervolume calculation in m dimensions.
 *
 * \author      T.Voss, O.Krause, T. Glasmachers
 * \date        2014-2016
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMECALCULATOR_MD_HOY_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUMECALCULATOR_MD_HOY_H

#include <shark/LinAlg/Base.h>

#include <algorithm>
#include <vector>
#include <map>

namespace shark {
/// \brief Implementation of the exact hypervolume calculation in m dimensions.
///
///  The algorithm is described in
///
/// Nicola Beume und Guenter Rudolph. 
/// Faster S-Metric Calculation by Considering Dominated Hypervolume as Klee's Measure Problem.
///  In: B. Kovalerchuk (ed.): Proceedings of the Second IASTED Conference on Computational Intelligence (CI 2006), 
/// pp. 231-236. ACTA Press: Anaheim, 2006.
struct HypervolumeCalculatorMDHOY{

	/// \brief Executes the algorithm.
	/// \param [in] set The set \f$S\f$ of points for which the following assumption needs to hold: \f$\forall s \in S: \lnot \exists s' \in S: s' \preceq s \f$
	/// \param [in] refPoint The reference point \f$\vec{r} \in \mathbb{R}^n\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$. .
	template<typename Set, typename VectorType >
	double operator()( Set const& points, VectorType const& refPoint){
		if(points.empty())
			return 0;
		SIZE_CHECK( points.begin()->size() == refPoint.size() );
		
		std::vector<VectorType> set;
		set.reserve(points.size());
		for(std::size_t i = 0; i != points.size(); ++i){
			set.push_back(points[i]);
		}
		std::sort( set.begin(), set.end(), [ ](VectorType const& x, VectorType const& y){return x.back() < y.back();});

		m_sqrtNoPoints = static_cast< std::size_t >( ::sqrt( static_cast<double>( points.size() ) ) );
		
		VectorType regLow( refPoint.size(), 1E15 );
		for( std::size_t i = 0; i < set.size(); i++ ){
			noalias(regLow) = min(regLow,set[i]);
		}
		return stream( regLow, refPoint, set, 0, refPoint.back() );	
	}

	template<typename VectorType>
	int covers( VectorType const& cuboid, VectorType const& regionLow ) {
		for( std::size_t i = 0; i < cuboid.size()-1; i++ ) {
			if( cuboid[i] > regionLow[i] )
				return 0;
		}
		return 1;
	}

	template<typename VectorType>
	int partCovers( VectorType const& cuboid, VectorType const& regionUp ) {
		for( std::size_t i = 0; i < cuboid.size()-1; i++) {
			if (cuboid[i] >= regionUp[i])
				return 0;
		}
		return 1;
	}

	template<typename VectorType>
	int containsBoundary( VectorType const& cub, VectorType const& regLow, int split ) {
		if( !( regLow[split] < cub[split] ) ) {
			return -1;
		} else {
			for ( int j = 0; j < split; j++) {
				if (regLow[j] < cub[j]) {
					return 1;
				}
			}
		}
		return 0;
	}

	template<typename VectorType>
	double getMeasure( const VectorType & regionLow, const VectorType & regionUp ) {
		double volume = 1.0;
		for( std::size_t i = 0; i < regionLow.size()-1; i++) {
			volume *= (regionUp[i] - regionLow[i]);
		}
		return volume;
	}

	template<typename VectorType>
	int isPile( const VectorType & cuboid, const VectorType & regionLow, const VectorType & regionUp ) {
		std::size_t pile = cuboid.size();
		for( std::size_t i = 0; i < cuboid.size()-1; i++ ) {
			if( cuboid[i] > regionLow[i] ) {
				if( pile != cuboid.size() ) {
					return (-1);
				}
				pile = i;
			}
		}

		return (int)pile;
	}

	template<typename VectorType>
	unsigned int binaryToInt( const VectorType & v ) {
		int result = 0;
		unsigned i;
		for (i = 0; i < v.size(); i++) {
			result += v[i] ? ( 1 << i ) : 0;
		}

		return result;
	}

	template<typename VectorType>
	void intToBinary(unsigned int i, VectorType & result) {
		for (std::size_t j = 0; j < result.size(); j++) 
			result[j] = 0;

		unsigned int rest = i;
		std::size_t idx = 0;

		while (rest != 0) {
			result[idx] = (rest % 2);

			rest = rest / 2;
			idx++;
		}
	}

	template<typename VectorType>
	double computeTrellis( const VectorType & regLow, const VectorType & regUp, const VectorType & trellis ) {
		std::vector<int> bs( regLow.size()-1, 1 );

		double result = 0;

		unsigned int noSummands = binaryToInt(bs);
		int oneCounter; double summand;

		for(unsigned i = 1; i <= noSummands; i++ ) {
			summand = 1;
			intToBinary(i, bs);
			oneCounter = 0;

			for(std::size_t j = 0; j < regLow.size()-1; j++ ) {
				if (bs[j] == 1) {
					summand *= regUp[j] - trellis[j];
					oneCounter++;
				} else
					summand *= regUp[j] - regLow[j];
			}

			if (oneCounter % 2 == 0)
				result -= summand ;
			else
				result += summand;
		}

		return result;
	}

	template<typename VectorType>
	double getMedian( const VectorType & bounds, int length) {
		if( length == 1 ) {
			return bounds[0];
		} else if( length == 2 ) {
			return bounds[1];
		}

		VectorType v( length );
		std::copy( bounds.begin(), bounds.begin() + length, v.begin() ); 
		std::sort( v.begin(), v.end() );

		return (length % 2 == 1) ? v[length/2] : (v[length/2-1] + v[(length/2)]) / 2;
	}

	template<typename Set, typename VectorType>
	double stream( const VectorType & regionLow,
		const VectorType & regionUp,
		const Set & points,
		int split,
		double cover 
	) {
		std::size_t numObjectives = regionLow.size();
		double coverOld;
		coverOld = cover;
		int coverIndex = 0;
		int coverIndexOld = -1;
		int c;

		double result = 0;

		double dMeasure = getMeasure(regionLow, regionUp);
		while( cover == coverOld && coverIndex < static_cast<int>( points.size() ) ) {
			if( coverIndexOld == coverIndex )
				break;

			coverIndexOld = coverIndex;

			if( covers( points[coverIndex], regionLow) ) {
				cover = points[coverIndex][numObjectives-1];
				result += dMeasure * (coverOld - cover);
			}
			else
				coverIndex++;
		}

		for (c = coverIndex; c > 0; c--) {
			if( points[c-1][numObjectives-1] == cover) {
				coverIndex--;
			}
		}

		if (coverIndex == 0)
			return (result);

		bool allPiles = true;

		std::vector<int> piles( coverIndex );

		for( int i = 0; i < coverIndex; i++ ) {
			piles[i] = isPile( points[i], regionLow, regionUp );
			if (piles[i] == -1) {
				allPiles = false;
				break;
			}
		}

		if( allPiles ) {
			VectorType trellis( regionUp );

			double current = 0.0;
			double next = 0.0;
			int i = 0;
			do {
				current = points[i][numObjectives-1];
				do {
					if( points[i][piles[i]] < trellis[piles[i]] ) {
						trellis[piles[i]] = points[i][piles[i]];
					}
					i++;
					if (i < coverIndex) {
						next = points[i][numObjectives-1];
					}
					else {
						next = cover;
						break;
					}

				}
				while (next == current);

				result += computeTrellis(regionLow, regionUp, trellis) * (next - current);
			} while (next != cover);

		} else {
			double bound = -1.0;
			std::vector<double> boundaries( coverIndex );
			std::vector<double> noBoundaries( coverIndex );
			unsigned boundIdx = 0;
			unsigned noBoundIdx = 0;

			do {
				for( int i = 0; i < coverIndex; i++ ) {
					int contained = containsBoundary(points[i], regionLow, split );
					if (contained == 1) {
						boundaries[boundIdx] = points[i][split];
						boundIdx++;
					} else if (contained == 0) {
						noBoundaries[noBoundIdx] = points[i][split];
						noBoundIdx++;
					}
				}

				if (boundIdx > 0) {
					bound = getMedian( boundaries, boundIdx );
				} else if( noBoundIdx > m_sqrtNoPoints ) {
					bound = getMedian( noBoundaries, noBoundIdx );
				} else {
					split++;
				}
			} while (bound == -1.0);

			Set pointsChildLow, pointsChildUp;

			VectorType regionUpC( regionUp );
			regionUpC[split] = bound;
			VectorType regionLowC( regionLow );
			regionLowC[split] = bound;

			for( int i = 0; i < coverIndex; i++) {
				if( partCovers( points[i], regionUpC) ) {
					pointsChildUp.push_back( points[i] );
				}

				if( partCovers( points[i], regionUp ) ) {
					pointsChildLow.push_back( points[i] );
				}
			}

			
			if( pointsChildUp.size() > 0 ) {					
				result += stream(regionLow, regionUpC, pointsChildUp, split, cover);
			}
			if (pointsChildLow.size() > 0) {
				result += stream(regionLowC, regionUp, pointsChildLow, split, cover);
			}
		}

		return result;
	}

	std::size_t m_sqrtNoPoints;
};

}
#endif
