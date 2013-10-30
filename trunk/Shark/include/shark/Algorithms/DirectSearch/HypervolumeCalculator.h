/**
*
* \brief Implementation of the exact hypervolume calculation in m dimensions.
*
*  The algorithm is described in
*
*  Nicola Beume und G�nter Rudolph. 
*  Faster S-Metric Calculation by Considering Dominated Hypervolume as Klee's Measure Problem.
*  In: B. Kovalerchuk (ed.): Proceedings of the Second IASTED Conference on Computational Intelligence (CI 2006), 
*  pp. 231-236. ACTA Press: Anaheim, 2006. 
*
* <BR><HR>
* This file is part of Shark. This library is free software;
* you can redistribute it and/or modify it under the terms of the
* GNU General Public License as published by the Free Software
* Foundation; either version 3, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this library; if not, see <http://www.gnu.org/licenses/>.
*/
#pragma once

#include <shark/Algorithms/DirectSearch/BoundingBoxCalculator.h>

#include <shark/LinAlg/Base.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include <cmath>

namespace shark {
	/**
	* \brief Implementation of the exact hypervolume calculation in m dimensions.
	*
	*  The algorithm is described in
	*
	*  Nicola Beume und G�nter Rudolph. 
	*  Faster S-Metric Calculation by Considering Dominated Hypervolume as Klee's Measure Problem.
	*  In: B. Kovalerchuk (ed.): Proceedings of the Second IASTED Conference on Computational Intelligence (CI 2006), 
	*  pp. 231-236. ACTA Press: Anaheim, 2006.
	*/
	struct HypervolumeCalculator {

		/**
		* \brief Returns an estimate on the runtime of the algorithm.
		* \param [in] noPoints The number of points n considered in the runtime estimation.
		* \param [in] noObjectives The number of points m considered in the runtime estimation.
		*/
		static double runtime( unsigned int noPoints, unsigned int noObjectives ) {
			if( noPoints < 10 )
				return HypervolumeCalculator::runtime( 10, noObjectives );
			return( 0.03 * noObjectives * noObjectives * ::exp( ::log( static_cast<double>( noPoints ) ) * noObjectives * 0.5 ) );
		}

		/**
		* \brief Default c'tor.
		*/
		HypervolumeCalculator() : m_useLogHyp( false ) {
		}

		/**
		* \brief Serializes/Deserializes the state of the calculator to the supplied archive.
		* \tparam Archive Archive type, needs to be a model of a boost::serialization archive.
		* \param [in,out] archive Archive to store to/load from.
		* \param [in] version Currently unused.
		*/
		template<typename Archive>
		void serialize( Archive & archive, const unsigned int version ) {
			archive & BOOST_SERIALIZATION_NVP( m_noObjectives );
			archive & BOOST_SERIALIZATION_NVP( m_sqrtNoPoints );
			archive & BOOST_SERIALIZATION_NVP( m_useLogHyp );
		}

		/**
		* \brief Executes the algorithm.
		* \param [in] extractor Function object \f$f\f$to "project" elements of the set to \f$\mathbb{R}^m\f$.
		* \param [in] set The set \f$S\f$ of points for which the following assumption needs to hold: \f$\forall s \in S: \lnot \exists s' \in S: f( s' ) \preceq f( s ) \f$
		* \param [in] refPoint The reference point \f$\vec{r} \in \mathbb{R}^m\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$. 
		* \param [in] noObjectives Dimensionality \f$m\f$, refPoint.size() == noObjectives needs to be satisfied.
		*/
		template<typename Set,typename Extractor, typename VectorType>
		double operator()( Extractor & extractor, const Set & set, const VectorType & refPoint, unsigned int noObjectives );

		/** \cond IMPL */
		template<typename VectorType>
		int covers( const VectorType & cuboid, const VectorType & regionLow );

		template<typename VectorType>
		int partCovers( const VectorType & cuboid, const VectorType & regionUp );

		template<typename VectorType>
		int containsBoundary ( const VectorType & cub, const VectorType & regLow, int split	);

		template<typename VectorType>
		double getMeasure( const VectorType & regionLow, const VectorType & regionUp );

		template<typename VectorType>
		int isPile( const VectorType & cuboid, const VectorType & regionLow, const VectorType & regionUp );

		template<typename VectorType>
		int binaryToInt( const VectorType & bs );

		template<typename VectorType>
		void intToBinary( int i, VectorType & result );

		template<typename VectorType>
		double computeTrellis( const VectorType & regLow, const VectorType & regUp, const VectorType & trellis );

		template<typename VectorType>
		double getMedian( const VectorType & bounds, int length	);

		template<typename Set, typename Extractor, typename VectorType>
		double 	stream	( const VectorType & regionLow, 	
			const VectorType & regionUp, 	
			const Set & points, 
			Extractor & extractor,
			int split, 			
			double cover		
			);

		unsigned int m_noObjectives;
		unsigned int m_sqrtNoPoints;
		bool m_useLogHyp;
		template<typename Extractor>
		struct LastObjectiveComparator {

			LastObjectiveComparator( Extractor & extractor ) : m_extractor( extractor ) {}

			template<typename VectorType>
			bool operator()( const VectorType & lhs, const VectorType & rhs ) {
				return( m_extractor( lhs ).back() < m_extractor( rhs ).back() );
			}

			Extractor & m_extractor;
		};
		/** \endcond IMPL */
	};

	/** \cond IMPL */
	template<typename Set,typename Extractor, typename VectorType >
	double HypervolumeCalculator::operator()( Extractor & extractor, const Set & constSet, const VectorType & refPoint, unsigned int noObjectives ) {

		m_noObjectives = noObjectives;
		m_sqrtNoPoints = static_cast< unsigned int >( ::sqrt( static_cast<double>( constSet.size() ) ) );

		Set set( constSet );

		std::stable_sort( set.begin(), set.end(), LastObjectiveComparator<Extractor>( extractor ) );

		if( noObjectives == 2 ) {

			double h;
			if( m_useLogHyp )
				h = ( ::log( refPoint[0] ) - ::log( extractor( set[0] )[0] ) ) * (::log( refPoint[1] ) - ::log( extractor( set[0] )[1] ) );
			else
				h = ( refPoint[0] - extractor( set[0] )[0] ) * ( refPoint[1] - extractor( set[0] )[1] );

			double diffDim1; unsigned int lastValidIndex = 0;
			for( unsigned int i = 1; i < set.size(); i++ ) {
				if( m_useLogHyp )
					diffDim1 = ::log( extractor( set[lastValidIndex] )[0] ) - ::log( extractor( set[i] )[0] );  // Might be negative, if the i-th solution is dominated.
				else
					diffDim1 = extractor( set[lastValidIndex] )[0] - extractor( set[i] )[0];

				if( diffDim1 > 0 ) {
					if( m_useLogHyp )
						h += diffDim1 * ( ::log( refPoint[1] ) - ::log( extractor( set[i] )[1] ) );
					else
						h += ( diffDim1 ) * ( refPoint[1] - extractor( set[i] )[1] );
					lastValidIndex = i;
				}
			}
			return h;
		}

		VectorType regUp( noObjectives, -1E15 );
		VectorType regLow( noObjectives, 1E15 );
		BoundingBoxCalculator<Extractor,VectorType> bbc( extractor, regLow, regUp );
		for( unsigned int i = 0; i < set.size(); i++ )
			bbc( set.at( i ) );
		//std::for_each( set.begin(), set.end(), bbc );
		
		return( stream( regLow, refPoint, set, extractor, 0, refPoint.back() ) );	
	}

	template<typename VectorType>
	int HypervolumeCalculator::covers( const VectorType & cuboid, const VectorType & regionLow ) {
		for( unsigned int i = 0; i < m_noObjectives-1; i++ ) {
			// for( unsigned int i = 0; i < std::min( cuboid.size(), regionLow.size() ); i++ ) {
			if( cuboid[i] > regionLow[i] )
				return (0);
		}
		return (1);
	}

	template<typename VectorType>
	int HypervolumeCalculator::partCovers( const VectorType & cuboid, const VectorType & regionUp ) {
		// for( unsigned int i = 0; i < std::min( cuboid.size(), regionUp.size() ); i++) {
		for( unsigned int i = 0; i < m_noObjectives-1; i++) {
			if (cuboid[i] >= regionUp[i])
				return (0);
		}
		return (1);
	}

	template<typename VectorType>
	int HypervolumeCalculator::containsBoundary( const VectorType & cub, const VectorType & regLow, int split ) {
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
	double HypervolumeCalculator::getMeasure( const VectorType & regionLow, const VectorType & regionUp ) {
		double volume = 1.0;
		// for ( unsigned int i = 0; i < regionLow.size(); i++ ) {
		for( unsigned int i = 0; i < m_noObjectives-1; i++) {
			volume *= (regionUp[i] - regionLow[i]);
		}

		// std::cout << "Get Measure: " << volume << std::endl;

		return( volume );
	}

	template<typename VectorType>
	int HypervolumeCalculator::isPile( const VectorType & cuboid, const VectorType & regionLow, const VectorType & regionUp ) {
		unsigned int pile = cuboid.size();
		// for( unsigned int i = 0; i < NO_OBJECTIVES - 1; i++) {
		for( unsigned int i = 0; i < m_noObjectives-1; i++ ) {
			if( cuboid[i] > regionLow[i] ) {
				if( pile != m_noObjectives ) {
					return (-1);
				}

				pile = i;
			}
		}

		return (pile);
	}

	template<typename VectorType>
	int HypervolumeCalculator::binaryToInt( const VectorType & v ) {
		int result = 0;
		unsigned i;
		for (i = 0; i < v.size(); i++) {
			result += v[i] ? ( 1 << i ) : 0;//::pow(2.0, (double)i);
		}

		return (result);
	}

	template<typename VectorType>
	void HypervolumeCalculator::intToBinary(int i, VectorType & result) {
		unsigned j;
		for (j = 0; j < m_noObjectives - 1; j++) 
			result[j] = 0;

		int rest = i;
		int idx = 0;

		while (rest != 0) {
			result[idx] = (rest % 2);

			rest = rest / 2;
			idx++;
		}
	}

	template<typename VectorType>
	double HypervolumeCalculator::computeTrellis( const VectorType & regLow, const VectorType & regUp, const VectorType & trellis ) {
		unsigned i, j;
		std::vector<int> bs( m_noObjectives-1, 1 );

		double result = 0;

		int noSummands = binaryToInt(bs);
		int oneCounter; double summand;

		for( i = 1; i <= (unsigned)noSummands; i++ ) {
			summand = 1;
			intToBinary(i, bs);
			oneCounter = 0;

			for( j = 0; j < m_noObjectives-1; j++ ) {
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

		return(result);
	}

	template<typename VectorType>
	double HypervolumeCalculator::getMedian( const VectorType & bounds, int length) {
		if( length == 1 ) {
			return bounds[0];
		} else if( length == 2 ) {
			return bounds[1];
		}

		VectorType v( length );
		std::copy( bounds.begin(), bounds.begin() + length, v.begin() ); 
		std::sort( v.begin(), v.end() );

		return(length % 2 == 1 ? v[length/2] : (v[length/2-1] + v[(length/2)]) / 2);
	}

	template<typename Set, typename Extractor, typename VectorType>
	double HypervolumeCalculator::stream( const VectorType & regionLow,
		const VectorType & regionUp,
		const Set & points,
		Extractor & extractor,
		int split,
		double cover ) {
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

				if( covers( extractor( points[coverIndex] ), regionLow) ) {
					cover = extractor( points[coverIndex] )[m_noObjectives-1]; //points[coverIndex * NO_OBJECTIVES + NO_OBJECTIVES - 1];
					result += dMeasure * (coverOld - cover);
				}
				else
					coverIndex++;
			}

			//  std::cout << "(II) No points: " << points.size() << std::endl;

			for (c = coverIndex; c > 0; c--) {
				// if (points[(c - 1) * NO_OBJECTIVES + NO_OBJECTIVES - 1] == cover) {
				/*std::cout << "\t " << extractor( points[c-1] )[m_noObjectives-1] << " vs. " << cover << std::endl;
				if( c == 1 )
				std::cout << "\t\t " << extractor( points[c-1] )[m_noObjectives-1] << " vs. " << cover << std::endl;*/
				if( extractor( points[c-1] )[m_noObjectives-1] == cover) {
					coverIndex--;
				}
			}

			// std::cout << "Cover index: " << coverIndex << ", split: " << split << std::endl;

			if (coverIndex == 0)
				return (result);

			bool allPiles = true; // int i;

			std::vector<int> piles( coverIndex );
			// int  * piles = (int*)malloc(coverIndex * sizeof(int));

			for( int i = 0; i < coverIndex; i++ ) {
				piles[i] = isPile( extractor( points[i] ), regionLow, regionUp );//isPile(points + i * NO_OBJECTIVES, regionLow, regionUp);
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
					current = extractor( points[i] )[m_noObjectives-1]; // [m_noObjectives-1]; // points[i * NO_OBJECTIVES + NO_OBJECTIVES - 1];
					do {
						if( extractor( points[i] )[piles[i]] < trellis[piles[i]] ) {
							trellis[piles[i]] = extractor( points[i] )[piles[i]];//points[i * NO_OBJECTIVES + piles[i]];
						}
						i++;
						if (i < coverIndex) {
							next = extractor( points[i] )[m_noObjectives-1];// points[i * NO_OBJECTIVES + NO_OBJECTIVES - 1];
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
						int contained = containsBoundary( extractor( points[i] ), regionLow, split ); // containsBoundary(points + i * NO_OBJECTIVES, regionLow, split);
						if (contained == 1) {
							// boundaries.push_back( extractor( points[i] )[split] ); //points[i * NO_OBJECTIVES + split];
							boundaries[boundIdx] = extractor( points[i] )[split];
							boundIdx++;
						} else if (contained == 0) {
							// noBoundaries.push_back( extractor( points[i] )[split] ); // points[i * NO_OBJECTIVES + split];
							noBoundaries[noBoundIdx] = extractor( points[i] )[split];
							noBoundIdx++;
						}
					}

					if (boundIdx > 0) {
						bound = getMedian( boundaries, boundIdx );
						// std::cout << "Median: " << bound << std::endl;
					} else if( noBoundIdx > m_sqrtNoPoints ) {
						bound = getMedian( noBoundaries, noBoundIdx );
						// std::cout << "Median: " << bound << std::endl;
					} else {
						split++;
					}
				} while (bound == -1.0);

				Set pointsChildLow, pointsChildUp;
				// pointsChildLow.reserve( coverIndex );
				// pointsChildUp.reserve( coverIndex );

				VectorType regionUpC( regionUp );
				regionUpC[split] = bound;
				VectorType regionLowC( regionLow );
				regionLowC[split] = bound;

				for( int i = 0; i < coverIndex; i++) {
					if( partCovers( extractor( points[i] ), regionUpC) ) {
						// pointsChildUp.append( points[i] );
						pointsChildUp.push_back( points[i] );
					}

					if( partCovers( extractor( points[i] ), regionUp ) ) {
						// pointsChildLow.append( points[i] );

						pointsChildLow.push_back( points[i] );
					}
				}

				// #pragma omp sections
				{

					// #pragma omp task
					{
						if( pointsChildUp.size() > 0 ) {					
							// #pragma omp atomic
							result += stream(regionLow, regionUpC, pointsChildUp, extractor, split, cover);
						}
					}

					// #pragma omp task
					{
						if (pointsChildLow.size() > 0) {
							// #pragma omp atomic
							result += stream(regionLowC, regionUp, pointsChildLow, extractor, split, cover);
						}
					}

				}
			}

			return (result);
	}
	/** \endcond IMPL */
}
