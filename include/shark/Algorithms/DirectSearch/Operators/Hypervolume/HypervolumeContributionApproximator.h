/*!
 *
 *
 * \brief       Approximately determines the individual contributing the least
 * hypervolume.
 *
 *
 *
 * \author      T.Voss, O.Krause
 * \date        2010-2016
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
#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUME_CONTRIBUTION_APPROXIMATOR_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_HYPERVOLUME_CONTRIBUTION_APPROXIMATOR_H

#include <boost/cstdint.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Domination/ParetoDominance.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator.h>
#include <algorithm>
#include <limits>
#include <vector>
#include <cmath>

namespace shark {

/// \brief Approximately determines the point of a set contributing the least hypervolume.
///
/// See K. Bringmann, T. Friedrich. Approximating the least hypervolume contributor: NP-hard in general, but fast in practice. Proc. of the 5th International Conference on Evolutionary Multi-Criterion Optimization (EMO 2009), Vol. 5467 of LNCS, pages 6-20, Springer-Verlag, 2009.
///
/// The algorithm works by estimating a bounding box for every point that includes all its contribution volume and then draw
/// sample inside the box to estimate which fraction of the box is covered by other points in the set.
///
/// The algorithm only implements the k=1 version of the smallest contribution. For the element A it returns holds the
/// following guarantue: with probability of 1-delta, Con(A) < (1+epsilon)Con(LC)
/// where LC is true least contributor. Note that there are no error guarantuees regarding the returned value of the contribution:
/// the algorithm stops when it is sure that the bound above holds, but depending on the setup, this might be very early.
///
/// Note that, while on average the algorithm performs reasonable well, the upper run-time is not bounded by the number of elements or dimensionality.
/// When two points have nearly the same(or exactly the same) contribution
/// the algorithm will run for many iterations, until the bound above holds. The same holds if the point with the smallest contribution
/// has a very large potential contribution as many samples are required to establish that allmost all of the box is covered.
///
///\tparam random The type of the random for sampling random points.
struct HypervolumeContributionApproximator{
	/// \brief Models a point and associated information for book-keeping purposes.
	template<typename VectorType>
	struct Point {
		Point( VectorType const& point, VectorType const& reference ) 
		: point( point )
		, sample( point.size() )
		, boundingBox( reference )
		, boundingBoxVolume( 0. )
		, approximatedContribution( 0. )
		, contributionLowerBound( 0. )
		, contributionUpperBound( 0. )
		, computedExactly(false)
		, noSamples( 0 )
		, noSuccessfulSamples( 0 )
		{}

		VectorType point;
		VectorType sample;
		VectorType boundingBox;
		std::vector< typename std::vector<Point>::const_iterator > influencingPoints;

		double boundingBoxVolume;
		double approximatedContribution;
		double contributionLowerBound;
		double contributionUpperBound;
		bool computedExactly;
		
		std::size_t noSamples;
		std::size_t noSuccessfulSamples;
	};

	double m_startDeltaMultiplier;
	double m_multiplierDelta;
	double m_minimumMultiplierDelta;
	
	double m_gamma;
	double m_errorProbability; ///<The error probability.
	double m_errorBound;  ///<The error bound

	/// \brief C'tor
	/// \param [in] delta the error probability of the least contributor
	/// \param [in] eps the error bound of the least contributor
	HypervolumeContributionApproximator()
	: m_startDeltaMultiplier( 0.1 )
	, m_multiplierDelta( 0.775 )
	, m_minimumMultiplierDelta( 0.2 )
	, m_gamma( 0.25 )
	, m_errorProbability(1.E-2)
	, m_errorBound(1.E-2)
	{}
		
	double delta()const{
		return m_errorProbability;
	}
	double& delta(){
		return m_errorProbability;
	}
	
	double epsilon()const{
		return m_errorBound;
	}
	double& epsilon(){
		return m_errorBound;
	}
	
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {
		archive & BOOST_SERIALIZATION_NVP(m_startDeltaMultiplier);
		archive & BOOST_SERIALIZATION_NVP(m_multiplierDelta);
		archive & BOOST_SERIALIZATION_NVP(m_minimumMultiplierDelta);
		archive & BOOST_SERIALIZATION_NVP(m_gamma);
		archive & BOOST_SERIALIZATION_NVP(m_errorProbability);
		archive & BOOST_SERIALIZATION_NVP(m_errorBound);
	}

	/// \brief Determines the point contributing the least hypervolume to the overall set of points.
	///
	/// \param [in] s pareto front of points
	/// \param [in] reference The reference point to consider for calculating individual points' contributions.
	template<class Set,class VectorType>
	std::vector<KeyValuePair<double,std::size_t> > smallest(Set const& points, std::size_t k, VectorType const& reference)const{
		SHARK_RUNTIME_CHECK(points.size() >= k, "There must be at least k points in the set");
		SHARK_RUNTIME_CHECK(k == 1, "Not implemented for k != 1");
				
		std::vector< Point<VectorType> > front;
		for(auto const& point: points) {
			front.emplace_back( point, reference );
		}
		computeBoundingBoxes( front );
		
		std::vector< typename std::vector< Point<VectorType> >::iterator > activePoints;
		for(auto it = front.begin(); it != front.end(); ++it ) {
			activePoints.push_back( it );
		}
		
		auto smallest = computeSmallest(activePoints, front.size());
		
		std::vector<KeyValuePair<double,std::size_t> > result(1);
		result[0].key = smallest->approximatedContribution;
		result[0].value = smallest-front.begin();
		return result;
	}
		
	
	/// \brief Returns the index of the points with smallest contribution.
	///
	/// As no reference point is given, the extremum points can not be computed and are never selected.
	///
	/// \param [in] points The set \f$S\f$ of points from which to select the smallest contributor.
	/// \param [in] k The number of points to select.
	template<class Set>
	std::vector<KeyValuePair<double,std::size_t> > smallest(Set const& points, std::size_t k)const{
		SHARK_RUNTIME_CHECK(points.size() >= k, "There must be at least k points in the set");
		SHARK_RUNTIME_CHECK(k == 1, "Not implemented for k != 1");
		
		//find reference point as well as points with lowest function value
		std::vector<std::size_t> minIndex(points[0].size(),0);
		RealVector minVal = points[0];
		RealVector reference=points[0];
		for(std::size_t i = 1; i != points.size(); ++i){
			noalias(reference) = max(reference,points[i]);
			for(std::size_t j = 0; j != minVal.size(); ++j){
				if(points[i](j)< minVal[j]){
					minVal[j] = points[i](j);
					minIndex[j]=i;
				}
			}
		}
		
		std::vector< Point<RealVector> > front;
		front.reserve( points.size() );
		for(auto const& point: points){
			front.emplace_back( point, reference );
		}
		computeBoundingBoxes( front );
		
		std::vector<std::vector< Point<RealVector> >::iterator > activePoints;
		for(auto it = front.begin(); it != front.end(); ++it ) {
			if(std::find(minIndex.begin(),minIndex.end(),it-front.begin()) != minIndex.end())
				continue;
			//~ //check whether point is on the boundary -> least contributor
			//~ for(std::size_t j = 0; j != minVal.size(); ++j){
				//~ if(it->point[j] == reference[j]){
					//~ return std::vector<KeyValuePair<double,std::size_t> >(1,{0.0,it-front.begin()});
				//~ }
			//~ }
			activePoints.push_back( it );
		
		}
		
		
		auto smallest = computeSmallest(activePoints, front.size());
		std::vector<KeyValuePair<double,std::size_t> > result(1);
		result[0].key = smallest->approximatedContribution;
		result[0].value = smallest-front.begin();
		return result;
	}

private:
	
	template<class Set>
	typename Set::value_type computeSmallest(Set& activePoints, std::size_t n)const{
		typedef typename Set::value_type SetIter;
		//compute initial guess for delta
		double delta = 0.;
		for( auto it = activePoints.begin(); it != activePoints.end(); ++it )
			delta = std::max( delta, (*it)->boundingBoxVolume );
		delta *= m_startDeltaMultiplier;
		
		unsigned int round = 0;
		while( true ) {
			round++;
			
			//check whether we spent so much time on sampling that computing the real volume is not much more expensive any more.
			//this guarantuees convergence even in cases that two points have the same hyper volume.
			SHARK_PARALLEL_FOR(int i = 0; i < (int)activePoints.size(); ++i){
				if(shouldCompute(*activePoints[i])){
					computeExactly(*activePoints[i]);
				}
			}
			
			//sample all active points so that their individual deviations are smaller than delta
			for( auto point: activePoints )
				sample( *point, round, delta, n );

			//find the current least contributor
			auto minimalElement = std::min_element(
				activePoints.begin(),activePoints.end(),
				[](SetIter const& a, SetIter const& b){return a->approximatedContribution < b->approximatedContribution;}
			);

			//section 3.4.1: push the least contributor: decrease its delta further to have a chance to end earlier.
			if( activePoints.size() > 2 ) {
				sample( **minimalElement, round, m_minimumMultiplierDelta * delta, n );
				minimalElement = std::min_element(
					activePoints.begin(),activePoints.end(),
					[](SetIter const& a, SetIter const& b){return a->approximatedContribution < b->approximatedContribution;}
				);
			}

			//remove all points whose confidence interval does not overlap with the current minimum any more.
			double erase_level = (*minimalElement)->contributionUpperBound;
			auto erase_start = std::remove_if(
				activePoints.begin(),activePoints.end(),
				[=](SetIter const& point){return point->contributionLowerBound > erase_level;}
			);
			activePoints.erase(erase_start,activePoints.end());
			
			//if the set only has one point left, we are done.
			if(activePoints.size() == 1)
				return activePoints.front();

			// stopping conditions: have we reached the desired accuracy? 
			// for this we need to know:
			// 1. contribution for all points are bounded above 0
			// 2. upperBound(LC) < (1+epsilon)*lowerBound(A) for all A
			double d = 0;
			for( auto it = activePoints.begin(); it != activePoints.end(); ++it ) {
				if( it == minimalElement )
					continue;
				double nom = (*minimalElement)-> contributionUpperBound;
				double den = (*it)->contributionLowerBound;
				if( den <= 0. )
					d = std::numeric_limits<double>::max();
				else
					d = std::max(d,nom/den);
			}
			
			//if the stopping condition is fulfilled, return the minimal element
			if(d < 1. + m_errorBound){
				return *minimalElement;
			}
			
			delta *= m_multiplierDelta;
		}
	}
	
	/// \brief Samples in the bounding box of the supplied point until a pre-defined threshold is reached.
	///
	/// \param [in] point Iterator to the point that should be sampled.
	/// \param [in] r The current round.
	/// \param [in] delta The delta that should be reached.
	/// \param [in] n the total number of points in the front. Required for proper calculation of bounds
	template<class VectorType>
	void sample( Point<VectorType>& point, unsigned int r, double delta, std::size_t n )const{
		if(point.computedExactly) return;//spend no time on points that are computed exactly
		
		double logFactor = std::log( 2. * n * (1. + m_gamma) / (m_errorProbability * m_gamma) );
		double logR = std::log( static_cast<double>( r ) );
		//compute how many samples we need until the bound of the current box is smaller than delta
		//this is formula (3) in the paper when used in an equality < delta and solving for noSamples
		//we add +1 to ensure that the inequality holds.
		double thresholdD= 1.0+0.5 * ( (1. + m_gamma) * logR + logFactor ) * sqr( point.boundingBoxVolume / delta );
		std::size_t threshold = static_cast<std::size_t>(thresholdD);
		//sample points inside the box of the current point
		for( ; point.noSamples < threshold; point.noSamples++ ) {
			//sample a point inside the box
			point.sample.resize(point.point.size());
			for( unsigned int i = 0; i < point.sample.size(); i++ ) {
				point.sample[ i ] =  random::uni(random::globalRng, point.point[ i ], point.boundingBox[ i ] );
			}
			++point.noSamples;
			//check if the point is not dominated by any of the influencing points
			if( !isPointDominated( point.influencingPoints, point.sample ) )
				point.noSuccessfulSamples++;
		}

		//current best guess for volume: fraction of accepted points inside the box imes the volume of the box. (2) in the paper
		point.approximatedContribution = (point.boundingBoxVolume * point.noSuccessfulSamples) /  point.noSamples;
		//lower and upper bounds for the best guess: with high probability it will be in this region. (3) in the paper.
		double deltaReached = std::sqrt( 0.5 * ((1. + m_gamma) * logR+ logFactor ) / point.noSamples ) * point.boundingBoxVolume;
		point.contributionLowerBound = point.approximatedContribution - deltaReached;
		point.contributionUpperBound = point.approximatedContribution + deltaReached;
	}
	
	/// \brief Checks whether a point is dominated by any point in a given set.
	///
	/// \tparam Set The type of the set of points.
	///
	/// \param [in] set The set of individuals to check the sampled point against.
	/// \param [in] point Point to test
	///
	/// \returns true if the point was non-dominated
	template<typename Set, typename VectorType>
	bool isPointDominated( Set const& set, VectorType const& point )const{
		
		for( unsigned int i = 0; i < set.size(); i++ ) {
			DominanceRelation rel = dominance(set[i]->point, point);
			if (rel == LHS_DOMINATES_RHS)
				return true;
		}
		return false;
	}

	/// \brief Computes bounding boxes and their volume for the range of points defined by the iterators.
	template<class Set>
	void computeBoundingBoxes(Set& set )const{
		for(auto it = set.begin(); it != set.end(); ++it ) {
			auto& p1 = *it;
			//first cut the bounding boxes on sides that are completely covered by other points
			for(auto itt = set.begin(); itt != set.end(); ++itt ) {

				if( itt == it )
					continue;

				auto& p2 = *itt;

				unsigned int coordCounter = 0;
				unsigned int coord = 0;
				for( unsigned int o = 0; o < p1.point.size(); o++ ) {
					if( p2.point[ o ] > p1.point[ o ] ) {
						coordCounter++;
						if( coordCounter == 2 )
							break;
						coord = o;
					}
				}

				if( coordCounter == 1 && p1.boundingBox[ coord ] > p2.point[ coord ] )
					p1.boundingBox[ coord ] = p2.point[ coord ];
			}
			
			//compute volume of the box
			it->boundingBoxVolume = 1.;
			for(unsigned int i = 0 ; i < it->boundingBox.size(); i++ ) {
				it->boundingBoxVolume *= it->boundingBox[ i ] - it->point[ i ];
			}

			//find all points that are partially covering this box
			for(auto itt = set.begin(); itt != set.end(); ++itt ) {
				if( itt == it )
					continue;

				bool isInfluencing = true;
				for( unsigned int i = 0; i < it->point.size(); i++ ) {
					if( itt->point[ i ] >= it->boundingBox[ i ] ) {
						isInfluencing = false;
						break;
					}
				}
				if( isInfluencing ) {
					it->influencingPoints.push_back( itt );
				}
			}
		}
	}
	template<class VectorType>
	bool shouldCompute(Point<VectorType> const& point)const{
		//we do not compute if it is already computed
		if(point.computedExactly) return false;
		std::size_t numPoints = point.influencingPoints.size();
		//point is on its own no need to sample.
		if(numPoints == 0) return true;
		std::size_t numObjectives = point.point.size();
		//runtime already spend on point
		double time = (double)point.noSamples * numObjectives;
		
		//estimate of algo run time
		double algoRunTime = 0.03 * numObjectives * numObjectives * std::pow(numPoints, numObjectives * 0.5 );
		return time > algoRunTime;
	}
	
	template<class VectorType>
	void computeExactly(Point<VectorType>& point)const{
		std::size_t numPoints = point.influencingPoints.size();
		//compute volume of the points inside the box
		std::vector<VectorType> transformedPoints(numPoints);
		for(std::size_t j = 0; j != numPoints; ++j){
			transformedPoints[j] = max(point.influencingPoints[j]->point, point.point);
		}
		HypervolumeCalculator vol;
		double volume = point.boundingBoxVolume - vol(transformedPoints, point.boundingBox);
		point.computedExactly = true;
		point.contributionLowerBound = volume;
		point.contributionUpperBound = volume;
		point.approximatedContribution = volume;
	}
};
}

#endif
