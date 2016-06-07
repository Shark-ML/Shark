/*!
 * 
 *
 * \brief       Determine the volume of the union of objects by an FPRAS
 * 
 * 
 *
 * \author      T.Voss
 * \date        2010-2011
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
#ifndef HYPERVOLUME_APPROXIMATOR_H
#define HYPERVOLUME_APPROXIMATOR_H

#include <shark/Algorithms/DirectSearch/Operators/Domination/ParetoDominance.h>
#include <shark/Statistics/Distributions/MultiNomialDistribution.h>

#include <shark/LinAlg/Base.h>

namespace shark {

/// \brief Implements an FPRAS for approximating the volume of a set of high-dimensional objects.
///  The algorithm is described in
///
/// Bringmann, Karl, and Tobias Friedrich. 
/// "Approximating the volume of unions and intersections of high-dimensional geometric objects."
/// Algorithms and Computation. Springer Berlin Heidelberg, 2008. 436-447.
///
/// The algorithm computes an approximation of the true Volume V, V' that fulfills
/// \f[ P((1-epsilon)V < V' <(1+epsilon)V') < 1-\delta \f]
///
struct HypervolumeApproximator {
	
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {
		archive & BOOST_SERIALIZATION_NVP(m_epsilon);
		archive & BOOST_SERIALIZATION_NVP(m_delta);
	}

	double epsilon()const{
		return m_epsilon;
	}
	double& epsilon(){
		return m_epsilon;
	}
	
	double delta()const{
		return m_delta;
	}
	
	double& delta(){
		return m_delta;
	}

	/// \brief Executes the algorithm.
	/// \param [in] e Function object \f$f\f$to "project" elements of the set to \f$\mathbb{R}^n\f$.
	/// \param [in] set The set \f$S\f$ of points for which the following assumption needs to hold: \f$\forall s \in S: \lnot \exists s' \in S: f( s' ) \preceq f( s ) \f$
	/// \param [in] refPoint The reference point \f$\vec{r} \in \mathbb{R}^n\f$ for the hypervolume calculation, needs to fulfill: \f$ \forall s \in S: s \preceq \vec{r}\f$. .
	template<typename Set, typename VectorType >
	double operator()( Set const& points, VectorType const& refPoint){
		std::size_t noPoints = points.size();

		if( noPoints == 0 )
			return 0;

		// runtime (O.K: added static_cast to prevent warning on VC10)
		boost::uint_fast64_t maxSamples=static_cast<boost::uint_fast64_t>( 12. * std::log( 1. / delta() ) / std::log( 2. ) * noPoints/sqr(epsilon()) ); 

		// calc separate volume of each box
		VectorType vol( noPoints, 1. );
		for( std::size_t p = 0; p != noPoints; ++p) {
			//guard against points which are worse than the reference
			if(min(refPoint - points[p] ) < 0){
				throw SHARKEXCEPTION("HyperVolumeApproximator: points must be better than reference point");
			}
			//taking the sum of logs instead of their product is numerically more stable in large dimensions were intermediate volumes can become very small or large
			vol[p] = std::exp(sum(log(refPoint[ 0 ] - points[p] )));
		}
		//calculate total sum of volumes
		double totalVolume = sum(vol);
		
		VectorType rndpoint( refPoint );
		boost::uint_fast64_t samples_sofar=0;
		boost::uint_fast64_t round=0;
		
		//we pick points randomly based on their volume
		MultiNomialDistribution pointDist(vol);

		while (true)
		{
			// sample ROI based on its volume. the ROI is defined as the Area between the reference point and a point in the front.
			auto point = points.begin() + pointDist(Rng::globalRng);
			
			// sample point in ROI   
			for( std::size_t i = 0; i < rndpoint.size(); i++ ){
				rndpoint[i] = (*point )[i] + Rng::uni() * ( refPoint[i] - (*point)[i] );
			}

			while (true)
			{
				if (samples_sofar>=maxSamples) return maxSamples * totalVolume / noPoints / round;
				auto candidate = points.begin() + static_cast<std::size_t>(noPoints*Rng::uni());
				samples_sofar++;
				DominanceRelation rel = dominance(*candidate, rndpoint);
				if (rel == LHS_DOMINATES_RHS || rel == EQUIVALENT) break;

			} 

			round++;
		}
	}
	
private:
	double m_epsilon;
	double m_delta;
};
}

#endif // HYPERVOLUME_APPROXIMATOR_H
