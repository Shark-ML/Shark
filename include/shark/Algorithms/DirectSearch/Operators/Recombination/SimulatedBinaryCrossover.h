/*!
 * 
 *
 * \brief       Simulated binary crossover operator.
 * 
 * 
 *
 * \author      T.Voss
 * \date        2010
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_CROSSOVER_SBX_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_CROSSOVER_SBX_H

#include <shark/Core/Random.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

namespace shark {

	/// \brief Simulated binary crossover operator.
	template<typename PointType>
	struct SimulatedBinaryCrossover {

		SimulatedBinaryCrossover() 
		: m_nc( 20.0 )
		, m_prob( 0.5 ) {}

		/// \brief Initializes the operator for the supplied box-constraints
		void init( RealVector const& lower, RealVector const& upper ) {
			SIZE_CHECK(lower.size() == upper.size());
			m_prob = 1./lower.size();
			m_lower = lower;
			m_upper = upper;
		}

		/// \brief Mates the supplied individuals.
		/// 
		/// \param [in,out] i1 Individual to be mated.
		/// \param [in,out] i2 Individual to be mated.
		template<class randomType, typename IndividualType>
		void operator()(randomType& rng, IndividualType & i1, IndividualType & i2 )const{	
			RealVector& point1 = i1.searchPoint();
			RealVector& point2 = i2.searchPoint();

			for( unsigned int i = 0; i < point1.size(); i++ ) {

				if( !random::coinToss(rng, m_prob ) )
					continue;
				
				double y1 = 0;
				double y2 = 0;
				if( point2[i] < point1[i] ) {
					y1 = point2[i];
					y2 = point1[i];
				} else {
					y1 = point1[i];
					y2 = point2[i];
				}
				
				double betaQ1 = 0.0;
				double betaQ2 = 0.0;
				if( std::abs(y2 - y1) < 1E-7 )continue;//equal
				
				// Find beta value2
				double beta1 = 1 + 2 * (y1 - m_lower( i )) / (y2 - y1);
				double beta2 = 1 + 2 * (m_upper( i ) - y2) / (y2 - y1);
				double expp = m_nc + 1.;
				// Find alpha
				double alpha1 = 2. - std::pow(beta1 , -expp);
				double alpha2 = 2. - std::pow(beta2 , -expp);

				double u = random::uni(rng, 0., 1. );
				alpha1 *=u;
				alpha2 *=u;
				if( u > 1. / alpha1 ) {
					alpha1 = 1. / (2. - alpha1);
				}
				if( u > 1. / alpha2 ) {
					alpha2 = 1. / (2. - alpha2);
				}
				betaQ1 = std::pow( alpha1, 1.0/expp );
				betaQ2 = std::pow( alpha2, 1.0/expp );

				//recombine points
				point1[i] = 0.5 * ((y1 + y2) - betaQ1 * (y2 - y1));
				point2[i] = 0.5 * ((y1 + y2) + betaQ2 * (y2 - y1));
				// randomly swap loci
				if( random::coinToss(rng,0.5) ) std::swap(point1[i], point2[i]);


				//  -> from Deb's implementation, not contained in any paper
				point1[i] = std::max( point1[i], m_lower( i ) );
				point1[i] = std::min( point1[i], m_upper( i ) );
				point2[i] = std::max( point2[i], m_lower( i ) );
				point2[i] = std::min( point2[i], m_upper( i ) );
			}

		}

		/// \brief Serializes this instance to the supplied archive.
		/// \tparam Archive The type of the archive the instance shall be serialized to.
		/// \param [in,out] archive The archive to serialize to.
		/// \param [in] version Version information (optional and not used here).
		template<typename Archive>
		void serialize( Archive & archive, const unsigned int version ) {
			archive & BOOST_SERIALIZATION_NVP( m_nc );
			archive & BOOST_SERIALIZATION_NVP( m_prob );
			archive & BOOST_SERIALIZATION_NVP( m_upper );
			archive & BOOST_SERIALIZATION_NVP( m_lower );
		}

		double m_nc; ///< Parameter nc.
		double m_prob; ///< Crossover probability.

		RealVector m_upper; ///< Upper bound (box constraint).
		RealVector m_lower; ///< Lower bound (box constraint).
	};
}

#endif
