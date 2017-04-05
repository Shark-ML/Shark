/*!
 * 
 *
 * \brief       Polynomial mutation operator.
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_MUTATION_POLYNOMIALMUTATION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_MUTATION_POLYNOMIALMUTATION_H

#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <shark/LinAlg/Base.h>
#include <shark/Core/Random.h>

namespace shark {

	/// \brief Polynomial mutation operator.
	struct PolynomialMutator {

		/// \brief Default c'tor.
		PolynomialMutator() : m_nm( 25.0 ) {}

		/// \brief Initializes the operator for the supplied box-constraints
		void init( RealVector const& lower, RealVector const& upper ) {
			SIZE_CHECK(lower.size() == upper.size());
			m_prob = 1./lower.size();
			m_lower = lower;
			m_upper = upper;
		}

		/// \brief Mutates the supplied individual.
		///
		///  for accessing the actual search point.
		/// \param [in,out] ind Individual to be mutated.
		template<typename IndividualType>
		void operator()(random::rng_type& rng, IndividualType & ind )const{
			RealVector& point = ind.searchPoint();
           
			for( unsigned int i = 0; i < point.size(); i++ ) {

				if( random::coinToss(rng, m_prob ) ) {
					if( point[i] < m_lower( i ) || point[i] > m_upper( i ) ) { 
						point[i] = random::uni(rng,m_lower(i),m_upper(i));
					} else {
						// Calculate normalized distance from boundaries
						double delta1 = (m_upper( i ) - point[i]) / (m_upper( i ) - m_lower( i ));
						double delta2 = (point[i] - m_lower( i ) ) / (m_upper( i ) - m_lower( i ));
						
						//compute change in delta
						double deltaQ=0;
						double u = random::uni(rng,0,1);
						if( u <= .5 ) {
							double delta = std::pow(delta1 , m_nm + 1.);
							deltaQ =  2.0 * u + (1.0 - 2.0 * u) * delta;
							deltaQ = std::pow(deltaQ, 1.0/(m_nm+1.0)) - 1. ;
						} else {
							double delta = std::pow(delta2 , m_nm + 1.);
							deltaQ = 2 * (1- u) + 2. * (u  - .5) * delta;
							deltaQ = 1. - std::pow(deltaQ , 1.0/(m_nm+1.0));
						}

						point[i] += deltaQ * (m_upper( i ) - m_lower( i ) );

						//  -> from Deb's implementation, not contained in any paper
						if (point[i] < m_lower( i ))
							point[i] = m_lower( i );

						if (point[i] > m_upper( i ))
							point[i] = m_upper( i );

					}
				}
			}                        
		}

		/// \brief Serializes this instance to the supplied archive.
		///
		/// \param [in,out] archive The archive to serialize to.
		/// \param [in] version Version information (optional and not used here).
		template<typename Archive>
		void serialize( Archive & archive, const unsigned int version ) {
			archive & BOOST_SERIALIZATION_NVP( m_nm );
			archive & BOOST_SERIALIZATION_NVP( m_prob );
			archive & BOOST_SERIALIZATION_NVP( m_upper );
			archive & BOOST_SERIALIZATION_NVP( m_lower );
		}

		double m_nm;
		double m_prob;

		RealVector m_upper;
		RealVector m_lower;
	};
}

#endif 
