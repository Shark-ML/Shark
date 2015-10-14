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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_MUTATION_POLYNOMIALMUTATION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_MUTATION_POLYNOMIALMUTATION_H

#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <shark/LinAlg/Base.h>
#include <shark/Rng/GlobalRng.h>

namespace shark {

	/// \brief Polynomial mutation operator.
	struct PolynomialMutator {

		/// \brief Default c'tor.
		PolynomialMutator() : m_nm( 20.0 ) {}

		/// \brief Initializes the operator for the supplied fitness function.
		///
		/// \param [in] f Instance of the objective function to initialize the operator for.
		template<typename Function>
		void init( const Function & f ) {
                        m_prob = 1./f.numberOfVariables();
			if(!f.isConstrained()){
				m_lower = blas::repeat(-1E20,f.numberOfVariables());
				m_upper = blas::repeat(1E20,f.numberOfVariables());
			}
			else if (f.hasConstraintHandler() && f.getConstraintHandler().isBoxConstrained()) {
				typedef BoxConstraintHandler<typename Function::SearchPointType> ConstraintHandler;
				ConstraintHandler  const& handler = static_cast<ConstraintHandler const&>(f.getConstraintHandler());
				
				m_lower = handler.lower();
				m_upper = handler.upper();

			} else{
				throw SHARKEXCEPTION("[PolynomialMutator::init] Algorithm does only allow box constraints");
			}                    
		}

		/// \brief Mutates the supplied individual.
		///
		///  for accessing the actual search point.
		/// \param [in,out] ind Individual to be mutated.
		template<typename IndividualType>
		void operator()( IndividualType & ind ) {
			double delta, deltaQ, expp,  u = 0.;
			
			RealVector& point = ind.searchPoint();
           
			for( unsigned int i = 0; i < point.size(); i++ ) {

				if( Rng::coinToss( m_prob ) ) {
					u  = Rng::uni( 0., 1. );
					if( point[i] <= m_lower( i ) || point[i] >= m_upper( i ) ) { 
						point[i] = u * (m_upper( i ) - m_lower( i ) ) + m_lower( i );
					} else {
						// Calculate delta
						if( (point[i] - m_lower( i ) ) < (m_upper( i ) - point[i]) )
							delta = (point[i] - m_lower( i ) ) / (m_upper( i ) - m_lower( i ) );
						else
							delta = (m_upper( i ) - point[i]) / (m_upper( i ) - m_lower( i ));

						delta = 1. - delta;
						expp  = (m_nm + 1.);
						delta = ::pow(delta , expp);
						expp  = 1. / expp;

						if( u <= .5 ) {
							deltaQ =  2. * u + (1 - 2. * u) * delta;
							deltaQ = ::pow(deltaQ, expp) - 1. ;
						} else {
							deltaQ = 2. - 2. * u + 2. * (u  - .5) * delta;
							deltaQ = 1. - ::pow(deltaQ , expp);
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
