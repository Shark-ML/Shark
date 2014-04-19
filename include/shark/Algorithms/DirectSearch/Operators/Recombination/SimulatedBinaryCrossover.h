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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_CROSSOVER_SBX_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_CROSSOVER_SBX_H

#include <shark/Rng/GlobalRng.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>

namespace shark {

	/// \brief Simulated binary crossover operator.
	template<typename PointType>
	struct SimulatedBinaryCrossover {

		SimulatedBinaryCrossover() 
		: m_nc( 20.0 )
		, m_prob( 0.5 ) {}

		/// \brief Initializes the operator for the supplied fitness function.
		///
		/// \param [in] f Instance of the objective function to initialize the operator for.
		template<typename Function>
		void init( const Function & f ) {
			m_prob = 0.5;
			if(!f.isConstrained()){
				m_lower = blas::repeat(-1E20,f.numberOfVariables());
				m_upper = blas::repeat(1E20,f.numberOfVariables());
			}
			else if (f.hasConstraintHandler() && f.getConstraintHandler().isBoxConstrained()) {
				typedef BoxConstraintHandler<PointType> ConstraintHandler;
				ConstraintHandler  const& handler = static_cast<ConstraintHandler const&>(f.getConstraintHandler());
				
				m_lower = handler.lower();
				m_upper = handler.upper();

			} else{
				throw SHARKEXCEPTION("[SimulatedBinaryCrossover::init] Algorithm does only allow box constraints");
			}
		}

		/// \brief Mates the supplied individuals.
		/// 
		/// \param [in,out] i1 Individual to be mated.
		/// \param [in,out] i2 Individual to be mated.
		template<typename IndividualType>
		void operator()( IndividualType & i1, IndividualType & i2 ) {	
			RealVector& point1 = i1.searchPoint();
			RealVector& point2 = i2.searchPoint();

			for( unsigned int i = 0; i < point1.size(); i++ ) {

				if( !Rng::coinToss( m_prob ) )
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
				
				double betaQ = 0.0;
				if( std::abs(y2 - y1) > 1E-7 ) {					
					// Find beta value
					double beta = 0;
					if( (y1 - m_lower( i )) > (m_upper( i ) - y2) )
						beta = 1 + 2 * (m_upper( i ) - y2) / (y2 - y1);
					else
						beta = 1 + 2 * (y1 - m_lower( i )) / (y2 - y1);


					double expp = m_nc + 1.;
					// Find alpha
					double alpha = 2. - std::pow(1.0/beta , expp);
					double u = Rng::uni( 0., 1. );

					if( u <= 1. / alpha ) {
						alpha *= u;
						betaQ = std::pow( alpha, 1.0/expp );
					} else {
						alpha *= u;
						alpha = 1. / (2. - alpha);
						betaQ = std::pow( alpha, 1.0/expp );
					}
				} else { // if genes are equal -> from Deb's implementation, not contained in any paper
					betaQ = 1.;
				}

				point1[i] = 0.5 * ((y1 + y2) - betaQ * (y2 - y1));
				point2[i] = 0.5 * ((y1 + y2) + betaQ * (y2 - y1));

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
