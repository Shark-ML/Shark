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

	/**
	* \brief Simulated binary crossover operator.
	* \tparam PointType Type of the point the operator acts upon.
	*/
	template<typename PointType>
	struct SimulatedBinaryCrossover {

		/**
		 * \brief Make PointType known to the outside world.
		*/
		typedef PointType point_type;

		/**
		* \brief Make this type known to the outside world.
		*/
		typedef SimulatedBinaryCrossover< point_type > this_type;

		/**
		* \brief Default value for the parameter nc.
		*/
		static double DEFAULT_NC() { return( 20. ); } 

		/**
		* \brief Default value for the crossover probability.
		*/
		static double DEFAULT_CROSSOVER_PROBABILITY() { return( 0.5 ); } 

		/**
		* \brief Default c'tor.
		*/
		SimulatedBinaryCrossover() : m_nc( this_type::DEFAULT_NC() ),
			m_prob( this_type::DEFAULT_CROSSOVER_PROBABILITY() ) {}

		/**
		* \brief Initializes the operator for the supplied fitness function.
		* \tparam Function Objective function type. Needs to be model of AbstractVectorSpaceObjectiveFunction.
		* \param [in] f Instance of the objective function to initialize the operator for.
		*/
		template<typename Function>
		void init( const Function & f ) {
			m_prob = 0.5;
			if(!f.isConstrained()){
				m_lower = blas::repeat(-1E20,f.numberOfVariables());
				m_upper = blas::repeat(1E20,f.numberOfVariables());
			}
			else if (f.hasConstraintHandler() &&! f.getConstraintHandler().isBoxConstrained()) {
				typedef BoxConstraintHandler<PointType> ConstraintHandler;
				ConstraintHandler  const& handler = static_cast<ConstraintHandler const&>(f.getConstraintHandler());
				
				m_lower = handler.lower();
				m_upper = handler.upper();

			} else{
				throw SHARKEXCEPTION("[SimulatedBinaryCrossover::init] Algorithm does only allow box constraints");
			}
		}

		/**
		* \brief Mates the supplied individuals.
		* \tparam IndividualType Type of the individual, needs to provider operator* 
		*  for accessing the actual search point.
		* \param [in,out] i1 Individual to be mated.
		* \param [in,out] i2 Individual to be mated.
		*/
		template<typename IndividualType>
		void operator()( IndividualType & i1, IndividualType & i2 ) {		
			double beta, betaQ, alpha, expp, y1 = 0, y2 = 0, u = 0.;

			for( unsigned int i = 0; i < (*i1).size(); i++ ) {

				if( !Rng::coinToss( m_prob ) )
					continue;

				if( (*i2)[i] < (*i1)[i] ) {
					y1 = (*i2)[i];
					y2 = (*i1)[i];
				} else {
					y1 = (*i1)[i];
					y2 = (*i2)[i];
				}

				if( ::fabs(y2 - y1) > 1E-7 ) {					
					// Find beta value
					if( (y1 - m_lower( i )) > (m_upper( i ) - y2) )
						beta = 1 + (2 * (m_upper( i ) - y2) / (y2 - y1));
					else
						beta = 1 + (2 * (y1 - m_lower( i )) / (y2 - y1));


					expp = (m_nc + 1.);
					beta = 1. / beta;

					// Find alpha
					alpha = 2. - ::pow(beta , expp);

					expp = 1. / expp;

					u = Rng::uni( 0., 1. );

					//  -> from Deb's implementation, not contained in any paper
					// do { u = Rng::uni(0, 1); } while(u == 1.);

					if( u <= 1. / alpha ) {
						alpha *= u;
						betaQ = ::pow( alpha, expp );
					} else {
						alpha *= u;
						alpha = 1. / (2. - alpha);
						betaQ = ::pow( alpha, expp );
					}
				} else { // if genes are equal -> from Deb's implementation, not contained in any paper
					betaQ = 1.;
				}

				(*i1)[i] = .5 * ((y1 + y2) - betaQ * (y2 - y1));
				(*i2)[i] = .5 * ((y1 + y2) + betaQ * (y2 - y1));

				//  -> from Deb's implementation, not contained in any paper
				(*i1)[i] = std::max( (*i1)[i], m_lower( i ) );
				(*i1)[i] = std::min( (*i1)[i], m_upper( i ) );
				(*i2)[i] = std::max( (*i2)[i], m_lower( i ) );
				(*i2)[i] = std::min( (*i2)[i], m_upper( i ) );
			}

		}

		/**
		* \brief Serializes this instance to the supplied archive.
		* \tparam Archive The type of the archive the instance shall be serialized to.
		* \param [in,out] archive The archive to serialize to.
		* \param [in] version Version information (optional and not used here).
		*/
		template<typename Archive>
		void serialize( Archive & archive, const unsigned int version ) {
			archive & BOOST_SERIALIZATION_NVP( m_nc );
			archive & BOOST_SERIALIZATION_NVP( m_prob );
			archive & BOOST_SERIALIZATION_NVP( m_upper );
			archive & BOOST_SERIALIZATION_NVP( m_lower );
		}

		double m_nc; ///< Parameter nc.
		double m_prob; ///< Crossover probability.

		point_type m_upper; ///< Upper bound (box constraint).
		point_type m_lower; ///< Lower bound (box constraint).
	};
}

#endif
