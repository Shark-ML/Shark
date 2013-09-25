/**
*
*  \brief Polynomial mutation operator.
*
*  \author T.Voss
*  \date 2010
*
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 3, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, see <http://www.gnu.org/licenses/>.
*
*/
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_MUTATION_POLYNOMIALMUTATION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_MUTATION_POLYNOMIALMUTATION_H

#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <shark/LinAlg/Base.h>
#include <shark/Rng/GlobalRng.h>

#include <vector>

namespace shark {

	/**
	* \brief Polynomial mutation operator.
	*/
	struct PolynomialMutator {

		/**
		* \brief Typedef for this type.
		*/
		typedef PolynomialMutator this_type;

		/**
		* \brief Default value for the parameter nm.
		*/
		static double DEFAULT_NM() { return( 20. ); } 

		/**
		* \brief Default c'tor.
		*/
		PolynomialMutator() : m_nm( this_type::DEFAULT_NM() ) {}

		/**
		* \brief Initializes the operator for the supplied fitness function.
		* \tparam Function Objective function type. Needs to be model of AbstractVectorSpaceObjectiveFunction.
		* \param [in] f Instance of the objective function to initialize the operator for.
		*/
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

		/**
		* \brief Mutates the supplied individual.
		* \tparam IndividualType Type of the individual, needs to provider operator* 
		*  for accessing the actual search point.
		* \param [in,out] ind Individual to be mutated.
		*/
		template<typename IndividualType>
		void operator()( IndividualType & ind ) {
			double delta, deltaQ, expp,  u = 0.;
           
			for( unsigned int i = 0; i < (*ind).size(); i++ ) {

				if( Rng::coinToss( m_prob ) ) {
					u  = Rng::uni( 0., 1. );
					if( (*ind)[i] <= m_lower( i ) || (*ind)[i] >= m_upper( i ) ) { 
						(*ind)[i] = u * (m_upper( i ) - m_lower( i ) ) + m_lower( i );
					} else {
						// Calculate delta
						if( ((*ind)[i] - m_lower( i ) ) < (m_upper( i ) - (*ind)[i]) )
							delta = ((*ind)[i] - m_lower( i ) ) / (m_upper( i ) - m_lower( i ) );
						else
							delta = (m_upper( i ) - (*ind)[i]) / (m_upper( i ) - m_lower( i ));

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

						(*ind)[i] += deltaQ * (m_upper( i ) - m_lower( i ) );

						//  -> from Deb's implementation, not contained in any paper
						if ((*ind)[i] < m_lower( i ))
							(*ind)[i] = m_lower( i );

						if ((*ind)[i] > m_upper( i ))
							(*ind)[i] = m_upper( i );

					}
				}
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
