//===========================================================================
/*!
*  \brief Objective function DTLZ2
*
*  \author T.Voss, T. Glasmachers, O.Krause
*  \date 2010-2011
*
*  \par Copyright (c) 1998-2007:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR>
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
//===========================================================================
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ2_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ2_H

#include <shark/Core/AbstractBoxConstraintsProvider.h>
#include <shark/ObjectiveFunctions/AbstractMultiObjectiveFunction.h>
#include <shark/Core/Traits/ObjectiveFunctionTraits.h>
#include <shark/Core/Traits/MultiObjectiveFunctionTraits.h>

#include <shark/Core/SearchSpaces/VectorSpace.h>
#include <shark/Rng/GlobalRng.h>

#include <boost/math/special_functions.hpp>

namespace shark {
	/**
	* \brief Implements the benchmark function DTLZ2.
	*
	* See: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.18.7531&rep=rep1&type=pdf
	* The benchmark function exposes the following features:
	*	- Scalable w.r.t. the searchspace and w.r.t. the objective space.
	*	- Highly multi-modal.
	*/
	struct DTLZ2 : public AbstractMultiObjectiveFunction< VectorSpace<double> >,
		public TraitsBoxConstraintsProvider< VectorSpace<double>::PointType, DTLZ2 > {
			typedef AbstractMultiObjectiveFunction< VectorSpace<double> > super;
			typedef TraitsBoxConstraintsProvider< VectorSpace<double>::PointType, DTLZ2 > meta;

			typedef super::ResultType ResultType;
			typedef super::SearchPointType SearchPointType;

			DTLZ2() : super( 2 ) {
				m_features |= CAN_PROPOSE_STARTING_POINT;
				m_features |= IS_CONSTRAINED_FEATURE;
				m_features |= CAN_PROVIDE_CLOSEST_FEASIBLE;
				m_name="DTLZ2";
			}

			void init() {
			}

			ResultType eval( const SearchPointType & x ) const {
				m_evaluationCounter++;

				RealVector value( noObjectives() );

				int k = numberOfVariables() - noObjectives() + 1 ;
				double g = 0.0;

				for( unsigned int i = numberOfVariables() - k + 1; i <= numberOfVariables(); i++ )
					g += ::pow( x( i-1 ) - 0.5, 2 );

				// g = 100 * (k + g);

				for (unsigned int i = 1; i <= noObjectives(); i++) {
					double f = 1. + g;
					for (int j = noObjectives() - i; j >= 1; j--)
						f *= ::cos(x( j-1 ) * M_PI / 2);

					if (i > 1)
						f *= ::sin(x( (noObjectives() - i + 1) - 1) * M_PI / 2);

					value[i-1] = f;
				}

				return( value );
			}

			void proposeStartingPoint( SearchPointType & x ) const {
				meta::proposeStartingPoint( x, m_numberOfVariables );
			}

			bool isFeasible( const SearchPointType & v ) const {
				return( meta::isFeasible( v ) );
			}

			void closestFeasible( SearchPointType & v ) const {
				meta::closestFeasible( v );
			}
	};

	/**
	 * \brief Specializes ObjectiveFunctionTraits for DTLZ2.
	 */
	template<> struct ObjectiveFunctionTraits<DTLZ2> {

		static DTLZ2::SearchPointType lowerBounds( unsigned int n ) {
			return( DTLZ2::SearchPointType( n, 0. ) );
		}

		static DTLZ2::SearchPointType upperBounds( unsigned int n ) {
			return( DTLZ2::SearchPointType( n, 1. ) );
		}

	};

	/**
	 * \brief Specializes MultiObjectiveFunctionTraits for DTLZ2.
	 */
	template<> struct MultiObjectiveFunctionTraits<DTLZ2> {

		/**
		* \brief Models the reference Pareto-front type.
		*/
		typedef std::vector< DTLZ2::ResultType > ParetoFrontType;

		/**
		* \brief Models the reference Pareto-set type.
		*/
		typedef std::vector< DTLZ2::SearchPointType > ParetoSetType;

		static std::vector< DTLZ2::ResultType > referenceFront( std::size_t noPoints, std::size_t n, std::size_t m ) {
			if( m != 2 )
				throw( shark::Exception( "DTLZ2: No reference front for no. of objectives other than 2 available." ) );
			std::vector< DTLZ2::ResultType > result( noPoints, DTLZ2::ResultType( m ) );
			for( std::size_t i = 0; i < result.size(); i++ ) {
				result[ i ][ 0 ] = static_cast< double >( i ) / static_cast< double >( result.size() - 1 );
				result[ i ][ 1 ] = ::sqrt( 1 - boost::math::pow<2>( result[ i ][ 0 ] ) );
			}

			return( result );
		}
	};

	ANNOUNCE_MULTI_OBJECTIVE_FUNCTION( DTLZ2, shark::moo::RealValuedObjectiveFunctionFactory );
}
#endif
