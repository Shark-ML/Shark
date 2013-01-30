//===========================================================================
/*!
*  \brief Objective function DTLZ1
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ1_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ1_H

#include <shark/Core/AbstractBoxConstraintsProvider.h>
#include <shark/ObjectiveFunctions/AbstractMultiObjectiveFunction.h>
#include <shark/Core/Traits/ObjectiveFunctionTraits.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>
#include <shark/Rng/GlobalRng.h>

namespace shark {

	/**
	* \brief Implements the benchmark function DTLZ1.
	*
	* See: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.18.7531&rep=rep1&type=pdf
	* The benchmark function exposes the following features:
	*	- Scalable w.r.t. the searchspace and w.r.t. the objective space.
	*	- Highly multi-modal.
	*/
	struct DTLZ1 : 
		public AbstractMultiObjectiveFunction< VectorSpace<double> >,
		public TraitsBoxConstraintsProvider< VectorSpace<double>::PointType, DTLZ1 > {
			typedef AbstractMultiObjectiveFunction< VectorSpace<double> > super;
			typedef TraitsBoxConstraintsProvider< VectorSpace<double>::PointType, DTLZ1 > meta;

			typedef super::ResultType ResultType;
			typedef super::SearchPointType SearchPointType;

			DTLZ1() : super( 2 ) {
				m_features |= CAN_PROPOSE_STARTING_POINT;
				m_features |= IS_CONSTRAINED_FEATURE;
				m_features |= CAN_PROVIDE_CLOSEST_FEASIBLE;
				m_name="DTLZ1";
			}

			void init() {
			}

			ResultType eval( const SearchPointType & x ) const {
				m_evaluationCounter++;

				ResultType value( noObjectives() );

				int k = numberOfVariables() - noObjectives() + 1 ;
				// TODO: Check k
				double g = 0.0;

				for( unsigned int i = numberOfVariables() - k + 1; i <= numberOfVariables(); i++ )
				    g += sqr( x( i-1 ) - 0.5 ) - std::cos( 20 * M_PI * ( x( i-1 ) - 0.5) );

				g = 100 * (k + g);

				for (unsigned int i = 1; i <= noObjectives(); i++) {
					double f = 0.5 * (1 + g);
					for( unsigned int j = noObjectives() - i; j >= 1; j--)
						f *= x( j-1 );

					if (i > 1)
						f *= 1 - x( (noObjectives() - i + 1) - 1);

					value[i-1] = f;
				}

				return value;
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
	* \brief Specializes objective function traits for the function DTLZ1.
	*/
	template<> 
	struct ObjectiveFunctionTraits<DTLZ1> {

		static DTLZ1::SearchPointType lowerBounds( unsigned int n ) {
                        return DTLZ1::SearchPointType( n, 0. );
		}

		static DTLZ1::SearchPointType upperBounds( unsigned int n ) {
			return DTLZ1::SearchPointType( n, 1. );
		}

	};

	ANNOUNCE_MULTI_OBJECTIVE_FUNCTION( DTLZ1, shark::moo::RealValuedObjectiveFunctionFactory );
}

#endif
