//===========================================================================
/*!
*  \brief Objective function DTLZ5
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ5_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DTLZ5_H

#include <shark/Core/AbstractBoxConstraintsProvider.h>
#include <shark/ObjectiveFunctions/AbstractMultiObjectiveFunction.h>
#include <shark/Core/Traits/ObjectiveFunctionTraits.h>
#include <shark/Core/Traits/MultiObjectiveFunctionTraits.h>

#include <shark/Core/SearchSpaces/VectorSpace.h>
#include <shark/Rng/GlobalRng.h>

#include <boost/math/special_functions.hpp>

namespace shark {
	/**
	* \brief Implements the benchmark function DTLZ5.
	*
	* See: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.18.7531&rep=rep1&type=pdf
	* The benchmark function exposes the following features:
	*	- Scalable w.r.t. the searchspace and w.r.t. the objective space.
	*	- Highly multi-modal.
	*/
	struct DTLZ5 : public AbstractMultiObjectiveFunction< VectorSpace<double> >,
		public TraitsBoxConstraintsProvider< VectorSpace<double>::PointType, DTLZ5 > {
			typedef AbstractMultiObjectiveFunction< VectorSpace<double> > super;
			typedef TraitsBoxConstraintsProvider< VectorSpace<double>::PointType, DTLZ5 > meta;

			typedef super::ResultType ResultType;
			typedef super::SearchPointType SearchPointType;



			DTLZ5() : super( 2 ) {
				m_features |= CAN_PROPOSE_STARTING_POINT;
				m_features |= IS_CONSTRAINED_FEATURE;
				m_features |= CAN_PROVIDE_CLOSEST_FEASIBLE;
				m_name = "DTLZ5";
			}

			void init() {
			}

			ResultType eval( const SearchPointType & x ) const {
				m_evaluationCounter++;

				ResultType value( noObjectives() );

				int    k ;
				double g ;

				std::vector<double> phi(noObjectives());

				k = numberOfVariables() - noObjectives() + 1 ;
				g = 0.0 ;

				for (unsigned int i = numberOfVariables() - k + 1; i <= numberOfVariables(); i++)
					g += boost::math::pow<2>( x(i-1) - 0.5 );

				double t = M_PI  / (4 * (1 + g));

				phi[0] = x( 0 ) * M_PI / 2;
				for (unsigned int i = 2; i <= (noObjectives() - 1); i++)
					phi[i-1] = t * (1 + 2 * g * x( i-1 ) );

				for (unsigned int i = 1; i <= noObjectives(); i++) {
					double f = (1 + g);

					for (unsigned int j = noObjectives() - i; j >= 1; j--)
						f *= ::cos(phi[j-1]);

					if (i > 1)
						f *= ::sin(phi[( noObjectives() - i + 1 ) - 1]);

					value[i-1] = f ;
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
	 * \brief Specializes ObjectiveFunctionTraits for DTLZ5.
	 */
	template<> struct ObjectiveFunctionTraits<DTLZ5> {

		static DTLZ5::SearchPointType lowerBounds( unsigned int n ) {
			return( DTLZ5::SearchPointType( n, 0. ) );
		}

		static DTLZ5::SearchPointType upperBounds( unsigned int n ) {
			return( DTLZ5::SearchPointType( n, 1. ) );
		}

	};

	/**
	 * \brief Specializes MultiObjectiveFunctionTraits for DTLZ5.
	 */
	template<> struct MultiObjectiveFunctionTraits<DTLZ5> {
	    
	    /**
	     * \brief Models the reference Pareto-front type.
	     */
	    typedef std::vector< DTLZ5::ResultType > ParetoFrontType;
	    
	    /**
	     * \brief Models the reference Pareto-set type.
	     */
	    typedef std::vector< DTLZ5::SearchPointType > ParetoSetType;
	};
	
	ANNOUNCE_MULTI_OBJECTIVE_FUNCTION( DTLZ5, shark::moo::RealValuedObjectiveFunctionFactory );
}
#endif
